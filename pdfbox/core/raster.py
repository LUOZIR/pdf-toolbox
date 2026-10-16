"""图片型 PDF 的编辑路径：实测墨迹 → 覆盖原像素 → 重绘文字。

与文本型 PDF 的本质区别：

    文本型：apply_redactions 从内容流里真正删除字形
            → 未改动区域可以做到字节级零差异
    图片型：文字是位图的一部分，没有任何字形可删
            → 只能用背景覆盖，原像素物理上无法移除

这个差异不可抹平，只能在产品里如实告知。所以本模块的设计目标是
"让覆盖看不出来"，而不是"假装无损"。

★ 核心原则：一切以像素实测为准，不信 OCR。
  OCR 的文本行框在本项目的真实样本上只覆盖了该行墨迹宽度的 24%
  （框宽 42.7pt，实际 175pt）。用它当覆盖区域会留下大片残影，
  用它反推字号会把新字画小一半。所以覆盖范围和字号都从墨迹实测得出。

覆盖策略分两档：
    纯色背景 → 直接填色，快且绝对干净
    复杂背景 → cv2.inpaint 从周边像素修补，并就地替换承载文字的图像对象
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .fonts import _find_file
from .model import Block

# 覆盖范围相对墨迹的外扩（pt）
PAD = 1.2
# 背景均匀性阈值：通道标准差超过此值判为复杂背景
UNIFORM_STD_MAX = 14.0
# 自动切粗体的墨量比阈值。
#
# ★ 这个自动判定在本项目的真实样本上被证伪了，所以阈值提得极高、实际不会触发：
#   原文墨量实测是 Arial 常规体的 1.55 倍，看上去"该用粗体"，
#   但把常规体重绘和粗体重绘并排给视觉复核，结论是**常规体明显更接近原文**。
#   原因是原文是缩放过的位图截图，边缘抗锯齿把墨量整体抬高了约 55%，
#   这个系统性偏差没法靠阈值区分开。
#
# 所以字重改成由用户显式选择（--weight bold / 界面上的字重下拉），
# 默认常规体。不把不可靠的自动判定当默认行为。
BOLD_SWITCH_RATIO = 5.0

# inpaint 的掩膜宽度上限（占图像宽度比例）。
# 超过此比例就不该用 inpaint：掩膜几乎横跨整幅图时，左右都没有可用像素
# 可参考，cv2.inpaint 只能从上下方向外推，实测会产出大片黑色伪影
# （一条横贯标题栏的改动把幻灯片两侧涂黑了）。这种情况用纯色填充更可靠。
INPAINT_MAX_MASK_RATIO = 0.6
# 墨迹判定：比背景暗这么多才算文字
INK_DARK_DELTA = 45
# 水平追踪文字段时允许的字符间空隙 = 比例 × 墨迹高度
BRIDGE_RATIO = 1.3
# 水平追踪的最远距离 = 比例 × 原框宽。
#
# ★ 这个值默认应该是 0（不追踪）。实测 OCR 的检测框本身相当准：
#   22 个块的平均内边距只有左右 0.8pt、上下 1.3~1.5pt，没有系统偏移。
#   早期为了修个别严重失真的框而开启大幅追踪，结果把本来准确的框
#   扩到了相邻内容上——反而制造了"文字框错位"这个更大的问题。
#   所以默认不扩展：以 OCR 框为准，只在框内收紧到真实墨迹。
#   确实需要扩展时由用户手工拖动覆盖框，或显式传 expand_cover=True。
MAX_EXPAND_RATIO = 0.0
# 实测倍率：足够高才能量准墨迹边界
MEASURE_ZOOM = 6.0

LATIN_INK_RATIO = 0.95      # 拉丁文含升降部时，墨迹高 ≈ 字号 × 0.95
CJK_INK_RATIO = 0.88

# 笔画粗细比的参照区间（供界面给用户建议用，不做自动判定）
REGULAR_STROKE = 0.13
BOLD_STROKE = 0.20


def _is_cjk_text(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


# --------------------------------------------------------------- 字号标定

_probe_cache: dict[tuple, float] = {}


def measure_text_ink_height(font: pymupdf.Font, text: str, size: float,
                            font_key: str = "") -> float:
    """离屏排一次文字，量出它的实际墨迹高度。

    为什么要真排一次：墨迹高度受字符组成影响极大 —— "ACE"（纯大写）
    和 "age"（带升降部）在同样字号下的墨迹高度能差一倍。
    按固定比例从原文字高度反推新文字字号，遇到字符组成不同就会明显偏大或偏小。
    """
    key = (font_key or getattr(font, "name", "?"), text, round(size, 2))
    if key in _probe_cache:
        return _probe_cache[key]

    probe_zoom = 8.0
    doc = pymupdf.open()
    page = doc.new_page(width=900, height=240)
    try:
        tw = pymupdf.TextWriter(page.rect)
        tw.append((20, 170), text, font=font, fontsize=size)
        tw.write_text(page, color=(0, 0, 0))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(probe_zoom, probe_zoom),
                              alpha=False)
        import numpy as np
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        mask = arr.mean(axis=2) < 160
        if not mask.any():
            h = 0.0
        else:
            rows = np.where(mask.any(axis=1))[0]
            h = (rows.max() - rows.min() + 1) / probe_zoom
    finally:
        doc.close()

    if len(_probe_cache) > 512:
        _probe_cache.clear()
    _probe_cache[key] = h
    return h


def calibrate_size(
    font: pymupdf.Font, text: str, target_ink_height: float,
    initial: float, font_key: str = "", max_iter: int = 4,
) -> tuple[float, bool]:
    """调整字号，使该字体排出的这段文字，墨迹高度等于原文实测高度。

    这是"用相似字体重绘到原位置"的关键一步：只有墨迹高度对上了，
    新文字看起来才和原文一个字号。返回 (字号, 是否收敛)，
    不收敛说明字体形状差异太大，调用方应当告警。
    """
    if target_ink_height <= 0.5 or not text.strip():
        return initial, True

    size = max(3.0, initial)
    for _ in range(max_iter):
        h = measure_text_ink_height(font, text, size, font_key)
        if h <= 0.1:
            return size, False
        ratio = target_ink_height / h
        if abs(ratio - 1.0) <= 0.02:
            return size, True
        new_size = max(3.0, min(400.0, size * ratio))
        if abs(new_size - size) < 0.02:
            return new_size, True
        size = new_size
    h = measure_text_ink_height(font, text, size, font_key)
    return size, (h > 0.1 and abs(target_ink_height / h - 1.0) <= 0.06)


# --------------------------------------------------------------- 字重判定


def _stroke_ratio_from_mask(mask) -> float:
    """从二值墨迹掩膜估计「笔画粗细 / 墨迹高度」之比。

    做法：笔画粗细 ≈ 墨迹像素数 ÷ 水平方向笔画穿越次数。
    每一行里每次「白→黑」的跳变都对应一次穿过笔画，统计总跳变数即可。
    这个比值与字号无关，只反映字重：常规体约 0.08~0.13，粗体约 0.14~0.22。
    """
    import numpy as np
    if mask.size == 0 or not mask.any():
        return 0.0
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return 0.0
    height = rows.max() - rows.min() + 1
    if height <= 0:
        return 0.0
    crossings = 0
    for r in range(rows.min(), rows.max() + 1):
        row = mask[r].astype(np.int8)
        crossings += int(np.sum(np.diff(row) == 1))
    if crossings == 0:
        return 0.0
    return (mask.sum() / crossings) / height


def _rule_rows(gray, bg_level: float, ink_mask):
    """找出"像横线"的像素行 —— 表格填空线、下划线。

    判据是**自适应**的两条，缺一不可：

      1. 这一行存在一条明显长于其他行的连续墨迹段
         （取所有行长度的中位数，超过它的 3.5 倍才算）
      2. 这条长段本身占行宽的一定比例

    为什么不能只用固定的"贯穿比例"：实测那份文档的填空线是很淡的灰线，
    用主阈值的掩膜去测只覆盖 31% 的宽度，过不了 55% 的门槛，
    横线一个都没检测到。所以这里用**更宽松的阈值**单独扫一遍，
    再靠"显著长于其他行"这个相对判据把它挑出来。

    为什么要专门识别它：「家庭住址　示例省示例市…」这类填写行下方常有一条灰色填空线，
    被当成墨迹算进高度后，字号推算成 15.15pt（实际 13.24pt），偏大 14%；
    覆盖时还会把这条线一起擦掉。
    """
    import numpy as np
    h, w = ink_mask.shape
    if w == 0 or h == 0:
        return np.zeros(h, dtype=bool)

    # 用更宽松的阈值再扫一遍，把淡线也抓出来
    lenient = gray < (bg_level - 12)
    runs: list[int] = []
    for r in range(h):
        row = lenient[r]
        best = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
        runs.append(best)

    nz = [r for r in runs if r > 0]
    if not nz:
        return np.zeros(h, dtype=bool)
    med = float(np.median(nz))
    thr = max(3.5 * med, 0.20 * w)

    is_rule = np.array([r >= thr for r in runs], dtype=bool)
    # 连续的行组才算法线；孤零零一行多半是文字里碰巧连上的一笔
    out = np.zeros(h, dtype=bool)
    i = 0
    while i < h:
        if is_rule[i]:
            j = i
            while j < h and is_rule[j]:
                j += 1
            if j - i >= 1:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _ink_mass(gray, bg_level: float, text_level: float) -> float:
    """墨量：灰度偏离背景的程度，在文字包围盒内的平均值，再归一化。

    ★ 为什么不用"墨迹像素占比"：原文是缩放过的截图，笔画边缘带大片
      抗锯齿过渡像素，二值化时全被算成墨迹，把密度虚高到 0.3~0.6；
      而参考探针是锐利的矢量渲染，同样字重只有 0.24~0.31。
      拿这两个数对比，几乎所有原文字都被误判成粗体。

      墨量对模糊和抗锯齿是稳健的 —— 模糊会把笔画摊开，但不改变总的墨量，
      所以两个来源可以直接比。
    """
    import numpy as np
    if gray.size == 0:
        return 0.0
    depth = np.clip(bg_level - gray, 0.0, None)
    denom = max(1.0, abs(bg_level - text_level))
    return float(depth.mean()) / denom


def measure_ink_density(font: pymupdf.Font, text: str, size: float,
                        font_key: str = "", effective_ppp: float = 0.0) -> float:
    """离屏排一次文字，量出它的墨量（用于和原文实测墨量比对字重）。

    ★ effective_ppp 是关键：原文是缩放过的位图截图（本例约 2.77 像素/pt），
      笔画边缘有大量抗锯齿过渡；而探针默认在 8 像素/pt 下渲染，边缘锐利。
      两者直接比墨量，原文永远显得更"粗"——实测把所有正文都误判成粗体。
      所以探针先按高分辨率渲染，再降采样到原文的有效分辨率，
      让两边的边缘模糊程度一致，墨量才有可比性。
    """
    key = ("mass", font_key or getattr(font, "name", "?"), text,
           round(size, 2), round(effective_ppp, 2))
    if key in _probe_cache:
        return _probe_cache[key]
    import numpy as np
    gray = _probe_gray(font, text, size, effective_ppp)
    if gray is None or gray.size == 0:
        val = 0.0
    else:
        mask = gray < 200
        if not mask.any():
            val = 0.0
        else:
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            sub = gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            val = _ink_mass(sub, 255.0, 0.0)
    if len(_probe_cache) > 512:
        _probe_cache.clear()
    _probe_cache[key] = val
    return val


def _probe_gray(font: pymupdf.Font, text: str, size: float,
                effective_ppp: float = 0.0):
    """在离屏页面上排一次文字，返回灰度图。

    effective_ppp > 0 时，把渲染结果降采样到该分辨率，
    以匹配原文位图的有效清晰度。
    """
    import numpy as np
    zoom = 8.0
    if effective_ppp > 0:
        zoom = max(zoom, effective_ppp)
    doc = pymupdf.open()
    page = doc.new_page(width=900, height=240)
    try:
        tw = pymupdf.TextWriter(page.rect)
        tw.append((20, 170), text, font=font, fontsize=size)
        tw.write_text(page, color=(0, 0, 0))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        gray = arr.mean(axis=2)

        # 降采样到原文的有效分辨率，让抗锯齿模糊程度一致
        if effective_ppp > 0 and zoom > effective_ppp * 1.05:
            from PIL import Image
            im = Image.fromarray(gray.astype(np.uint8))
            w = max(1, int(round(im.width * effective_ppp / zoom)))
            h = max(1, int(round(im.height * effective_ppp / zoom)))
            gray = np.asarray(im.resize((w, h), Image.BOX)).astype(float)
        return gray
    finally:
        doc.close()


def measure_stroke_ratio(font: pymupdf.Font, text: str, size: float,
                         font_key: str = "") -> float:
    """在离屏页面上排一次文字，量出它的笔画粗细比。"""
    key = ("stroke", font_key or getattr(font, "name", "?"), text, round(size, 2))
    if key in _probe_cache:
        return _probe_cache[key]

    zoom = 8.0
    doc = pymupdf.open()
    page = doc.new_page(width=900, height=240)
    try:
        tw = pymupdf.TextWriter(page.rect)
        tw.append((20, 170), text, font=font, fontsize=size)
        tw.write_text(page, color=(0, 0, 0))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        import numpy as np
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        mask = arr.mean(axis=2) < 160
        val = _stroke_ratio_from_mask(mask)
    finally:
        doc.close()

    if len(_probe_cache) > 512:
        _probe_cache.clear()
    _probe_cache[key] = val
    return val


# 字重变体：给定一个字体文件，推测同族的其他字重文件
_WEIGHT_PAIRS: dict[str, list[str]] = {
    "arial":      ["arial.ttf", "arialbd.ttf"],
    "arialbd":    ["arialbd.ttf", "arial.ttf"],
    "calibri":    ["calibri.ttf", "calibrib.ttf"],
    "calibrib":   ["calibrib.ttf", "calibri.ttf"],
    "times":      ["times.ttf", "timesbd.ttf"],
    "timesbd":    ["timesbd.ttf", "times.ttf"],
    "msyh":       ["msyh.ttc", "msyhbd.ttc"],
    "msyhbd":     ["msyhbd.ttc", "msyh.ttc"],
    "deng":       ["Deng.ttf", "Dengb.ttf"],
    "dengb":      ["Dengb.ttf", "Deng.ttf"],
    "simsun":     ["simsun.ttc", "simhei.ttf"],       # 宋体没有粗体，用黑体近似
    "simhei":     ["simhei.ttf", "msyhbd.ttc"],
    "notosanssc": ["NotoSansSC-VF.ttf"],
    "notoserifsc":["NotoSerifSC-VF.ttf"],
}


def weight_variants(font_file: str) -> list[str]:
    """列出同族的候选字重文件，供按笔画粗细挑选。"""
    stem = Path(font_file).stem.lower()
    if stem in _WEIGHT_PAIRS:
        return _WEIGHT_PAIRS[stem]
    # 名字里含 bold/bd 的，试着去掉后找常规体
    for tag in ("-bold", "bold", "bd", "-b"):
        if tag in stem:
            base = stem.replace(tag, "").strip("-_")
            for key, cands in _WEIGHT_PAIRS.items():
                if key == base:
                    return cands
    # 兜底：同族常规体本身
    return [Path(font_file).name]


def _effective_ppp(page: pymupdf.Page, rect) -> float:
    """算出该区域原文位图的有效分辨率（像素 / pt）。

    用于让字重探针在与原文相同的清晰度下比较墨量。
    """
    host = find_host_image(page, rect)
    if host is None or host.bbox[2] <= host.bbox[0]:
        return 0.0
    return host.width / (host.bbox[2] - host.bbox[0])


def pick_weight(
    base_font_file: str, family_candidates: list[str], text: str,
    target_stroke_ratio: float, size: float, target_density: float = 0.0,
    effective_ppp: float = 0.0,
) -> tuple[str, float, str]:
    """在候选字体文件里挑一个笔画粗细最接近原文的。

    原文若是粗体而我用常规体重绘，即使字高完全一致，看起来也会"偏大偏松散"——
    实测中被明确指出「上图字体较粗且紧凑，下图明显更舒展」。
    所以字重必须按实测笔画粗细来选，而不是一律用常规体。

    返回 (选中的文件路径, 该字体的笔画比, 字重标签)。
    """
    if target_density <= 0.0 and target_stroke_ratio <= 0.0:
        return base_font_file, 0.0, "regular"

    # 先用"常规体"作基准做一次测量，再判断是否需要切粗体。
    # 不能直接比绝对值 —— 原文的字体可能只是比基准字体本身更重
    # （本例幻灯片的正文用的是几何无衬线体，墨量天然高于 Arial），
    # 那不代表它在本族里属于粗体。所以要求"明显更重"才切。
    base_path = _find_file(Path(base_font_file).name) or base_font_file
    try:
        base_font = pymupdf.Font(fontfile=base_path)
        base_metric = measure_ink_density(base_font, text, size,
                                          Path(base_path).name, effective_ppp)
    except Exception:
        base_metric = 0.0
    if base_metric > 0:
        ratio_to_base = target_density / base_metric
        if ratio_to_base <= BOLD_SWITCH_RATIO:
            return base_path, base_metric, "regular"

    best_file = base_font_file
    best_ratio = 0.0
    best_err = float("inf")
    best_label = "regular"

    for cand in family_candidates:
        path = _find_file(cand)
        if not path:
            continue
        try:
            f = pymupdf.Font(fontfile=path)
        except Exception:
            continue
        # 以墨迹密度为唯一判据：它总是可算，且对字重敏感
        metric = measure_ink_density(f, text, size, Path(path).name, effective_ppp)
        target = target_density
        if metric <= 0 or target <= 0:
            continue
        err = abs(metric / target - 1.0)
        if err < best_err:
            best_err = err
            best_file = path
            best_ratio = metric
            low = Path(path).stem.lower()
            if "bd" in low or "bold" in low or "hei" in low:
                best_label = "bold"
            else:
                best_label = "regular"

    return best_file, best_ratio, best_label


@dataclass
class InkMetrics:
    """从像素墨迹实测出的文字度量。"""
    rect: tuple[float, float, float, float]      # 墨迹实际包围盒（pt）
    height: float
    color: tuple[float, float, float]            # 文字颜色（0-1）
    bg: tuple[int, int, int]                     # 背景色（0-255）
    bg_std: float                                # 背景均匀性
    ink_pixels: int
    run_extended: float = 0.0                    # 相对原始框向外扩展了多少 pt
    stroke_ratio: float = 0.0                    # 笔画粗细 / 墨迹高度，用于判定字重
    ink_density: float = 0.0                     # 墨量，字重判定的核心指标
    rule_span: float = 0.0                       # 检测到的横线（下划线）厚度，0=没有

    def size_for(self, text: str) -> float:
        ratio = CJK_INK_RATIO if _is_cjk_text(text) else LATIN_INK_RATIO
        return max(4.0, self.height / ratio)

    def summary(self) -> str:
        return (f"墨迹 x={self.rect[0]:.1f}..{self.rect[2]:.1f}"
                f"(宽 {self.rect[2]-self.rect[0]:.1f}pt) 高 {self.height:.1f}pt"
                + (f" 较原框外扩 {self.run_extended:.1f}pt" if self.run_extended > 1 else ""))


@dataclass
class RasterEditResult:
    page_no: int
    old_text: str
    new_text: str
    method: str
    ink: InkMetrics | None = None
    used_size: float = 0.0
    covered_rects: list[tuple[float, float, float, float]] = field(default_factory=list)
    lines_after: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        m = "周边像素修补(inpaint)" if self.method == "inpaint" else "纯色填充"
        return (f"p{self.page_no+1} 覆盖方式={m}  字号={self.used_size:.1f}pt  "
                f"覆盖 {len(self.covered_rects)} 处")


# --------------------------------------------------------------- 像素实测


def measure_ink(
    page: pymupdf.Page,
    rect: tuple[float, float, float, float],
    *,
    zoom: float = MEASURE_ZOOM,
    y_pad_ratio: float = 0.25,
    bridge_ratio: float = BRIDGE_RATIO,
    expand: bool = True,
) -> InkMetrics:
    """实测 OCR 框附近文字墨迹的真实范围、颜色、背景。

    expand=True 时沿水平方向追踪文字连续段 —— 这是修复 OCR 框偏小的关键：
    从原框中心向左右走，只要列上有墨迹就继续，遇到超过一个字高的空白才停。
    """
    import numpy as np

    x0, y0, x1, y1 = rect
    h = max(1.0, y1 - y0)

    band_y0 = max(0.0, y0 - h * y_pad_ratio)
    band_y1 = min(page.rect.height, y1 + h * y_pad_ratio)

    span = max(1.0, x1 - x0)
    search_x0 = max(0.0, x0 - span * MAX_EXPAND_RATIO) if expand else x0
    search_x1 = min(page.rect.width, x1 + span * MAX_EXPAND_RATIO) if expand else x1
    search_x1 = max(search_x1, search_x0 + 1.0)

    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3]

    sub = arr[max(0, int(band_y0 * zoom)):int(band_y1 * zoom),
              max(0, int(search_x0 * zoom)):int(search_x1 * zoom)]
    if sub.size == 0:
        return InkMetrics(rect=rect, height=h, color=(0.0, 0.0, 0.0),
                          bg=(255, 255, 255), bg_std=0.0, ink_pixels=0)

    gray = sub.mean(axis=2).astype(float)

    # ---- 第一步：在紧贴 OCR 框的小区域上标定「背景 / 文字色 / 极性」
    #
    # 为什么必须分两段：宽搜索区（为了追踪扩展而开得很大）里常混着大片
    # 彩色元素，用它的统计量算"背景色"根本不是正文的背景。实测过三种做法都翻车：
    #  ① 只按"深色算墨迹"：幻灯片标题是深青底配白字，整条标题栏被当成文字，
    #     墨迹宽度从真实的 ~200pt 虚报成 437pt，覆盖范围超出图像边界。
    #  ② "少数派算墨迹"：白底黑字的区域中位数已是 255，"比背景亮"的像素为 0，
    #     于是选中空掩膜 —— 有 6926 个文字像素却被判成"没有检测到墨迹"。
    #  ③ 宽区取众数：区域里的大片彩色抢走众数，墨迹判定整体反过来，
    #     整片白色背景被当成文字。
    #
    # 小区域里局部背景占绝对主导，统计量才可信；标定出的阈值再拿去宽区扩展。
    calib = arr[int(band_y0 * zoom):int(band_y1 * zoom),
                max(0, int((x0 - 2) * zoom)):int((x1 + 2) * zoom)]
    if calib.size == 0:
        calib = sub
    g_cal = calib.mean(axis=2).astype(float)
    bg_level = float(np.median(g_cal))

    dark_n = int((g_cal < bg_level - INK_DARK_DELTA).sum())
    light_n = int((g_cal > bg_level + INK_DARK_DELTA).sum())
    if dark_n == 0 and light_n == 0:
        text_is_dark = True
    elif light_n == 0:
        text_is_dark = True
    elif dark_n == 0:
        text_is_dark = False
    else:
        text_is_dark = dark_n <= light_n

    if text_is_dark:
        ink_mask = gray < (bg_level - INK_DARK_DELTA)
    else:
        ink_mask = gray > (bg_level + INK_DARK_DELTA)

    bg_pixels = sub[~ink_mask]
    if bg_pixels.size:
        bg = tuple(int(v) for v in np.median(bg_pixels, axis=0))
        bg_std = float(np.mean(np.std(bg_pixels.astype(float), axis=0)))
    else:
        bg, bg_std = (255, 255, 255), 0.0

    # 文字颜色：取墨迹中最"纯"的那三分之一 —— 深字取最暗、浅字取最亮，
    # 避开抗锯齿造成的中间过渡色
    ink_px = sub[ink_mask]
    if ink_px.size:
        lum = ink_px.mean(axis=1)
        order = np.argsort(lum)
        k = max(1, len(ink_px) // 3)
        core = ink_px[order[:k]] if text_is_dark else ink_px[order[-k:]]
        color = tuple(float(v) / 255.0 for v in np.median(core, axis=0))
    else:
        color = (0.0, 0.0, 0.0)

    if not ink_mask.any():
        return InkMetrics(rect=rect, height=h, color=color, bg=bg,
                          bg_std=bg_std, ink_pixels=0)

    # ---- 先识别并排除横线（下划线/填空线）
    rule_rows = _rule_rows(gray, bg_level, ink_mask)
    rule_ratio = float(rule_rows.sum()) / max(1, ink_mask.shape[0])
    if rule_rows.any() and rule_ratio < 0.6:
        # 只有"少数几行是横线"才排除；整块都是横线说明判断错了，别乱排除
        ink_mask = ink_mask.copy()
        ink_mask[rule_rows] = False
    else:
        rule_rows = np.zeros(ink_mask.shape[0], dtype=bool)

    # ---- 先用行投影锁定"属于这一行"的行范围
    # 竖直搜索带为了兜住升降部会上下各放宽一些，但那样会连带框进相邻行。
    # 实测中带内的下一行文字会让水平追踪桥接过去，把覆盖范围从 42pt
    # 一路扩到 244pt，误伤旁边的内容。所以先按行投影把范围收回本行。
    row_ink = ink_mask.sum(axis=1)
    rows_active = row_ink > 0
    n_rows = len(rows_active)
    center_row = int(((y0 + y1) / 2 - band_y0) * zoom)
    center_row = min(max(center_row, 0), n_rows - 1)
    if not rows_active[center_row]:
        near = np.where(rows_active)[0]
        if len(near):
            center_row = int(near[np.argmin(np.abs(near - center_row))])

    max_row_gap = max(1, int(0.18 * h * zoom))
    r0 = r1 = center_row
    gap = 0
    while r0 > 0:
        if rows_active[r0 - 1]:
            r0, gap = r0 - 1, 0
        else:
            gap += 1
            if gap > max_row_gap:
                break
            r0 -= 1
    gap = 0
    while r1 < n_rows - 1:
        if rows_active[r1 + 1]:
            r1, gap = r1 + 1, 0
        else:
            gap += 1
            if gap > max_row_gap:
                break
            r1 += 1
    row_band = ink_mask[r0:r1 + 1, :]

    # ---- 在该行范围内做水平追踪，扩展 OCR 偏小的框
    n_cols = row_band.shape[1]
    if expand:
        has_ink = row_band.sum(axis=0) > 0
        center_px = int(((x0 + x1) / 2 - search_x0) * zoom)
        center_px = min(max(center_px, 0), n_cols - 1)
        if not has_ink[center_px]:
            near = np.where(has_ink)[0]
            if len(near):
                center_px = int(near[np.argmin(np.abs(near - center_px))])

        max_gap = max(1, int(bridge_ratio * h * zoom))
        left, right, gap = center_px, center_px, 0
        while left > 0:
            if has_ink[left - 1]:
                left, gap = left - 1, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
                left -= 1
        gap = 0
        while right < n_cols - 1:
            if has_ink[right + 1]:
                right, gap = right + 1, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
                right += 1
    else:
        left, right = 0, n_cols - 1

    run = row_band[:, left:right + 1]
    cols = np.where(run.any(axis=0))[0]
    rows = np.where(run.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return InkMetrics(rect=rect, height=h, color=color, bg=bg,
                          bg_std=bg_std, ink_pixels=int(ink_mask.sum()))

    # 收紧到真实墨迹的边界 —— 不管是否做了扩展，最终都用实际墨迹列/行范围，
    # 而不是搜索窗或行带的边界。这样覆盖框才是"正好框住这行字"。
    ink_rect = (
        search_x0 + (left + cols.min()) / zoom,
        band_y0 + (r0 + rows.min()) / zoom,
        search_x0 + (left + cols.max() + 1) / zoom,
        band_y0 + (r0 + rows.max() + 1) / zoom,
    )
    extended = max(0.0, x0 - ink_rect[0]) + max(0.0, ink_rect[2] - x1)

    # 用收紧后的墨迹区域算笔画粗细比，供字重判定使用
    px0 = int((ink_rect[0] - search_x0) * zoom)
    px1 = int((ink_rect[2] - search_x0) * zoom)
    py0 = int((ink_rect[1] - band_y0) * zoom)
    py1 = int((ink_rect[3] - band_y0) * zoom)
    tight = ink_mask[max(0, py0):py1, max(0, px0):px1]
    stroke = _stroke_ratio_from_mask(tight)
    text_level = float(sub[ink_mask].mean()) if ink_mask.any() else bg_level
    gray_tight = gray[max(0, py0):py1, max(0, px0):px1]
    density = _ink_mass(gray_tight, bg_level, text_level)

    rule_span = 0.0
    if rule_rows.any():
        idx = np.where(rule_rows)[0]
        rule_span = float((idx.max() - idx.min() + 1)) / zoom

    return InkMetrics(
        rect=ink_rect, height=ink_rect[3] - ink_rect[1],
        color=color, bg=bg, bg_std=bg_std,
        ink_pixels=int(ink_mask.sum()), run_extended=extended,
        stroke_ratio=stroke, ink_density=density,
        rule_span=rule_span,
    )


# --------------------------------------------------------------- 覆盖


def replace_image_preserving_flags(doc, page, xref: int, arr, rect=None) -> None:
    """把改好的像素写回图像对象 —— 直接改流，不调用 page.replace_image。

    ★ 这是唯一正确的做法。前面在 replace_image 外面打了三种补丁全部失败，
    根本原因在这里：

        page.replace_image() 会**新建**图像对象、并把色彩空间写成 ICCBased
        （[ /ICCBased 引用一个独立的 ICC 配置流 ]）。带来三个后果：
          1. xref 会漂移 —— 第一次就地更新，第二次新建对象，
             而内容流的 Do 仍指向旧对象，于是"改了但渲染不变"，且不报错；
          2. ICCBased 与原图的 DeviceRGB 渲染有细微差异 ——
             实测未改动区域劣化 9.5%；
          3. 那个独立的 ICC 配置流在某些情况下失效 ——
             渲染抛 invalid ICC colorspace。

    直接 update_stream 则：
        · 同一个 xref，不产生新对象（连续编辑天然可用）
        · /ColorSpace 原样保留（无色差）
        · /Interpolate 等标记一并保留
        · /Filter 从 DCTDecode 自动换成无损的 FlateDecode
        · /Length 自动更新
    """
    import numpy as np
    raw = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    doc.update_stream(xref, raw, compress=True)


def pixmap_from_array(arr) -> pymupdf.Pixmap:
    """把 numpy 图像数组转成 Pixmap。

    这里踩过一个坑：PyMuPDF 的 `Pixmap.set_rect()` 看名字像"填入原始像素"，
    实际文档写的是 "Set color of all pixels in bbox" —— 它把矩形内所有像素
    设成同一种颜色。用它灌数据会得到一张所有像素都等于首像素的纯色图
    （实测：一张白底幻灯片变成了全黑图，因为该图首像素是深色）。

    PyMuPDF 1.28 也不接受 numpy 数组直接构造，`samples` 属性只读。
    可靠路径是经 PNG 无损中转 —— 1152×558 约 40ms，代价可接受。
    """
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    pix = pymupdf.Pixmap(buf.getvalue())

    # ★ 再包一层 DeviceRGB，剥掉可能带上的 ICC 色彩配置引用。
    # PNG 里若夹带 ICC profile，PyMuPDF 会把它存成一个独立对象挂到图像上；
    # 存盘时 garbage=4 会把这个对象回收掉，图像就变成
    # 「invalid ICC colorspace」—— 加载失败、渲染静默回退到旧图。
    # 症状极具迷惑性：图像数据确实改了，渲染却完全不变，
    # 连续编辑时表现为"改了但 PDF 没动"。
    if pix.colorspace is not None and str(pix.colorspace) != "Colorspace(CS_RGB) - DeviceRGB":
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix


@dataclass
class HostImage:
    """承载文字的页面图像对象。"""
    xref: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    # PDF 变换矩阵 [a b c d e f]，把图像的单位方形 (0..1)² 映射到页面坐标。
    # 有了它才能处理旋转/镜像放置的图像 —— 早期只处理轴对齐的情况，
    # 遇到 90° 旋转的图像（实测第 5 页就是）直接跳过，
    # 那页的编辑会静默走错路径，用户看到的结果就是"覆盖位置不对"。
    matrix: tuple[float, float, float, float, float, float] | None = None

    @property
    def rotated(self) -> bool:
        if self.matrix is None:
            return False
        a, b, c, d = self.matrix[:4]
        return abs(b) > 1e-6 or abs(c) > 1e-6

    def to_pixels(self, rect) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        pts = [self._point(x0, y0), self._point(x1, y0),
               self._point(x0, y1), self._point(x1, y1)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def _point(self, x: float, y: float) -> tuple[float, float]:
        """PDF 坐标 → 图像像素坐标（支持任意旋转/镜像）。"""
        if self.matrix is None:
            ix0, iy0, ix1, iy1 = self.bbox
            sx = self.width / (ix1 - ix0) if ix1 > ix0 else 1.0
            sy = self.height / (iy1 - iy0) if iy1 > iy0 else 1.0
            return (x - ix0) * sx, (y - iy0) * sy

        a, b, c, d, e, f = self.matrix
        det = a * d - b * c
        if abs(det) < 1e-9:
            return 0.0, 0.0
        dx, dy = x - e, y - f
        # 2×2 矩阵求逆：(u,v) = inv([[a,c],[b,d]]) · (dx,dy)
        u = (d * dx - c * dy) / det
        v = (-b * dx + a * dy) / det
        return u * self.width, v * self.height


def find_host_image(page: pymupdf.Page, rect) -> HostImage | None:
    """找出完全包住该矩形的那张页面图像。

    支持旋转/镜像放置的图像：坐标映射走完整的矩阵求逆。
    早期只处理轴对齐的情况，遇到 90° 旋转的图像直接跳过 ——
    实测第 5 页就是旋转的，那页的编辑会静默走错路径。
    """
    x0, y0, x1, y1 = rect
    for info in page.get_image_info(xrefs=True):
        ib = info["bbox"]
        if ib[0] - 1 <= x0 and ib[1] - 1 <= y0 and ib[2] + 1 >= x1 and ib[3] + 1 >= y1:
            tr = info.get("transform")
            mat = tuple(float(v) for v in tr) if tr is not None else None
            return HostImage(xref=info["xref"], bbox=tuple(ib),
                             width=info["width"], height=info["height"],
                             matrix=mat)
    return None


def _common_host(page: pymupdf.Page, rects) -> HostImage | None:
    xs0 = min(r[0] for r in rects)
    ys0 = min(r[1] for r in rects)
    xs1 = max(r[2] for r in rects)
    ys1 = max(r[3] for r in rects)
    return find_host_image(page, (xs0, ys0, xs1, ys1))


def _mask_too_wide(doc, page: pymupdf.Page, rect) -> bool:
    """覆盖区域是否横跨了承载图像的大部分宽度。"""
    host = find_host_image(page, rect)
    if host is None:
        return False
    try:
        pix = pymupdf.Pixmap(doc, host.xref)
    except Exception:
        return False
    if pix.width <= 0:
        return False
    local = HostImage(xref=host.xref, bbox=host.bbox,
                      width=pix.width, height=pix.height)
    px0, _, px1, _ = local.to_pixels(rect)
    return (px1 - px0) > INPAINT_MAX_MASK_RATIO * pix.width


def _pixel_or_fallback_bg(arr, local, rect, fallback=(255, 255, 255)):
    """从该字上下窄带取背景色中位数。"""
    import numpy as np
    x0, y0, x1, y1 = local.to_pixels(rect)
    x0 = max(0, x0); x1 = min(arr.shape[1], x1)
    y0 = max(0, y0); y1 = min(arr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return fallback
    strips = []
    if y0 > 3:
        strips.append(arr[max(0, y0 - 4):y0, x0:x1].reshape(-1, 3))
    if y1 < arr.shape[0] - 3:
        strips.append(arr[y1:min(arr.shape[0], y1 + 4), x0:x1].reshape(-1, 3))
    if not strips:
        strips = [arr[y0:y1, x0:x1].reshape(-1, 3)]
    px = np.concatenate([s for s in strips if s.size])
    if px.size == 0:
        return fallback
    return tuple(int(v) for v in np.median(px, axis=0))


def _cover_solid(page: pymupdf.Page, rects, bg: tuple[int, int, int]) -> None:
    color = tuple(v / 255.0 for v in bg)
    for (x0, y0, x1, y1) in rects:
        page.draw_rect(
            pymupdf.Rect(x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD),
            color=None, fill=color, overlay=True,
        )


def _cover_inpaint_in_image(doc, page, host: HostImage, rects) -> bool:
    """就地修补承载文字的那张图像，再用 replace_image 写回。

    比"把整页渲染成一张大图盖上去"干净得多：只动那一张图，
    页面其余内容、z 序、其他图像对象全不受影响，文件也不会暴涨。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    try:
        pix = pymupdf.Pixmap(doc, host.xref)
    except Exception:
        return False
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.n < 3:
        return False

    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].copy()

    local = HostImage(xref=host.xref, bbox=host.bbox,
                      width=pix.width, height=pix.height)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for r in rects:
        px0, py0, px1, py1 = local.to_pixels(r)
        m = int(PAD * 2)
        px0, py0 = max(0, px0 - m), max(0, py0 - m)
        px1, py1 = min(arr.shape[1], px1 + m), min(arr.shape[0], py1 + m)
        if px1 > px0 and py1 > py0:
            mask[py0:py1, px0:px1] = 255
    if mask.max() == 0:
        return False

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    repaired = cv2.cvtColor(cv2.inpaint(bgr, mask, 3.0, cv2.INPAINT_TELEA),
                            cv2.COLOR_BGR2RGB)
    try:
        replace_image_preserving_flags(doc, page, host.xref, repaired,
                                   rect=host.bbox)
    except Exception:
        return False
    return True


def _cover_inpaint_fullpage(page: pymupdf.Page, rects, zoom: float = 3.0) -> bool:
    """整页修补后作为一张图盖上去 —— 图像带复杂变换时的退路。

    代价明确：这一页会被整体栅格化，文件变大、原有可选文字失效。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].copy()

    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for (x0, y0, x1, y1) in rects:
        px0 = max(0, int((x0 - PAD * 2) * zoom))
        py0 = max(0, int((y0 - PAD * 2) * zoom))
        px1 = min(arr.shape[1], int((x1 + PAD * 2) * zoom))
        py1 = min(arr.shape[0], int((y1 + PAD * 2) * zoom))
        mask[py0:py1, px0:px1] = 255
    if mask.max() == 0:
        return False

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    repaired = cv2.cvtColor(cv2.inpaint(bgr, mask, 4.0, cv2.INPAINT_TELEA),
                            cv2.COLOR_BGR2RGB)
    page.insert_image(page.rect, pixmap=pixmap_from_array(repaired), overlay=True)
    return True


# --------------------------------------------------------------- 主流程


def edit_raster_block(
    doc: pymupdf.Document,
    block: Block,
    new_text: str,
    font: pymupdf.Font,
    *,
    fontsize: float | None = None,
    color: tuple[float, float, float] | None = None,
    method: str | None = None,
    cover_rect: tuple[float, float, float, float] | None = None,
    match_size: bool = True,
    expand_cover: bool = False,
    font_file: str | None = None,
    weight: str = "auto",
) -> RasterEditResult:
    """图片型页面的块编辑。

    流程就是「用底色盖住原文字 → 用相似字体在原文位置重绘」：

      1. 在 OCR 框内实测真实墨迹，得到紧致的覆盖范围、字号、文字色、背景色
      2. 按实测背景色覆盖（纯色）或用周边像素修补（复杂背景）
      3. 按实测字号把新文字画回原位置

    OCR 的框实测相当准（平均内边距左右 0.8pt、上下 1.3~1.5pt），所以
    默认不向外扩展 —— 早期为修个别失真框而做的大幅扩展，反而把准确的框
    扩到了相邻内容上，制造出更明显的错位。需要改范围时用 cover_rect 手工指定。
    """
    if not block.lines:
        raise ValueError("块内没有文本行")

    page = doc[block.page_no]

    # ---- 1. 实测墨迹（覆盖范围、字号、颜色的唯一依据）
    if cover_rect is not None:
        # 用户既然手工画了框，框就是权威：既不做水平追踪，也不做竖直外扩。
        # 外扩会把框外的页面底色也卷进来，让"墨迹"判定失准
        # （实测把一个 12pt 的标题高估成 24.6pt，画出来直接换行，
        # 第二行白字落到白底上完全看不见）。
        ink = measure_ink(page, cover_rect, expand=False, y_pad_ratio=0.0)
        ink.rect = cover_rect
    elif len(block.lines) == 1:
        ink = measure_ink(page, block.lines[0].bbox, expand=expand_cover)
    else:
        ink = _merge_ink(page, block, expand_cover)

    # ---- 字重：按实测笔画粗细在候选字重里挑最接近的
    #
    # 原文若是粗体而用常规体重绘，即使字高完全一致也会显得"偏大偏松散"。
    # 实测中被明确指出「上图字体较粗且紧凑，下图明显更舒展」，
    # 所以字重必须按笔画实测来选，不能一律用常规体。
    weight_label = ""
    chosen_file = font_file
    if font_file and weight == "bold":
        cands = [c for c in weight_variants(font_file)
                 if any(t in Path(c).stem.lower() for t in ("bd", "bold", "hei"))]
        p0 = _find_file(cands[0]) if cands else None
        if p0:
            chosen_file, weight_label = p0, "bold"
        else:
            weight_label = "regular"
    elif font_file and weight == "regular":
        weight_label = "regular"
    elif font_file and ink.stroke_ratio > 0:
        cands = weight_variants(font_file)
        ppp = _effective_ppp(page, ink.rect)
        chosen_file, chosen_ratio, weight_label = pick_weight(
            font_file, cands, new_text, ink.stroke_ratio,
            fontsize if fontsize is not None else ink.size_for(new_text),
            target_density=ink.ink_density, effective_ppp=ppp,
        )
        if chosen_file != font_file:
            try:
                font = pymupdf.Font(fontfile=chosen_file)
            except Exception:
                weight_label = ""

    # ---- 墨迹测量的合理性护栏
    #
    # 墨迹**不可能**超出 OCR 文字框太多。一旦超出，说明这一页的墨迹测量失效了
    # （实测第 5 页旋转图像那页，整条搜索带都有墨迹，行投影分不开，
    #  测出的墨迹高 41.3pt 比框高 27.6pt 还大）。
    # 后果很严重：字号按 41.3pt 标定 → 得到 31.3pt（实际约 12pt），
    # 文字被判定"放不下"而折成 4 行，覆盖范围也炸开。
    # 这里把墨迹框钳制回文字框，高度按框高的 0.78 估（CJK 墨迹约占字号 0.88，
    # 但 OCR 框本身含上下留白，0.78 是实测较稳的系数）。
    box_h = max((l.bbox[3] - l.bbox[1] for l in block.lines), default=0.0)
    if box_h > 0 and ink.height > box_h * 1.35:
        clamp_h = box_h * 0.78
        cy = (ink.rect[1] + ink.rect[3]) / 2
        ink.rect = (ink.rect[0], cy - clamp_h / 2, ink.rect[2], cy + clamp_h / 2)
        ink.height = clamp_h
        result_note = True
    else:
        result_note = False

    # ---- 字号：先粗估，再用离屏实测标定到"墨迹高度等于原文实测高度"
    size_converged = True
    if fontsize is not None:
        size = fontsize
    elif match_size:
        size = ink.size_for(new_text)
        size, size_converged = calibrate_size(
            font, new_text, ink.height, size,
            font_key=Path(chosen_file).name if font_file else "",
        )
    else:
        size = block.size
    use_color = color if color is not None else ink.color

    result = RasterEditResult(
        page_no=block.page_no, old_text=block.text, new_text=new_text,
        method="", ink=ink, used_size=size,
    )
    result.covered_rects = [ink.rect]

    if result_note:
        result.warnings.append(
            "本页的墨迹测量超出文字框（该页墨迹密集或图像有旋转），"
            "字号已按文字框推算，结果可能略有偏差")
    if ink.ink_pixels == 0:
        result.warnings.append("该区域没有检测到墨迹，覆盖范围可能不准确")
    if weight_label == "bold":
        result.warnings.append("已按指定使用粗体")
    if not size_converged:
        result.warnings.append(
            f"选用的字体与原文形状差异较大，字号只能标定到 {size:.1f}pt 近似值；"
            f"换一个更接近原文字的字体文件可改善"
        )

    # OCR 框与实测墨迹差得离谱时，说明这一行的识别不可信 —— 必须让用户知道
    orig_w = max(1.0, block.bbox[2] - block.bbox[0])
    ink_w = ink.rect[2] - ink.rect[0]
    if cover_rect is None and ink_w > orig_w * 2.0:
        result.warnings.append(
            f"OCR 框（宽 {orig_w:.0f}pt）与实测墨迹（宽 {ink_w:.0f}pt）相差悬殊，"
            f"这一行的识别可能不准。建议在界面上手工调整覆盖范围，"
            f"确认覆盖框正好框住要改的那行字"
        )

    if method:
        chosen = method
    else:
        chosen = "solid" if ink.bg_std <= UNIFORM_STD_MAX else "inpaint"
        if chosen == "inpaint" and _mask_too_wide(doc, page, ink.rect):
            chosen = "solid"
            result.warnings.append(
                "覆盖区域横跨图像大部分宽度，改用纯色填充"
                "（这种形状做 inpaint 会从上下方向外推，产生伪影）"
            )

    # ---- 2. 覆盖
    if chosen == "inpaint":
        host = _common_host(page, [ink.rect])
        ok = _cover_inpaint_in_image(doc, page, host, [ink.rect]) if host else False
        method_note = "inpaint"
        if not ok:
            ok = _cover_inpaint_fullpage(page, [ink.rect])
            if ok:
                method_note = "inpaint"
                result.warnings.append("该页被整体栅格化以完成修补，原有可选文字会失效")
        if not ok:
            result.warnings.append("inpaint 不可用，已退化为纯色填充")
            chosen = "solid"
            method_note = "solid"
    else:
        method_note = "solid"

    if method_note == "solid":
        _cover_solid(page, [ink.rect], ink.bg)
        result.method = "solid"
    else:
        result.method = "inpaint"

    # ---- 3. 重绘
    single_line_source = len(block.lines) == 1
    width = max(1.0, ink.rect[2] - ink.rect[0])
    lines = _layout(new_text, font, size, ink)
    if single_line_source and len(lines) > 1:
        # 单行源块被撑成多行通常不是用户想要的 —— 图片型文档里换行的第二行
        # 会落到覆盖框外（框外还是原底色），白字落白底上就凭空消失了。
        # 所以优先缩字号压成一行；缩到 70% 仍放不下才允许换行。
        floor = size * 0.70
        while len(lines) > 1 and size > floor:
            size = max(floor, size * 0.94)
            lines = _layout(new_text, font, size, ink)
        result.used_size = size
        if len(lines) > 1:
            result.warnings.append(
                f"新内容比原文字长，已缩到 {size:.1f}pt 仍放不进一行，已折行显示"
            )
    result.lines_after = len(lines)
    _draw(page, ink, lines, font, size, use_color)

    if result.method == "inpaint":
        result.warnings.append(
            "文字原像素是被周边像素修补盖住的，不是真正移除；放大后可能有痕迹"
        )
    return result


# --------------------------------------------------------------- 排版与绘制


def _layout(text: str, font: pymupdf.Font, size: float, ink: InkMetrics) -> list[str]:
    from .editor import _normalize_input, break_lines

    text = _normalize_input(text)
    width = max(1.0, ink.rect[2] - ink.rect[0])
    return break_lines(text, font, size, width, first_indent=0.0, kinsoku=True)


def _draw(page: pymupdf.Page, ink: InkMetrics, lines: list[str],
          font: pymupdf.Font, size: float, color) -> None:
    """在实测墨迹的位置绘制新文字。

    基线由字体的 ascender 反推，让字顶对齐墨迹顶部 ——
    比"从墨迹底部减一个固定比例"更贴合不同字体的度量。
    """
    gap = size * 1.35
    left = ink.rect[0]
    asc = getattr(font, "ascender", 0.86) or 0.86
    first_baseline = ink.rect[1] + asc * size

    tw = pymupdf.TextWriter(page.rect)
    for i, text in enumerate(lines):
        tw.append((left, first_baseline + i * gap), text, font=font, fontsize=size)
    tw.write_text(page, color=color, overlay=True)


def _cluster_runs(band):
    """墨迹的列游程（真实字簇）。返回 [(起px, 止px), ...]。"""
    import numpy as np
    if band is None or band.size == 0:
        return []
    col = band.sum(axis=0)
    runs = []
    a = None
    for i, v in enumerate(col > 0):
        if v and a is None:
            a = i
        elif not v and a is not None:
            runs.append((a, i))
            a = None
    if a is not None:
        runs.append((a, len(col)))
    return runs


def _boundary_ink_score(col, bounds) -> float:
    """内部边界处的平均墨迹量 —— 越小说明边界越贴合真实空隙。

    这就是"参考字体选得对不对"的判据：字宽比例正确时，DP 会把每条边界
    送进字符之间的空隙；比例错了，边界只能落在笔画上。
    """
    import numpy as np
    inner = bounds[1:-1]
    if not inner:
        return float("inf")
    n = len(col)
    tot = 0.0
    for v in inner:
        lo = max(0, v - 1)
        hi = min(n, v + 2)
        tot += float(col[lo:hi].sum())
    return tot / len(inner)


def _font_width_residual(runs, text: str, font, font_path: str,
                         size_pt: float, zoom: float) -> float:
    """某个字体能不能解释观测到的墨迹游程 —— 返回残差的中位绝对偏差（pt）。

    ★ 这是选参考字体的主判据，因为它直接问对了问题：
      "把字符按顺序分配进各条墨迹游程，这个字体的字形墨迹宽能不能解释游程的宽度？"
      每条游程的宽度应约等于「其中各字形墨迹宽之和 + 抗锯齿外扩量」，
      外扩量对所有游程是同一个常数，所以残差应当**小且一致** ——
      取中位绝对偏差，正好把那个未知常数抵消掉。

    为什么不用"边界处的墨迹量"当主判据：它被数量占多数的字符主导。
    实测某邮箱串 17 个字里 10 个是数字，数字边界在两种字体下差别很小，
    真正能区分字体的是 '@'（Arial 约 1 em、宋体 0.5 em）——
    结果边界判据选了宋体，'@' 的框被切成 5.3pt，而它的墨迹有 9.3pt 宽。
    残差判据里 '@' 那一条游程直接把宋体否掉：MAD 1.76 vs Arial 0.68。

    逐字分配必须**按顺序贪心**，不能用"段中点落在游程内"：
    段中点是切分结果，而切分正是待验证的东西 —— 用它来分配会自我印证。
    """
    if not runs or not font_path or size_pt <= 0.5:
        return float("inf")
    off = glyph_ink_offsets(font, text, size_pt, font_path)
    if not off:
        return float("inf")

    # ★ 分配要同时容忍**并**与**拆**：
    #   并 —— 相邻字贴在一起，一条游程里有好几个字；
    #   拆 —— 一个字内部有全高竖向空隙（如「川」），一条游程放不下一个字。
    # 早先按"整字贪心"分配，遇到拆的情况就整体错位一格，
    # 反而把**正确**字体的残差算大 —— 真值样本行 2 是 36 字 37 游程，
    # 结果选中了字宽比例更差的微软雅黑，左边界误差从 0.19 涨到 0.75。
    # 用"剩余未分配墨迹宽"做进位，并和拆就都自然成立。
    res = []
    ci = 0
    n = len(text)
    rem = 0.0
    for (ca, cb) in runs:
        rw = (cb - ca) / zoom
        acc = 0.0
        while acc < rw - 0.10:
            if rem <= 0.10:
                if ci >= n:
                    break
                a, b = off.get(text[ci], (0.0, 0.0))
                rem = b - a
                ci += 1
                if rem <= 0.10:
                    continue
            take = min(rem, rw - acc)
            acc += take
            rem -= take
            if rem > 0.10:
                break            # 本游程用完，剩下的墨迹留给下一条游程
            rem = 0.0
        if acc > 0.10:
            res.append(rw - acc)
    if len(res) < 3:
        return float("inf")
    med = sorted(res)[len(res) // 2]
    return sum(abs(r - med) for r in res) / len(res)


def _fit_ref_font(band, x_l: float, zoom: float, text: str):
    """在所有候选参考字体里挑**切分质量最好**的那个。

    返回 (font, 路径, 边界px列表, 推定字号pt)，都不成立时返回 None。

    ★ 参考字体是拟合出来的参数，不是猜出来的。
    早先按"文本是不是 ASCII"在 Arial 和宋体之间二选一，
    对一份中文文档里的邮箱串就选错了 —— 实测同一行：
      宋体先验 → '@' 只分到 5.3pt 的字格，而它的墨迹有 9.3pt 宽
      Arial 先验 → '@' 分到 9.8pt 的字格，与墨迹相符
    判据只用像素，不需要知道真相，所以对任何文档都成立。
    """
    import numpy as np
    if band is None or band.size == 0 or not text:
        return None
    col = band.sum(axis=0).astype(float)
    if col.size < 3:
        return None
    nz = np.where(col > 0)[0]
    if nz.size == 0:
        return None
    st, en = int(nz.min()), int(nz.max()) + 1
    if en - st < len(text):
        return None
    runs = _cluster_runs(band)

    try:
        from .ocr import _ref_font_candidates
        cands = list(_ref_font_candidates()) + _ref_font_candidates_hybrid()
    except Exception:
        cands = []
    best = None
    for font, path in cands:
        # ★ 缺字的候选必须**硬剔除**。text_length 对缺字会返回回退宽度
        #   （不是 0），光看它筛不掉 —— 实测 Arial 在一行纯中文上
        #   照样能通过宽度检查，然后靠"残差恰好一致"挤掉正确的字体，
        #   整行框全被切坏。has_glyph 对系统字体是可靠的。
        try:
            if any(not font.has_glyph(ord(c)) for c in text):
                continue
        except Exception:
            pass
        try:
            raw = [max(0.1, font.text_length(c, fontsize=100.0)) for c in text]
        except Exception:
            continue
        # ★ 先用"网格拟合"或"竖向高度"定出真正的文字范围，
        #   把首尾混进来的非文字墨迹（下划线、边框、杂墨）排除掉。
        #   拟合跨度必须"拟合得住"才采用（残差 ≤ 0.6pt）：残差大说明这个
        #   范围/字体组合解释不了墨迹，硬用会把边界切乱。
        span = None
        col_span = _span_by_column_height(band, zoom)
        if col_span is not None:
            span = (col_span[0], col_span[1], 0.0)
        if span is None:
            ext_span = _span_by_extent(band, runs, len(text), zoom)
            if ext_span is not None:
                span = (ext_span[0], ext_span[1], 0.0)
        if span is None:
            _s = _fit_text_span(runs, text, font, path, zoom, x_l)
            if _s is not None and _s[3] / zoom <= 0.6:
                span = (_s[0], _s[1], _s[3])
        if span is not None:
            st_f, en_f = span[0], span[1]
            if en_f - st_f < len(text):
                continue
        else:
            st_f, en_f = float(st), float(en)

        total = sum(raw) or 1.0
        scale = (en_f - st_f) / total
        exp_w = [max(0.5, r * scale) for r in raw]
        b = _dp_align_boundaries(col, int(round(st_f)), int(round(en_f)),
                                 exp_w, win_ratio=0.45)
        if b is None or len(b) != len(text) + 1:
            continue
        size_pt = scale / zoom * 100.0
        resid = _font_width_residual(runs, text, font, path, size_pt, zoom)
        tie = _boundary_ink_score(col, b)
        score = (round(resid, 3), tie)
        if best is None or score < best[0]:
            best = (score, font, path, b, size_pt)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _segment_by_clusters(runs, zoom: float, text: str,
                         font, font_path: str, size_pt: float):
    """墨迹簇与字符**一对一**时，直接用簇边界 —— 墨迹本身就是答案。

    ★ 这条路径不需要任何字宽先验，所以精度上限最高。
    实测某表单行「乙方（承诺方）：姓名（参与学生）」：
    16 个墨迹簇正好对 16 个字，簇宽与参考字形墨迹宽的相对误差中位 5%。
    但当时的代码走的是 DP，DP 为了让"段宽之和 = 墨迹跨度"把
    「方」切成 6.2pt（真实 11.5pt），整行前半段右偏 2~5pt。

    必须**验证**再采用，不能光看数量相等：
      · 中文里「川」这类字本身就有全高竖向空隙，会分裂成多个簇；
      · 相邻字贴在一起时会并成一个簇。
    所以逐字比对"簇宽 vs 参考字形墨迹宽"，中位相对误差超过阈值就不用。
    """
    if not runs or not text or len(runs) != len(text):
        return None
    if not font_path or size_pt <= 0.5:
        return None
    off = glyph_ink_offsets(font, text, size_pt, font_path)
    if not off:
        return None
    devs = []
    for (ca, cb), ch in zip(runs, text):
        cw = (cb - ca) / zoom
        a, b = off.get(ch, (0.0, 0.0))
        ew = b - a
        if ew <= 0.15:          # 空白/极窄字形不参与判定
            continue
        devs.append(abs(cw - ew) / ew)
    if len(devs) < max(3, int(len(text) * 0.6)):
        return None
    devs.sort()
    if devs[len(devs) // 2] > 0.35:
        return None
    return [r[0] for r in runs] + [runs[-1][1]]


def _fit_text_span(runs, text: str, font, font_path: str, zoom: float,
                   x_l: float, max_trim: int = 3):
    """用"字宽比例网格的最小二乘拟合"定出**真正的文字范围**。

    返回 (st_px, en_px, scale, rms_px) 或 None。

    ★ 为什么需要它：
    墨迹里常混着不属于文字的东西 —— 实测某数字串两侧各有一段下划线，
    它们被算进墨迹跨度后整条字格网格被撑宽 15%，11 个字累计漂移 7pt；
    「电子邮箱」右侧一段杂墨同样把网格撑宽。这些都是**首尾多出来的游程**，
    所以按"首尾各裁掉若干游程"枚举，用网格拟合的残差挑出正确的组合。

    ★ 模型里必须带**每个字形的左边距**：游程起点是墨迹左沿、不是笔位原点，
      各字形左边距差很多（'1' 的墨迹窄而居中、左边距大），
      不带它时这个差异会淹没信号、裁错范围。
    ★ 还要带上**游程宽度**：宽度模式是字体无关的强信号 ——
      实测某数字串里 '1' 的窄游程出现在第 1/4/10 条，
      与该串中 '1' 的位置 0/3/9 精确对应，而错位的候选对不上。
    """
    import numpy as np

    n = len(text)
    if n < 2 or not runs:
        return None
    try:
        raw = [max(0.5, font.text_length(c, fontsize=100.0)) for c in text]
    except Exception:
        return None
    offs = glyph_ink_offsets(font, text, 100.0, font_path)
    if not offs:
        return None
    lsb = np.array([max(0.0, offs.get(c, (0.0, 0.0))[0])
                    for c in text], dtype=float)
    ink_w100 = np.array([max(0.0, offs.get(c, (0.0, 0.0))[1]
                            - offs.get(c, (0.0, 0.0))[0])
                         for c in text], dtype=float)
    if float(ink_w100.sum()) <= 1.0:
        ink_w100 = None
    tot = sum(raw) or 1.0
    cum = np.concatenate([[0.0], np.cumsum(raw)])[:n]
    extra = len(runs) - n
    if extra < 0 or extra > 2 * max_trim:
        return None
    best = None
    for tl in range(0, min(extra, max_trim) + 1):
        tr = extra - tl
        if tr < 0 or tr > max_trim:
            continue
        sub = runs[tl:len(runs) - tr] if tr else runs[tl:]
        if len(sub) != n:
            continue
        starts = np.array([a for a, _ in sub], dtype=float)
        widths = np.array([b - a for a, b in sub], dtype=float)
        A = np.vstack([np.ones(n), cum + lsb]).T
        try:
            coef, *_ = np.linalg.lstsq(A, starts, rcond=None)
        except Exception:
            continue
        x0, k = float(coef[0]), float(coef[1])
        if k <= 0.2:
            continue
        rms_pos = float(np.sqrt(((starts - A @ coef) ** 2).mean()))
        rms_w = (float(np.sqrt(((widths - ink_w100 * k) ** 2).mean()))
                 if ink_w100 is not None else 0.0)
        rms = rms_pos + rms_w
        if best is None or rms < best[0]:
            best = (rms, x0, x0 + tot * k, k)
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def _span_by_column_height(band, zoom: float, ratio: float = 0.55):
    """按**逐列竖向高度**定出文字范围，返回 (st_px, en_px) 或 None。

    ★ 这是处理"表单下划线"的关键，而且是唯一在**粘连**情况下仍然有效的判据。

    实测这份中文表单的填写栏两侧都有下划线（视觉模型确认图里是
    `_E12345678_`、`_13812340000_`）。它们被算进墨迹跨度后整条字格网格
    被撑宽 12~19%（实测字格间距 5.0pt，参考字体却推出 5.73pt），
    误差沿行累积到 5~6pt。

    已有的两条规则都失效：
      · 按游程数裁剪（_fit_text_span）—— 需要游程数 ≥ 字数，
        而这些块往往**粘连**（`2023117550` 10 个字只有 1 条游程）；
      · 按游程竖向高度剔除（_span_by_extent）—— 需要下划线自成一条游程，
        但它常常与数字糊在一起。

    逐列看就分得开：下划线的每一列只有 1.2~1.5pt 高，
    正文满高列有 7.3~9.3pt —— 差 5 倍以上，糊在一起也改变不了这一点。
    """
    import numpy as np

    if band is None or band.size == 0:
        return None
    cols = np.where(band.any(axis=0))[0]
    if cols.size < 3:
        return None
    ext = np.zeros(band.shape[1], dtype=float)
    for k in cols:
        rows = np.where(band[:, k])[0]
        if rows.size:
            ext[k] = rows.max() - rows.min() + 1
    tall = ext[cols]
    ref = float(np.median(tall[tall >= 0.5 * tall.max()])) if tall.size else 0.0
    if ref <= 0:
        return None
    keep = cols[ext[cols] >= ratio * ref]
    if keep.size < 2:
        return None
    st, en = int(keep.min()), int(keep.max()) + 1
    if st <= int(cols.min()) and en >= int(cols.max()) + 1:
        return None                      # 没剔掉任何东西
    # 至少保留一半宽度：剔太多说明判据在这行不成立
    if (en - st) < 0.5 * (int(cols.max()) + 1 - int(cols.min())):
        return None
    # ★ 只当裁掉的宽度**确实显著**时才采用。
    #   正常文字的边缘也可能有零星矮列（汉字首笔偏内、标点本身矮），
    #   那点差别不该动跨度 —— 实测不加这道守卫时真值样本从 137/137
    #   退化到 135/137、最大误差 0.88→1.46pt。
    #   下划线的宽度是 3.5pt 量级，与这类零星矮列差一个数量级。
    removed = (st - int(cols.min())) + (int(cols.max()) + 1 - en)
    if removed < 2.0 * zoom:
        return None
    return st, en


def _span_by_extent(band, runs, n_chars: int, zoom: float,
                    ratio: float = 0.35):
    """用**竖向高度**剔除首尾的非文字墨迹（下划线、边框、杂墨）。

    返回 (st_px, en_px) 或 None。

    专治"游程数少于字数"（墨迹粘连）—— 那时按游程数的裁剪搜索用不上。

    ★ 判据自带保护：下划线的竖向高度只有正文的 1/5（实测 1.5pt vs 7.8pt），
      但文字里的逗号、冒号同样是矮的 —— 把它们一并剔除后剩下的游程数
      就对不上字数了，于是这条规则**自动放弃**，不会误伤真正的标点。
    """
    import numpy as np

    if not runs or n_chars <= 0:
        return None
    ext = []
    for a, b in runs:
        rows = np.where(band[:, a:b].any(axis=1))[0]
        ext.append(float(rows.max() - rows.min() + 1) if rows.size else 0.0)
    if not ext:
        return None
    med = float(np.median(ext))
    if med <= 0:
        return None
    keep = [i for i, e in enumerate(ext) if e >= ratio * med]
    if len(keep) != n_chars:
        return None
    if keep != list(range(keep[0], keep[-1] + 1)):
        return None
    if keep[0] == 0 and keep[-1] == len(runs) - 1:
        return None
    return runs[keep[0]][0], runs[keep[-1]][1]


def _segment_by_ink(band, x_l: float, zoom: float, n_chars: int,
                    text: str = "", ref_font=None):
    """按墨迹切出一行里 n_chars 个字符的边界（pt）。

    ★ 两轮迭代才走对路：

    第一轮用"相对阈值 + 按字宽比例合并"，问题是把真实字间空隙（1.13pt）
    也合并掉了，边界整体漂移（误差中位 10.7pt，接近一个字）。
    第二轮改成零阈值 + 按空隙中位数合并，又切过头 ——
    中文字内部的笔画空隙被当成字间空隙，一个字被切成两段。

    所以最终不用"先切再修补"，而是**一次用 DP 求出最优边界**：
      代价 = Σ|段宽 - 预期字宽|  +  λ × 落点处的墨迹量
    DP 天然保证边界单调递增、数量正好 n_chars，不需要事后合并或拆分。

    预期字宽不再是"行宽 ÷ 字数"（混了半角会偏），而是按**参考字体的
    字宽比例**分配：中文 1.0 em、数字/字母约 0.5 em —— 再整体缩放到墨迹跨度。
    """
    import numpy as np
    if band is None or band.size == 0 or n_chars <= 0:
        return None

    col = band.sum(axis=0).astype(float)
    if col.size < 3:
        return None
    nz = np.where(col > 0)[0]
    if nz.size == 0:
        return None
    start_px, end_px = int(nz.min()), int(nz.max()) + 1
    if end_px - start_px < n_chars:
        return None

    # 预期字宽：按参考字体的比例分，再缩放到墨迹跨度
    if ref_font is not None and text and len(text) == n_chars:
        try:
            raw = [max(0.1, ref_font.text_length(c, fontsize=100.0)) for c in text]
        except Exception:
            raw = [1.0] * n_chars
    else:
        raw = [1.0 if ("一" <= c <= "鿿") else 0.5 for c in text]             if len(text) == n_chars else [1.0] * n_chars
    total_raw = sum(raw) or 1.0
    scale = (end_px - start_px) / total_raw
    exp_w = [r * scale for r in raw]

    b = _dp_align_boundaries(col, start_px, end_px, exp_w, win_ratio=0.45)
    if b is None or len(b) != n_chars + 1:
        return None
    return [x_l + v / zoom for v in b]


_GLYPH_INK_CACHE: dict = {}
_GLYPH_MASS_CACHE: dict = {}
_GLYPH_HEIGHT_CACHE: dict = {}


def line_tilt_measure(band, zoom: float):
    """量一行的倾斜程度：返回 (整行垂直落差 pt, 字高参照 pt)。

    落差：把墨迹跨度分左右两半，各自求"墨迹行的质心 y"，两点相隔半行，
    乘以 2 即为整行落差。用质心而不是"顶行/底行"，是因为质心由整行墨量
    加权，不会被个别粗笔画或标点带偏。

    字高参照：只取**有墨**的列，各列墨迹竖向跨度的中位数。
    不能用 OCR 框高 —— 倾斜文字的 OCR 框本身就很高（斜着包住字），
    拿它当参照会把阈值抬高，倾斜行反而判成水平。

    这个量决定两件事：倾斜行既不能走"按水平字宽分配"的竖列切分，
    也不能用"墨迹顶到格子边界"来判断粘连带 ——
    倾斜行的字间空隙是一条斜带，任何竖直列都能切到它。
    """
    import numpy as np

    if band is None or getattr(band, "size", 0) == 0 or band.shape[1] < 8:
        return 0.0, 0.0
    try:
        xs = np.where(band.any(axis=0))[0]
        if xs.size < 8:
            return 0.0, 0.0
        mid = (int(xs.min()) + int(xs.max())) // 2

        def centroid_y(a: int, b_: int):
            sub = band[:, a:b_ + 1]
            w = sub.sum(axis=1).astype(float)
            tot = float(w.sum())
            if tot <= 0.0:
                return None
            return float((w * np.arange(w.size)).sum() / tot)

        cl = centroid_y(int(xs.min()), mid)
        cr = centroid_y(mid, int(xs.max()))
        drop = 0.0 if (cl is None or cr is None) else abs(cr - cl) * 2.0 / zoom

        ext = []
        for k in xs:
            rows = np.where(band[:, k])[0]
            if rows.size:
                ext.append(float(rows.max() - rows.min() + 1))
        height = float(np.median(ext)) / zoom if ext else 0.0
        return drop, height
    except Exception:
        return 0.0, 0.0


def is_cjk_char(ch: str) -> bool:
    """汉字与全角标点（它们的字宽恒为 1 em，与西文字体无关）。"""
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF or 0x3000 <= o <= 0x303F
            or 0xFF00 <= o <= 0xFFEF)


class HybridRefFont:
    """中西混排的参考字体：汉字用中文字体、ASCII 用西文字体。

    ★ 为什么必须有它：真实中文文档里汉字与 ASCII 常来自两个字体。
      实测某样本：汉字推进 11.0pt、数字推进 6.0pt（比值 0.545），
      而宋体自己的 ASCII 比值是 0.500、Arial 是 0.556 ——
      说明汉字来自宋体、数字来自 Arial 类字体。
      用**单一字体**建模时，数字那一段的字格必然偏窄或偏宽：
      按宋体算，每个数字窄 0.5pt，32 个数字累计偏差 16pt；
      按 Arial 算，每个汉字又宽 0.6pt。误差沿行累积，
      这正是"中西混排行错位"的根源。
    """

    def __init__(self, cjk, latin, cjk_path: str, latin_path: str):
        self._cjk, self._latin = cjk, latin
        self._cjk_path, self._latin_path = cjk_path, latin_path
        short = lambda f: getattr(f, "name", "?").split()[0]  # noqa: E731
        self.name = f"{short(cjk)}+{short(latin)}"

    def path_for(self, ch: str) -> str:
        return self._cjk_path if is_cjk_char(ch) else self._latin_path

    def font_for(self, ch: str):
        """返回该字符真正使用的字体对象（不是混合体本身）。

        ★ 必须返回**非混合体**：字形渲染函数看到 path_for 就会逐字委托，
          若把混合体本身再传进去就会无限递归 ——
          实测 RecursionError 被 except 吞掉，混合字体的字形墨迹一直是空的
          （只有字宽比例生效，粘连修正与验证全部失效）。
        """
        return self._cjk if is_cjk_char(ch) else self._latin

    def _pick(self, ch: str):
        return self._cjk if is_cjk_char(ch) else self._latin

    def text_length(self, ch: str, fontsize: float = 11.0) -> float:
        return self._pick(ch).text_length(ch, fontsize=fontsize)

    def has_glyph(self, cp: int) -> bool:
        return self._pick(chr(cp)).has_glyph(cp)

    def valid_for(self, text: str) -> bool:
        """两个字体都覆盖得住这段文字才算可用。"""
        try:
            for ch in text:
                f = self._pick(ch)
                if ch.isascii() or is_cjk_char(ch):
                    if not f.has_glyph(ord(ch)):
                        return False
        except Exception:
            return False
        return True


def _ref_font_candidates_hybrid():
    """在候选字体里再生成一批"中文×西文"配对，供字体拟合挑选。"""
    from .ocr import _ref_font_candidates
    out = []
    cands = _ref_font_candidates()
    cjk = [(f, p) for f, p in cands if not f.name.lower().startswith(
        ("arial", "calibri", "times"))]
    latin = [(f, p) for f, p in cands if f.name.lower().startswith(
        ("arial", "calibri", "times"))]
    for cf, cp in cjk:
        for lf, lp in latin:
            out.append((HybridRefFont(cf, lf, cp, lp), cp))
    return out


def glyph_ink_offsets(ref_font, chars: str, size_pt: float, font_path: str = ""):
    """参考字体里每个字形的墨迹相对**笔位原点**的偏移 (a, b)，单位 pt。

    ★ 这是"粘连字"唯一可用的信息源。
    低分辨率扫描件上，相邻数字的字间空隙只有 1.29pt ≈ 0.83 个源像素 ——
    **空隙在像素里根本不存在**，两个数字糊成一团。此时无论怎么量墨迹，
    量到的都是整团的外沿，不是单个字的边界。
    唯一还知道的，是这个字本身长什么样、占多宽 —— 即参考字体的字形度量。

    做法：把去重后的字按固定间距排在一张临时页上渲染一次，量各自的墨迹范围。
    返回 {字符: (a, b)}，a/b 是墨迹左右沿相对笔位原点的距离（pt）。

    ★ 渲染一律在**基准字号 100pt** 做，再按比例缩放，缓存键里不带字号。
    字形墨迹随字号线性缩放，所以同一字体同一字符集只需要渲染一次 ——
    否则每换一个字号就重渲一遍（字体拟合阶段要试 6 个候选 × 每行一个字号）。
    """
    import numpy as np

    uniq = "".join(dict.fromkeys([c for c in chars if c and not c.isspace()]))
    if not uniq or size_pt <= 0.5 or not font_path:
        return {}
    base = _glyph_ink_at100(ref_font, uniq, font_path)
    if not base:
        return {}
    k = size_pt / 100.0
    return {ch: (a * k, b * k) for ch, (a, b) in base.items()}


_BASE_SIZE = 100.0


def _glyph_ink_at100(ref_font, uniq: str, font_path: str) -> dict:
    """字形墨迹在基准字号 100pt 下的左右沿偏移 {字符: (a, b)}（缓存）。"""
    import numpy as np

    key = (getattr(ref_font, "name", ""), uniq)
    hit = _GLYPH_INK_CACHE.get(key)
    if hit is not None:
        return hit
    # 混合字体（中文+西文配对）：逐字用各自的字体渲染后合并。
    # 渲染很便宜且带缓存，所以不必按字体分组。
    if hasattr(ref_font, "path_for") and hasattr(ref_font, "font_for"):
        out_h: dict = {}
        mass_h: dict = {}
        hgt_h: dict = {}
        for ch in uniq:
            f2 = ref_font.font_for(ch)
            sub = _glyph_ink_at100(f2, ch, ref_font.path_for(ch))
            if ch in sub:
                out_h[ch] = sub[ch]
            k2 = (getattr(f2, "name", ""), ch)
            if k2 in _GLYPH_MASS_CACHE:
                mass_h[ch] = _GLYPH_MASS_CACHE[k2].get(ch, 0.0)
            if k2 in _GLYPH_HEIGHT_CACHE:
                hgt_h[ch] = _GLYPH_HEIGHT_CACHE[k2].get(ch, 0.0)
        _GLYPH_INK_CACHE[key] = out_h
        _GLYPH_MASS_CACHE[key] = mass_h
        _GLYPH_HEIGHT_CACHE[key] = hgt_h
        return out_h

    size_pt = _BASE_SIZE
    cell = size_pt * 2.0
    pad_x = size_pt
    zoom = 2.0                     # 100pt 的字用 2 倍就够，省渲染时间
    try:
        d = pymupdf.open()
        page = d.new_page(width=pad_x * 2 + cell * len(uniq),
                          height=size_pt * 4.0)
        base_y = size_pt * 2.5
        for i, ch in enumerate(uniq):
            page.insert_text((pad_x + i * cell, base_y), ch,
                             fontname="R", fontfile=font_path,
                             fontsize=size_pt)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        d.close()
    except Exception:
        return {}
    gray = arr.mean(axis=2)
    bg = float(np.median(gray))
    mask = gray < (bg - 40)

    out: dict = {}
    for i, ch in enumerate(uniq):
        x0 = int((pad_x + i * cell - size_pt) * zoom)
        x1 = int((pad_x + i * cell + size_pt) * zoom)
        x0 = max(0, x0)
        x1 = min(mask.shape[1], x1)
        if x1 <= x0:
            continue
        sub = mask[:, x0:x1]
        cols = np.where(sub.any(axis=0))[0]
        if cols.size == 0:
            continue
        pen = (pad_x + i * cell) * zoom
        out[ch] = ((x0 + cols.min() - pen) / zoom,
                   (x0 + cols.max() + 1 - pen) / zoom)
        # 顺便收集墨迹量与墨迹高：验证工具要用它们做"自标定"判据
        # （框内墨迹量 ÷ 参考字形墨迹量，同行比值应当一致）。
        rows = np.where(sub.any(axis=1))[0]
        if rows.size:
            _GLYPH_HEIGHT_CACHE.setdefault(key, {})[ch] = (
                float(rows.max() - rows.min() + 1) / zoom)
        _GLYPH_MASS_CACHE.setdefault(key, {})[ch] = float(sub.sum())
    _GLYPH_INK_CACHE[key] = out
    return out


def glyph_mass(font, chars: str, font_path: str) -> dict:
    """参考字形的**墨迹像素数**（基准字号 100pt 下）：{字符: 像素数}。

    验证工具用它做"自标定"判据：正确的框里装的正好是一个字的墨迹，
    "框内墨迹量 ÷ 参考字形墨迹量"应当等于同一个常数（扫描的整体模糊/加粗
    带来的放大，对整行每个字都一样）。某个字明显偏离本行中位数，
    说明它的框要么吃了邻居、要么切掉了自己。
    """
    uniq = "".join(dict.fromkeys([c for c in chars if c and not c.isspace()]))
    if not uniq or not font_path:
        return {}
    _glyph_ink_at100(font, uniq, font_path)
    return _GLYPH_MASS_CACHE.get((getattr(font, "name", ""), uniq), {})


def glyph_ink_height(font, chars: str, font_path: str) -> dict:
    """参考字形的**墨迹高度**（基准字号 100pt 下）：{字符: 高度 pt}。

    用于独立于字宽地估字号：字宽推算会被"跨度里混进非文字墨迹"带偏，
    而墨迹高度只取决于字形本身。
    """
    uniq = "".join(dict.fromkeys([c for c in chars if c and not c.isspace()]))
    if not uniq or not font_path:
        return {}
    _glyph_ink_at100(font, uniq, font_path)
    return _GLYPH_HEIGHT_CACHE.get((getattr(font, "name", ""), uniq), {})


def text_ink_height(band, zoom: float) -> float:
    """一行文字的**字形墨迹高度**（pt）：各墨迹游程竖向跨度的中位数。

    ★ 不能用"逐列竖向跨度的中位数"：列与列之间大量是笔画空隙，
      中位数会被拉低一半以上（实测 13.8pt 的中文字量出 7.83pt）。
      按**游程**取中位数才对 —— 一条游程就是一个字（或字的一部分）。
    """
    import numpy as np

    runs = _cluster_runs(band)
    if not runs:
        return 0.0
    ext = []
    for a, b in runs:
        rows = np.where(band[:, a:b].any(axis=1))[0]
        if rows.size:
            ext.append(float(rows.max() - rows.min() + 1) / zoom)
    return float(np.median(ext)) if ext else 0.0


def _dp_align_boundaries(col_ink, start_px: int, end_px: int,
                         exp_widths: list[float], win_ratio: float = 0.40):
    """用动态规划把「预期字宽序列」和「实际墨迹列投影」做最优对齐。

    ★ 这是逐字框精度问题的正解。

    原来逐字边界纯粹来自"比例估算"：把参考字体的字宽按行宽等比缩放。
    参考字体和实际字体不同、行宽里又混着下划线和前后留白时，
    误差会平摊到每个字上，框就整体偏移（实测有 2~3pt 的偏差）。

    这里改成：以估算位置为基准开一个搜索窗，在窗内用 DP 找一组
    **单调递增**的边界，使代价最小。代价由两项组成：
      · 宽度偏差 —— 各段宽度要接近预期字宽
      · 落点惩罚 —— 边界要落在**墨迹空隙**上（字符分界处本来就是空白列）

    这样边界会自己"吸附"到真实的字间空隙上，而不是停在估算位置。
    """
    import numpy as np
    n = len(exp_widths)
    if n == 0 or end_px <= start_px:
        return None
    # 每行（seg 段）的墨迹总量，用于"落点惩罚"
    if col_ink is None or len(col_ink) == 0:
        return None

    total_exp = sum(exp_widths) or 1.0
    scale = (end_px - start_px) / total_exp
    cum = [0.0]
    for w in exp_widths:
        cum.append(cum[-1] + w * scale)

    win = max(2, int(win_ratio * (end_px - start_px) / max(1, n)))
    INF = float("inf")

    # ★ 两项代价必须归一到同一量级，否则其中一项会完全压制另一项。
    # 早先落点惩罚是"墨迹像素数 × 6"，而宽度代价是"像素偏差 × 1" ——
    # 一行 13pt 的字在 6 倍下每个字的墨迹像素数有好几百，
    # 惩罚动辄上千，宽度代价最大才十几。结果 DP 完全被墨迹的微小起伏牵着走，
    # 字宽先验形同不存在：重度粘连的行（25 字只有 5 条墨迹游程）里
    # 边界落在哪里只看剖面怎么抖，个别字错位 3~4pt。
    # 改成用"本列墨迹 ÷ 该行墨迹列的中位数"，再乘一个权重：
    # 落在真空隙上是 0，落在笔画正中约等于权重，与宽度代价可比。
    _ink_ref = 1.0
    try:
        import numpy as _np
        _nz = _np.asarray(col_ink)[_np.asarray(col_ink) > 0]
        if _nz.size:
            _ink_ref = max(1.0, float(_np.median(_nz)))
    except Exception:
        pass
    _INK_W = 20.0

    def gap_penalty(pos: int) -> float:
        """边界落在有墨迹的列上要罚分（按本行墨迹量归一）。"""
        lo = max(0, pos - 1)
        hi = min(len(col_ink), pos + 2)
        if hi <= lo:
            return 0.0
        return float(col_ink[lo:hi].sum()) / (3.0 * _ink_ref) * _INK_W

    # dp[i][pos] = 放完第 i 个边界（落在 pos）的最小代价
    prev = {}
    p0 = start_px
    prev[p0] = gap_penalty(p0)
    for i in range(1, n):
        cur = {}
        exp_pos = start_px + cum[i]
        lo = max(prev_pos_min := start_px + int(cum[i - 1]) - win,
                 start_px + 1)
        _ = prev_pos_min
        lo = max(start_px + 1, int(exp_pos) - win)
        hi = min(end_px - (n - i), int(exp_pos) + win)
        if hi <= lo:
            return None
        for pos in range(lo, hi + 1):
            best = INF
            for ppos, pcost in prev.items():
                seg = pos - ppos
                if seg <= 0:
                    continue
                c = pcost + abs(seg - exp_widths[i - 1] * scale) * 1.0
                if c < best:
                    best = c
            if best < INF:
                cur[pos] = best + gap_penalty(pos)
        if not cur:
            return None
        prev = cur

    # 收尾：最后一个边界固定在 end_px
    best, best_pos = INF, None
    for ppos, pcost in prev.items():
        seg = end_px - ppos
        if seg <= 0:
            continue
        c = pcost + abs(seg - exp_widths[-1] * scale)
        if c < best:
            best, best_pos = c, ppos
    if best_pos is None:
        return None

    # 回溯
    bounds = [end_px]
    pos = best_pos
    bounds.append(pos)
    for i in range(n - 2, 0, -1):
        exp_pos = start_px + cum[i]
        lo = max(start_px + 1, int(exp_pos) - win)
        hi = min(end_px - (n - i), int(exp_pos) + win)
        cand = None
        cbest = INF
        for ppos in range(lo, hi + 1):
            if ppos >= bounds[-1]:
                break
            seg = bounds[-1] - ppos
            if seg <= 0:
                continue
            c = abs(seg - exp_widths[i] * scale)
            if c < cbest:
                cbest, cand = c, ppos
        if cand is None:
            return None
        bounds.append(cand)
    bounds.append(start_px)
    bounds.reverse()
    return bounds


BAND_MARGIN = 2.0   # band 在 OCR 框外留的余量（pt）—— 见 BAND_MARGIN 的说明


def per_char_boxes(page: pymupdf.Page, block: Block,
                   *, zoom: float = 6.0, pad: float | None = None):
    """逐字算出各自的紧致墨迹框 —— 应对文字倾斜、弯曲、行内字号不一致。

    ★ 两条铁律，都是踩坑换来的：

    1. **必须按行来算，不能按块**。
       早期版本在一个字的列范围里、**整个块的高度上**找墨迹 ——
       两行的块里，同一个列范围上下两行都有墨迹，框就被拉成贯穿两行的竖条
       （用户截图里"标题两行被识别成一行"其实就是这个：块本来是两行，
        但逐字框画成了贯穿两行的条）。

    2. **列范围必须和邻居划清界线**。
       逐字列范围来自 OCR 的比例估算，相邻字常互相重叠，
       于是框会盖住旁边的字（实测"家庭住址"那行多数字的框都横跨两个字）。
       用相邻字 bbox 的中点作为分界，各自守住自己的那一格。

    返回 [(Char, (x0,y0,x1,y1)), ...]。
    """
    import numpy as np

    # ★ 外扩量只留 1 个测量像素。
    # 早先写死 0.4pt，理由是"留点余量免得切掉字的边缘"，但真值实测显示
    # 这个余量是**四边误差的主要来源**：中位误差 0.36~0.52pt，几乎正好等于它。
    # 用户要的是"边框贴着字"，而墨迹边界本身就是答案，余量只会让框虚胖。
    if pad is None:
        pad = 1.0 / zoom

    y_top = min(l.bbox[1] for l in block.lines) - BAND_MARGIN
    y_bot = max(l.bbox[3] for l in block.lines) + BAND_MARGIN
    x_l = min(l.bbox[0] for l in block.lines) - BAND_MARGIN
    x_r = max(l.bbox[2] for l in block.lines) + BAND_MARGIN

    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                             clip=pymupdf.Rect(x_l, y_top, x_r, y_bot), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
    except Exception:
        return [(c, tuple(c.bbox)) for line in block.lines
                for sp in line.spans for c in sp.chars]

    if arr.size == 0:
        return [(c, tuple(c.bbox)) for line in block.lines
                for sp in line.spans for c in sp.chars]

    gray = arr.mean(axis=2)
    bg = float(np.median(gray))
    ink_all = gray < (bg - 40)

    out: list = []
    for line in block.lines:
        pads = [c for sp in line.spans for c in sp.chars]
        if not pads:
            continue
        widths_px = [max(1, int((c.bbox[2] - c.bbox[0]) * zoom)) for c in pads]
        med_w = float(np.median(widths_px))
        min_w = max(2, int(med_w * 0.45))

        n0 = len(out)          # 本行的框从 out[n0:] 开始
        # 只在这一行的竖向条带里找墨迹 —— 铁律 1
        ly0 = max(0, int((line.bbox[1] - y_top) * zoom))
        ly1 = min(ink_all.shape[0], int((line.bbox[3] - y_top) * zoom))
        if ly1 <= ly0:
            ly0, ly1 = 0, ink_all.shape[0]
        band = ink_all[ly0:ly1]

        # ★ 先判本行是否倾斜 —— 这个判断要**在两处**用到，必须在切分之前做。
        # 倾斜行上墨迹跨度是斜向的：竖列切分与"墨迹顶到格子边界"的粘连判据
        # 都会失效（字间空隙成了斜带，任何一条竖直列都能切到它）。
        line_drop_pt, line_h_pt = line_tilt_measure(band, zoom)
        line_tilted = bool(line_drop_pt > 0.35 * max(3.0, line_h_pt))

        # ★ 首选「墨迹簇一对一」：墨迹本身就是答案，不需要任何字宽先验，
        #   精度上限最高。仅当簇数与字数相等**且**逐字宽度对得上才采用。
        # ★ 其次「拟合参考字体 + DP」：参考字体用实测打分挑，不按脚本猜。
        # ★ 两者都假设文字水平 —— 倾斜行上字间空隙成了斜带，竖列切分与
        #   粘连判据都会失效（实测倾斜行误差从 0.44/0.18 劣化到 0.72/1.29），
        #   所以倾斜行跳过它们，改用下面以 OCR 框位为搜索起点的那条 DP。
        line_bounds = None
        trust_bounds = False
        glyph_offsets: dict = {}
        # 先置空：下面只在"非倾斜且拟合成功"时赋值，
        # 倾斜行或拟合失败时它们仍必须是已绑定的名字
        _font = _path = None
        _size = 0.0
        if not line_tilted:
            try:
                _txt = "".join(c.c for c in pads)
                fit = _fit_ref_font(band, x_l, zoom, _txt)
                if fit is not None:
                    _font, _path, _dp_bounds, _size = fit
                    cb = _segment_by_clusters(
                        _cluster_runs(band), zoom, _txt, _font, _path, _size)
                    if cb is not None:
                        line_bounds = [x_l + v / zoom for v in cb]
                        trust_bounds = True
                    else:
                        line_bounds = [x_l + v / zoom for v in _dp_bounds]
                    glyph_offsets = glyph_ink_offsets(_font, _txt, _size, _path)
            except Exception:
                line_bounds = None

        # 墨迹分段失败时退回旧的簇/DP 方案
        try:
            col_has = band.any(axis=0)
            runs = []
            st = None
            for _i, _v in enumerate(col_has):
                if _v and st is None:
                    st = _i
                elif not _v and st is not None:
                    runs.append([st, _i])
                    st = None
            if st is not None:
                runs.append([st, len(col_has)])
            if runs and line_bounds is None:
                wpx = [b - a for a, b in runs]
                medpx = float(np.median(wpx))
                merge_px = max(0, int(0.18 * medpx))
                merged = [runs[0][:]]
                for a, b in runs[1:]:
                    if a - merged[-1][1] <= merge_px:
                        merged[-1][1] = b
                    else:
                        merged.append([a, b])
                if len(merged) == len(pads):
                    line_bounds = [x_l + m[0] / zoom for m in merged]
                    line_bounds.append(x_l + merged[-1][1] / zoom)
        except Exception:
            line_bounds = None

        # ★ 其次做一次 DP 对齐，把逐字边界吸附到真实的墨迹空隙上。
        # 边界由"比例估算"给出的位置只是搜索起点，真正的落点由
        # "各段宽度接近预期 + 边界落在空白列"这两个条件共同决定。
        try:
            col_ink = band.sum(axis=0).astype(float)
            # 把列墨迹做一点平滑，避免单个噪声像素影响落点
            if len(col_ink) > 2:
                k = np.ones(3) / 3.0
                col_ink = np.convolve(col_ink, k, mode="same")
            sx0 = int((pads[0].bbox[0] - x_l) * zoom)
            sx1 = int((pads[-1].bbox[2] - x_l) * zoom)
            sx0 = max(0, min(sx0, band.shape[1] - 2))
            sx1 = max(sx0 + 1, min(sx1, band.shape[1]))
            exp_w = [max(1.0, (c.bbox[2] - c.bbox[0]) * zoom) for c in pads]
            if line_bounds is None:
                b = _dp_align_boundaries(col_ink, sx0, sx1, exp_w)
                if b is not None and len(b) == len(pads) + 1:
                    line_bounds = [x_l + v / zoom for v in b]
        except Exception:
            line_bounds = None

        # 参考字体的字形墨迹偏移 —— 只给「粘连边修正」用。
        # 水平行上面已经连同字体拟合一起算好了；倾斜行或拟合失败时在这里补一次。
        line_size_pt = 0.0
        if not glyph_offsets and line_bounds is not None \
                and len(line_bounds) == len(pads) + 1:
            try:
                from .ocr import _ref_font_for, _ref_font_path_for
                _txt2 = "".join(c.c for c in pads)
                _rf = _ref_font_for(_txt2)
                if _rf is not None:
                    _raw100 = [max(0.1, _rf.text_length(ch, fontsize=100.0))
                               for ch in _txt2]
                    line_size_pt = (line_bounds[-1] - line_bounds[0]) \
                        * 100.0 / (sum(_raw100) or 1.0)
                    glyph_offsets = glyph_ink_offsets(
                        _rf, _txt2, line_size_pt, _ref_font_path_for(_txt2))
            except Exception:
                glyph_offsets = {}

        # 粘连修正只在水平行上启用（倾斜行的空隙是斜带，判据会误触发）。
        # 用簇边界时也跳过：簇边界已经把字切开，墨迹本来就顶在格子边上，
        # 再"修正"等于把正确的边界又推走。
        fuse_fix_ok = bool(glyph_offsets) and not line_tilted \
            and not trust_bounds

        for i, c in enumerate(pads):
            # 列范围：以相邻字 bbox 的中点为界，守住自己那一格 —— 铁律 2
            cx0 = c.bbox[0]
            cx1 = c.bbox[2]
            if i > 0:
                prev = pads[i - 1]
                mid = (prev.bbox[2] + c.bbox[0]) / 2
                if mid > prev.bbox[0]:
                    cx0 = max(cx0, mid)
            if i + 1 < len(pads):
                nxt = pads[i + 1]
                mid = (c.bbox[2] + nxt.bbox[0]) / 2
                if mid > c.bbox[0]:
                    cx1 = min(cx1, mid)
            if cx1 <= cx0:
                cx0, cx1 = c.bbox[0], c.bbox[2]
            if line_bounds is not None:
                # DP 给出的边界优先 —— 它落在真实空隙上，比估算准
                cx0 = line_bounds[i]
                cx1 = line_bounds[i + 1]

            px0 = int((cx0 - x_l) * zoom)
            px1 = int((cx1 - x_l) * zoom)
            # 簇边界是实测出来的答案，不能再被"最小宽度"撑开 ——
            # 括号、冒号本来就窄（实测「（」只有 3.7pt，而最小宽度是 6.1pt），
            # 一撑就比原字宽出 65%
            if px1 - px0 < min_w and not trust_bounds:
                mid = (px0 + px1) // 2
                px0, px1 = mid - min_w // 2, mid + (min_w - min_w // 2)
            px0 = max(0, min(px0, band.shape[1] - 1))
            px1 = max(px0 + 1, min(px1, band.shape[1]))

            col = band[:, px0:px1]
            # ★ 竖向范围要取**列范围的中段**来量，不能直接用整个列范围。
            #   列范围的边缘常含进邻居的一两列墨迹，而邻居在整行高度上都有墨，
            #   矮字形于是被拉成整行高 —— 实测「一」（单一横画，真值高 1.17pt）
            #   被量成 11.16pt，四边均值 2.65pt 直接超标。
            #   试过"按行覆盖率阈值"剔除（要求一行墨迹横跨足够多列），
            #   反而更差（137→133/138）：汉字侧笔所在的行覆盖率本来就低，
            #   被一并裁掉了。取中段则两全：邻居的渗墨只在边缘，
            #   而字自己的笔画（含上下横画）都穿过中段。
            _w = px1 - px0
            _a = px0 + int(_w * 0.2)
            _b = px1 - int(_w * 0.2)
            if _b <= _a:
                _a, _b = px0, px1
            rows = np.where(band[:, _a:_b].any(axis=1))[0]
            if len(rows) == 0:
                out.append((c, tuple(c.bbox)))
                continue
            ty0 = y_top + (ly0 + rows.min()) / zoom - pad
            ty1 = y_top + (ly0 + rows.max() + 1) / zoom + pad
            cols = np.where(col.any(axis=0))[0]
            tx0 = x_l + (px0 + cols.min()) / zoom - pad
            tx1 = x_l + (px0 + cols.max() + 1) / zoom + pad

            # ★ 粘连边的度量修正。
            # 低分辨率扫描件上，相邻数字的字间空隙不足一个源像素，
            # 两个字的墨迹在像素里连成一体 —— 此时"量墨迹"量到的是整团的外沿：
            # 框的左边切掉自己、右边吃进邻居（真值实测右偏 2.88pt）。
            # 判据：本字格的墨迹一直顶到格子边界，说明邻居的墨渗进来了。
            # 这时改用参考字体的字形墨迹偏移，并借**没粘连**的那一侧反推笔位原点。
            _off = glyph_offsets.get(c.c) if fuse_fix_ok else None
            if _off is not None:
                _a, _b = _off
                _pred_w = _b - _a
                _fused_l = cols.min() <= 0
                _fused_r = cols.max() >= (px1 - px0) - 1
                if (_fused_l or _fused_r) and (tx1 - tx0 + 2 * pad) > _pred_w + 0.25:
                    if not _fused_l:
                        _pen = (tx0 + pad) - _a
                    elif not _fused_r:
                        _pen = (tx1 - pad) - _b
                    else:
                        _pen = line_bounds[i]
                    if _fused_l:
                        tx0 = _pen + _a - pad
                    if _fused_r:
                        tx1 = _pen + _b + pad
            out.append((c, (tx0, ty0, tx1, ty1)))

        # ★ 竖向范围复测：细笔画在低分辨率下被反锯齿摊薄，
        #   一个字自己的列切片可能只有 1 像素宽，量不到笔画最淡的上下端，
        #   框就偏矮 —— 实测某行 'l' 只量到 3.7pt 高，而同行的 'h' 是 7.5pt
        #   （'l' 是带升部的字母，本该与 'h' 同高）。
        #   对"高度明显偏矮且不是标点"的字，用略宽的列带重测竖向范围：
        #   同一行的相邻列竖向范围基本一致，所以借邻居的列不会把框拉高，
        #   只会把被摊薄而漏掉的笔画端点补回来。
        line_boxes = out[n0:]
        if len(line_boxes) >= 3:
            hs = [r[3] - r[1] for _c, r in line_boxes]
            med_h = float(np.median(hs))
            # ★ 判据是"这个字形本该有多高"，不是"它比同行矮多少"。
            #   只看"矮于同行"会把本来就矮的字补高 —— 实测「一」（单一横画，
            #   真值高 1.17pt）被补成 11.16pt，四边均值 2.65pt 直接超标。
            #   所以拿参考字形的墨迹高做基准：预期高度不够的，本来就不该满高。
            _exp_h: dict = {}
            _line_txt = "".join(c.c for c in pads)
            if _font is not None and _path:
                try:
                    _raw_h = glyph_ink_height(_font, _line_txt, _path)
                    _k = _size / 100.0
                    _exp_h = {ch: v * _k for ch, v in _raw_h.items()}
                except Exception:
                    _exp_h = {}
            for k, (c, r) in enumerate(line_boxes):
                if r[3] - r[1] >= 0.55 * med_h:
                    continue
                _eh = _exp_h.get(c.c, 0.0)
                if _eh <= 0 or _eh < 0.55 * med_h:
                    continue          # 这个字形本来就矮（一、标点、下划线）
                a = max(0, int((r[0] - x_l) * zoom) - 2)
                b = min(band.shape[1], int((r[2] - x_l) * zoom) + 2)
                if b <= a:
                    continue
                sub = band[:, a:b]
                rows = np.where(sub.any(axis=1))[0]
                if not rows.size:
                    continue
                ty0b = y_top + (ly0 + rows.min()) / zoom - pad
                ty1b = y_top + (ly0 + rows.max() + 1) / zoom + pad
                # 只向上补：补出来的高度不得超过同行的中位高，避免把
                # 小字号字被邻居的大字号带高
                if (ty1b - ty0b) <= med_h * 1.15 and (ty1b - ty0b) > (r[3] - r[1]):
                    line_boxes[k] = (c, (r[0], ty0b, r[2], ty1b))

    # 塌缩兜底：明显矮于同行的框，用它自己原框的中心补一个。
    #
    # ★ 兜底必须**按字形预期高度**判断，不能只看"比同行矮"。
    #   有些字本来就矮：单一横画的「一」（墨迹高约 0.1 em）、
    #   全角冒号（两个点）、下划线。拿"矮于同行一半"去判它们，
    #   会把一个 1.2pt 的框补成整行高 —— 实测「一」被补成 11.16pt，
    #   四边均值 2.65pt 直接超标。
    #   兜底真正要救的是"测量失败"（框塌成一条线），那种情况远小于预期高度。
    _hh_cache: dict = {}
    if _font is not None and _path:
        try:
            _hh_cache = glyph_ink_height(
                _font, "".join(c.c for c in
                               [cc for line2 in block.lines
                                for sp2 in line2.spans for cc in sp2.chars]),
                _path)
        except Exception:
            _hh_cache = {}
    by_line_heights: dict[int, list[float]] = {}
    for idx, (c, r) in enumerate(out):
        by_line_heights.setdefault(c_line_index(block, c), []).append(r[3] - r[1])
    fixed = []
    for c, r in out:
        hs = by_line_heights.get(c_line_index(block, c)) or [r[3] - r[1]]
        med = float(np.median(hs)) if hs else (r[3] - r[1])
        if med > 0 and (r[3] - r[1]) < 0.5 * med:
            exp_h = _hh_cache.get(c.c, 0.0) * (_size / 100.0)
            if 0.0 < exp_h < 0.5 * med:
                fixed.append((c, r))       # 这个字形本来就矮，不该补
                continue
            cy = (c.bbox[1] + c.bbox[3]) / 2
            fixed.append((c, (r[0], cy - med / 2, r[2], cy + med / 2)))
        else:
            fixed.append((c, r))
    return fixed


def c_line_index(block: Block, ch) -> int:
    """字符属于第几行。"""
    for i, line in enumerate(block.lines):
        for sp in line.spans:
            for c in sp.chars:
                if c is ch:
                    return i
    return 0


def measure_block_ink(
    page: pymupdf.Page, block: Block, *, expand: bool = True,
) -> InkMetrics:
    """对外暴露：量一个块的墨迹，供 UI 预览覆盖范围。"""
    if not block.lines:
        raise ValueError("块内没有文本行")
    if len(block.lines) == 1:
        return measure_ink(page, block.lines[0].bbox, expand=expand)
    return _merge_ink(page, block, expand)


def _merge_ink(page: pymupdf.Page, block: Block, expand: bool) -> InkMetrics:
    """多行块：逐行实测后合并为一个墨迹矩形。

    基准行取墨迹像素最多的那一行 —— 若取第一行而它恰好是空行或标点行，
    背景色和文字色都会退化成默认值（白底黑字），覆盖上去就是一块白斑。
    """
    parts = [measure_ink(page, l.bbox, expand=expand) for l in block.lines]
    base = max(parts, key=lambda p: p.ink_pixels)
    merged = InkMetrics(
        rect=(min(p.rect[0] for p in parts),
              min(p.rect[1] for p in parts),
              max(p.rect[2] for p in parts),
              max(p.rect[3] for p in parts)),
        height=max(p.height for p in parts),
        color=base.color, bg=base.bg, bg_std=base.bg_std,
        ink_pixels=sum(p.ink_pixels for p in parts),
        run_extended=max(p.run_extended for p in parts),
        # 字重判据必须从基准行带过来 —— 早期漏了这两个字段，
        # 所有多行块的字重判定都退化成"常规体"，粗体原文会被画成细体
        stroke_ratio=base.stroke_ratio,
        ink_density=base.ink_density,
    )
    return merged
