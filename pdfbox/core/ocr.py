"""OCR 引擎：让图片型 PDF 也能编辑。

适用于没有文字层的 PDF（扫描件、截图、幻灯片导出）。
这类文档的内容是位图，PDF 里压根没有可擦除的字形 ——
所以编辑方式与文本型 PDF 有本质区别：

    文本型 PDF：擦除字形（真删除）→ 用匹配字体重画
    图片型 PDF：用背景色覆盖原像素 → 用匹配字体重画

前者可以做到"未改动区域字节级零差异"，后者做不到 ——
原像素无法真正移除，只能盖住。这是物理限制，任何工具都一样，
UI 和报告里必须如实告知用户。

流程：
    1. 检测页面是否为图片型（无文字层 或 文字极少但图像占满版面）
    2. 对页面渲染图跑 OCR，得到带坐标的文本行
    3. 按行/段落规则聚合成块，接入统一的结构模型
    4. 编辑时：采样背景色 → 覆盖原区域 → 绘制新文字
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
from pathlib import Path

import pymupdf

from .model import (
    KIND_HEADING,
    KIND_PARAGRAPH,
    LAYER_FLOW,
    Block,
    DocStructure,
    Line,
    PageStructure,
    Span,
)

# 判定为"图片型页面"的条件
TEXT_CHAR_MIN = 20          # 页面文字少于这么多字符，视为没有文字层
IMAGE_COVER_MIN = 0.5       # 图像覆盖页面面积比例下限

# OCR 渲染倍率：越高越准但越慢。3.0 对应约 216 DPI，中文识别质量足够
OCR_ZOOM = 3.0

_engine = None


def _get_engine():
    """惰性加载 RapidOCR，避免没用到 OCR 时也拖慢启动。"""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


@dataclass
class OcrLine:
    text: str
    bbox: tuple[float, float, float, float]      # PDF 坐标（pt）
    confidence: float
    quad: list[tuple[float, float]] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


def is_image_page(page: pymupdf.Page) -> tuple[bool, str]:
    """判断页面是否为图片型（无可用文字层）。返回 (是否图片型, 理由)。"""
    text = page.get_text().strip()
    # OCR 出来的空格字符不算文字
    chars = sum(1 for ch in text if not ch.isspace())

    page_area = page.rect.width * page.rect.height
    img_area = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        img_area += r.get_area()
    img_ratio = img_area / page_area if page_area else 0.0

    if chars < TEXT_CHAR_MIN and img_ratio >= IMAGE_COVER_MIN:
        return True, f"文字仅 {chars} 字符、图像覆盖 {img_ratio*100:.0f}% 版面 → 图片型"
    if chars < TEXT_CHAR_MIN:
        return True, f"文字仅 {chars} 字符（低于阈值 {TEXT_CHAR_MIN}）→ 视为无文字层"
    return False, f"文字 {chars} 字符 → 文本型"


def ocr_page(page: pymupdf.Page, zoom: float = OCR_ZOOM) -> list[OcrLine]:
    """对页面渲染图做 OCR，返回 PDF 坐标下的文本行。"""
    engine = _get_engine()
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    import numpy as np
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]

    result, _elapse = engine(img)
    if not result:
        return []

    inv = 1.0 / zoom
    lines: list[OcrLine] = []
    for item in result:
        quad, text, conf = item[0], item[1], item[2]
        xs = [p[0] * inv for p in quad]
        ys = [p[1] * inv for p in quad]
        lines.append(OcrLine(
            text=text,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            confidence=float(conf),
            quad=[(p[0] * inv, p[1] * inv) for p in quad],
        ))
    return lines


# --------------------------------------------------------------- 行→块聚合

LINE_MERGE_GAP = 0.8        # 行间距小于 比例×行高 视为同段
LEFT_TOL = 6.0              # 左边界对齐容差
SIZE_RATIO = 1.25           # 行高比超过此值判为标题


def _median_height(lines: list[OcrLine]) -> float:
    hs = [l.height for l in lines if l.height > 0]
    return statistics.median(hs) if hs else 12.0


def group_ocr_blocks(lines: list[OcrLine], page_no: int,
                     pixel_source=None) -> list[Block]:
    """把 OCR 行聚合成块，接入统一的结构模型。

    规则比文本型 PDF 简单：OCR 只给出文字和方框，没有字体/字号信息，
    所以只能靠行高、行距、左边界对齐这几个版面特征。
    """
    if not lines:
        return []

    lines = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    body_h = _median_height(lines)

    text_width = 0.0
    for l in lines:
        text_width = max(text_width, l.width)

    blocks: list[Block] = []
    cur: list[OcrLine] = []
    expected_gap: float | None = None

    def flush():
        nonlocal cur, expected_gap
        if cur:
            blocks.append(_lines_to_block(cur, page_no, body_h, pixel_source))
        cur = []
        expected_gap = None

    for line in lines:
        if not cur:
            cur = [line]
            continue
        prev = cur[-1]
        gap = line.bbox[1] - prev.bbox[3]
        overlap = _h_overlap(prev, line)
        aligned = abs(line.bbox[0] - prev.bbox[0]) <= LEFT_TOL
        height_ok = abs(line.height - prev.height) <= 0.35 * max(prev.height, line.height)
        gap_ok = gap <= LINE_MERGE_GAP * body_h
        if expected_gap is not None and gap_ok:
            gap_ok = abs(gap - expected_gap) <= 0.6 * max(expected_gap, 1.0)

        # ★ 光看"水平重叠"会把表单行错并成一个段落。
        # 实测「丙方：（…）」「家庭住址…」「紧急电话…」三行是三个独立表单项，
        # 它们从 x=96 / 126 / 302 起 —— 位置完全不同，但水平重叠率高达 0.886，
        # 按重叠就合并了，于是框变成一个横跨错乱的大块。
        #
        # 加两条判据：
        #   左边界 —— 首行允许 2.5 字宽的缩进，其余行必须紧贴同一左边界
        #   右边界 —— 同段落各行要么右边界接近（两端对齐/满行），
        #             要么本行明显更短且左边界对齐（段落末行）
        left_ok = aligned
        if not left_ok and len(cur) == 1:
            left_ok = abs(line.bbox[0] - prev.bbox[0]) <= 2.5 * max(prev.height, line.height)

        right_diff = abs(line.bbox[2] - prev.bbox[2])
        right_ok = right_diff <= 2.0 * max(prev.height, line.height)
        if not right_ok:
            # 本行明显短于上一行且左边界对齐 → 视为段落末行
            right_ok = (line.bbox[2] < prev.bbox[2] - 0.5 * prev.height) and aligned

        if overlap > 0.3 and height_ok and gap_ok and left_ok and right_ok:
            cur.append(line)
            gaps = [cur[i + 1].bbox[1] - cur[i].bbox[3] for i in range(len(cur) - 1)]
            expected_gap = sum(gaps) / len(gaps) if gaps else None
        else:
            flush()
            cur = [line]
    flush()
    return blocks


def _h_overlap(a: OcrLine, b: OcrLine) -> float:
    lo = max(a.bbox[0], b.bbox[0])
    hi = min(a.bbox[2], b.bbox[2])
    if hi <= lo:
        return 0.0
    shorter = min(a.width, b.width)
    return (hi - lo) / shorter if shorter > 0 else 0.0


def _pixel_source(page: pymupdf.Page, zoom: float = OCR_ZOOM):
    """渲染页面像素，供逐字边界吸附到墨迹空隙用。

    吸附能让逐字边界准到像素级 —— 比例估算只是近似，
    而字符之间的真实间隙是"空白列"，吸过去就准了。
    """
    import numpy as np
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        return (arr, zoom)
    except Exception:
        return None


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF or 0x3000 <= o <= 0x303F
            or 0xFF00 <= o <= 0xFFEF)


_REF_FONTS: dict[str, object] = {}
_REF_FONT_PATHS: dict[str, str] = {}
_REF_FONT_ALL: list | None = None


def _ref_font_candidates() -> list:
    """所有候选参考字体，按 (Font, 路径) 返回。

    ★ 为什么不再按"文本是不是 ASCII"挑字体：
      一份中文文档里的数字、邮箱、电话，用的往往还是同一个中文字体。
      照脚本切成 Arial，字宽比例会整体错位 —— 实测某邮箱串里
      '.' 的框被 Arial 的 0.25em 先验撑到 2.7pt，而墨迹实测只有 1.7pt；
      同一行数字被切成 5.3pt，实测字宽是 5.2pt。
      改成把候选字体的**切分质量实测打分**，让数据决定用哪个。
    """
    global _REF_FONT_ALL
    if _REF_FONT_ALL is not None:
        return _REF_FONT_ALL
    from .fonts import _find_file
    out: list = []
    for c in ("simsun.ttc", "NotoSansSC-VF.ttf", "msyh.ttc",
              "arial.ttf", "calibri.ttf", "times.ttf"):
        p = _find_file(c)
        if not p:
            continue
        try:
            out.append((pymupdf.Font(fontfile=p), p))
        except Exception:
            continue
    _REF_FONT_ALL = out
    return out


def _ref_font_path_for(text: str) -> str:
    """返回该脚本的参考字体**文件路径**（没有则空串）。

    单独暴露路径，是因为有的地方需要重新渲染这个字体来取字形墨迹范围
    （pymupdf.Font 对象本身不留字体文件路径）。
    """
    has_cjk = any(_is_cjk(c) for c in text)
    key = "cjk" if has_cjk else "latin"
    if key in _REF_FONT_PATHS:
        return _REF_FONT_PATHS[key]
    from .fonts import _find_file
    cands = (("simsun.ttc", "NotoSansSC-VF.ttf", "msyh.ttc") if has_cjk
             else ("arial.ttf", "calibri.ttf", "times.ttf"))
    path = ""
    for c in cands:
        p = _find_file(c)
        if p:
            path = p
            break
    _REF_FONT_PATHS[key] = path
    return path


def _ref_font_for(text: str):
    """按文字脚本挑一个参考字体，用于估算逐字宽度。

    必须用真字体的度量：西文比例字体的字符宽度差好几倍
    （i 窄 W 宽），均匀等分会让逐字边界完全切错。
    """
    has_cjk = any(_is_cjk(c) for c in text)
    key = "cjk" if has_cjk else "latin"
    if key in _REF_FONTS:
        return _REF_FONTS[key]
    font = None
    path = _ref_font_path_for(text)
    if path:
        try:
            font = pymupdf.Font(fontfile=path)
        except Exception:
            font = None
    _REF_FONTS[key] = font
    return font


def detect_glyph_clusters(has_ink, zoom: float, x0: float,
                          merge_ratio: float = 0.22):
    """从列墨迹投影里找出「字形簇」——每一簇对应一个视觉上的字符。

    做法：把"有墨迹的列"连成段，相邻段之间的小空隙（小于字符宽度的
    merge_ratio 倍）合并 —— 字母 'i' 的点与竖笔、'%' 的上下两半
    都会被拆成多段，必须合回一个字。

    这是"从间隙反推空格"的基础：簇之间的**大**空隙就是空格。
    """
    runs: list[list[int]] = []
    st = None
    for i, v in enumerate(has_ink):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append([st, i])
            st = None
    if st is not None:
        runs.append([st, len(has_ink)])
    if not runs:
        return []

    widths = [b - a for a, b in runs]
    med_w = float(np.median(widths)) if widths else 1.0
    merge_px = max(1, int(merge_ratio * med_w))

    merged: list[list[int]] = [runs[0][:]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= merge_px:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(x0 + a / zoom, x0 + b / zoom) for a, b in merged]


def find_space_positions(has_ink, zoom: float, x0: float):
    """从墨迹列投影里找出「词间空隙」的位置（pt）。

    判据：先把所有墨迹间隙量出来，取中位数当"字内间隙"的基准，
    明显大于它的就是词间空隙。

    为什么不用"字形簇数量 == 字符数量"来对齐：密集正文里相邻字母常粘连，
    簇数远少于字数（实测一整行 41 个字母只分出 12 个簇），
    按簇对齐会直接失败退回比例估算，等于没修。
    而"哪里有空隙"这个信号和字符怎么切分无关，是可靠的。
    """
    gaps: list[tuple[float, float, float]] = []   # (起点pt, 终点pt, 宽度pt)
    st = None
    for i, v in enumerate(has_ink):
        if not v and st is None:
            st = i
        elif v and st is not None:
            gaps.append((x0 + st / zoom, x0 + i / zoom, (i - st) / zoom))
            st = None
    if not gaps:
        return []

    widths = [g[2] for g in gaps if g[2] > 0]
    if not widths:
        return []
    med = float(np.median(widths))
    # 词间空隙一般明显宽于字内笔画间隙；同时给一个绝对下限防误判
    thr = max(0.9, 2.2 * med)
    return [(g[0] + g[1]) / 2 for g in gaps if g[2] >= thr]


def reinsert_spaces(text: str, space_pts: list[float], x0: float,
                    advances: list[float]) -> tuple[str, list[float]]:
    """按空隙位置把空格补回 OCR 文字，并给出逐字边界。

    做法：用字体度量逐字向右推进；每当推进位置越过一个词间空隙的中心，
    就在那里插一个空格，并把位置跳到空隙之后。

    不依赖"簇数 == 字数"这个前提，因此密集粘连的正文也能处理。
    """
    chars = [c for c in text if not c.isspace()]
    if not chars or not space_pts:
        return text, []
    if len(advances) != len(chars):
        return text, []

    out: list[str] = []
    bounds: list[float] = [x0]
    x = x0
    si = 0
    for ch, adv in zip(chars, advances):
        # 把落在当前字之前的空隙都变成空格
        while si < len(space_pts) and space_pts[si] <= x + adv * 0.5:
            if out:
                out.append(" ")
            x = max(x, space_pts[si] + adv * 0.35)
            bounds[-1] = x
            si += 1
        out.append(ch)
        bounds.append(x + adv)
        x += adv
    while si < len(space_pts):
        out.append(" ")
        si += 1
    return "".join(out), bounds


def align_text_to_clusters(text: str, clusters: list[tuple[float, float]]):
    """把 OCR 文字和字形簇对齐，按簇间大空隙插入空格，并给出逐字边界。

    ★ 这是解决"OCR 丢空格"的关键：OCR 常把整行识别成一个连在一起的词
      （实测 'Start doing background research on your project'
      被识别成 'Startdoingbackgroundresearchonyourproject'）。
      字符数虽然对得上，但按比例估算的逐字位置和实际字形完全错位，
      后续"只改动的字""复用原像素"全都建立在错的坐标上。

      改为：从墨迹簇量出每个字的真实位置，再看簇间空隙 ——
      大空隙就是空格，直接补回去。文字内容仍用 OCR 的结果（它认字是准的），
      位置和空格则用像素说了算。

    返回 (带空格的文字, 逐字边界列表)；对不齐时返回 (None, None) 让调用方兜底。
    """
    chars = [c for c in text if not c.isspace()]
    spaces_hint = [c.isspace() for c in text]
    if not chars or not clusters:
        return None, None

    bounds = [c[0] for c in clusters] + [clusters[-1][1]]

    if len(clusters) != len(chars):
        return None, None   # 连字/粘连等情况，交回比例估算

    # 簇间空隙 → 判定空格
    gaps = [clusters[i + 1][0] - clusters[i][1] for i in range(len(clusters) - 1)]
    widths = [b - a for a, b in clusters]
    med_w = float(np.median(widths)) if widths else 1.0
    # 拉丁文的空格宽度通常约等于 0.25~0.5 个字宽；
    # 字内笔画间隙一般不到 0.15 个字宽，取 0.28 作分界
    space_thr = max(0.9, 0.28 * med_w)

    out: list[str] = []
    for i, ch in enumerate(chars):
        out.append(ch)
        if i < len(gaps) and gaps[i] > space_thr:
            out.append(" ")
    return "".join(out), bounds


def _proportional_bounds(text: str, x0: float, x1: float,
                         font: pymupdf.Font | None, size: float) -> list[float]:
    """按字体度量给出每个字符的 x 边界，再归一化到 [x0, x1]。

    为什么不能均匀等分：`Pitch to Industry` 是比例字体，每个字符宽度天差地别
    （i 窄、W 宽）。均匀等分会把每个字都当 2.69pt 宽 —— 实测逐字定位完全切错，
    而「只改动的那几个字」这个精度要求恰恰依赖逐字边界准确。
    """
    n = len(text)
    if n == 0:
        return [x0, x1]
    advances: list[float]
    if font is not None:
        try:
            advances = [max(0.1, font.text_length(ch, fontsize=size)) for ch in text]
        except Exception:
            advances = [1.0] * n
    else:
        # 没有字体就按「CJK 全宽、西文半宽」估
        advances = [(1.0 if _is_cjk(ch) else 0.5) for ch in text]

    total = sum(advances) or 1.0
    scale = (x1 - x0) / total

    xs = [x0]
    for a in advances:
        xs.append(xs[-1] + a * scale)
    xs[-1] = x1
    return xs


def _snap_to_ink_gaps(arr, zoom: float, y0: float, y1: float,
                      xs: list[float], window: float = 0.6) -> list[float]:
    """把字符边界吸附到最近的空白列。

    比例估算只是近似，而字符之间的真实间隙是"空白列"。
    把边界吸到最近的空白列上，逐字边界就准到像素级。
    """
    import numpy as np

    if arr is None:
        return xs
    h = arr.shape[0]
    py0 = max(0, int(y0 * zoom))
    py1 = min(h, int(y1 * zoom))
    if py1 <= py0:
        return xs
    band = arr[py0:py1]
    has_ink = (band.mean(axis=2) < 200).any(axis=0)
    win = max(1, int(window * zoom))

    snapped = [xs[0]]
    for x in xs[1:-1]:
        c = int(x * zoom)
        lo, hi = max(0, c - win), min(len(has_ink), c + win)
        best = None
        for i in range(lo, hi):
            if not has_ink[i]:
                if best is None or abs(i - c) < abs(best - c):
                    best = i
        snapped.append((best / zoom) if best is not None else x)
    snapped.append(xs[-1])
    return snapped


def _lines_to_block(ocr_lines: list[OcrLine], page_no: int, body_h: float,
                    pixel_source=None) -> Block:
    """把一组 OCR 行包装成结构模型的 Block。

    OCR 没有字体信息，所以构造一个"伪 Span"承载文字与坐标，
    font 留空、size 由行高反推 —— 字体匹配阶段会走"未知字体"的兜底路径。
    """
    omlines: list[Line] = []
    for ol in ocr_lines:
        x0, y0, x1, y1 = ol.bbox
        h = ol.height
        # 字号近似：OCR 行高包含上下留白，经验上约为 0.78 倍行高
        size = max(4.0, h * 0.78)
        # 基线近似：行底减去降部空间
        baseline = y1 - h * 0.18

        # 逐字边界：优先「字体度量推进 + 墨迹空隙补空格」，这才是像素级的
        text = ol.text
        xs = None
        ref_font = _ref_font_for(text)
        if pixel_source is not None:
            arr, zoom = pixel_source
            py0 = max(0, int(y0 * zoom))
            py1 = min(arr.shape[0], int(y1 * zoom))
            px0 = max(0, int(x0 * zoom))
            px1 = min(arr.shape[1], int(x1 * zoom) + 1)
            if py1 > py0 and px1 > px0:
                # ★ 必须裁到该行的 x 范围。
                # 取整页宽度会把页边距也算成"空隙"——实测检测出的 4 个"词间空隙"
                # 全在文字之外（x=45/411/429），真正的词间空隙一个都没抓到。
                band = arr[py0:py1, px0:px1]
                has_ink = (band.mean(axis=2) < 200).any(axis=0)
                space_pts = find_space_positions(has_ink, zoom, x0)
                advances = None
                if ref_font is not None:
                    try:
                        advances = [ref_font.text_length(c, fontsize=size)
                                    for c in text if not c.isspace()]
                        total = sum(advances)
                        if total > 0:      # 归一化到实际行宽，吸收字体差异
                            scale = (x1 - x0) / total
                            advances = [a * scale for a in advances]
                    except Exception:
                        advances = None
                if advances is not None:
                    aligned, bounds = reinsert_spaces(text, space_pts, x0, advances)
                    if bounds and len(bounds) == len(aligned) + 1:
                        text = aligned
                        xs = bounds
                        ol.text = aligned

        if xs is None:
            xs = _proportional_bounds(text, x0, x1, _ref_font_for(text), size)
            if pixel_source is not None:
                arr, zoom = pixel_source
                xs = _snap_to_ink_gaps(arr, zoom, y0, y1, xs)

        chars = []
        from .model import Char
        if len(xs) == len(text) + 1:
            for i, ch in enumerate(text):
                chars.append(Char(c=ch, glyph_id=0,
                                  origin=(xs[i], baseline),
                                  bbox=(xs[i], y0, xs[i + 1], y1)))
        else:
            # 兜底：字符数与边界数不一致时按比例铺开
            cw = (x1 - x0) / max(1, len(text))
            for i, ch in enumerate(text):
                chars.append(Char(c=ch, glyph_id=0,
                                  origin=(x0 + i * cw, baseline),
                                  bbox=(x0 + i * cw, y0, x0 + (i + 1) * cw, y1)))
        span = Span(
            chars=chars, font="", size=size,
            color=(0.0, 0.0, 0.0), direction=(1.0, 0.0), flags=0,
            opacity=1.0, bbox=(x0, y0, x1, y1), seqno=0, page_no=page_no,
        )
        omlines.append(Line(
            spans=[span], page_no=page_no, direction=(1.0, 0.0),
            cross=baseline, along0=x0, along1=x1, bbox=(x0, y0, x1, y1),
        ))

    xs0 = [l.bbox[0] for l in omlines]
    ys0 = [l.bbox[1] for l in omlines]
    xs1 = [l.bbox[2] for l in omlines]
    ys1 = [l.bbox[3] for l in omlines]

    avg_conf = sum(l.confidence for l in ocr_lines) / len(ocr_lines)
    text = "\n".join(l.text for l in ocr_lines)
    is_heading = (
        len(omlines) == 1
        and om_lines_height(omlines) > body_h * SIZE_RATIO
    ) or (len(text) <= 30 and not text.rstrip().endswith(("。", ".", "；", ";")))

    block = Block(
        lines=omlines, page_no=page_no,
        bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
        kind=KIND_HEADING if is_heading else KIND_PARAGRAPH,
        layer=LAYER_FLOW, z=0,
        confidence=round(avg_conf, 3),
        reasons=[f"OCR 识别（平均置信度 {avg_conf:.2f}）"],
        raw={"ocr": True},
    )
    return block


def om_lines_height(lines: list[Line]) -> float:
    return max((l.bbox[3] - l.bbox[1] for l in lines), default=0.0)


# --------------------------------------------------------------- 页面/文档


def ocr_page_structure(
    page: pymupdf.Page, page_no: int, zoom: float = OCR_ZOOM,
) -> PageStructure:
    """对单个图片型页面做 OCR，构造页面结构。"""
    structure = PageStructure(
        page_no=page_no, width=page.rect.width, height=page.rect.height,
    )
    lines = ocr_page(page, zoom=zoom)
    structure.blocks.extend(
        group_ocr_blocks(lines, page_no, _pixel_source(page, zoom)))

    # 图片作为整体块记录（编辑时它们是被覆盖的对象，不参与重排）
    for info in page.get_image_info():
        structure.blocks.append(Block(
            page_no=page_no, bbox=tuple(info["bbox"]),
            kind="image", layer=LAYER_FLOW,
            raw={"width": info.get("width"), "height": info.get("height")},
            confidence=1.0, reasons=["页面图像对象"],
        ))
    return structure


def ocr_document(path: str, zoom: float = OCR_ZOOM, progress=None) -> DocStructure:
    """对整个图片型 PDF 做 OCR，产出与文本型一致的结构模型。

    progress: 可选回调 (已完成页数, 总页数)。
    """
    doc = pymupdf.open(path)
    structure = DocStructure(path=path)

    # 全部页面的行高统计出正文主体字号，供标题判定使用
    all_lines: list[OcrLine] = []
    per_page: dict[int, list[OcrLine]] = {}
    for pno in range(len(doc)):
        lines = ocr_page(doc[pno], zoom=zoom)
        per_page[pno] = lines
        all_lines.extend(lines)
        if progress:
            progress(pno + 1, len(doc))

    body_h = _median_height(all_lines)

    for pno in range(len(doc)):
        page = doc[pno]
        ps = PageStructure(
            page_no=pno, width=page.rect.width, height=page.rect.height,
        )
        ps.blocks.extend(
            group_ocr_blocks(per_page[pno], pno, _pixel_source(page, zoom)))
        for info in page.get_image_info():
            ps.blocks.append(Block(
                page_no=pno, bbox=tuple(info["bbox"]), kind="image",
                layer=LAYER_FLOW,
                raw={"width": info.get("width"), "height": info.get("height")},
                confidence=1.0, reasons=["页面图像对象"],
            ))
        structure.pages.append(ps)

    doc.close()
    structure.body_size = round(body_h * 0.78, 1)
    structure.font_xrefs = {}
    return structure
