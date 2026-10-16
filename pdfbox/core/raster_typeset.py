"""图片型 PDF 的活字编辑：自动 diff、只改动的字、优先复制原像素块。

与矢量版的 `typeset.py` 是同一套思路，只是"活字"从矢量字形换成了像素块：

    矢量型：字盘里存的是每个字的矢量原件，复用 = 把矢量内容搬过来
    图片型：字盘里存的是每个字的像素块，复用 = 把那一小块像素搬过来

复用的价值对图片型更大 —— 图片型本来就不知道原字体是什么，
用字体"仿造"再怎么调也是假的；而复制像素块是原封不动的原件。

流程（对齐用户确认的四条）：
    1. 输入框预填 OCR 文字，用户在上面改
    2. 工具自动 diff 出改动范围，不做查找/替换界面
    3. 只覆盖改动那几个字的位置，其余像素一个不动
    4. 每个新字：字盘里有就复制原件，没有再退到字体渲染

字盘范围默认限制在同一段落内（用户指定），可选跨段落。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from .model import Block
from .raster import (
    HostImage,
    _is_cjk_text,
    find_host_image,
    pixmap_from_array,
    replace_image_preserving_flags,
)
from .typeset import common_affix

# 逐字覆盖时上下左右的外扩
# 覆盖用的外扩：删除文字时必须盖干净，宁可稍大
CHAR_PAD_X = 0.8
CHAR_PAD_Y = 1.2


@dataclass
class CharPlacement:
    """一个字符在页面上的实际落点（编辑时画在哪，就是哪）。

    ★ 这是本轮结构改动的核心：编辑引擎知道每个字确切画在哪，
      就应该把这份坐标直接回写进结构模型，而不是事后用"字号 × 字数"去猜。
      此前四类 bug（界面不更新、多行被压成一行、范围跑出图像、空白处多出框）
      全都是"事后估算"造成的。
    """
    ch: str
    x0: float
    x1: float
    baseline: float
    line_idx: int
    changed: bool = False       # 是否是本次新画/改动的字


@dataclass
class GlyphPatch:
    """一个可复用的像素块。"""
    ch: str
    rect: tuple[float, float, float, float]     # 在原页面上的位置（pt）
    page_no: int
    label: str = ""              # 来源描述，给用户看（如 "第1页 第13块"）
    size: float = 0.0            # 该字所在行的字号，用于排序与判断能否复用
    in_block: bool = False       # 是否来自本段落（优先级最高）
    color: tuple[float, float, float] | None = None   # 来源块的文字色
    baseline_offset: float = 0.0   # 从 rect 顶边到基线的距离（pt）

    def describe(self) -> str:
        where = "本段落" if self.in_block else (self.label or f"第{self.page_no+1}页")
        return f"{self.ch!r}　{where}　字号 {self.size:.1f}pt　x={self.rect[0]:.0f}"


@dataclass
class PatchCase:
    """字盘：字符 → 候选像素块列表（按优先级排序）。

    默认优先用本段落里的字 —— 同段落字号字体最一致。
    跨段落的候选排在后面，只在需要时（本段落没有这个字）才用。
    同一字符有多个候选时，界面让用户挑。
    """
    by_char: dict[str, list[GlyphPatch]] = field(default_factory=dict)

    def candidates(self, ch: str) -> list[GlyphPatch]:
        return self.by_char.get(ch, [])

    def pick(self, ch: str, choice: GlyphPatch | None = None) -> GlyphPatch | None:
        if choice is not None and choice.ch == ch:
            return choice
        lst = self.by_char.get(ch)
        return lst[0] if lst else None

    def ambiguous(self, ch: str) -> bool:
        """该字是否有多个"同样优先"的候选，需要用户挑。"""
        lst = self.candidates(ch)
        if len(lst) < 2:
            return False
        return lst[0].in_block == lst[1].in_block

    def add(self, patch: GlyphPatch) -> None:
        self.by_char.setdefault(patch.ch, []).append(patch)

    def sort_all(self, target_size: float = 0.0) -> None:
        """排序：本段落优先 → 字号接近优先 → 页码靠前优先。"""
        for ch, lst in self.by_char.items():
            lst.sort(key=lambda g: (
                0 if g.in_block else 1,
                abs(g.size - target_size),
                g.page_no,
                g.rect[1],
                g.rect[0],
            ))

    def filter_compatible(self, doc, target_size: float,
                          target_color: tuple[float, float, float],
                          size_tol: float = 0.18,
                          color_tol: float = 0.30) -> int:
        """剔除风格明显不匹配的候选，返回剔除了多少个。

        ★ 跨段落复用必须做这个筛选。实测：给正文的 'W' 找到的第一个候选
          来自第 2 页标题 "WEEK 3" —— 那是绿色底上的白色大字，
          和正文深灰小字完全是两种风格。不看兼容性直接搬过来，
          结果贴出一个突兀的白字，视觉复核一眼就看出不对。

        判据两条：字号相差不超过 18%、文字颜色距离不超过 0.30。
        颜色从候选的像素块里实测 —— 不能只看字号，
        同字号的白字和黑字照样不能互换。
        """
        removed = 0
        for ch, lst in list(self.by_char.items()):
            keep = []
            for gp in lst:
                if gp.in_block:
                    keep.append(gp)
                    continue
                if target_size > 0 and gp.size > 0:
                    if abs(gp.size - target_size) / target_size > size_tol:
                        continue
                col = gp.color
                if col is not None and target_color:
                    d = sum(abs(a - b) for a, b in zip(col, target_color))
                    if d > color_tol:
                        continue
                keep.append(gp)
            removed += len(lst) - len(keep)
            if keep:
                self.by_char[ch] = keep
            else:
                self.by_char.pop(ch, None)
        return removed

    _color_cache: dict = None

    def _patch_color(self, doc, gp: GlyphPatch):
        """实测候选像素块里的文字颜色（取最暗/最亮那部分的中位数）。"""
        if self._color_cache is None:
            self._color_cache = {}
        key = (gp.page_no, tuple(round(v, 1) for v in gp.rect))
        if key in self._color_cache:
            return self._color_cache[key]
        val = None
        try:
            import numpy as np
            patch = extract_patch_from_doc(doc, gp)
            if patch is not None and patch.size:
                px = patch.reshape(-1, 3).astype(float)
                lum = px.mean(axis=1)
                k = max(1, len(px) // 3)
                # 深色文字取最暗那部分、浅色文字取最亮那部分
                core = (px[np.argsort(lum)[-k:]] if lum.mean() > 127
                        else px[np.argsort(lum)[:k]])
                val = tuple(float(v) / 255.0 for v in np.median(core, axis=0))
        except Exception:
            val = None
        self._color_cache[key] = val
        return val

    def stats(self, text: str) -> tuple[int, int]:
        n_reuse = sum(1 for ch in text if ch in self.by_char)
        return n_reuse, len(text) - n_reuse

    def ambiguous_chars(self, text: str) -> list[str]:
        seen, out = set(), []
        for ch in text:
            if ch not in seen and self.ambiguous(ch):
                seen.add(ch)
                out.append(ch)
        return out


@dataclass
class RasterTypePlan:
    mode: str                     # inplace | shift | whole
    prefix_len: int = 0
    suffix_len: int = 0
    kept_chars: int = 0
    changed_old: str = ""
    changed_new: str = ""
    reusable: int = 0             # 改动的新字里，字盘能提供的个数
    need_render: int = 0          # 需要字体渲染的个数
    width_delta: float = 0.0
    ambiguous: list[str] = field(default_factory=list)   # 多个候选、需用户挑的字
    start_rect: tuple[float, float, float, float] = (0, 0, 0, 0)
    end_rect: tuple[float, float, float, float] = (0, 0, 0, 0)
    start_line: int = 0

    def summary(self) -> str:
        names = {"inplace": "只改动的字（其余像素不动）",
                 "shift": "改动的字 + 后续整体平移",
                 "whole": "整行重画"}
        miss = f"，其中 {self.need_render} 字需用字体渲染" if self.need_render else ""
        return (f"{names.get(self.mode, self.mode)}："
                f"前 {self.prefix_len} 字不动、后 {self.suffix_len} 字不动、"
                f"改动 {len(self.changed_old)} → {len(self.changed_new)} 字"
                f"（复用原像素 {self.reusable} 字{miss}）")


def _flat_chars(block: Block):
    """块内所有字符，按阅读顺序拉平。"""
    out = []
    for line in block.lines:
        for span in line.spans:
            for ch in span.chars:
                out.append(ch)
    return out


def build_patch_case(
    page: pymupdf.Page,
    block: Block,
    *,
    scope: str = "paragraph",
    doc: pymupdf.Document | None = None,
    structure=None,
    block_label: str = "",
) -> PatchCase:
    """建「字盘」：字符 → 候选像素块列表。

    scope="paragraph" 只收本段落的字（默认）。同段落字号字体最一致，
    复用出来的效果最自然。
    scope="page"      再收本页其他段落的字。
    scope="document"  再收整个文档的字（跨页），字号可能不同，需要用户确认。

    收集时把候选全部留下而不是只留第一个 —— 同一字符常有多个来源，
    哪个最合适取决于字号是否接近、是否同段落，界面可以让用户挑。
    """
    case = PatchCase()
    label = block_label or f"第{block.page_no + 1}页"

    # 本段落
    for line in block.lines:
        size = line.size
        for span in line.spans:
            for ch in span.chars:
                if not ch.c.strip():
                    continue
                case.add(GlyphPatch(
                    ch=ch.c, rect=tuple(ch.bbox), page_no=block.page_no,
                    label=f"{label}（本段落）", size=size, in_block=True,
                    baseline_offset=ch.origin[1] - ch.bbox[1],
                ))

    if scope == "paragraph":
        case.sort_all(block.size)
        return case

    pages = [block.page_no] if scope == "page" else None

    # 优先用结构模型里的块 —— 图片型 PDF 没有文字层，get_text() 拿不到任何东西，
    # 字盘只能从 OCR 建出来的结构里取（那才有逐字坐标）。
    # 这是"跨段落复用"能否成立的关键：一开始直接用 rawdict，结果图片型文档
    # 一个字都收不到，跨段落字盘始终是空的。
    if structure is not None:
        from . import raster as _raster
        color_cache: dict = {}

        def block_color(b):
            """来源块的文字色。

            必须按"整个块"实测，不能按单个字采 —— 单个字的框很窄，
            里面背景像素占多数，采出来的是底色而不是文字色
            （实测拿到 (255,229,203) 浅橙，那是背景）。
            """
            key = (b.page_no, round(b.bbox[1], 1), round(b.bbox[0], 1))
            if key in color_cache:
                return color_cache[key]
            val = None
            if doc is not None:
                try:
                    ink = _raster.measure_block_ink(doc[b.page_no], b, expand=False)
                    if ink.ink_pixels:
                        val = ink.color
                except Exception:
                    val = None
            color_cache[key] = val
            return val

        for ps in structure.pages:
            if pages is not None and ps.page_no not in pages:
                continue
            for b in ps.blocks:
                if b is block:
                    continue
                if (b.page_no == block.page_no
                        and abs(b.bbox[1] - block.bbox[1]) < 1.0
                        and abs(b.bbox[0] - block.bbox[0]) < 1.0):
                    continue   # 本段落已收过
                bcol = block_color(b)
                for line in b.lines:
                    for span in line.spans:
                        for ch in span.chars:
                            if not ch.c.strip():
                                continue
                            case.add(GlyphPatch(
                                ch=ch.c, rect=tuple(ch.bbox), page_no=b.page_no,
                                label=f"第{b.page_no + 1}页（{b.kind}）",
                                size=line.size, in_block=False, color=bcol,
                                baseline_offset=ch.origin[1] - ch.bbox[1],
                            ))
        case.sort_all(block.size)
        return case

    # 没有结构模型时退回 rawdict（文本型 PDF 用得上）
    pages_idx = [block.page_no] if pages is not None else range(len(doc or []))
    for pno in pages_idx:
        try:
            pg = (doc or page.parent)[pno]
        except Exception:
            continue
        try:
            raw = pg.get_text("rawdict")
        except Exception:
            continue
        for bi, b in enumerate(raw.get("blocks", [])):
            if b.get("type", 0) != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    spansize = span.get("size", 0.0)
                    for c in span.get("chars", []):
                        ch = c.get("c", "")
                        if not ch.strip():
                            continue
                        if pno == block.page_no and bi == _block_index(page, block):
                            continue
                        case.add(GlyphPatch(
                            ch=ch, rect=tuple(c["bbox"]), page_no=pno,
                            label=f"第{pno + 1}页 第{bi + 1}块",
                            size=spansize, in_block=False,
                        ))
    case.sort_all(block.size)
    return case


def _block_index(page: pymupdf.Page, block: Block) -> int:
    """块在页面 rawdict 里的序号，用于跳过已收录的本段落。"""
    try:
        for i, b in enumerate(page.get_text("rawdict").get("blocks", [])):
            if b.get("type", 0) != 0:
                continue
            if b.get("bbox") and abs(b["bbox"][1] - block.bbox[1]) < 1.0:
                return i
    except Exception:
        pass
    return -1


def plan_raster_typeset(
    page: pymupdf.Page, block: Block, new_text: str,
    case: PatchCase,
    choices: dict[str, GlyphPatch] | None = None,
    structure=None,
) -> RasterTypePlan:
    """算改动范围。不做任何修改，便于先给用户看。"""
    chars = _flat_chars(block)
    old = "".join(c.c for c in chars)
    new = new_text
    if old == new:
        return RasterTypePlan(mode="inplace")

    p, s = common_affix(old, new)
    old_mid = chars[p:len(chars) - s] if s else chars[p:]
    new_mid = new[p:len(new) - s] if s else new[p:]

    plan = RasterTypePlan(
        mode="inplace", prefix_len=p, suffix_len=s,
        kept_chars=p + s, changed_old="".join(c.c for c in old_mid),
        changed_new=new_mid,
    )
    if not old_mid:
        # 纯插入（只看不删）：在末尾或中间加字。
        # 不能退回"整块重画" —— 用户明确要求加长时能保留的必须保留。
        # 按 shift 处理：在插入点画新字，插入点之后的内容整体右移。
        # （矢量路径早已这么处理，图片型这边漏了，实测末尾加两个字
        #   会触发整块覆盖重画 2903 像素，而不是只画那两个字）
        if not new_mid:
            plan.mode = "inplace"
            return plan
        if p > 0:
            anchor = chars[p - 1]
            plan.start_rect = anchor.bbox
        elif chars:
            plan.start_rect = chars[0].bbox
        # 插入点之后的字符要右移
        plan.end_rect = chars[-1].bbox if chars else plan.start_rect
        est = 0.0
        for ch in new_mid:
            gp = case.pick(ch)
            est += ((gp.rect[2] - gp.rect[0]) if gp
                    else max(4.0, block.size * 0.9))
        plan.width_delta = est
        plan.mode = "shift"
        return plan

    plan.start_rect = old_mid[0].bbox
    plan.end_rect = old_mid[-1].bbox
    _lmap = {}
    for _li, _ln in enumerate(block.lines):
        for _sp in _ln.spans:
            for _c in _sp.chars:
                _lmap[id(_c)] = _li
    plan.start_line = _lmap.get(id(old_mid[0]), 0)
    old_w = plan.end_rect[2] - plan.start_rect[0]

    choices = choices or {}
    for ch in new_mid:
        if case.pick(ch, choices.get(ch)) is not None:
            plan.reusable += 1
        else:
            plan.need_render += 1
    plan.ambiguous = case.ambiguous_chars(new_mid)

    # 宽度差先按"复用取原宽、需渲染的按字号估"算个大概
    est = 0.0
    for ch in new_mid:
        gp = case.pick(ch, choices.get(ch))
        est += (gp.rect[2] - gp.rect[0]) if gp else old_mid[0].bbox[2] - old_mid[0].bbox[0]
    plan.width_delta = est - old_w
    # 宽度明显变化时后续文字会错位，走刚性平移（后缀整体挪位，不重新断行）
    if abs(plan.width_delta) > max(0.6, 0.06 * old_w):
        plan.mode = "shift"
    return plan


def apply_raster_typeset(
    doc: pymupdf.Document,
    block: Block,
    new_text: str,
    font: pymupdf.Font,
    plan: RasterTypePlan,
    case: PatchCase,
    *,
    bg_color: tuple[int, int, int] | None = None,
    size: float | None = None,
    choices: dict[str, GlyphPatch] | None = None,
) -> dict:
    """执行图片型活字编辑：只覆盖改动的字，优先贴原像素块。

    实现要点：直接把改动合成进承载文字的那张图像对象，
    而不是在页面上叠新对象 —— 这样新文字和原图是同一层位图，
    像素密度一致，不会有"一层新、一层旧"的割裂感。
    """
    page = doc[block.page_no]
    host = find_host_image(page, block.bbox)
    if host is None:
        return {"ok": False, "reason": "找不到承载文字的图像"}

    try:
        pix = pymupdf.Pixmap(doc, host.xref)
    except Exception:
        return {"ok": False, "reason": "无法读取承载图像"}
    if pix.n < 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    import numpy as np
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].copy()

    local = HostImage(xref=host.xref, bbox=host.bbox,
                      width=pix.width, height=pix.height)

    def px(rect):
        x0, y0, x1, y1 = local.to_pixels(rect)
        return (max(0, x0), max(0, y0),
                min(arr.shape[1], x1), min(arr.shape[0], y1))

    chars = _flat_chars(block)
    old_mid = chars[plan.prefix_len:len(chars) - plan.suffix_len] \
        if plan.suffix_len else chars[plan.prefix_len:]

    # ★ 逐字记录实际落点：前缀保持原位、改动段用新位置、后缀整体平移。
    # 这份坐标会被直接回写进结构模型 —— 不再事后用字号×字数去估。
    placements: list[CharPlacement] = []
    _line_of: dict[int, int] = {}
    for _li, _ln in enumerate(block.lines):
        for _sp in _ln.spans:
            for _c in _sp.chars:
                _line_of[id(_c)] = _li
    for _c in chars[:plan.prefix_len]:
        placements.append(CharPlacement(
            _c.c, _c.bbox[0], _c.bbox[2], _c.origin[1],
            _line_of.get(id(_c), 0), False))
    if not old_mid:
        # 纯插入：没有要覆盖的字，直接在插入点画新字、把后面的内容整体右移。
        # 走这条路才符合"加长时能保留的必须保留"——
        # 早期这里直接报错退回整块重画，用户看到的就是整个段落都变了。
        return _apply_pure_insert(doc, page, block, chars, plan, case,
                                  font, size, bg_color)

    # 字号必须用「实测墨迹高度反推的真实字号」，不能用 OCR 框高度。
    # OCR 框比文字本身高不少（实测 9.7pt vs 真实 5.6pt），
    # 用框高当字号会把字画得巨大 —— 视觉复核里新字比周围大一圈就是这个原因。
    if size is None:
        # 字号必须按**整行**实测推，不能按单个字推。
        # 单个字的框很窄，里面墨迹少且容易被切边 ——
        # 实测同一个 'I' 单字推出来 4.56pt，整行推出来 5.61pt，
        # 差 23%，画出来明显偏小。
        try:
            from .raster import measure_block_ink
            _ink = measure_block_ink(page, block, expand=False)
            size = _font_size_hint(block, _ink, "".join(c.c for c in old_mid))
            # ★ 合理性护栏：墨迹**不可能**超出 OCR 文字框。
            # 一旦超出，说明这一页的行投影失效了（实测第 5 页整条带都有墨迹，
            # 测出的墨迹高 41.3pt 比框高 27.6pt 还大），推算出的字号会翻倍，
            # 结果是文字被判定"放不下"而折成好几行、覆盖范围也炸开。
            # 这时退回按 OCR 框高估字号 —— 不准但不会失控。
            box_h = max(c.bbox[3] - c.bbox[1] for c in old_mid)
            if _ink.height > box_h * 1.35:
                size = box_h * 0.78
        except Exception:
            size = (old_mid[0].bbox[3] - old_mid[0].bbox[1]) * 0.78

    # 文字色从**整块**采样，不能从单个字采。
    # 单个字的 bbox 很窄（实测 'P' 只有 2.7pt 宽），中位色采到的是背景而不是笔画
    # —— 实测拿到 (255,229,201) 浅橙，那是底色。整行采才有足够的墨迹像素。
    text_color = (0.0, 0.0, 0.0)
    try:
        from .raster import measure_block_ink
        _blk_ink = measure_block_ink(page, block, expand=False)
        if _blk_ink.ink_pixels:
            text_color = _blk_ink.color
    except Exception:
        pass

    # ---------- 0. shift 模式：先把后缀整段切下来备用
    #
    # 必须在新字贴上之前切，否则切到的就是已经改过的内容。
    # 后缀用"整段一块"而不是逐字切：整段切下来是一张完整的矩形切片，
    # 平移后原封不动地贴回去，字距、笔画、抗锯齿全部保持，
    # 这是"刚性平移"能成立的关键 —— 逐字切再逐字贴容易在接缝处出错。
    tail_patch = None
    tail_src = None
    if plan.mode == "shift" and plan.suffix_len > 0:
        tail = chars[plan.prefix_len + len(old_mid):]
        if tail:
            tx0 = min(c.bbox[0] for c in tail) - CHAR_PAD_X
            ty0 = min(c.bbox[1] for c in tail) - CHAR_PAD_Y
            tx1 = max(c.bbox[2] for c in tail) + CHAR_PAD_X
            ty1 = max(c.bbox[3] for c in tail) + CHAR_PAD_Y
            tail_src = (tx0, ty0, tx1, ty1)
            ax, ay = _pdf_to_px(local, tx0, ty0)
            bx, by = _pdf_to_px(local, tx1, ty1)
            ax = max(0, ax); ay = max(0, ay)
            bx = min(arr.shape[1], bx); by = min(arr.shape[0], by)
            if bx > ax and by > ay:
                tail_patch = arr[ay:by, ax:bx].copy()
            else:
                tail_src = None

    # ---------- 背景色：整块采一次，不要逐字采
    #
    # 逐字采容易采偏 —— 每个字的采样带只覆盖自己那一小段宽度，
    # 万一那一段正好压着别的元素（上标、下划线、色块边缘），
    # 采出来的"背景"就是错的，填上去成一块异色补丁。
    # 整块采的采样带横跨整行，中位数由占多数的真实背景决定，稳得多。
    block_bg = None
    if bg_color is None:
        try:
            bx0 = min(c.bbox[0] for c in chars) - 2
            by0 = min(c.bbox[1] for c in chars)
            bx1 = max(c.bbox[2] for c in chars) + 2
            by1 = max(c.bbox[3] for c in chars)
            block_bg = _local_bg(arr, local, (bx0, by0, bx1, by1))
        except Exception:
            block_bg = None

    # ---------- 1. 覆盖改动的那几个字
    #
    # 覆盖范围用**逐字墨迹框**而不是 OCR 框 + 小外扩。
    # 原因：删除文字时若覆盖不完整，原字会露出来（用户反馈"缺少底色覆盖被删除的字"）。
    # OCR 的文字框是按内容估算的，笔画可能超出它；逐字墨迹框是贴着实际笔画量出来的，
    # 再向外留一点余量，才能保证盖干净。
    ink_boxes = {}
    try:
        from .raster import per_char_boxes
        for _c, _r in per_char_boxes(page, block, zoom=4.0):
            ink_boxes[id(_c)] = _r
    except Exception:
        ink_boxes = {}

    n_bg = 0
    for ch in old_mid:
        base = ink_boxes.get(id(ch), tuple(ch.bbox))
        rx0, ry0, rx1, ry1 = px((base[0] - CHAR_PAD_X, base[1] - CHAR_PAD_Y,
                                 base[2] + CHAR_PAD_X, base[3] + CHAR_PAD_Y))
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        bg = bg_color if bg_color is not None else block_bg
        if bg is None:
            # 背景色取"该字上方一带"的**中位数**。
            #
            # 只取上方不取下方：下方常压着下一行文字或下划线。
            # 用中位数而不是"最亮百分位"：实测该字上方是浅橙色背景 (255,229,201)，
            # 但同一带里混着别处的白色像素，取最亮的 30% 会抓到白色，
            # 于是在文字周围填出一个白方块——视觉复核一眼就看出来了。
            # 中位数由占多数的背景决定，不受少数异色像素影响。
            h = ry1 - ry0
            up0 = max(0, ry0 - max(3, h))
            up1 = max(0, ry0 - 1)
            strip = arr[up0:up1, rx0:rx1].reshape(-1, 3) if up1 > up0 else None
            if strip is None or strip.size == 0:
                strip = arr[ry0:ry1, rx0:rx1].reshape(-1, 3)
            if strip.size:
                bg = tuple(int(v) for v in np.median(strip, axis=0))
        if bg is not None:
            arr[ry0:ry1, rx0:rx1] = bg
        n_bg += 1

    # ---------- 2. 贴新字：优先复制原像素块，没有才用字体渲染
    x = plan.start_rect[0]
    baseline = old_mid[0].origin[1]
    reused = 0
    rendered = 0
    fallback = []

    def flush_fallback():
        nonlocal fallback, rendered
        if not fallback:
            return
        txt = "".join(ch for ch, _ in fallback)
        patch = _render_text_patch(txt, font, size, page, local, color=text_color)
        if patch is not None:
            paste_patch(arr, patch, fallback[0][1], old_mid[0].origin[1], local)
        rendered += len(fallback)
        fallback = []

    choices = choices or {}
    for ch in plan.changed_new:
        gp = case.pick(ch, choices.get(ch))
        if gp is not None:
            flush_fallback()
            patch = _extract_patch(doc, arr, local, host.xref, gp,
                                   block.page_no)
            if patch is not None:
                same_area = (gp.page_no == block.page_no and gp.in_block)
                place_patch(arr, patch, x, old_mid[0].origin[1], local,
                            mask=not same_area,
                            baseline_offset=gp.baseline_offset)
                reused += 1
                x += (gp.rect[2] - gp.rect[0])
                placements.append(CharPlacement(
                    ch, x, x + (gp.rect[2] - gp.rect[0]), baseline,
                    plan.start_line, True))
                continue
        if not fallback:
            fallback = []
        fallback.append((ch, x))
        placements.append(CharPlacement(
            ch, x, x + font.text_length(ch, fontsize=size), baseline,
            plan.start_line, True))
        x += font.text_length(ch, fontsize=size)
    flush_fallback()

    # ---------- 3. shift：把后缀整段刚性平移到新位置
    #
    # 后缀整块切下来、整块贴回去，只改 x —— 纵向一点不动。
    # 用底边对齐（不传 baseline_offset）。
    # 早先这里留过一个 `tail_src[3] - origin[1] - PAD + ...` 的偏移公式，
    # 算出来偏约 20pt，后缀被整体抬高一行，视觉上后半段文字飞到了上面。
    if tail_patch is not None and tail_src is not None:
        bg2 = _local_bg(arr, local, tail_src)
        if bg2 is not None:
            ax, ay = _pdf_to_px(local, tail_src[0], tail_src[1])
            bx, by = _pdf_to_px(local, tail_src[2], tail_src[3])
            ax = max(0, ax); ay = max(0, ay)
            bx = min(arr.shape[1], bx); by = min(arr.shape[0], by)
            if bx > ax and by > ay:
                arr[ay:by, ax:bx] = bg2
        place_patch(arr, tail_patch, tail_src[0] + plan.width_delta,
                    tail_src[3], local)

    # ---------- 3.5 后缀落点：**无论哪种模式都要记**
    #
    # 早先这段只在 shift 模式跑过（因为后缀切片的代码在 shift 分支里），
    # 于是 inplace（等长改字）时后缀一个字都没进结构 ——
    # 表现为：改了几个字之后，编辑框里改动位置之后的内容全部消失、
    # 框也异常缩短，但 PDF 上那些字还在。
    # 后缀的位移：shift 是 width_delta，inplace 是 0。
    _dx = plan.width_delta if plan.mode == "shift" else 0.0
    for _c in chars[plan.prefix_len + len(old_mid):]:
        placements.append(CharPlacement(
            _c.c, _c.bbox[0] + _dx, _c.bbox[2] + _dx,
            _c.origin[1], _line_of.get(id(_c), 0), False))

    # ---------- 4. 写回图像
    try:
        replace_image_preserving_flags(doc, page, host.xref, arr, rect=host.bbox)
    except Exception as e:
        return {"ok": False, "reason": f"写回图像失败：{e}"}

    return {
        "ok": True, "kept": plan.kept_chars,
        "covered": n_bg, "reused": reused, "rendered": rendered,
        "width_delta": plan.width_delta,
        "placements": placements,
    }


def _apply_pure_insert(doc, page, block, chars, plan, case,
                       font, size, bg_color) -> dict:
    """纯插入：只在插入点画新字，后面的内容整体右移。"""
    import numpy as np
    host = find_host_image(page, block.bbox)
    if host is None:
        return {"ok": False, "reason": "找不到承载文字的图像"}
    try:
        pix = pymupdf.Pixmap(doc, host.xref)
    except Exception:
        return {"ok": False, "reason": "无法读取承载图像"}
    if pix.n < 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].copy()
    local = HostImage(xref=host.xref, bbox=host.bbox,
                      width=pix.width, height=pix.height, matrix=host.matrix)

    _line_of: dict[int, int] = {}
    for _li, _ln in enumerate(block.lines):
        for _sp in _ln.spans:
            for _c in _sp.chars:
                _line_of[id(_c)] = _li
    placements: list[CharPlacement] = []
    for _c in chars[:plan.prefix_len]:
        placements.append(CharPlacement(
            _c.c, _c.bbox[0], _c.bbox[2], _c.origin[1],
            _line_of.get(id(_c), 0), False))

    at = plan.prefix_len
    if size is None:
        size = block.size
    if at > 0:
        anchor = chars[at - 1]
        x_start = anchor.bbox[2]
        baseline = anchor.origin[1]
    else:
        anchor = chars[0] if chars else None
        x_start = anchor.bbox[0] if anchor else block.bbox[0]
        baseline = anchor.origin[1] if anchor else block.bbox[3]

    # 插入点之后的整段先切下来
    tail = chars[at:]
    tail_src = None
    tail_patch = None
    if tail:
        tx0 = min(c.bbox[0] for c in tail) - CHAR_PAD_X
        ty0 = min(c.bbox[1] for c in tail) - CHAR_PAD_Y
        tx1 = max(c.bbox[2] for c in tail) + CHAR_PAD_X
        ty1 = max(c.bbox[3] for c in tail) + CHAR_PAD_Y
        ax, ay = _pdf_to_px(local, tx0, ty0)
        bx, by = _pdf_to_px(local, tx1, ty1)
        ax, ay = max(0, ax), max(0, ay)
        bx, by = min(arr.shape[1], bx), min(arr.shape[0], by)
        if bx > ax and by > ay:
            tail_patch = arr[ay:by, ax:bx].copy()
            tail_src = (tx0, ty0, tx1, ty1)

    reused = 0
    rendered = 0
    x = x_start
    fallback: list[tuple[str, float]] = []

    def flush():
        nonlocal fallback, rendered
        if not fallback:
            return
        txt = "".join(ch for ch, _ in fallback)
        patch = _render_text_patch(txt, font, size, page, local,
                                   color=(0.0, 0.0, 0.0))
        if patch is not None:
            paste_patch(arr, patch, fallback[0][1], baseline, local)
        rendered += len(fallback)
        fallback = []

    for ch in plan.changed_new:
        gp = case.pick(ch)
        if gp is not None:
            flush()
            patch = _extract_patch(doc, arr, local, host.xref, gp, block.page_no)
            if patch is not None:
                place_patch(arr, patch, x, baseline, local,
                            mask=not gp.in_block,
                            baseline_offset=gp.baseline_offset)
                placements.append(CharPlacement(
                    ch, x, x + (gp.rect[2] - gp.rect[0]), baseline,
                    plan.start_line, True))
                reused += 1
                x += (gp.rect[2] - gp.rect[0])
                continue
        fallback.append((ch, x))
        placements.append(CharPlacement(
            ch, x, x + font.text_length(ch, fontsize=size), baseline,
            plan.start_line, True))
        x += font.text_length(ch, fontsize=size)
    flush()

    # 后面的内容整体右移
    if tail_patch is not None and tail_src is not None:
        bg2 = _local_bg(arr, local, tail_src)
        if bg2 is not None:
            ax, ay = _pdf_to_px(local, tail_src[0], tail_src[1])
            bx, by = _pdf_to_px(local, tail_src[2], tail_src[3])
            ax, ay = max(0, ax), max(0, ay)
            bx, by = min(arr.shape[1], bx), min(arr.shape[0], by)
            if bx > ax and by > ay:
                arr[ay:by, ax:bx] = bg2
        place_patch(arr, tail_patch, tail_src[0] + plan.width_delta,
                    tail_src[3], local)
        for _c in tail:
            placements.append(CharPlacement(
                _c.c, _c.bbox[0] + plan.width_delta,
                _c.bbox[2] + plan.width_delta, _c.origin[1],
                _line_of.get(id(_c), 0), False))

    try:
        replace_image_preserving_flags(doc, page, host.xref, arr, rect=host.bbox)
    except Exception as e:
        return {"ok": False, "reason": f"写回图像失败：{e}"}

    return {"ok": True, "kept": plan.prefix_len, "covered": 0,
            "reused": reused, "rendered": rendered,
            "width_delta": plan.width_delta, "placements": placements}


# --------------------------------------------------------------- 像素块操作


def _extract_patch(doc: pymupdf.Document, arr, local: HostImage, host_xref: int,
                   gp: GlyphPatch, target_page_no: int):
    """取出一小块像素。

    ★ 判断依据必须是「候选是否落在同一张承载图像里」，而不是「是否同一页」。
      一页里常有多张图（本样本每页 3 张幻灯片位图），
      同页但不同图的两个字，坐标系完全不同 ——
      早期按"同页"判断，结果从当前工作数组里按错误坐标取像素，
      取到的是别的图片内容，复用自然全部失败（实测复用数一直是 0）。

    同一张图：直接从当前工作数组取（快，且和后续合成在同一份数据上）
    不同图：从文档里把那张图单独取出来，按它自己的坐标映射取像素
    """
    if gp.page_no == target_page_no:
        src_host = find_host_image(doc[gp.page_no], gp.rect)
        if src_host is not None and src_host.xref == host_xref:
            x0, y0, x1, y1 = local.to_pixels(gp.rect)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(arr.shape[1], x1); y1 = min(arr.shape[0], y1)
            if x1 <= x0 or y1 <= y0:
                return None
            return arr[y0:y1, x0:x1].copy()

    return extract_patch_from_doc(doc, gp)


def _local_bg(arr, local: HostImage, rect):
    """在给定矩形上方一带取背景中位数。"""
    import numpy as np
    x0, y0 = _pdf_to_px(local, rect[0], rect[2] if False else rect[1])
    x1, _ = _pdf_to_px(local, rect[2], rect[1])
    _, y1 = _pdf_to_px(local, rect[2], rect[3])
    x0 = max(0, x0); x1 = min(arr.shape[1], x1)
    y0 = max(0, y0); y1 = min(arr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    h = y1 - y0
    up0 = max(0, y0 - max(3, h))
    up1 = max(0, y0 - 1)
    strip = arr[up0:up1, x0:x1].reshape(-1, 3) if up1 > up0 else None
    if strip is None or strip.size == 0:
        strip = arr[y0:y1, x0:x1].reshape(-1, 3)
    if strip.size == 0:
        return None
    return tuple(int(v) for v in np.median(strip, axis=0))


def _font_size_hint(block, ink, text: str) -> float:
    """估计字号。

    ★ 中文走「行宽 ÷ 字数」，那是**精确值** —— 中文等宽字体下每字步进就等于字号。
      而按墨迹高度反推会偏大：中文的全高字墨迹约占 0.97 em，不是 0.88，
      用 0.88 会算出大 10% 的字号，画出来的字比原字宽一圈（实测量实测 16.3pt vs 真实 14.8pt）。

      西文比例字体没有这个等宽关系，仍用墨迹高度反推。
    """
    # ★ 只拿**中文字符自己的宽度**当字号。
    # 中文等宽，一个字占 1 em，宽度就等于字号 —— 不受同行有没有西文影响。
    # 早先用的是「整行宽 ÷ 总字数」，遇到 '41'、'311811'、'_' 这类半角字符时
    # 平均值被拉低，算出的字号偏小，重绘出来的字明显比原文小一圈。
    import statistics
    widths = []
    for line in block.lines:
        for sp in line.spans:
            for c in sp.chars:
                if _is_cjk_text(c.c):
                    widths.append(c.bbox[2] - c.bbox[0])
    if widths:
        unit = statistics.median(widths)
        if unit > 1.0:
            return unit
    return ink.size_for(text)


def _pdf_to_px(local: HostImage, x: float, y: float) -> tuple[int, int]:
    """PDF 坐标 → 承载图像的像素坐标。

    ★ 必须走 HostImage 的完整映射（含旋转矩阵），不能自己按 bbox 做轴对齐换算。
      轴对齐换算只对"图像正着放"的情况成立；实测第 5 页的图像是 90° 旋转放置的，
      按轴对齐换算出来的像素坐标完全不对，覆盖和粘贴会落到错误的位置
      —— 这也是"覆盖位置不对"的一个直接来源。
    """
    px, py = local._point(x, y)
    return int(px), int(py)


def extract_patch_from_doc(doc, gp: GlyphPatch):
    """直接从文档里取出某候选字形的像素块 —— 不依赖当前工作数组。

    用于兼容性筛选时实测候选的文字色：同字号的白字和黑字不能互换，
    只看字号是不够的。
    """
    try:
        import numpy as np
        host = find_host_image(doc[gp.page_no], gp.rect)
        if host is None:
            return None
        pix = pymupdf.Pixmap(doc, host.xref)
        if pix.n < 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        local = HostImage(xref=host.xref, bbox=host.bbox,
                          width=pix.width, height=pix.height)
        ax, ay = _pdf_to_px(local, gp.rect[0], gp.rect[1])
        bx, by = _pdf_to_px(local, gp.rect[2], gp.rect[3])
        ax = max(0, ax); ay = max(0, ay)
        bx = min(arr.shape[1], bx); by = min(arr.shape[0], by)
        if bx <= ax or by <= ay:
            return None
        return arr[ay:by, ax:bx].copy()
    except Exception:
        return None


def _patch_mask(patch):
    """从像素块里标出「哪些像素属于文字」。

    ★ 背景色用「整块像素的众数」判定，不用边框中位数。
      边框中位数在源字带高亮底色时会失准 —— 实测复用带黄色高亮底的 'W' 时，
      采到的"背景"不是黄色，于是黄色像素被当成文字一起贴了过去，
      结果贴出来一个带黄底的怪字。

    判据：把像素按灰度聚成两类，占多数的那一类是背景；
    明显偏离背景的算文字。这样不管源位置是白底、彩底还是高亮底都能处理。
    """
    import numpy as np
    if patch.size == 0:
        return None
    px = patch.reshape(-1, 3).astype(int)
    lum = px.mean(axis=1)
    # 用中位数把像素粗分两簇，取像素多的那簇当背景
    mid = float(np.median(lum))
    lo, hi = px[lum <= mid], px[lum > mid]
    bg = np.median(hi if len(hi) >= len(lo) else lo, axis=0)
    diff = np.abs(px - bg).sum(axis=1)
    # 阈值取"偏离量"的较高分位，把抗锯齿过渡也排除掉
    thr = max(90.0, float(np.percentile(diff, 55)))
    return (diff > thr).reshape(patch.shape[:2])


def place_patch(arr, patch, target_x: float, baseline_y: float,
                local: HostImage, mask: bool = False,
                baseline_offset: float | None = None):
    """把原像素块贴到目标位置。

    ★ 必须按**基线**对齐，不能按包围盒底边。
      包围盒底边包含降部空间，而"有降部的字"和"没降部的字"底边位置差别很大
      （"Industry" 的 y 有降部、"Webinar" 没有）。按底边对齐会把新字整体压下去，
      渲染出来墨迹高度虚高、看着比周围大一圈 —— 实测行内墨迹高从 5.40pt 涨到 6.90pt。

      基线位置是已知的：每个字符的 origin[1] 就是基线 y。
      源像素块顶边到基线的距离记在 baseline_offset 里，照它反推粘贴位置即可。

    mask=True 时只贴文字像素、保留目标位置的背景。
    ★ 跨区域复用时必须开：像素块是**连同源位置的背景一起**搬过来的，
      第2页的字块带着第2页的底色，直接整块贴到第1页会带过去一块异色底
      —— 实测视觉复核一眼就看出"贴上去的"。
      同段落内复用则关掉它：那条路径本来就是像素级精确的（实测差异 0），
      按掩膜贴反而会损失抗锯齿边缘。
    """
    m = _patch_mask(patch) if mask else None
    h, w = patch.shape[:2]
    tx, ty_anchor = _pdf_to_px(local, target_x, baseline_y)
    if baseline_offset is None:
        # 不传偏移量 → 把 anchor_y 当作补丁的**底边**。
        # 整段刚性平移用这个语义：原后缀切片的底边对齐到新位置的底边。
        # 早期这里默认按基线对齐、偏移量取 0，结果补丁从基线往下铺，
        # 超出文字区域十几 pt（实测把 y=442 的内容都改了）。
        ty = ty_anchor - h
    else:
        scale = local.height / max(1e-6, local.bbox[3] - local.bbox[1])
        ty = ty_anchor - int(round(baseline_offset * scale))
    if tx + w <= 0 or ty + h <= 0:
        return
    sx0 = max(0, -tx)
    sy0 = max(0, -ty)
    tx0 = max(0, tx)
    ty0 = max(0, ty)
    tw = min(w - sx0, arr.shape[1] - tx0)
    th = min(h - sy0, arr.shape[0] - ty0)
    if tw <= 0 or th <= 0:
        return
    if m is None:
        arr[ty0:ty0 + th, tx0:tx0 + tw] = patch[sy0:sy0 + th, sx0:sx0 + tw]
    else:
        sub_m = m[sy0:sy0 + th, sx0:sx0 + tw]
        sub_p = patch[sy0:sy0 + th, sx0:sx0 + tw]
        region = arr[ty0:ty0 + th, tx0:tx0 + tw]
        region[sub_m] = sub_p[sub_m]


def paste_patch(arr, patch, target_x: float, baseline_y: float,
                local: HostImage):
    """把渲染出的字形按掩膜合成到目标位置 —— 只覆盖墨迹像素，不铺背景。"""
    rgb, mask, base_off = patch
    h, w = rgb.shape[:2]
    tx, ty_base = _pdf_to_px(local, target_x, baseline_y)
    ty = ty_base - int(round(base_off))
    sx0 = max(0, -tx)
    sy0 = max(0, -ty)
    tx0 = max(0, tx)
    ty0 = max(0, ty)
    tw = min(w - sx0, arr.shape[1] - tx0)
    th = min(h - sy0, arr.shape[0] - ty0)
    if tw <= 0 or th <= 0:
        return
    sub_m = mask[sy0:sy0 + th, sx0:sx0 + tw]
    sub_rgb = rgb[sy0:sy0 + th, sx0:sx0 + tw]
    region = arr[ty0:ty0 + th, tx0:tx0 + tw]
    region[sub_m] = sub_rgb[sub_m]


def _render_text_patch(text: str, font: pymupdf.Font, size: float,
                       page: pymupdf.Page, local: HostImage,
                       color=(0.0, 0.0, 0.0)):
    """把一段文字按承载图像的分辨率渲染成像素块。

    关键：按原图的分辨率渲染，让新字的像素密度和原图一致 ——
    直接在页面上叠矢量文字会显得"太清晰"，和周围的位图格格不入。
    """
    import numpy as np
    # 承载图像每 pt 多少像素
    ppp = local.width / max(1e-6, local.bbox[2] - local.bbox[0])
    zoom = max(1.0, min(12.0, ppp))
    probe = pymupdf.open()
    try:
        pg = probe.new_page(width=400, height=200)
        tw = pymupdf.TextWriter(pg.rect)
        tw.append((20, 140), text, font=font, fontsize=size)
        tw.write_text(pg, color=color)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        mask = a.mean(axis=2) < 200
        if not mask.any():
            return None
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        sub = a[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()
        m = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()
        # 基线在探针页面上的 y=140，换回补丁内的像素偏移
        base_off_px = 140.0 * zoom - rows.min()
        # 只返回墨迹像素 + 掩膜，不填背景色。
        # 早期版本把未着墨处填成白色再整块贴上，结果在新字周围贴出一个白方块
        # —— 视觉复核里那个"W 贴在白色标签上"就是这个原因。
        # 正确做法是只把墨迹像素按掩膜覆盖上去，背景保持原样。
        return sub, m, base_off_px
    except Exception:
        return None
    finally:
        probe.close()
