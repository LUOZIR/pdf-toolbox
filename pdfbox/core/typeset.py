"""活字印刷式编辑：把段落切成最小单字单元，只重排改动的那一段。

原来的做法是「块里改一个字 → 整块擦除重排重绘」，未改动的部分也跟着
失去原始字节、可能产生细微位移。这个模块把它改成活字印刷的思路：

    把段落拆成一个个"活字"（每个字符带着自己的精确原位），
    编辑时只把改动的那几个字取出来换掉，
    其余的字原地不动 —— 一个 bit 都不碰。

三种结果模式，按改动影响的范围从小到大自动选择：

    inplace  新字串宽度与原来一致 → 后续的字原地不动
             （改一个错别字、数字位数不变、同长度词语替换都属于这类）
    shift    宽度变了但行数不变 → 后续的字整体刚性平移，
             内部间距完全保持（不是重新断行！）
    reflow   行数变了 → 退回整块重排（这里就没有活字的好处了）

保真度的差别很大：inplace 模式下，一段 100 字的段落改 2 个字，
只有那 2 个字所在的像素会变，其余 98 个字保留原始字节。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymupdf

from .model import Block

# 宽度容差：新字串与旧字串宽度差在这个比例内视为一致，可走 inplace
WIDTH_TOL_RATIO = 0.004      # 0.4%
MIN_WIDTH_TOL = 0.35         # 但绝对容差至少 0.35pt


@dataclass
class TypeUnit:
    """一个"活字"：单字符 + 它在页面上的精确位置 + 归属信息。"""
    ch: str
    origin: tuple[float, float]
    bbox: tuple[float, float, float, float]
    line_idx: int
    font: str
    size: float


@dataclass
class TypeCase:
    """整块的活字盘。"""
    units: list[TypeUnit] = field(default_factory=list)
    lines: list[list[TypeUnit]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(u.ch for u in self.units)

    def line_of(self, unit_idx: int) -> int:
        return self.units[unit_idx].line_idx


def build_type_case(block: Block) -> TypeCase:
    """从块的结构模型建出活字盘。

    每个字符的精确原点、包围盒、字号都来自 L0 字符层，
    是 get_texttrace + rawdict 融合后的准确数据。
    """
    case = TypeCase()
    for li, line in enumerate(block.lines):
        line_units: list[TypeUnit] = []
        for span in line.spans:
            for ch in span.chars:
                u = TypeUnit(
                    ch=ch.c, origin=ch.origin, bbox=ch.bbox,
                    line_idx=li, font=span.font, size=span.size,
                )
                case.units.append(u)
                line_units.append(u)
        case.lines.append(line_units)
    return case


def common_affix(old: str, new: str) -> tuple[int, int]:
    """求公共前缀长度与公共后缀长度（前后缀不重叠）。"""
    n = min(len(old), len(new))
    p = 0
    while p < n and old[p] == new[p]:
        p += 1
    s = 0
    while s < n - p and old[len(old) - 1 - s] == new[len(new) - 1 - s]:
        s += 1
    return p, s


@dataclass
class TypeSetPlan:
    """活字排版方案。"""
    mode: str                       # inplace | shift | reflow | none
    prefix_len: int = 0
    suffix_len: int = 0
    changed_old: list[TypeUnit] = field(default_factory=list)
    changed_new: str = ""
    keep_before: int = 0            # 前半段保持不动的字符数
    keep_after: int = 0             # 后半段保持不动的字符数
    width_delta: float = 0.0        # 变动段的宽度差
    start_origin: tuple[float, float] = (0.0, 0.0)
    start_line: int = 0
    end_line: int = 0
    lines_after: int = 0
    reused_exact: bool = False      # 新字串与旧字串逐字相同（含错位复用的字）
    room_width: float = 0.0         # 允许加长到的宽度
    can_extend: bool = False        # 右侧是否有空间可加长

    def summary(self) -> str:
        if self.mode == "none":
            return "内容未变化"
        names = {"inplace": "原地换字（后续文字不动）",
                 "shift": "换字 + 后续刚性平移",
                 "reflow": "换字导致行数变化，整块重排"}
        return (f"{names.get(self.mode, self.mode)}："
                f"前 {self.keep_before} 字不动、后 {self.keep_after} 字不动、"
                f"改动 {len(self.changed_old)} → {len(self.changed_new)} 字"
                + (f"，宽度差 {self.width_delta:+.2f}pt" if abs(self.width_delta) > 0.01 else ""))


def available_width(block: Block, structure=None, page_width: float = 0.0,
                    margin: float = 24.0) -> tuple[float, str]:
    """算出这个块最多能用多宽 —— 优先考虑"向右加长"。

    用户明确要求：文字装不下时，先试着把文本框加长，
    检查是否碰到别的文本框；没碰到就该加长，而不是换行。

    返回 (可用宽度, 说明)。判断依据：
      1. 右边的最近邻块 —— 不能压到它
      2. 页面右边距 —— 不能超出
    两者取更近的那个。
    """
    cur = max((l.along1 - l.along0 for l in block.lines), default=0.0)
    left = min((l.along0 for l in block.lines), default=block.bbox[0])
    right_limit = (page_width - margin) if page_width > 0 else (left + cur)

    if structure is None:
        return max(cur, right_limit - left), "无结构信息，只受页面边距限制"

    blockers = []
    for pg in structure.pages:
        if pg.page_no != block.page_no:
            continue
        for b in pg.blocks:
            if b is block:
                continue
            # 只考虑竖直方向有重叠的块（同一水平带上的邻居）
            if b.bbox[3] <= block.bbox[1] + 1 or b.bbox[1] >= block.bbox[3] - 1:
                continue
            if b.bbox[0] >= left + 2:      # 在本块右侧
                blockers.append(b.bbox[0])

    if blockers:
        nearest = min(blockers)
        room = nearest - 6.0 - left       # 留 6pt 间隙
        note = f"右侧最近内容在 x={nearest:.0f}pt，最多可加长到那儿"
        return max(cur, min(room, right_limit - left)), note
    return max(cur, right_limit - left), "右侧无其他内容，可加长到页边距"


def plan_typeset(
    block: Block, new_text: str, font: pymupdf.Font,
    *, force_reflow: bool = False, structure=None, page_width: float = 0.0,
) -> TypeSetPlan:
    """算出这个改动该怎么排。

    只做规划，不改文档 —— 方便先给用户看"这次改动会影响多少文字"。
    """
    case = build_type_case(block)
    old = case.text
    # 活字盘里的文本是没有换行符的连续字符流，
    # 而 block.text 是用换行符拼的、用户输入也可能带换行。
    # 不规整化的话两边长度和内容都对不上，逐字 diff 会完全失真
    # （实测把"只改 2 个字"误判成"改动 42 个字"）。
    from .editor import _normalize_input
    new = _normalize_input(new_text)

    if old == new:
        return TypeSetPlan(mode="none")

    p, s = common_affix(old, new)
    plan = TypeSetPlan(
        mode="inplace", prefix_len=p, suffix_len=s,
        keep_before=p, keep_after=s,
    )
    plan.changed_new = new[p:len(new) - s] if s else new[p:]
    plan.changed_old = case.units[p:len(old) - s] if s else case.units[p:]
    plan.start_line = case.line_of(p) if p < len(case.units) else (len(case.lines) - 1)
    end_idx = len(old) - s - 1
    plan.end_line = case.line_of(end_idx) if 0 <= end_idx < len(case.units) else plan.start_line

    if not plan.changed_old:
        # 纯插入（只看不删）：不能当成"没变化"提前返回，
        # 否则插进去的字没地方放、后面的文字也不挪位。
        # 按 shift 处理：在插入点画新字，后缀整体右移。
        if not plan.changed_new:
            return plan
        idx = min(p, len(case.units) - 1)
        anchor = case.units[idx]
        plan.start_origin = anchor.origin
        plan.start_line = anchor.line_idx
        plan.end_line = anchor.line_idx
        # 插入点之后的字符整体右移插入内容的宽度
        plan.width_delta = font.text_length(plan.changed_new, fontsize=anchor.size)
        plan.mode = "shift"
        block_width = max((l.along1 - l.along0 for l in block.lines), default=0.0)
        room, _ = available_width(block, structure, page_width)
        plan.room_width = room
        plan.can_extend = room > block_width + 0.5
        plan.lines_after = plan.end_line - plan.start_line + 1
        return plan

    first = plan.changed_old[0]
    last = plan.changed_old[-1]
    plan.start_origin = first.origin
    old_width = last.bbox[2] - first.bbox[0]

    new_width = font.text_length(plan.changed_new, fontsize=first.size) if plan.changed_new else 0.0
    plan.width_delta = new_width - old_width

    tol = max(MIN_WIDTH_TOL, old_width * WIDTH_TOL_RATIO)

    if force_reflow:
        plan.mode = "reflow"
        return plan

    # 行数是否变化：把新的变动段按原块宽度排一遍看占几行
    block_width = max((l.along1 - l.along0 for l in block.lines), default=old_width)
    from .editor import break_lines
    laid = break_lines(plan.changed_new, font, first.size, block_width) \
        if plan.changed_new else [""]
    plan.lines_after = len(laid)
    old_lines = plan.end_line - plan.start_line + 1

    if abs(plan.width_delta) <= tol and len(laid) == old_lines:
        plan.mode = "inplace"
    elif len(laid) == old_lines:
        plan.mode = "shift"
    else:
        plan.mode = "reflow"
    return plan


# --------------------------------------------------------------- 执行


def _unit_rect(units: list[TypeUnit], pad: float = 1.0):
    x0 = min(u.bbox[0] for u in units) - 0.5
    y0 = min(u.bbox[1] for u in units) - pad
    x1 = max(u.bbox[2] for u in units) + 0.5
    y1 = max(u.bbox[3] for u in units) + pad
    return pymupdf.Rect(x0, y0, x1, y1)


def _redact_units(page: pymupdf.Page, units: list[TypeUnit], size: float) -> int:
    """只擦除这些活字 —— 按行分组，每行一个紧贴的擦除矩形。"""
    from .editor import commit_redactions

    by_line: dict[int, list[TypeUnit]] = {}
    for u in units:
        by_line.setdefault(u.line_idx, []).append(u)

    pad = min(1.5, max(0.4, 0.12 * size))
    n = 0
    for _, group in by_line.items():
        r = _unit_rect(group, pad)
        page.add_redact_annot(r, fill=False)
        n += 1
    if n:
        commit_redactions(page)
    return n


def apply_typeset(
    doc: pymupdf.Document,
    block: Block,
    new_text: str,
    font: pymupdf.Font,
    plan: TypeSetPlan | None = None,
    *,
    color: tuple[float, float, float] | None = None,
) -> dict:
    """按活字方案执行编辑。

    inplace：只擦改动的那几个字，后面的字原地不动（保真度最高）
    shift  ：擦掉改动段及其之后的部分，改动段之后整体刚性平移
    reflow ：退回整块重排
    """
    if plan is None:
        plan = plan_typeset(block, new_text, font)
    if plan.mode == "none":
        return {"mode": "none", "redacted": 0, "drawn": 0, "kept": 0}

    page = doc[block.page_no]
    use_color = color if color is not None else block.color
    case = build_type_case(block)

    if plan.mode == "reflow":
        return {"mode": "reflow", "delegate": True}

    # ---------- 采集"活字盘"：段落里已有的字形，改动时直接复用
    #
    # 必须在擦除之前做快照 —— 擦除之后原来的字形就没了，没法再复制。
    # 复用是像素级完美的（实测复制前后像素差异为 0），
    # 而用字体匹配"仿造"只是近似。所以能复用就复用。
    inventory = build_glyph_inventory(block)
    source_doc = pymupdf.open("pdf", doc.tobytes())

    # ---------- inplace / shift 共用的擦除范围
    redact_units = (plan.changed_old if plan.mode == "inplace"
                    else case.units[plan.prefix_len:])
    size = plan.changed_old[0].size if plan.changed_old else block.size
    n_rect = _redact_units(page, redact_units, size)

    # ---------- 绘制：优先复用原字形，字盘里没有的才用字体画
    x, baseline = plan.start_origin
    reused, drawn = render_glyph_run(
        page, source_doc, block.page_no, plan.changed_new,
        font, size, x, baseline, inventory,
    )

    total_drawn = drawn
    tail_reused = 0
    if plan.mode == "shift" and plan.suffix_len > 0:
        # 后续的字整体刚性平移 —— 关键是"刚性"：内部间距完全保持，
        # 不是重新断行。重新断行会让整段的换行位置全变，那就不叫活字了。
        tail = case.units[plan.prefix_len + len(plan.changed_old):]
        new_end_x = x + font.text_length(plan.changed_new, fontsize=size)
        # 纯插入时 changed_old 为空，位移基准就是插入点本身
        old_end_x = (plan.changed_old[-1].bbox[2] if plan.changed_old
                     else plan.start_origin[0])
        dx = new_end_x - old_end_x
        # 逐字复用原件做刚性平移 —— 不能整段用字体重排，
        # 那样没改的字也会跟着走样（用户明确要求保留能保留的部分）
        tail_reused, tail_drawn = shift_glyph_run(
            page, source_doc, block.page_no, tail, inventory, dx,
            font, size, fallback_color=use_color,
        )
        total_drawn += tail_drawn

    source_doc.close()
    return {
        "mode": plan.mode,
        "redacted": n_rect,
        "drawn": total_drawn,
        "reused": reused + (tail_reused if plan.mode == "shift" else 0),
        "tail_reused": tail_reused if plan.mode == "shift" else 0,
        "kept": plan.keep_before + plan.keep_after,
        "width_delta": plan.width_delta,
    }


# --------------------------------------------------------------- 字形复用
#
# 用户提出的核心思路：段落里已经有的字，直接复制一份用，不要用字体去"仿造"。
#
# 做成之后比字体匹配好得多 —— 复用是像素级完美的，匹配永远只是近似。
# 实测：同一个字用 Form XObject 从原位置复制到新位置，像素差异为 0；
# 而用匹配字体重绘，即使度量比 1.0000 也会有细微差别。
#
# 实现方式：page.show_pdf_page(target_rect, source_doc, pno, clip=source_rect)
# 把原字形的区域整块作为 Form XObject 搬过去。
# 试过直接往内容流追加原始操作符（更"正统"），但 MuPDF 不渲染，只有这条路可靠。

GLYPH_CLIP_PAD = 1.2      # 裁剪框外扩（pt）。实测 1.2 时像素差异为 0；
                          # 0.6 会因亚像素取整产生十几像素差异。


@dataclass
class GlyphSource:
    """一个可复用的字形：它在原页面的位置 + 相对基线的偏移。"""
    ch: str
    clip: tuple[float, float, float, float]      # 原位裁剪框（已外扩）
    offset: tuple[float, float, float, float]    # 裁剪框相对 origin 的偏移
    size: float
    advance: float
    font: str


def build_glyph_inventory(block: Block) -> dict[str, GlyphSource]:
    """采集段落里每个字的字形来源，形成"活字盘"。

    同一个字出现多次时取第一个 —— 字形完全相同，取哪个都一样。
    """
    inv: dict[str, GlyphSource] = {}
    for line in block.lines:
        for span in line.spans:
            for ch in span.chars:
                if ch.c in inv or not ch.c.strip():
                    continue
                x0, y0, x1, y1 = ch.bbox
                ox, oy = ch.origin
                clip = (x0 - GLYPH_CLIP_PAD, y0 - GLYPH_CLIP_PAD,
                        x1 + GLYPH_CLIP_PAD, y1 + GLYPH_CLIP_PAD)
                inv[ch.c] = GlyphSource(
                    ch=ch.c, clip=clip,
                    offset=(clip[0] - ox, clip[1] - oy,
                            clip[2] - ox, clip[3] - oy),
                    size=span.size,
                    advance=(ch.bbox[2] - ch.bbox[0]),
                    font=span.font,
                )
    return inv


def shift_glyph_run(
    page: pymupdf.Page,
    source_doc: pymupdf.Document,
    source_page_no: int,
    units: list,
    inventory: dict[str, GlyphSource],
    dx: float,
    font: pymupdf.Font,
    size: float,
    fallback_color=(0.0, 0.0, 0.0),
) -> tuple[int, int]:
    """把一段文字整体平移 dx，**逐字复用原件**。

    用户明确要求：文本框加长时，能保留的应该保留，不能整段重画。
    所以这里是"刚性平移"——每个字都用它自己的矢量原件，只改横向位置，
    字距、笔画、抗锯齿全部和原来一模一样。

    早期版本是用 TextWriter 把这段文字用**字体重新排一遍**，
    虽然位置对了，但字形是重画的：遇到字体不匹配、系统缺原字体时，
    整段后半部分都会跟着走样 —— 明明没改它，看起来却变了。

    返回 (复用字数, 新绘字数)。
    """
    reused = 0
    drawn = 0
    fallback: list[tuple[str, float, float]] = []

    def flush():
        nonlocal fallback, drawn
        if not fallback:
            return
        tw = pymupdf.TextWriter(page.rect)
        for ch, fx, fy in fallback:
            tw.append((fx, fy), ch, font=font, fontsize=size)
        tw.write_text(page, color=fallback_color, overlay=True)
        drawn += len(fallback)
        fallback = []

    for u in units:
        src = inventory.get(u.ch) if u.ch.strip() else None
        if src is not None and abs(src.size - size) < 0.6:
            flush()
            tx = u.origin[0] + dx
            ty = u.origin[1]
            target = pymupdf.Rect(
                tx + src.offset[0], ty + src.offset[1],
                tx + src.offset[2], ty + src.offset[3],
            )
            try:
                page.show_pdf_page(target, source_doc, source_page_no,
                                   clip=pymupdf.Rect(*src.clip))
                reused += 1
                continue
            except Exception:
                pass
        fallback.append((u.ch, u.origin[0] + dx, u.origin[1]))
    flush()
    return reused, drawn


def render_glyph_run(
    page: pymupdf.Page,
    source_doc: pymupdf.Document,
    source_page_no: int,
    text: str,
    font: pymupdf.Font,
    size: float,
    start_x: float,
    baseline: float,
    inventory: dict[str, GlyphSource],
) -> tuple[int, int]:
    """把一段文字画到指定位置：字盘里有的复用原字形，没有的用字体绘制。

    返回 (复用字数, 新绘字数)。

    为什么复用比匹配好：复用是原封不动地把那个字的矢量内容搬过来，
    连渲染 hinting、灰度、笔画粗细都完全一致；匹配只能做到"度量对得上"，
    遇到字体版本不同、有变形、或者本身就是仿制字体时就会看得出差异。
    """
    reused = 0
    drawn = 0
    x = start_x
    fallback_buf: list[str] = []
    fallback_x = x

    def flush_fallback():
        nonlocal fallback_buf, fallback_x, drawn
        if fallback_buf:
            tw = pymupdf.TextWriter(page.rect)
            tw.append((fallback_x, baseline), "".join(fallback_buf),
                      font=font, fontsize=size)
            tw.write_text(page, color=(0, 0, 0), overlay=True)
            drawn += len(fallback_buf)
            fallback_buf = []

    for ch in text:
        src = inventory.get(ch) if ch.strip() else None
        if src is not None and abs(src.size - size) < 0.6:
            flush_fallback()
            target = pymupdf.Rect(
                x + src.offset[0], baseline + src.offset[1],
                x + src.offset[2], baseline + src.offset[3],
            )
            try:
                page.show_pdf_page(target, source_doc, source_page_no,
                                   clip=pymupdf.Rect(*src.clip))
                reused += 1
                x += font.text_length(ch, fontsize=size)
                fallback_x = x
                continue
            except Exception:
                pass   # 复用失败就退回字体绘制
        if not fallback_buf:
            fallback_x = x
        fallback_buf.append(ch)
        x += font.text_length(ch, fontsize=size)
    flush_fallback()
    return reused, drawn
