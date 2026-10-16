"""重排引擎：块变高之后，把下游内容推到新位置。

核心是「最小位移原则」：
  重排后逐块比对新旧位置，位置没变的块（偏差 < 阈值）一律不碰，
  保留原文件字节。只有真正需要移动的块才擦除重画。

这条原则把保真度从"整页重建"抢回到"局部重建"：
  改一个字但没撑开行 → 整页字节级零变化
  改长了撑开三行    → 只有被顶下去的部分重新生成，页面上半部分和
                      所有固定层元素一个 bit 都没动

固定层（页眉/页脚/水印/背景）永不参与重排 —— 这是方案的生命线，
否则改一个字就会把印章和页码顶飞。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from .editor import (
    _block_width,
    _is_justified,
    _original_indent,
    commit_redactions,
    draw_block,
    draw_rotated_block,
    plan_lines,
    redact_block,
)
from .fonts import FontMatch, resolve_for_block
from .model import KIND_IMAGE, KIND_TABLE, LAYER_FIXED, LAYER_FLOW, Block, DocStructure

# 位移阈值：小于此值的位移视为"位置没变"，跳过重画
SHIFT_EPSILON = 0.5


@dataclass
class BlockMove:
    block: Block
    old_top: float
    new_top: float
    shift: float
    action: str          # edited | redraw | keep
    lines: list[str] = field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:
        mark = {"edited": "✎ 编辑", "redraw": "↕ 重画", "keep": "✓ 保持"}[self.action]
        return (f"  {mark}  y {self.old_top:7.1f} → {self.new_top:7.1f} "
                f"(位移 {self.shift:+6.2f})  {self.block.text[:30]!r}")


@dataclass
class ReflowPlan:
    page_no: int
    moves: list[BlockMove] = field(default_factory=list)
    overflow_pages: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return sum(1 for m in self.moves if m.action == "keep")

    @property
    def redrawn(self) -> int:
        return sum(1 for m in self.moves if m.action == "redraw")

    @property
    def edited(self) -> int:
        return sum(1 for m in self.moves if m.action == "edited")

    def report(self) -> str:
        lines = [
            f"重排计划 p{self.page_no+1}："
            f"编辑 {self.edited}  重画 {self.redrawn}  保持原样 {self.kept}",
        ]
        for m in self.moves:
            lines.append(str(m))
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def _reading_order(blocks: list[Block]) -> list[Block]:
    """按阅读顺序排列：先从上到下，同一水平带内从左到右。"""
    return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))


def plan_reflow(
    structure: DocStructure,
    page_no: int,
    edited_block: Block,
    new_text: str,
    match: FontMatch,
) -> ReflowPlan:
    """计算重排方案：哪些块要动、移到哪里、哪些可以原样保留。"""
    page_struct = structure.pages[page_no]
    plan = ReflowPlan(page_no=page_no)

    # 只重排流式层；图片和表格作为整体刚性移动，不重排内部
    flow = [b for b in page_struct.blocks if b.layer == LAYER_FLOW]
    ordered = _reading_order(flow)

    # 编辑块的新高度
    size = edited_block.size
    width = _block_width(edited_block)
    indent = _original_indent(edited_block)
    gap = edited_block.line_gap if len(edited_block.lines) > 1 else size * 1.5
    if gap <= 0:
        gap = size * 1.5

    right_edge = max(l.along1 for l in edited_block.lines)
    margin_room = page_struct.width - 24.0 - right_edge
    hang = max(0.0, min(size, margin_room))

    new_lines, _ = plan_lines(edited_block, new_text, match.font, size, width, indent, hang)
    old_height = (len(edited_block.lines) - 1) * gap
    new_height = (len(new_lines) - 1) * gap
    height_delta = new_height - old_height

    # 编辑块底部（用于判断哪些块在它下游）
    edit_bottom = max(l.bbox[3] for l in edited_block.lines)

    cumulative = 0.0
    for block in ordered:
        if block is edited_block:
            plan.moves.append(BlockMove(
                block=block, old_top=block.bbox[1], new_top=block.bbox[1],
                shift=0.0, action="edited", lines=new_lines,
                reason=f"用户编辑：{len(edited_block.lines)} 行 → {len(new_lines)} 行",
            ))
            cumulative += height_delta
            continue

        # 编辑块上游的块不受影响
        if block.bbox[3] <= edit_bottom + 0.5:
            plan.moves.append(BlockMove(
                block=block, old_top=block.bbox[1], new_top=block.bbox[1],
                shift=0.0, action="keep",
                reason="位于编辑位置上游，不受影响",
            ))
            continue

        # 下游的块按累计位移下移
        shift = cumulative
        new_top = block.bbox[1] + shift
        if abs(shift) < SHIFT_EPSILON:
            action = "keep"
            reason = f"累计位移 {shift:+.2f}pt 小于阈值，保留原字节"
            new_top = block.bbox[1]
        else:
            action = "redraw"
            reason = f"被上游内容顶下 {shift:+.2f}pt"

        plan.moves.append(BlockMove(
            block=block, old_top=block.bbox[1], new_top=new_top,
            shift=shift, action=action,
            lines=[l.text for l in block.lines],
            reason=reason,
        ))

        # 图片/表格按刚性整体处理，自身高度不变
        if block.kind not in (KIND_IMAGE, KIND_TABLE):
            pass

    # 溢出检查：最后一个块是否超出页面
    if plan.moves:
        lowest = max(m.new_top for m in plan.moves)
        last = max(plan.moves, key=lambda m: m.old_top)
        new_bottom = last.block.bbox[3] + last.shift
        page_height = page_struct.height
        if new_bottom > page_height - 60:
            plan.overflow_pages += 1
            plan.warnings.append(
                f"重排后内容底部 {new_bottom:.1f} 超出页面安全区 "
                f"({page_height - 60:.1f})，需要溢出到下一页"
            )
    return plan


def endangered_fixed_blocks(
    structure: DocStructure, page_no: int,
    redact_rects: list[tuple[float, float, float, float]],
) -> list[Block]:
    """找出会被擦除波及的固定层文本块。

    为什么必须有这一步：apply_redactions 会删除「任何包围盒与擦除矩形相交的
    字符」，不管它属于哪一层。水印是斜跨整页的旋转文本，包围盒巨大，
    擦除正文行时必然命中它 —— 实测擦一条正文行就会削掉水印的一两个字，
    擦满一页水印就整个消失。

    所以固定层保护不能只是"不去碰它"，必须在擦除后把被波及的元素原样重绘回来。
    """
    out: list[Block] = []
    rects = [pymupdf.Rect(r) for r in redact_rects]
    for b in structure.pages[page_no].blocks:
        if b.layer != LAYER_FIXED or not b.lines:
            continue
        bb = pymupdf.Rect(b.bbox)
        if any(bb.intersects(r) for r in rects):
            out.append(b)
    return out


def apply_reflow(
    doc: pymupdf.Document,
    structure: DocStructure,
    plan: ReflowPlan,
    edited_block: Block,
    new_text: str,
    match: FontMatch,
) -> dict:
    """执行重排。

    关键顺序：
      1. 擦完所有旧位置（先擦后画，否则后一块的擦除会毁掉前一块刚画好的内容）
      2. 重绘被擦除波及的固定层元素（水印等），且画在内容下层
      3. 再画所有正文新位置（覆盖在水印之上，z 序正确）
    """
    page = doc[plan.page_no]
    font = match.font

    # ---- 阶段一：擦除所有需要动的块的旧位置
    to_redraw = [m for m in plan.moves if m.action in ("edited", "redraw")]
    redact_rects: list[tuple[float, float, float, float]] = []
    for m in to_redraw:
        redact_rects.extend(redact_block(page, m.block))

    # 先记下会被波及的固定层元素 —— 必须在 apply_redactions 之前采集
    endangered = endangered_fixed_blocks(structure, plan.page_no, redact_rects)

    if to_redraw:
        commit_redactions(page)

    # ---- 阶段二：把被波及的固定层元素原样重绘回来（画在下层）
    stats = {"redrawn": 0, "kept": 0, "edited": 0, "fixed_restored": 0}
    for b in endangered:
        if b.is_rotated:
            if draw_rotated_block(page, b):
                stats["fixed_restored"] += 1
        else:
            sub = resolve_for_block(b.font, b)
            if sub.ok:
                draw_block(page, b, [l.text for l in b.lines], sub.font, b.size,
                           justify=_is_justified(b), indent=_original_indent(b))
                stats["fixed_restored"] += 1

    # ---- 阶段三：在新位置绘制正文
    for m in plan.moves:
        if m.action == "keep":
            stats["kept"] += 1
            continue
        if m.action == "edited":
            draw_block(
                page, m.block, m.lines, font, m.block.size,
                y_offset=0.0, justify=_is_justified(m.block),
                indent=_original_indent(m.block),
            )
            stats["edited"] += 1
            continue
        # redraw：同一文字，整体位移
        sub = resolve_for_block(m.block.font, m.block)
        if not sub.ok:
            continue
        draw_block(
            page, m.block, m.lines, sub.font, m.block.size,
            y_offset=m.shift, justify=_is_justified(m.block),
            indent=_original_indent(m.block),
        )
        stats["redrawn"] += 1
    return stats
