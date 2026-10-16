"""擦除 + 重画闭环：把结构模型里的一段文字换掉，其余像素一个不动。

流程：
  1. 按行精确计算擦除矩形（紧贴原文字，不多占一分）
  2. add_redact_annot + apply_redactions —— 真正从内容流删除字形
  3. 用匹配字体重排版新文本，按原基线与行距绘制

两个必须显式设置的参数，否则会出事：
  - graphics=PDF_REDACT_LINE_ART_NONE
    默认值 REMOVE_IF_COVERED 会把被擦除框覆盖的矢量图形一起删掉，
    表格边框、下划线、页眉分隔线会全部消失。
  - images=PDF_REDACT_IMAGE_NONE
    默认值 PIXELS 会把覆盖到的图像像素涂白，背景图会被打洞。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymupdf

from .fonts import FontMatch
from .model import Block

# 行内不允许出现在行首的标点（简易禁则处理，避免标点孤悬行首）
NO_LINE_START = set("。，、；：？！）】》」』%,.;:?!)]}>”’")
NO_LINE_END = set("（【《「『([{<“‘")

# 行间距与块间距
REDACT_PAD_RATIO = 0.12      # 擦除矩形上下留白 = 比例 × 字号
MIN_REDACT_PAD = 0.4
MAX_REDACT_PAD = 1.5


@dataclass
class EditResult:
    page_no: int
    old_text: str
    new_text: str
    redact_rects: list[tuple[float, float, float, float]] = field(default_factory=list)
    drawn_lines: list[dict] = field(default_factory=list)
    lines_before: int = 0
    lines_after: int = 0
    baseline_first: float = 0.0
    line_gap: float = 0.0
    overflow: bool = False
    preserved_breaks: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def height_before(self) -> float:
        return self.lines_before * self.line_gap

    @property
    def height_after(self) -> float:
        return self.lines_after * self.line_gap

    @property
    def height_delta(self) -> float:
        """正数表示变高了，需要把下面的内容下推。"""
        return self.height_after - self.height_before

    def summary(self) -> str:
        d = self.height_delta
        arrow = f"高度变化 {d:+.1f}pt" if abs(d) > 0.01 else "高度不变"
        keep = "  [沿用原始断行]" if self.preserved_breaks else ""
        return (f"p{self.page_no+1}  {self.lines_before} 行 → {self.lines_after} 行  {arrow}{keep}"
                + ("  ⚠ 溢出块边界" if self.overflow else ""))


# --------------------------------------------------------------- 文本折行


def _normalize_input(text: str) -> str:
    """规整输入文本。

    block.text 是用 \\n 把各行拼起来的，那是"显示用"的产物。
    直接喂回排版引擎会把 \\n 当成真字符画出来（SimSun 里没有它的字形，
    会变成空白或豆腐块），所以必须先把行断符还原成连续的流。
    中文字之间直接相连，西文之间补一个空格，避免单词粘连。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in text:
        return text
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch != "\n":
            out.append(ch)
            continue
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if prev and nxt and not _is_cjk(prev) and not _is_cjk(nxt):
            out.append(" ")
    return "".join(out)


def _tokenize(text: str) -> list[str]:
    """切分成可断行的最小单元：CJK 单字成组，拉丁词整体成组。"""
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if _is_cjk(ch):
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch == " ":
            buf += ch
            tokens.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF or 0x3000 <= o <= 0x303F
            or 0xFF00 <= o <= 0xFFEF)


def _advance(font: pymupdf.Font, token: str, size: float) -> float:
    return font.text_length(token, fontsize=size)


def break_lines(
    text: str, font: pymupdf.Font, size: float, width: float,
    first_indent: float = 0.0, hang: float = 0.0, kinsoku: bool = True,
) -> list[str]:
    """按可用宽度折行。

    kinsoku=False 时用朴素断行（填满就断，不管标点位置）。

    为什么要能关掉禁则：本工具的目标是保真，不是把排版改得更规范。
    原文若用朴素断行（行首出现 '、' 这类标点就是证据），我们重排时
    也必须用朴素断行，否则同一段文字重排后断行位置会变，
    版式就漂移了 —— 哪怕一个字都没改。

    hang 是允许标点悬挂的最大额外宽度。即使开启禁则，悬挂量也受
    页面边距约束，不能让标点溢出版心压到页边距。
    """
    tokens = _tokenize(text)
    lines: list[str] = []
    cur: list[str] = []
    cur_w = 0.0
    limit = width - first_indent
    indent = first_indent

    for tok in tokens:
        w = _advance(font, tok, size)
        if cur and cur_w + w > limit + 0.01:
            if kinsoku and tok and tok[0] in NO_LINE_START:
                # 禁则一：标点在允许范围内悬挂到行末
                if cur_w + w <= limit + hang + 0.01:
                    cur.append(tok)
                    lines.append("".join(cur))
                    cur, cur_w, limit, indent = [], 0.0, width, 0.0
                    continue
                # 悬挂放不下 → 把前一个字符一起挪到下一行（避头尾）
                moved = cur.pop()
                if cur:
                    lines.append("".join(cur))
                cur = [moved, tok]
                cur_w = _advance(font, moved, size) + w
                limit, indent = width, 0.0
                continue
            # 禁则二：本行末尾是禁止行尾的标点 → 挪到下一行
            if kinsoku and cur and cur[-1] and cur[-1][-1] in NO_LINE_END:
                moved = cur.pop()
                lines.append("".join(cur))
                cur = [moved, tok]
                cur_w = _advance(font, moved, size) + w
                limit, indent = width, 0.0
                continue
            lines.append("".join(cur))
            cur, cur_w, limit, indent = [tok], w, width, 0.0
        else:
            cur.append(tok)
            cur_w += w
    if cur:
        lines.append("".join(cur))
    return lines


def detect_kinsoku(block: Block) -> bool:
    """从原文档的断行结果反推它是否应用了禁则。

    行首出现「禁止行首」的标点，就是原文用朴素断行的直接证据。
    照搬原文的行为，重排才不会产生额外漂移。
    """
    for line in block.lines:
        t = line.text.lstrip()
        if t and t[0] in NO_LINE_START:
            return False
    return True


def plan_lines(
    block: Block, text: str, font: pymupdf.Font, size: float,
    width: float, indent: float, hang: float,
) -> tuple[list[str], bool]:
    """决定新文本怎么断行，返回 (行列表, 是否沿用了原始断行)。

    最小位移原则的第一个应用点：文字没变就不重新断行。
    重新断行会因为禁则规则、宽度取整等细节产生细微差异，
    哪怕文字一字未改，版式也会漂移。文字未变时直接沿用原始断行，
    结果是真正的零漂移。
    """
    if text == _normalize_input(block.text) and block.lines:
        return [l.text for l in block.lines], True
    kinsoku = detect_kinsoku(block)
    return break_lines(text, font, size, width, first_indent=indent,
                       hang=hang, kinsoku=kinsoku), False


# --------------------------------------------------------------- 版式计算


def _original_indent(block: Block) -> float:
    """原块首行相对左边界的缩进量。"""
    if not block.lines:
        return 0.0
    lefts = [l.along0 for l in block.lines]
    return max(0.0, block.lines[0].along0 - min(lefts))


def _block_width(block: Block) -> float:
    return max((l.width for l in block.lines), default=0.0)


def _is_justified(block: Block) -> bool:
    """块是否两端对齐：多行且多数行宽度一致且接近满宽。"""
    if len(block.lines) < 3:
        return False
    ws = [l.width for l in block.lines[:-1]]
    if not ws:
        return False
    mx = max(ws)
    full = sum(1 for w in ws if abs(w - mx) < 1.0)
    return full >= max(2, len(ws) * 0.7)


# --------------------------------------------------------------- 主流程


def redact_block(page: pymupdf.Page, block: Block) -> list[tuple[float, float, float, float]]:
    """擦除块的原始文字，返回用到的擦除矩形。

    单独抽出来是因为重排场景必须先擦完所有旧位置、再画所有新位置。
    若边擦边画，后面块的擦除会把前面块刚画好的内容一并抹掉。
    """
    size = block.size
    pad = min(MAX_REDACT_PAD, max(MIN_REDACT_PAD, REDACT_PAD_RATIO * size))
    rects: list[tuple[float, float, float, float]] = []
    for line in block.lines:
        x0, y0, x1, y1 = line.bbox
        rect = pymupdf.Rect(x0 - 0.5, y0 - pad, x1 + 0.5, y1 + pad)
        page.add_redact_annot(rect, fill=False)
        rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
    return rects


def commit_redactions(page: pymupdf.Page) -> bool:
    """执行已登记的擦除。这两个参数必须显式设置，理由见模块文档。"""
    return page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
    )


def draw_block(
    page: pymupdf.Page,
    block: Block,
    lines: list[str],
    font: pymupdf.Font,
    size: float,
    *,
    y_offset: float = 0.0,
    justify: bool = False,
    indent: float | None = None,
) -> list[dict]:
    """按块的原始几何绘制文本行，返回每行的实际落点。

    旋转块（水印）走另一条路径 —— 见 draw_rotated_block。
    """
    color = block.color
    indent = _original_indent(block) if indent is None else indent
    width = _block_width(block)
    gap = block.line_gap if len(block.lines) > 1 else size * 1.5
    if gap <= 0:
        gap = size * 1.5

    first_cross = block.lines[0].cross + y_offset
    left = min(l.along0 for l in block.lines)
    drawn: list[dict] = []

    tw = pymupdf.TextWriter(page.rect)
    for i, text in enumerate(lines):
        baseline = first_cross + i * gap
        start_x = (left + indent) if i == 0 else left
        natural = font.text_length(text, fontsize=size)
        last_line = (i == len(lines) - 1)

        if justify and not last_line and natural < width - 0.5:
            extra = (width - natural) / max(1, len(text) - 1)
            x = start_x
            for ch in text:
                tw.append((x, baseline), ch, font=font, fontsize=size)
                x += font.text_length(ch, fontsize=size) + extra
        else:
            tw.append((start_x, baseline), text, font=font, fontsize=size)

        drawn.append({"index": i, "text": text, "baseline": baseline,
                      "x": start_x, "width": natural})

    tw.write_text(page, color=color, overlay=True)
    return drawn


def draw_rotated_block(
    page: pymupdf.Page,
    block: Block,
    text: str | None = None,
    font: pymupdf.Font | None = None,
) -> bool:
    """重绘旋转文本（水印等）。

    做法：在首字符原点追加文字，再以该点为定点做旋转。
    关键在旋转符号 —— PyMuPDF 的 Matrix(θ) 与方向向量 (cosθ, sinθ) 符号相反，
    直接传 atan2 得到的角度会把字形转反（原点对、字形镜像）。
    实测传 -angle 才能逐像素复现原水印。

    font 不传时按块的 PDF 字体名去匹配系统字体。
    """
    if not block.lines:
        return False
    line = block.lines[0]
    span = line.spans[0]
    if not span.chars:
        return False

    if font is None:
        from .fonts import resolve_for_block
        m = resolve_for_block(span.font, block)
        if not m.ok:
            return False
        font = m.font

    origin = span.chars[0].origin
    angle = math.degrees(math.atan2(span.direction[1], span.direction[0]))
    content = text if text is not None else "".join(c.c for c in span.chars)

    tw = pymupdf.TextWriter(page.rect)
    tw.append(origin, content, font=font, fontsize=span.size)
    tw.write_text(
        page,
        color=span.color,
        opacity=span.opacity if span.opacity < 1.0 else None,
        morph=(pymupdf.Point(origin[0], origin[1]), pymupdf.Matrix(-angle)),
        overlay=False,   # 水印在内容下层
    )
    return True


def replace_block_text(
    doc: pymupdf.Document,
    block: Block,
    new_text: str,
    match: FontMatch,
    *,
    fontsize: float | None = None,
    keep_indent: bool = True,
    y_offset: float = 0.0,
) -> EditResult:
    """把块的文字整体替换为 new_text，版式沿用原块几何。"""
    if not match.ok:
        raise ValueError(f"没有可用字体，无法重画：{match.warnings}")
    if not block.lines:
        raise ValueError("块内没有文本行")

    page = doc[block.page_no]
    font = match.font
    size = fontsize if fontsize is not None else block.size
    new_text = _normalize_input(new_text)

    result = EditResult(
        page_no=block.page_no,
        old_text=block.text,
        new_text=new_text,
        lines_before=len(block.lines),
    )

    # ---- 1. 擦除：逐行紧贴原文字
    result.redact_rects = redact_block(page, block)
    commit_redactions(page)

    # ---- 2. 版式计算
    width = _block_width(block)
    indent = _original_indent(block) if keep_indent else 0.0
    justify = _is_justified(block)

    first_cross = block.lines[0].cross
    gap = block.line_gap if len(block.lines) > 1 else size * 1.5
    if gap <= 0:
        gap = size * 1.5
    result.baseline_first = first_cross
    result.line_gap = gap

    # 标点悬挂的额度：最多探出一个字宽，且不能吃掉页边距
    right_edge = max(l.along1 for l in block.lines)
    margin_room = page.rect.width - 24.0 - right_edge
    hang = max(0.0, min(size, margin_room))

    lines, preserved = plan_lines(block, new_text, font, size, width, indent, hang)
    result.lines_after = len(lines)
    result.preserved_breaks = preserved

    # ---- 3. 重画
    result.drawn_lines = draw_block(
        page, block, lines, font, size,
        y_offset=y_offset, justify=justify, indent=indent,
    )

    # ---- 4. 溢出检查
    new_bottom = first_cross + y_offset + (len(lines) - 1) * gap + size * 0.3
    old_bottom = max(l.bbox[3] for l in block.lines)
    if new_bottom > old_bottom + 0.5:
        result.overflow = True
        result.warnings.append(
            f"新内容底部 {new_bottom:.1f} 超出原块底部 {old_bottom:.1f}，需要下推后续内容"
        )
    if match.warnings:
        result.warnings.extend(match.warnings)
    return result


def delete_block_text(doc: pymupdf.Document, block: Block) -> EditResult:
    """只擦除不重画 —— 用于删除整段。"""
    page = doc[block.page_no]
    result = EditResult(
        page_no=block.page_no, old_text=block.text, new_text="",
        lines_before=len(block.lines), lines_after=0,
    )
    result.redact_rects = redact_block(page, block)
    commit_redactions(page)
    result.line_gap = block.line_gap
    return result
