"""规则引擎：给 L2 块判定「类型 kind」和「图层 layer」，并给出置信度。

设计要点：
  1. 每条规则输出置信度而非硬判定。低于阈值的块被标记出来，交用户一键修正。
  2. 图层判定（flow / fixed）是最关键的一条 —— 固定层元素永不参与重排，
     否则改一个字就会把页眉、页码、水印顶飞。
  3. 每条判定都记录 reasons，让用户看得见"为什么这么判"，而不是黑盒。
"""

from __future__ import annotations

import math
from collections import Counter

from .model import (
    FIXED_KINDS,
    KIND_FOOTER,
    KIND_GRAPHICS,
    KIND_HEADING,
    KIND_HEADER,
    KIND_IMAGE,
    KIND_LIST,
    KIND_PARAGRAPH,
    KIND_TABLE,
    KIND_UNKNOWN,
    KIND_WATERMARK,
    LAYER_FIXED,
    LAYER_FLOW,
    Block,
    DocStructure,
)

# 阈值
HEADER_BAND = 70.0          # 页眉条带高度
FOOTER_BAND = 70.0          # 页脚条带高度
HEADING_SIZE_RATIO = 1.15   # 标题字号 / 正文字号
HEADING_MAX_CHARS = 40      # 标题最大字符数
WATERMARK_LIGHT = 0.72      # 水印灰度下限
WATERMARK_MIN_SIZE = 24.0   # 水印最小字号
CONFIDENCE_FLAG = 0.75      # 低于此值标记为待确认

# 列表行首模式
_LIST_PREFIXES = ("•", "·", "○", "●", "▪", "-", "—", "－", "*")
import re
_LIST_NUM = re.compile(r"^\s*(\(?\d{1,2}\)?[.、)]|\(?[一二三四五六七八九十]{1,3}\)?[.、)]|[A-Za-z][.)])\s")


def _is_light(color: tuple[float, float, float]) -> bool:
    r, g, b = color
    return r > WATERMARK_LIGHT and g > WATERMARK_LIGHT and b > WATERMARK_LIGHT


def _is_list_start(text: str) -> bool:
    t = text.lstrip()
    if not t:
        return False
    if t[0] in _LIST_PREFIXES:
        return True
    return bool(_LIST_NUM.match(t))


def _band_of(block: Block, height: float) -> str:
    """块整体落在页眉/页脚条带内？"""
    if block.bottom <= HEADER_BAND:
        return "header"
    if block.top >= height - FOOTER_BAND:
        return "footer"
    return ""


def _collect_repeated_bands(doc: DocStructure) -> tuple[set[str], set[str]]:
    """跨页重复出现的页眉/页脚文本 —— 用于提高判定置信度。

    真正的页眉页脚在多页文档里会反复出现（页码除外，所以要去掉数字再比）。
    """
    head_texts: Counter[str] = Counter()
    foot_texts: Counter[str] = Counter()
    npages = len(doc.pages)

    for page in doc.pages:
        seen_h, seen_f = set(), set()
        for b in page.blocks:
            band = _band_of(b, page.height)
            norm = _normalize_band_text(b.text)
            if not norm:
                continue
            if band == "header":
                seen_h.add(norm)
            elif band == "footer":
                seen_f.add(norm)
        head_texts.update(seen_h)
        foot_texts.update(seen_f)

    need = max(2, int(npages * 0.6))
    heads = {t for t, c in head_texts.items() if c >= need}
    foots = {t for t, c in foot_texts.items() if c >= need}
    return heads, foots


_DIGITS = re.compile(r"\d+")


def _normalize_band_text(text: str) -> str:
    """去掉数字后比较，这样"第 1 页"和"第 2 页"能识别为同一页脚。"""
    return _DIGITS.sub("#", text.replace("\n", " ").strip())


def classify(doc: DocStructure) -> DocStructure:
    """给整个文档的块打上 kind / layer / confidence。"""
    repeated_heads, repeated_foots = _collect_repeated_bands(doc)
    body_size = doc.body_size or 10.0

    for page in doc.pages:
        # 文本块按阅读顺序排列，让判定规则能看到上下文
        text_blocks = [
            b for b in page.blocks
            if b.kind not in (KIND_IMAGE, KIND_TABLE, KIND_GRAPHICS)
        ]
        text_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        for i, block in enumerate(text_blocks):
            prev_b = text_blocks[i - 1] if i > 0 else None
            next_b = text_blocks[i + 1] if i + 1 < len(text_blocks) else None
            _classify_text_block(
                block, page.height, body_size,
                repeated_heads, repeated_foots, prev_b, next_b,
            )

    # 统一：固定层类型强制归入 fixed 层；其余按各块自身判定
    for page in doc.pages:
        for block in page.blocks:
            if block.kind in FIXED_KINDS:
                block.layer = LAYER_FIXED
            elif block.layer != LAYER_FIXED:
                block.layer = LAYER_FLOW

    return doc


def _classify_text_block(
    block: Block,
    page_height: float,
    body_size: float,
    repeated_heads: set[str],
    repeated_foots: set[str],
    prev_block: Block | None = None,
    next_block: Block | None = None,
) -> None:
    reasons: list[str] = []
    confidence = 1.0

    # ---- 规则 1：旋转文本 → 水印（最强信号，方向向量不会骗人）
    if block.is_rotated:
        angle = math.degrees(math.atan2(*reversed(block.lines[0].direction)))
        block.kind = KIND_WATERMARK
        block.layer = LAYER_FIXED
        block.confidence = 0.98
        block.reasons = [f"旋转文本 {angle:.1f}°（非水平/垂直）"]
        return

    # ---- 规则 2：低饱和 + 大字号 → 水印（非旋转水印的兜底）
    if block.size >= WATERMARK_MIN_SIZE and _is_light(block.color):
        block.kind = KIND_WATERMARK
        block.layer = LAYER_FIXED
        block.confidence = 0.8
        block.reasons = [
            f"大字号 {block.size:.1f}pt 且颜色很浅 "
            f"({block.color[0]:.2f},{block.color[1]:.2f},{block.color[2]:.2f})"
        ]
        return

    # ---- 规则 3：页眉 / 页脚（位置 + 跨页重复）
    band = _band_of(block, page_height)
    if band == "header":
        norm = _normalize_band_text(block.text)
        repeated = norm in repeated_heads
        block.kind = KIND_HEADER
        block.layer = LAYER_FIXED
        block.confidence = 0.95 if repeated else 0.7
        block.reasons = [
            f"位于页眉条带（底部 {block.bottom:.1f} ≤ {HEADER_BAND}）",
            "跨页重复出现" if repeated else "未发现跨页重复，置信度降低",
        ]
        return
    if band == "footer":
        norm = _normalize_band_text(block.text)
        repeated = norm in repeated_foots
        block.kind = KIND_FOOTER
        block.layer = LAYER_FIXED
        block.confidence = 0.95 if repeated else 0.7
        block.reasons = [
            f"位于页脚条带（顶部 {block.top:.1f} ≥ {page_height - FOOTER_BAND:.1f}）",
            "跨页重复出现" if repeated else "未发现跨页重复，置信度降低",
        ]
        return

    # ---- 规则 4：标题（字号显著大于正文）
    text = block.text.strip()
    if block.size >= body_size * HEADING_SIZE_RATIO and len(text) <= HEADING_MAX_CHARS:
        ratio = block.size / body_size
        block.kind = KIND_HEADING
        block.layer = LAYER_FLOW
        block.confidence = 0.92 if ratio >= 1.3 else 0.8
        block.reasons = [f"字号 {block.size:.1f}pt 为正文 {body_size:.1f}pt 的 {ratio:.2f} 倍"]
        if block.lines and block.lines[0].spans[0].is_bold:
            block.reasons.append("加粗")
        return

    # ---- 规则 5：列表
    first_line = block.lines[0].text if block.lines else ""
    if _is_list_start(first_line):
        block.kind = KIND_LIST
        block.layer = LAYER_FLOW
        block.confidence = 0.85
        block.reasons = ["行首存在列表标记"]
        return

    # ---- 规则 6：短行独立成块 → 疑似小标题（与正文同字号，只能靠版面判断）
    #      关键结构信号：短行 + 紧跟一个缩进的正文段落。
    #      这是中文公文/合同里"第X条 标题"的典型形态。
    if len(block.lines) == 1 and len(text) <= 24 and not text.endswith(("。", "；", "，", ".")):
        next_text = next_block.text if next_block is not None else ""
        next_indented = next_text.startswith(("　", " ", "\t"))
        next_is_longer = (
            next_block is not None
            and next_block.kind not in FIXED_KINDS
            and len(next_text) > len(text) * 1.8
        )
        if next_indented and next_is_longer:
            block.kind = KIND_HEADING
            block.layer = LAYER_FLOW
            block.confidence = 0.88
            block.reasons = [
                "单行短文本、无句末标点",
                "紧随其后的是缩进正文段落 → 判定为条款小标题",
            ]
        else:
            block.kind = KIND_HEADING
            block.layer = LAYER_FLOW
            block.confidence = 0.6
            block.reasons = ["单行短文本且无句末标点，疑似小标题（缺少结构佐证）"]
        return

    # ---- 规则 7：默认段落
    block.kind = KIND_PARAGRAPH
    block.layer = LAYER_FLOW
    block.confidence = 0.9
    block.reasons = [f"常规文本块（{len(block.lines)} 行，字号 {block.size:.1f}pt）"]


def summarize(doc: DocStructure) -> str:
    """生成人类可读的分类摘要。"""
    lines = []
    lines.append(f"文档：{doc.path}")
    lines.append(f"页数：{len(doc.pages)}   正文主导字号：{doc.body_size}pt")
    lines.append(f"嵌入字体：{', '.join(f'{k}(xref={v})' for k, v in doc.font_xrefs.items())}")
    lines.append("")

    kind_counter: Counter[str] = Counter()
    low_conf: list[tuple[int, Block]] = []

    for page in doc.pages:
        lines.append(f"── 第 {page.page_no + 1} 页 ({page.width:.0f}×{page.height:.0f}) ──")
        ordered = sorted(page.blocks, key=lambda b: (b.layer != LAYER_FIXED, b.bbox[1], b.bbox[0]))
        for b in ordered:
            kind_counter[b.kind] += 1
            flag = " ⚠" if b.confidence < CONFIDENCE_FLAG else ""
            tag = "固定层" if b.layer == LAYER_FIXED else "流式层"
            preview = b.text.replace("\n", "⏎")[:44]
            lines.append(
                f"  [{tag}][{b.kind:<9}]{flag} {b.confidence:.2f}  "
                f"y={b.bbox[1]:6.1f}  {b.size:5.1f}pt  {preview}"
            )
            if b.confidence < CONFIDENCE_FLAG:
                low_conf.append((page.page_no + 1, b))
        lines.append("")

    lines.append("── 统计 ──")
    for k, c in kind_counter.most_common():
        lines.append(f"  {k:<10} {c}")
    lines.append(f"  待确认（置信度 < {CONFIDENCE_FLAG}）：{len(low_conf)} 个块")
    for pno, b in low_conf:
        lines.append(f"    p{pno} [{b.kind}] {b.text[:40]!r}  ← {b.reasons[-1]}")
    return "\n".join(lines)
