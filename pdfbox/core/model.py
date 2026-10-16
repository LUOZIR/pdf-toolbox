"""文档结构模型 —— 五层结构中的 L0~L2 数据结构。

L0 字符层  Char   每个字符：unicode、字形 ID、基线原点、包围盒
L1 行层    Line   同一基线上的字符组成行
L2 块层    Block  行按规则聚合成块（段落/标题/表格/图片/水印…）

L3 流层（阅读顺序、框架归属）与 L4 图层（flow/fixed）在 classify.py 中赋值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- L0 字符层


@dataclass(slots=True)
class Char:
    c: str
    glyph_id: int
    origin: tuple[float, float]      # 基线起点（未旋转的坐标系下）
    bbox: tuple[float, float, float, float]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass(slots=True)
class Span:
    """同一字体、字号、颜色、方向的一段连续文字 —— 直接来自 get_texttrace()。"""

    chars: list[Char]
    font: str                        # 基础字体名，如 "SimSun Regular"（已去掉子集前缀）
    size: float
    color: tuple[float, float, float]
    direction: tuple[float, float]   # (cos, sin)，水平文本为 (1, 0)
    flags: int
    opacity: float
    bbox: tuple[float, float, float, float]
    seqno: int                       # 内容流中的绘制顺序 = z 序
    page_no: int

    @property
    def text(self) -> str:
        return "".join(c.c for c in self.chars)

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 16)

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2)

    @property
    def is_rotated(self) -> bool:
        """方向偏离水平/垂直即视为旋转文本（水印的判定依据）。"""
        return abs(self.direction[1]) > 1e-3 and abs(self.direction[0] - 1.0) > 1e-3

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.direction[1], self.direction[0]))


# ---------------------------------------------------------------- L1 行层


@dataclass(slots=True)
class Line:
    spans: list[Span]
    page_no: int
    direction: tuple[float, float]
    cross: float                     # 沿法线方向的坐标；水平文本即基线 y
    along0: float                    # 行首沿向坐标
    along1: float                    # 行尾沿向坐标
    bbox: tuple[float, float, float, float]

    @property
    def text(self) -> str:
        return "".join(s.text for s in sorted(self.spans, key=lambda s: self._key(s)))

    def _key(self, s: Span) -> float:
        cos, sin = self.direction
        return min((c.origin[0] * cos + c.origin[1] * sin) for c in s.chars)

    @property
    def size(self) -> float:
        return max(s.size for s in self.spans)

    @property
    def font(self) -> str:
        # 取字符数最多的 span 的字体作为主导字体
        return max(self.spans, key=lambda s: len(s.chars)).font

    @property
    def color(self) -> tuple[float, float, float]:
        return max(self.spans, key=lambda s: len(s.chars)).color

    @property
    def seqno(self) -> int:
        return min(s.seqno for s in self.spans)

    @property
    def is_rotated(self) -> bool:
        return abs(self.direction[1]) > 1e-3 and abs(self.direction[0] - 1.0) > 1e-3

    @property
    def width(self) -> float:
        return self.along1 - self.along0

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


# ---------------------------------------------------------------- L2 块层

# 块类型
KIND_PARAGRAPH = "paragraph"
KIND_HEADING = "heading"
KIND_LIST = "list"
KIND_TABLE = "table"
KIND_IMAGE = "image"
KIND_GRAPHICS = "graphics"
KIND_HEADER = "header"
KIND_FOOTER = "footer"
KIND_WATERMARK = "watermark"
KIND_UNKNOWN = "unknown"

# 图层
LAYER_FLOW = "flow"      # 参与重排
LAYER_FIXED = "fixed"    # 钉死原位，永不参与重排

# 这些类型必然是固定层 —— 它们的存在本身就是"不随正文流动"的。
# 注意 KIND_GRAPHICS 不在此列：图形该不该固定取决于位置
# （页眉分隔线是固定的，表格框线要跟着表格走），由 extract 阶段按条带判定。
FIXED_KINDS = {KIND_HEADER, KIND_FOOTER, KIND_WATERMARK}


@dataclass
class Block:
    lines: list[Line] = field(default_factory=list)
    page_no: int = 0
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    kind: str = KIND_UNKNOWN
    layer: str = LAYER_FLOW
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)
    z: int = 0
    # 非文本块（图片/表格）直接携带原始对象
    raw: dict | None = None

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def size(self) -> float:
        return max((l.size for l in self.lines), default=0.0)

    @property
    def font(self) -> str:
        return self.lines[0].font if self.lines else ""

    @property
    def color(self) -> tuple[float, float, float]:
        return self.lines[0].color if self.lines else (0.0, 0.0, 0.0)

    @property
    def is_rotated(self) -> bool:
        return bool(self.lines) and self.lines[0].is_rotated

    @property
    def line_gap(self) -> float:
        """块内相邻行的平均跨行间距。"""
        if len(self.lines) < 2:
            return 0.0
        cross = [l.cross for l in self.lines]
        gaps = [abs(cross[i + 1] - cross[i]) for i in range(len(cross) - 1)]
        return sum(gaps) / len(gaps)

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def bottom(self) -> float:
        return self.bbox[3]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "layer": self.layer,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "bbox": [round(v, 2) for v in self.bbox],
            "z": self.z,
            "lines": len(self.lines),
            "font": self.font,
            "size": round(self.size, 2),
            "text": self.text[:120],
        }


# ---------------------------------------------------------------- 页面/文档


@dataclass
class PageStructure:
    page_no: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    drawings: list[dict] = field(default_factory=list)

    def by_layer(self, layer: str) -> list[Block]:
        return [b for b in self.blocks if b.layer == layer]

    def by_kind(self, kind: str) -> list[Block]:
        return [b for b in self.blocks if b.kind == kind]


@dataclass
class DocStructure:
    path: str
    pages: list[PageStructure] = field(default_factory=list)
    body_size: float = 0.0           # 正文主导字号（文档级统计量）
    font_xrefs: dict[str, int] = field(default_factory=dict)  # 字体名 → xref

    def flow_blocks(self, page_no: int) -> list[Block]:
        return [b for b in self.pages[page_no].blocks if b.layer == LAYER_FLOW]
