"""核心引擎层：与 UI 无关，可被 CLI 或 GUI 复用。"""

from .classify import classify, summarize
from .editor import draw_block, draw_rotated_block, replace_block_text, delete_block_text
from .fonts import FontMatch, describe, resolve_for_block
from .model import (
    KIND_FOOTER, KIND_GRAPHICS, KIND_HEADING, KIND_HEADER, KIND_IMAGE,
    KIND_LIST, KIND_PARAGRAPH, KIND_TABLE, KIND_UNKNOWN, KIND_WATERMARK,
    LAYER_FIXED, LAYER_FLOW, Block, Char, DocStructure, Line, PageStructure, Span,
)
from .structure import extract_document
from .verify import check_fixed_layer, compare, fixed_layer_signature

__all__ = [
    "extract_document", "classify", "summarize",
    "resolve_for_block", "describe", "FontMatch",
    "replace_block_text", "delete_block_text", "draw_block", "draw_rotated_block",
    "compare", "check_fixed_layer", "fixed_layer_signature",
    "Block", "Char", "Line", "Span", "DocStructure", "PageStructure",
    "KIND_PARAGRAPH", "KIND_HEADING", "KIND_LIST", "KIND_TABLE", "KIND_IMAGE",
    "KIND_GRAPHICS", "KIND_HEADER", "KIND_FOOTER", "KIND_WATERMARK", "KIND_UNKNOWN",
    "LAYER_FLOW", "LAYER_FIXED",
]
