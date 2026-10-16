"""把结构分类结果可视化回页面上，用于人工校验规则准不准。

不同颜色代表不同块类型，虚线框代表固定层（不参与重排），
黄色粗框代表置信度低于阈值、需要用户确认的块。
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from .model import (
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
    Block,
    DocStructure,
)

COLORS: dict[str, tuple[int, int, int]] = {
    KIND_HEADING:   (230, 57, 70),
    KIND_PARAGRAPH: (29, 53, 87),
    KIND_LIST:      (69, 123, 157),
    KIND_HEADER:    (244, 162, 97),
    KIND_FOOTER:    (244, 162, 97),
    KIND_WATERMARK: (155, 93, 229),
    KIND_IMAGE:     (42, 157, 143),
    KIND_TABLE:     (0, 180, 216),
    KIND_GRAPHICS:  (150, 150, 150),
    KIND_UNKNOWN:   (0, 0, 0),
}

ZOOM = 2.0
CONFIDENCE_FLAG = 0.75


def _dashed_rect(draw: ImageDraw.ImageDraw, box, color, width=2, dash=8, gap=5):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in (
        (x0, y0, x1, y0), (x1, y0, x1, y1),
        (x1, y1, x0, y1), (x0, y1, x0, y0),
    ):
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        steps = int(length // (dash + gap)) + 1
        for i in range(steps):
            t0 = i * (dash + gap) / length
            t1 = min((i * (dash + gap) + dash) / length, 1.0)
            draw.line(
                [ax + (bx - ax) * t0, ay + (by - ay) * t0,
                 ax + (bx - ax) * t1, ay + (by - ay) * t1],
                fill=color, width=width,
            )


def render_page(
    pdf_path: str,
    doc: DocStructure,
    page_no: int,
    out_path: Path,
    show_labels: bool = True,
) -> Path:
    pdf = pymupdf.open(pdf_path)
    page = pdf[page_no]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pdf.close()

    draw = ImageDraw.Draw(img)
    structure = doc.pages[page_no]

    for b in structure.blocks:
        color = COLORS.get(b.kind, COLORS[KIND_UNKNOWN])
        box = tuple(v * ZOOM for v in b.bbox)
        is_fixed = b.layer == LAYER_FIXED
        low_conf = b.confidence < CONFIDENCE_FLAG

        if is_fixed:
            _dashed_rect(draw, box, color, width=2)
        else:
            draw.rectangle(box, outline=color, width=2)

        if low_conf:
            draw.rectangle(
                (box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3),
                outline=(255, 200, 0), width=3,
            )

        if show_labels:
            label = f"{b.kind[:4]} {b.confidence:.2f}"
            tx, ty = box[0], max(0, box[1] - 13)
            tw = 8 * len(label)
            draw.rectangle((tx, ty, tx + tw, ty + 12), fill=(255, 255, 255))
            draw.text((tx + 1, ty + 1), label, fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_all(pdf_path: str, doc: DocStructure, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in doc.pages:
        out = out_dir / f"classify_p{p.page_no + 1}.png"
        paths.append(render_page(pdf_path, doc, p.page_no, out))
    return paths


def render_plain(pdf_path: str, page_no: int, out_path: Path) -> Path:
    """无标注的原图，用于和标注图对照。"""
    pdf = pymupdf.open(pdf_path)
    pix = pdf[page_no].get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), alpha=False)
    pdf.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out_path)
    return out_path
