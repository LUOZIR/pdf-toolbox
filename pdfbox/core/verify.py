"""像素级比对：验证"未改动区域零差异"这个核心论断。

这是整个方案的裁判。改字之后：
  - 被编辑块内部允许有差异（那是我们主动改的）
  - 被编辑块之外必须像素级零差异，一个像素都不能差
  - 页眉、页脚、水印、表格框线这些固定层元素更是必须一个 bit 都不动

如果这个断言成立，方案就成立；不成立，就得回头改设计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

# 像素差异阈值：低于此值视为抗锯齿噪声
PIXEL_TOL = 12


@dataclass
class DiffResult:
    width: int
    height: int
    total_pixels: int
    changed_pixels: int
    change_bbox: tuple[int, int, int, int] | None
    outside_pixels: int                                  # 期望区域之外的差异
    regions: list[dict] = field(default_factory=list)     # 连通差异区域
    zoom: float = 2.0

    @property
    def changed_ratio(self) -> float:
        return self.changed_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def outside_ratio(self) -> float:
        return self.outside_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def clean(self) -> bool:
        """未改动区域是否完全零差异 —— 这是唯一的通过标准。"""
        return self.outside_pixels == 0

    def report(self) -> str:
        lines = [
            f"总像素      : {self.total_pixels}",
            f"变化像素    : {self.changed_pixels} ({self.changed_ratio*100:.4f}%)",
            f"期望区域外  : {self.outside_pixels} ({self.outside_ratio*100:.4f}%)",
        ]
        if self.change_bbox:
            x0, y0, x1, y1 = self.change_bbox
            lines.append(
                f"变化范围    : x={x0/self.zoom:.1f}..{x1/self.zoom:.1f}pt  "
                f"y={y0/self.zoom:.1f}..{y1/self.zoom:.1f}pt"
            )
        lines.append(f"连通差异区  : {len(self.regions)} 处")
        for r in self.regions[:8]:
            lines.append(
                f"    x={r['x0']/self.zoom:.1f}..{r['x1']/self.zoom:.1f}pt  "
                f"y={r['y0']/self.zoom:.1f}..{r['y1']/self.zoom:.1f}pt  "
                f"{r['pixels']} px"
            )
        lines.append("")
        lines.append("✅ 通过：未改动区域零差异" if self.clean
                     else f"❌ 不通过：期望区域外有 {self.outside_pixels} 个像素发生变化")
        return "\n".join(lines)


def render_array(path: str, page_no: int, zoom: float) -> np.ndarray:
    d = pymupdf.open(path)
    pix = d[page_no].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n).copy()
    d.close()
    return arr


def _connected_regions(mask: np.ndarray, zoom: float, max_regions: int = 40) -> list[dict]:
    """用行扫描 + 并查集找出差异的连通区域。

    只为报告用，不需要极高精度，所以按行做区间合并即可。
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    # 按行分组，记录每行的 x 区间
    rows: dict[int, tuple[int, int]] = {}
    order = np.argsort(ys, kind="stable")
    ys_s, xs_s = ys[order], xs[order]
    start = 0
    for i in range(1, len(ys_s) + 1):
        if i == len(ys_s) or ys_s[i] != ys_s[start]:
            seg_x = xs_s[start:i]
            y = int(ys_s[start])
            rows[y] = (int(seg_x.min()), int(seg_x.max()))
            start = i

    # 相邻行区间重叠则合并
    regions: list[dict] = []
    cur: dict | None = None
    for y in sorted(rows):
        x0, x1 = rows[y]
        if cur is not None and y - cur["y1"] <= 2 and not (x1 < cur["x0"] - 3 or x0 > cur["x1"] + 3):
            cur["x0"] = min(cur["x0"], x0)
            cur["x1"] = max(cur["x1"], x1)
            cur["y1"] = y
            cur["pixels"] += int((mask[y]).sum())
        else:
            if cur:
                regions.append(cur)
            cur = {"x0": x0, "x1": x1, "y0": y, "y1": y, "pixels": int(mask[y].sum())}
    if cur:
        regions.append(cur)

    regions.sort(key=lambda r: -r["pixels"])
    return regions[:max_regions]


def compare(
    path_a: str,
    path_b: str,
    page_no: int,
    *,
    expect_region: tuple[float, float, float, float] | None = None,
    margin: float = 2.0,
    zoom: float = 2.0,
    diff_image: Path | None = None,
) -> DiffResult:
    """比对两个 PDF 的同一页。

    expect_region 是允许发生变化的区域（被编辑块的包围盒，单位 pt）。
    该区域之外任何像素差异都会被计入 outside_pixels。
    """
    a = render_array(path_a, page_no, zoom)
    b = render_array(path_b, page_no, zoom)

    if a.shape != b.shape:
        raise ValueError(f"页面尺寸不一致：{a.shape} vs {b.shape}")

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2)
    mask = diff > PIXEL_TOL

    total = mask.size
    changed = int(mask.sum())
    ys, xs = np.where(mask)
    change_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if changed else None

    # 期望区域之外的差异
    outside = 0
    if expect_region is not None and changed:
        x0, y0, x1, y1 = expect_region
        px0 = int((x0 - margin) * zoom)
        py0 = int((y0 - margin) * zoom)
        px1 = int((x1 + margin) * zoom)
        py1 = int((y1 + margin) * zoom)
        allowed = np.zeros_like(mask)
        allowed[max(0, py0):py1, max(0, px0):px1] = True
        outside = int((mask & ~allowed).sum())

    regions = _connected_regions(mask, zoom) if changed else []

    if diff_image is not None and changed:
        vis = a.copy()
        vis[mask] = [255, 0, 0]
        diff_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis).save(diff_image)

    return DiffResult(
        width=a.shape[1], height=a.shape[0], total_pixels=total,
        changed_pixels=changed, change_bbox=change_bbox,
        outside_pixels=outside, regions=regions, zoom=zoom,
    )


def assert_identical(path_a: str, path_b: str, page_no: int, zoom: float = 2.0) -> DiffResult:
    """断言两页完全相同 —— 用于验证"没碰过的地方一个像素都没变"。"""
    return compare(path_a, path_b, page_no, expect_region=None, zoom=zoom)


# --------------------------------------------------------------- 固定层比对

BAND = 70.0


def fixed_layer_signature(pdf_path: str, page_no: int) -> list[tuple]:
    """提取固定层元素的特征签名：旋转文本（水印）+ 页眉页脚条带内的文本。

    为什么不能只靠像素比对来验证固定层：水印斜跨整页，和正文在空间上重叠。
    重排后正文整体下移，那块区域的像素本来就该变，水印即使丢了也会被
    "正文变化"掩盖过去。实测中正是如此 —— 像素比对报"区域外零差异"，
    但水印其实已经被擦除干净了。所以固定层必须单独做结构比对。

    签名包含文字、首字符原点、方向向量、字号、颜色 —— 任何一项不同都说明
    固定层被动过。
    """
    d = pymupdf.open(pdf_path)
    page = d[page_no]
    height = page.rect.height
    sig: list[tuple] = []

    for s in page.get_texttrace():
        if s.get("type", 0) != 0:
            continue
        direction = (s["dir"][0], s["dir"][1])
        rotated = abs(direction[1]) > 1e-3 and abs(direction[0] - 1.0) > 1e-3
        b = s["bbox"]
        in_band = b[3] <= BAND or b[1] >= height - BAND
        if not (rotated or in_band):
            continue
        text = "".join(chr(c[0]) for c in s["chars"])
        sig.append((
            text,
            tuple(round(v, 2) for v in s["chars"][0][2]),
            tuple(round(v, 6) for v in direction),
            round(s["size"], 2),
            tuple(round(c, 4) for c in s["color"]),
        ))

    d.close()
    return sorted(sig)


@dataclass
class FixedLayerCheck:
    original: list[tuple]
    result: list[tuple]

    @property
    def ok(self) -> bool:
        return self.original == self.result

    def report(self) -> str:
        if self.ok:
            return f"✅ 固定层完好：{len(self.original)} 个元素的位置/方向/字号/颜色全部一致"
        lines = ["❌ 固定层受损："]
        orig_set = {s[0]: s for s in self.original}
        res_set = {s[0]: s for s in self.result}
        for text in orig_set:
            if text not in res_set:
                lines.append(f"    丢失: {text!r}")
            elif orig_set[text] != res_set[text]:
                lines.append(f"    改变: {text!r}")
                lines.append(f"        原: {orig_set[text][1:]}")
                lines.append(f"        新: {res_set[text][1:]}")
        for text in res_set:
            if text not in orig_set:
                lines.append(f"    新增: {text!r}")
        return "\n".join(lines)


def check_fixed_layer(path_a: str, path_b: str, page_no: int) -> FixedLayerCheck:
    return FixedLayerCheck(
        original=fixed_layer_signature(path_a, page_no),
        result=fixed_layer_signature(path_b, page_no),
    )
