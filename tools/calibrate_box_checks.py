"""标定逐字框错位检测器的阈值。

★ 为什么必须先标定：
检查器是"只用像素、不看真相"的判据。要用它去判真实文档，
就得先知道**框对的时候**它给出什么数、**框错的时候**它给出什么数。
真值样本同时具备这两者 —— 有精确答案，也能人为把框移错。
不标定就定阈值，等于拿一把没刻度的尺子去量。

做法：
  1. 在真值样本上算出每个字的「墨迹量比相对本行中位数的偏离」（框对时的分布）；
  2. 人为把框平移 ±1/±2/±3pt，再算同一个量（框错时的分布）；
  3. 取两簇之间的分界作为阈值，并报告它能抓住多少人为错位。

跑：python -m tools.calibrate_box_checks
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pdfbox.core.raster import (                              # noqa: E402
    _fit_ref_font, glyph_mass, per_char_boxes,
)
from pdfbox.core.session import Session                        # noqa: E402
from verify_char_boxes import TRUTH, SAMPLE                    # noqa: E402


def _band(s, pno, block, zoom=6.0):
    line = block.lines[0]
    y_top = min(l.bbox[1] for l in block.lines) - 2
    y_bot = max(l.bbox[3] for l in block.lines) + 2
    x_l = min(l.bbox[0] for l in block.lines) - 2
    x_r = max(l.bbox[2] for l in block.lines) + 2
    pix = s.doc[pno].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                                clip=pymupdf.Rect(x_l, y_top, x_r, y_bot),
                                alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3]
    gray = arr.mean(axis=2)
    bg = float(np.median(gray))
    ink = gray < (bg - 40)
    ly0 = max(0, int((line.bbox[1] - y_top) * zoom))
    ly1 = min(ink.shape[0], int((line.bbox[3] - y_top) * zoom))
    return (ink[ly0:ly1] if ly1 > ly0 else ink), x_l, zoom


def main() -> int:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    s = Session(str(SAMPLE))
    s.load()
    good: list[float] = []
    shifted: dict[float, list[float]] = {1.0: [], 2.0: [], 3.0: []}
    try:
        for tl in truth["lines"]:
            by = tl["baseline_y"]
            cand = None
            for pno in range(s.page_count):
                for b in s.editable_blocks(pno):
                    if b.lines and abs(b.bbox[1] - by) < 25:
                        cand = b
                        break
                if cand:
                    break
            if cand is None:
                continue
            band, x_l, zoom = _band(s, cand.page_no, cand)
            txt = "".join(c.c for sp in cand.lines[0].spans for c in sp.chars)
            fit = _fit_ref_font(band, x_l, zoom, txt)
            if fit is None:
                continue
            font, fpath, _b, size_pt = fit
            mass100 = glyph_mass(font, txt, fpath)
            k = (size_pt / 100.0) ** 2
            boxes = per_char_boxes(s.doc[cand.page_no], cand, zoom=zoom)

            def obs(a_px: int, b_px: int) -> float:
                a_px = max(0, min(a_px, band.shape[1]))
                b_px = max(a_px, min(b_px, band.shape[1]))
                return float(band[:, a_px:b_px].sum()) / (zoom * zoom)

            # 本行比值中位数 = 模糊系数（框正确时的参照）
            rr = []
            for c, r in boxes:
                if r[2] - r[0] < 3.0:
                    continue
                exp = mass100.get(c.c, 0.0) * k
                if exp <= 0.5:
                    continue
                o = obs(int(round((r[0] - x_l) * zoom)),
                        int(round((r[2] - x_l) * zoom)))
                if o > 0:
                    rr.append(o / exp)
            if len(rr) < 3:
                continue
            med = statistics.median(rr)

            for c, r in boxes:
                if r[2] - r[0] < 3.0:
                    continue
                exp = mass100.get(c.c, 0.0) * k
                if exp <= 0.5:
                    continue
                for dx in (0.0, 1.0, 2.0, 3.0):
                    for sgn in ((0,) if dx == 0.0 else (1, -1)):
                        o = obs(int(round((r[0] + sgn * dx - x_l) * zoom)),
                                int(round((r[2] + sgn * dx - x_l) * zoom)))
                        if o <= 0:
                            continue
                        dev = abs(o / exp - med) / med
                        (good if dx == 0.0 else shifted[dx]).append(dev)
    finally:
        s.close()

    def stat(name: str, v: list[float]) -> None:
        if not v:
            print(f"  {name:<20} 无数据")
            return
        v = sorted(v)
        q = lambda p: v[min(len(v) - 1, int(len(v) * p))]
        print(f"  {name:<20} n={len(v):<5} 中位 {statistics.median(v):.3f}  "
              f"90% {q(0.90):.3f}  95% {q(0.95):.3f}  "
              f"99% {q(0.99):.3f}  最大 {v[-1]:.3f}")

    print("════ 墨迹量比「相对本行中位数的偏离」分布 ════")
    stat("框正确", good)
    for dx in (1.0, 2.0, 3.0):
        stat(f"框平移 ±{dx:.0f}pt", shifted[dx])

    if not good:
        return 1
    g = sorted(good)
    p95 = g[min(len(g) - 1, int(len(g) * 0.95))]
    p99 = g[min(len(g) - 1, int(len(g) * 0.99))]
    thr = round(max(p99, p95 * 1.3), 2)
    print()
    print(f"框正确时 95% 分位 {p95:.3f}、99% 分位 {p99:.3f}")
    print(f"→ 建议 MASS_DEV_MAX = {thr}")
    print()
    for dx in (3.0, 2.0, 1.0):
        v = shifted[dx]
        if not v:
            continue
        caught = sum(1 for x in v if x > thr) / len(v)
        print(f"  人为平移 ±{dx:.0f}pt：{caught:.0%} 的字会被判为错位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
