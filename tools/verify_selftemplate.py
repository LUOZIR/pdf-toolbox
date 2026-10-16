"""验证"自模板"是否比系统字体更准 —— 在真值样本上量。

思路：从扫描件**自身**收集字形模板，而不是拿系统字体去凑。
模板来源是那些切分走了"墨迹簇一对一"路径的行 ——
游程数与字数相等且逐字宽度与参考字形墨迹宽吻合，它们的框就是墨迹游程的边界，
精确、不需要任何字宽推算。

★ 为什么值得试：系统字体去凑时有两个消不掉的误差源 ——
  参考字体与真实字体的**度量差异**（字宽比例、边距），
  以及低分辨率下**形状差异**（笔画粗细、衬线）。自模板同字体、同扫描特性，
  两个误差源一起消失。

★ 为什么必须先在这里验证：前几轮吃过"直接上生产、结果反而更差"的亏
  （模板匹配替代切分在真值样本上从 99% 退化到 87%）。
  所以先在**有精确答案**的数据上量：用 A 组行的模板去放 B 组的字，
  与真值逐字比对，并和"不精修"的基线对照。

跑：python -m tools.verify_selftemplate
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
    _cluster_runs, _effective_ppp, _fit_ref_font, _segment_by_clusters,
    per_char_boxes,
)
from pdfbox.core.scan_match import (                          # noqa: E402
    GlyphBank, extract_patch, ncc_peak,
)
from pdfbox.core.session import Session                        # noqa: E402
from verify_char_boxes import TRUTH, SAMPLE, glyph_box          # noqa: E402

Z = 6.0


def _line_ctx(s, pno, block):
    """取一行的墨迹掩膜、坐标原点、参考字体与是否可信。"""
    line = block.lines[0]
    y_top = min(l.bbox[1] for l in block.lines) - 2
    x_l = min(l.bbox[0] for l in block.lines) - 2
    x_r = max(l.bbox[2] for l in block.lines) + 2
    y_bot = max(l.bbox[3] for l in block.lines) + 2
    pix = s.doc[pno].get_pixmap(matrix=pymupdf.Matrix(Z, Z),
                                clip=pymupdf.Rect(x_l, y_top, x_r, y_bot),
                                alpha=False)
    g = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].mean(axis=2)
    bd = g < (float(np.median(g)) - 40)
    txt = "".join(c.c for sp in line.spans for c in sp.chars)
    fit = _fit_ref_font(bd, x_l, Z, txt)
    trusted = False
    if fit is not None:
        cb = _segment_by_clusters(_cluster_runs(bd), Z, txt,
                                  fit[0], fit[1], fit[3])
        trusted = cb is not None
    return dict(band=bd, x_l=x_l, y_top=y_top, txt=txt, fit=fit,
                trusted=trusted, rect=(x_l, y_top, x_r, y_bot))


def main() -> int:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    s = Session(str(SAMPLE))
    s.load()
    before: list[float] = []
    after: list[float] = []
    before_hit: list[float] = []
    after_hit: list[float] = []
    bank = GlyphBank()
    n_from_bank = 0
    n_total = 0
    try:
        # 第一遍：从可信行收集模板
        blocks = []
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
            ctx = _line_ctx(s, cand.page_no, cand)
            boxes = per_char_boxes(s.doc[cand.page_no], cand)
            blocks.append((tl, cand, ctx, boxes))
            if ctx["trusted"]:
                ppp = _effective_ppp(s.doc[cand.page_no], ctx["rect"]) or 2.0
                for i, (c, r) in enumerate(boxes):
                    if i >= len(tl["chars"]):
                        break
                    patch = extract_patch(s.doc[cand.page_no], r, ppp)
                    bank.add(c.c, ctx["fit"][3], patch)
        print(f"模板库：{len(bank)} 个不同字形（来自 {len(blocks)} 段中可信的"
              f"{sum(1 for _t, _b, x, _x in blocks if x['trusted'])} 段）")

        # 第二遍：对**不可信**行用自模板重放，与真值比对
        for tl, cand, ctx, boxes in blocks:
            if ctx["fit"] is None:
                continue
            size_pt = ctx["fit"][3]
            ppp = _effective_ppp(s.doc[cand.page_no], ctx["rect"]) or 2.0
            img = None
            if not ctx["trusted"]:
                pix = s.doc[cand.page_no].get_pixmap(
                    matrix=pymupdf.Matrix(ppp, ppp),
                    clip=pymupdf.Rect(*ctx["rect"]), alpha=False)
                a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)[:, :, :3].mean(
                        axis=2).astype(np.float32)
                img = 255.0 - a
                img -= img.mean()
            win = max(2, int(round(8.0 * ppp)))
            for i, (c, r) in enumerate(boxes):
                if i >= len(tl["chars"]):
                    break
                gb = glyph_box(tl["chars"][i], tl.get("tilt", 0.0))
                if gb is None:
                    continue
                e0 = sum(abs(v) for v in (
                    r[0] - gb[0], r[2] - gb[2],
                    r[1] - gb[1], r[3] - gb[3])) / 4
                before.append(e0)
                nr = r
                n_total += 1
                if img is not None:
                    t = bank.get(c.c, size_pt)
                    if t is not None:
                        tmpl, tcx, tcy, iw, ih = t
                        cx = ((r[0] + r[2]) / 2.0 - ctx["x_l"]) * ppp
                        cy = ((r[1] + r[3]) / 2.0 - ctx["y_top"]) * ppp
                        dx, dy, sc = ncc_peak(img, tmpl, cx - tcx,
                                              cy - tcy, win)
                        if sc >= 0.45:
                            n_from_bank += 1
                            before_hit.append(e0)
                            ncx = (cx + dx) / ppp + ctx["x_l"]
                            ncy = (cy + dy) / ppp + ctx["y_top"]
                            hw = iw / 2.0 / ppp + 1.0 / Z
                            hh = ih / 2.0 / ppp + 1.0 / Z
                            nr = (ncx - hw, ncy - hh, ncx + hw, ncy + hh)
                e1 = sum(abs(v) for v in (
                    nr[0] - gb[0], nr[2] - gb[2],
                    nr[1] - gb[1], nr[3] - gb[3])) / 4
                after.append(e1)
                if len(before_hit) > len(after_hit):
                    after_hit.append(e1)
    finally:
        s.close()

    def stat(name, v):
        if not v:
            print(f"  {name}: 无数据")
            return
        v = sorted(v)
        ok = sum(1 for x in v if x <= 1.0)
        print(f"  {name:<14} n={len(v):<4} 中位 {statistics.median(v):.2f}  "
              f"90% {v[min(len(v) - 1, int(len(v) * 0.9))]:.2f}  "
              f"最大 {v[-1]:.2f}  ≤1pt {ok}/{len(v)}"
              f"（{100 * ok / len(v):.0f}%）")

    print("════ 自模板精修前后的四边平均误差（真值样本）════")
    stat("精修前", before)
    stat("自模板后", after)
    print()
    print("════ 只看自模板命中的那些字（其余字不受影响）════")
    stat("命中前", before_hit)
    stat("命中后", after_hit)
    print(f"  自模板命中 {n_from_bank}/{n_total} 字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
