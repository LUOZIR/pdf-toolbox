"""用模板匹配独立测出每个字在扫描件上的真实位置。

这是无真相情况下能拿到的最强证据：把参考字体渲染成**与扫描件同分辨率**的
字形模板，在当前位置附近的窗口里做归一化互相关，峰值就是该字真正的位置。
它不依赖任何切分假设 —— 切分结果是要被检验的对象，不能同时当参照。

## 为什么低分辨率下仍然可行

112 DPI 时一个 10.5pt 的汉字约 16×16 个源像素、几十个墨迹像素。
二维灰度互相关用的是全部这些像素，而"列剖面"只用一维投影 ——
所以模板匹配比剖面判据稳得多（剖面判据实测两簇完全重叠，已废弃）。

## 必须先标定

真值样本有精确答案，所以可以先量出模板匹配自身的定位误差，
再拿它去判真实文档。不标定就用，等于拿没刻度的尺子量。

跑：
    python -m tools.locate_chars --calibrate        # 在真值样本上标定
    python -m tools.locate_chars form_sample.pdf 1    # 检查某页每个字
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pdfbox.core.raster import (                              # noqa: E402
    _effective_ppp, _fit_ref_font, per_char_boxes,
)
from pdfbox.core.session import Session                        # noqa: E402

FIX = ROOT / "tests" / "fixtures"
FONT_CACHE: dict = {}

# ★ 模板与互相关必须**直接用核心的实现**，不能在这里留一份副本。
#   早先这里是复制粘贴的版本，核心修好之后副本没跟着改，
#   于是"验证器"拿旧代码去判新代码，两边口径不一致 ——
#   这种"两份实现悄悄漂移"的问题已经在本项目里出现过多次。
from pdfbox.core.scan_match import glyph_template, ncc_peak  # noqa: E402

# 阈值 —— 由 --calibrate 在真值样本上标定后写回。
# 实测模板匹配自身精度：横向 95% 0.91pt / 最大 1.54pt，
# 纵向 95% 0.98pt / 最大 2.00pt。取 1.5pt 作判据，
# 即"框中心与独立测出的字心相差超过 1.5pt 就算错位"。
DX_MAX = 1.5
DY_MAX = 1.5
SCORE_MIN = 0.36          # 互相关低于此值说明匹配本身不可信，不据此判错位


@dataclass
class Located:
    index: int
    ch: str
    box: tuple[float, float, float, float]
    dx: float                 # 模板匹配位置 - 现有框中心（pt）
    dy: float
    score: float              # 归一化互相关峰值


@dataclass
class PageLocate:
    text: str
    ppp: float
    ref_font: str = ""
    ref_size: float = 0.0
    chars: list[Located] = field(default_factory=list)

    @property
    def bad(self) -> list[Located]:
        return [c for c in self.chars if c.score >= SCORE_MIN
                and (abs(c.dx) > DX_MAX or abs(c.dy) > DY_MAX)]

    @property
    def weak(self) -> list[Located]:
        return [c for c in self.chars if c.score < SCORE_MIN]


def locate_block(s: Session, pno: int, block, zoom: float = 4.0,
                 win_pt: float = 6.0) -> PageLocate:
    """对块里每个字做模板匹配，返回"匹配位置 - 现有框中心"的偏差。"""
    line = block.lines[0] if block.lines else None
    pads = [c for sp in line.spans for c in sp.chars] if line else []
    out = PageLocate(text=block.text, ppp=0.0)
    if not pads:
        return out

    # 用与 per_char_boxes 相同的墨迹定义，但保留灰度（互相关用灰度更稳）
    y_top = min(l.bbox[1] for l in block.lines) - 2
    y_bot = max(l.bbox[3] for l in block.lines) + 2
    x_l = min(l.bbox[0] for l in block.lines) - 2
    x_r = max(l.bbox[2] for l in block.lines) + 2
    ppp = _effective_ppp(s.doc[pno], (x_l, y_top, x_r, y_bot)) or 2.0
    out.ppp = ppp
    pz = ppp
    pix = s.doc[pno].get_pixmap(matrix=pymupdf.Matrix(pz, pz),
                                clip=pymupdf.Rect(x_l, y_top, x_r, y_bot),
                                alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].mean(axis=2)
    img = 255.0 - arr.astype(np.float32)
    img -= img.mean()

    txt = "".join(c.c for c in pads)
    # 参考字体拟合要用与 per_char_boxes 相同的墨迹掩膜与倍率
    z6 = 6.0
    pix6 = s.doc[pno].get_pixmap(matrix=pymupdf.Matrix(z6, z6),
                                 clip=pymupdf.Rect(x_l, y_top, x_r, y_bot),
                                 alpha=False)
    g6 = np.frombuffer(pix6.samples, dtype=np.uint8).reshape(
        pix6.height, pix6.width, pix6.n)[:, :, :3].mean(axis=2)
    band6 = g6 < (float(np.median(g6)) - 40)
    ly0 = max(0, int((line.bbox[1] - y_top) * z6))
    ly1 = min(band6.shape[0], int((line.bbox[3] - y_top) * z6))
    fit = _fit_ref_font(band6[ly0:ly1] if ly1 > ly0 else band6,
                        x_l, z6, txt)
    if fit is None:
        return out
    font, fpath, _b, size_pt = fit
    out.ref_font = getattr(font, "name", "")
    out.ref_size = size_pt

    boxes = per_char_boxes(s.doc[pno], block)
    win = max(2, int(round(win_pt * pz)))
    for i, (c, r) in enumerate(boxes):
        if c.c.isspace():
            continue
        tmpl, tcx0, tcy0, _iw, _ih = glyph_template(fpath, c.c, size_pt, pz)
        if tmpl is None:
            continue
        # 现有框中心 → 图像像素坐标
        cx_pt = (r[0] + r[2]) / 2.0
        cy_pt = (r[1] + r[3]) / 2.0
        cx = (cx_pt - x_l) * pz
        cy = (cy_pt - y_top) * pz
        # 把模板摆到"框中心"上：模板墨迹质心对齐框中心
        ox = cx - tcx0
        oy = cy - tcy0
        dx, dy, sc = ncc_peak(img, tmpl, ox, oy, win)
        out.chars.append(Located(i, c.c, tuple(r), dx / pz, dy / pz, sc))
    return out


def calibrate() -> int:
    """在真值样本上量模板匹配自身的定位误差。"""
    from verify_char_boxes import TRUTH, SAMPLE, glyph_box
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    s = Session(str(SAMPLE))
    s.load()
    errs, scores = [], []
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
            res = locate_block(s, cand.page_no, cand, win_pt=4.0)
            for lc in res.chars:
                gb = glyph_box(tl["chars"][lc.index],
                               tl.get("tilt", 0.0)) if \
                    lc.index < len(tl["chars"]) else None
                if gb is None:
                    continue
                # 模板匹配给出的字心 vs 真值字心
                tx = (gb[0] + gb[2]) / 2.0
                ty = (gb[1] + gb[3]) / 2.0
                bx = (lc.box[0] + lc.box[2]) / 2.0
                byy = (lc.box[1] + lc.box[3]) / 2.0
                # lc.dx 是"匹配位置 - 框中心"，所以匹配字心 = 框中心 + dx
                errs.append((abs(bx + lc.dx - tx), abs(byy + lc.dy - ty)))
                scores.append(lc.score)
    finally:
        s.close()

    if not errs:
        print("没有量到任何字")
        return 1
    ex = sorted(e[0] for e in errs)
    ey = sorted(e[1] for e in errs)
    q = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]
    print("════ 模板匹配定位精度（相对真值，n=%d）════" % len(errs))
    print(f"  横向误差  中位 {statistics.median(ex):.2f}pt  "
          f"90% {q(ex, 0.9):.2f}pt  95% {q(ex, 0.95):.2f}pt  "
          f"最大 {ex[-1]:.2f}pt")
    print(f"  纵向误差  中位 {statistics.median(ey):.2f}pt  "
          f"90% {q(ey, 0.9):.2f}pt  95% {q(ey, 0.95):.2f}pt  "
          f"最大 {ey[-1]:.2f}pt")
    sc = sorted(scores)
    print(f"  互相关分数 中位 {statistics.median(sc):.3f}  "
          f"5% {q(sc, 0.05):.3f}  最小 {sc[0]:.3f}")
    print()
    print(f"→ 建议 DX_MAX = {round(max(1.0, q(ex, 0.95) * 1.5), 2)}pt，"
          f"DY_MAX = {round(max(1.0, q(ey, 0.95) * 1.5), 2)}pt")
    print(f"→ 建议 SCORE_MIN = {round(max(0.2, q(sc, 0.05) * 0.8), 2)}")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--calibrate":
        return calibrate()
    src = argv[0] if argv else str(FIX / "form_sample.pdf")
    pages = [int(argv[1]) - 1] if len(argv) > 1 else [0]
    s = Session(src)
    s.load()
    try:
        for pno in pages:
            print(f"════ {Path(src).name} 第 {pno + 1} 页 ════")
            nbad = ntot = 0
            for b in s.editable_blocks(pno):
                if not b.lines or len(b.text.strip()) < 2:
                    continue
                res = locate_block(s, pno, b)
                bad = res.bad
                ntot += len(res.chars)
                nbad += len(bad)
                flag = "✓" if not bad else "✗"
                worst = max((max(abs(c.dx), abs(c.dy))
                             for c in res.chars), default=0.0)
                print(f"  {flag} {res.text[:30]!r:<34} "
                      f"ppp={res.ppp:.2f} {res.ref_font} "
                      f"{res.ref_size:.1f}pt  最差偏移 {worst:.2f}pt")
                for c in bad[:8]:
                    print(f"      第 {c.index} 字 {c.ch!r} "
                          f"dx={c.dx:+.2f} dy={c.dy:+.2f} "
                          f"相关={c.score:.3f}")
            print(f"  共 {ntot} 字，偏移超标 {nbad} 字")
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
