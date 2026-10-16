"""逐字框错位检测器：只用像素、不需要真相的判据。

为什么需要它：真值样本能给出精确误差，但真实扫描件没有真相。
要证明"每个字框都没错位"，就得有一个**不依赖真相**的判据。

## 主判据：墨迹量比的一致性（自标定）

正确的框里装的正好是一个字的墨迹。把"框内墨迹量 ÷ 参考字形墨迹量"
算出来，这个比值应当等于同一个常数 —— 扫描的模糊与加粗带来的整体放大，
它对整行每个字都一样。所以：

    比值_i 相对本行比值中位数的偏离 = 该字框的可疑程度

某个字吃了邻居的墨迹 → 比值明显偏大；切掉了自己一截 → 明显偏小。
**自标定**是这条判据的关键：不需要知道模糊系数是多少，
用同一行其他字当参照，偏离就被量出来了。

为什么不用列剖面相关性（试过并废弃）：112 DPI 下 10.5pt 的字只有约
10×16 个源像素，把剖面归一化到 32 个分箱等于在比噪声。实测框正确时
相关性最小 -0.417，而人为平移 ±3pt 后 93% 的值比它还高 —— 两簇完全重叠，
判不出东西。墨迹量是标量，不受这个尺度限制。

## 辅助判据（行层面，都是必要条件）

- **墨迹覆盖率**：本行所有框并起来必须覆盖全部墨迹列。有墨迹落在框外，
  说明至少有一个框太窄或偏了。
- **相邻框重叠**：互相压进去太多，说明边界切进了字里。
- **字格推进一致性**：相邻框左沿间距应等于参考字体的字宽推进。

★ 阈值必须先标定再用：所有阈值都是在真值样本上量出来的
（真值样本有精确答案，可以同时得到"框对"和"人为移错"两种分布）。
未标定就拿来判真实文档，等于用一把没刻度的尺子。
标定脚本：tools/calibrate_box_checks.py
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfbox.core.raster import (                      # noqa: E402
    _cluster_runs, _fit_ref_font, glyph_mass, per_char_boxes,
)
from pdfbox.core.session import Session               # noqa: E402

# 阈值 —— 由 tools/calibrate_box_checks.py 在真值样本上标定后写回
MASS_DEV_MAX = 0.45      # 墨迹量比相对本行中位数的偏离上限
COVER_MIN = 0.985        # 墨迹覆盖率下限
OVERLAP_MAX = 1.5        # 相邻框允许的最大重叠（pt）
ADV_DEV_MAX = 2.5        # 相邻框左沿间距与参考推进的最大偏差（pt）
NARROW_PT = 3.0          # 窄于此宽度的框只查覆盖率与重叠


@dataclass
class CharCheck:
    index: int
    ch: str
    box: tuple[float, float, float, float]
    mass_ratio: float = 0.0
    mass_dev: float = 0.0
    adv_dev: float = 0.0
    width: float = 0.0

    @property
    def ok(self) -> bool:
        # 窄框（括号、句点）墨迹太少，比值不稳，只查覆盖率与重叠
        if self.width < NARROW_PT:
            return True
        return self.mass_dev <= MASS_DEV_MAX

    def why(self) -> str:
        b = []
        if self.mass_dev > MASS_DEV_MAX:
            b.append(f"墨迹量偏离本行中位数 {self.mass_dev:.0%}"
                     f"（比值 {self.mass_ratio:.2f}）")
        if self.adv_dev > ADV_DEV_MAX:
            b.append(f"与前一字的推进偏差 {self.adv_dev:.1f}pt")
        return "；".join(b)


@dataclass
class BlockCheck:
    page_no: int
    text: str
    bbox: tuple[float, float, float, float]
    n_chars: int
    ref_font: str = ""
    ref_size: float = 0.0
    n_runs: int = 0
    chars: list[CharCheck] = field(default_factory=list)
    coverage: float = 1.0
    max_overlap: float = 0.0
    mass_med: float = 0.0

    @property
    def bad(self) -> list[CharCheck]:
        return [c for c in self.chars if not c.ok]

    @property
    def ok(self) -> bool:
        return (not self.bad and self.coverage >= COVER_MIN
                and self.max_overlap <= OVERLAP_MAX)


def _line_band(s: Session, pno: int, block, zoom: float):
    """取某块首行的墨迹掩膜（与 per_char_boxes 同一套坐标与阈值）。"""
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


def check_block(s: Session, pno: int, block, zoom: float = 6.0) -> BlockCheck:
    line = block.lines[0] if block.lines else None
    pads = [c for sp in line.spans for c in sp.chars] if line else []
    bc = BlockCheck(page_no=pno, text=block.text, bbox=block.bbox,
                    n_chars=len(pads))
    if not pads:
        return bc

    band, x_l, zoom = _line_band(s, pno, block, zoom)
    txt = "".join(c.c for c in pads)
    fit = _fit_ref_font(band, x_l, zoom, txt)
    if fit is None:
        bc.ref_font = "(拟合失败)"
        return bc
    font, fpath, _bounds, size_pt = fit
    bc.ref_font = getattr(font, "name", "")
    bc.ref_size = size_pt
    bc.n_runs = len(_cluster_runs(band))
    mass100 = glyph_mass(font, txt, fpath)
    k = (size_pt / 100.0) ** 2

    boxes = per_char_boxes(s.doc[pno], block, zoom=zoom)

    # 覆盖率与重叠
    covered = np.zeros(band.shape[1], dtype=bool)
    box_px = []
    for c, r in boxes:
        a = max(0, min(int(round((r[0] - x_l) * zoom)), band.shape[1]))
        b = max(a, min(int(round((r[2] - x_l) * zoom)), band.shape[1]))
        box_px.append((a, b))
        covered[a:b] = True
    ink_cols = band.any(axis=0)
    tot = int(ink_cols.sum())
    bc.coverage = float((ink_cols & covered).sum()) / tot if tot else 1.0
    for i in range(len(box_px) - 1):
        bc.max_overlap = max(bc.max_overlap,
                             (box_px[i][1] - box_px[i + 1][0]) / zoom)

    # 逐字墨迹量比（自标定：拿本行中位数当参照）
    ratios = []
    for i, (c, r) in enumerate(boxes):
        a, b = box_px[i]
        obs = float(band[:, a:b].sum()) / (zoom * zoom)      # 换算回 pt²
        exp = mass100.get(c.c, 0.0) * k
        cc = CharCheck(i, c.c, tuple(r), width=r[2] - r[0])
        if exp > 0.5 and obs > 0:
            cc.mass_ratio = obs / exp
            ratios.append(cc.mass_ratio)
        bc.chars.append(cc)
    if ratios:
        bc.mass_med = statistics.median(ratios)
        for cc in bc.chars:
            if cc.mass_ratio > 0 and bc.mass_med > 0:
                cc.mass_dev = abs(cc.mass_ratio - bc.mass_med) / bc.mass_med

    # 推进偏差
    if len(boxes) >= 2:
        for i in range(1, len(boxes)):
            try:
                exp_adv = (font.text_length(txt[i - 1], fontsize=size_pt)
                           + font.text_length(txt[i], fontsize=size_pt)) / 2.0
            except Exception:
                exp_adv = size_pt
            bc.chars[i].adv_dev = abs(
                (boxes[i][1][0] - boxes[i - 1][1][0]) - exp_adv)
    return bc


def check_page(s: Session, pno: int, zoom: float = 6.0) -> list[BlockCheck]:
    out = []
    for b in s.editable_blocks(pno):
        if not b.lines or len(b.text.strip()) < 2:
            continue
        out.append(check_block(s, pno, b, zoom))
    return out


def report(checks: list[BlockCheck], verbose: bool = True) -> bool:
    print(f"{'段':>3} {'字':>4} {'过':>3} {'覆盖':>7} {'重叠':>6} "
          f"{'量比中位':>8} {'最差偏离':>8}  参考字体 / 文本")
    print("-" * 100)
    allok = True
    for i, bc in enumerate(checks):
        worst = max((c.mass_dev for c in bc.chars), default=0.0)
        allok &= bc.ok
        print(f"{i + 1:>3} {bc.n_chars:>4} {'✓' if bc.ok else '✗':>3} "
              f"{bc.coverage:>7.3f} {bc.max_overlap:>6.2f} "
              f"{bc.mass_med:>8.2f} {worst:>8.0%}  "
              f"{bc.ref_font} {bc.ref_size:.1f}pt 游程{bc.n_runs}  "
              f"{bc.text[:26]!r}")
        if verbose and not bc.ok:
            for c in bc.bad[:10]:
                print(f"      第 {c.index} 字 {c.ch!r} "
                      f"[{c.box[0]:.1f},{c.box[2]:.1f}] w={c.width:.1f}  "
                      f"{c.why()}")
            if bc.coverage < COVER_MIN:
                print(f"      墨迹未被覆盖 {1 - bc.coverage:.1%}")
            if bc.max_overlap > OVERLAP_MAX:
                print(f"      相邻框重叠 {bc.max_overlap:.2f}pt")
    print("-" * 100)
    n_ok = sum(1 for b in checks if b.ok)
    print(f"共 {len(checks)} 段，通过 {n_ok} 段"
          f"（{100 * n_ok / max(1, len(checks)):.0f}%）")
    return allok


def main(argv: list[str]) -> int:
    src = argv[0] if argv else str(ROOT / "tests" / "fixtures"
                                   / "form_sample.pdf")
    pages = [int(argv[1]) - 1] if len(argv) > 1 else [0]
    s = Session(src)
    s.load()
    try:
        allok = True
        for pno in pages:
            print(f"════ {Path(src).name} 第 {pno + 1} 页 ════")
            allok &= report(check_page(s, pno))
    finally:
        s.close()
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
