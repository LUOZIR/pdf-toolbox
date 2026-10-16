"""逐字框精度验证器：把算法输出与「标准答案」逐字比对。

这是唯一可信的度量。此前试过的两种都不成立：
  · "与 OCR 估算位置的偏差" —— 估算位置正是要修正的对象，量它没有意义
  · "中文框宽均匀性" —— 紧致框的宽度本来就随字形变化，指标被混淆
  · 视觉模型 —— 在密集小字上的判断前后不一致

真值样本由 tools/make_truth_sample.py 生成：
先用已知字体在已知坐标写字并记录每字的精确位置与字形包围盒，
再光栅化成 1.55 像素/pt 的位图 —— 既像扫描件，又知道真相。

误差 = 算法给出的框 与 真实字形包围盒 在四条边上的差值。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfbox.core.raster import per_char_boxes          # noqa: E402
from pdfbox.core.session import Session                # noqa: E402

TRUTH = ROOT / "tests" / "fixtures" / "truth_sample.truth.json"
SAMPLE = ROOT / "tests" / "fixtures" / "truth_sample.pdf"
FONT = "C:/Windows/Fonts/simsun.ttc"


def glyph_box(tc: dict, tilt_deg: float = 0.0):
    """真值：该字的真实墨迹框（页面坐标）。

    真值由生成阶段逐字单独渲染量得（见 tools/make_truth_sample.py），
    存成相对插入原点的偏移；这里再按字符位置换算到页面坐标。
    倾斜行绕字符起点旋转。
    """
    import math
    ia = tc.get("ink_abs")
    if ia:
        return tuple(ia)      # 直接是页面坐标的实测真值
    ink = tc.get("ink")
    if not ink:
        return None
    cx, base = tc["x0"], tc["baseline"]
    x0, y0, x1, y1 = ink
    if abs(tilt_deg) < 1e-6:
        return (cx + x0, base + y0, cx + x1, base + y1)
    rad = math.radians(tilt_deg)
    pts = []
    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        rx = px * math.cos(rad) - py * math.sin(rad)
        ry = px * math.sin(rad) + py * math.cos(rad)
        pts.append((cx + rx, base + ry))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def main() -> int:
    if not TRUTH.exists():
        print("缺少真值样本，先运行 tools/make_truth_sample.py")
        return 2
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))

    s = Session(str(SAMPLE))
    s.load(progress=lambda m: print("  ", m, file=sys.stderr))

    print("════ 逐字框误差（相对真实字形包围盒）════")
    print(f"{'行':<4}{'字数':>5}{'匹配':>5}  {'左':>7}{'右':>7}{'上':>7}{'下':>7}"
          f"  {'四边均值':>9}")
    print("-" * 62)

    all_err = []
    for tl in truth["lines"]:
        tchars = tl["chars"]
        size, by, tilt = tl["size"], tl["baseline_y"], tl["tilt"]
        # 找对应的块：y 范围重叠
        cand = None
        for pno in range(s.page_count):
            for b in s.editable_blocks(pno):
                if not b.lines:
                    continue
                if abs(b.bbox[1] - by) < 25:
                    cand = b
                    break
            if cand:
                break
        if cand is None:
            print(f"{tl['text'][:3]:<4}{len(tchars):>5}{'—':>5}  找不到对应块")
            continue

        boxes = per_char_boxes(s.doc[cand.page_no], cand)
        # 按顺序配对：真值字 vs 算法框
        n = min(len(boxes), len(tchars))
        dl = dr = dt = db = 0.0
        for i in range(n):
            algo = boxes[i][1]
            tc = tchars[i]
            gb = glyph_box(tc, tilt)
            if gb is None:
                continue
            el = abs(algo[0] - gb[0])
            er = abs(algo[2] - gb[2])
            et = abs(algo[1] - gb[1])
            eb = abs(algo[3] - gb[3])
            dl += el; dr += er; dt += et; db += eb
            all_err.append((el, er, et, eb, size))
        if n:
            print(f"{tl['text'][:3]:<4}{len(tchars):>5}{n:>5}  "
                  f"{dl/n:7.2f}{dr/n:7.2f}{dt/n:7.2f}{db/n:7.2f}"
                  f"  {(dl+dr+dt+db)/(4*n):8.2f}")

    print("-" * 62)
    if all_err:
        L = [e[0] for e in all_err]
        R = [e[1] for e in all_err]
        T = [e[2] for e in all_err]
        B = [e[3] for e in all_err]
        A = [sum(e[:4]) / 4 for e in all_err]

        def line(name, v):
            print(f"  {name:<12} 均值 {statistics.mean(v):6.2f}  "
                  f"中位 {statistics.median(v):6.2f}  90分位 "
                  f"{sorted(v)[int(len(v)*0.9)-1]:6.2f}  最大 {max(v):6.2f}")

        print(f"共 {len(all_err)} 个字")
        line("左边界", L)
        line("右边界", R)
        line("上边界", T)
        line("下边界", B)
        line("四边均值", A)
        tight = sum(1 for a in A if a <= 1.0)
        print(f"  四边均值 ≤1pt 的字: {tight}/{len(A)} ({100*tight/len(A):.0f}%)")
    s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
