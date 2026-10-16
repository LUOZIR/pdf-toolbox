"""转换质量验证器：量"转换到底损失了什么"，而不是量像素差。

★ 为什么不用整页 PSNR 衡量文字转换：
绝对定位的 PDF 与流式排版的 Word 之间，文字天然不可能逐像素对齐 ——
哪怕版式还原得完全正确，每个字的像素也基本对不上，PSNR 会一直很低
（实测一份 3 页文档文字重合 98.6%、页数一致，PSNR 仍只有 16dB）。
拿它当指标会把"对了"判成"错了"。

真正该量的是这三件事，它们各自独立、都能追到具体原因：
  1. **文字内容**    —— 去空白后的字符重合度（漏字/多字/乱码会立刻掉下来）
  2. **页数**        —— 溢出与丢失的直接信号
  3. **行位漂移**    —— 逐页把两边的文本行按 y 排序配对，
                       量每行顶端、左端、宽度的偏差。
                       它才是"版面跑没跑"的度量：
                       行顶端偏 3pt 是正常折行，偏 60pt 是版面崩了。
"""

from __future__ import annotations

import difflib
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class LineDrift:
    """一行文字在两份 PDF 里的位置偏差。"""

    page: int
    text: str
    dx: float
    dy: float
    dw: float


@dataclass
class CompareReport:
    src: str
    dst: str
    src_pages: int = 0
    dst_pages: int = 0
    src_chars: int = 0
    dst_chars: int = 0
    text_overlap: float = 0.0
    missing_chars: int = 0
    extra_chars: int = 0
    drifts: list[LineDrift] = field(default_factory=list)
    src_no_text: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def dy_median(self) -> float:
        return statistics.median([d.dy for d in self.drifts]) if self.drifts else 0.0

    @property
    def dx_median(self) -> float:
        return statistics.median([d.dx for d in self.drifts]) if self.drifts else 0.0

    def verdict(self) -> str:
        """按实测阈值给结论 —— 不是"好不好"的观感，是可复现的判据。"""
        if self.src_no_text:
            return ("源文件没有文字层（扫描件）：只能比页数 —— "
                    + ("页数一致" if self.src_pages == self.dst_pages
                       else f"页数 {self.src_pages}→{self.dst_pages}"))
        if self.text_overlap >= 0.98 and self.src_pages == self.dst_pages \
                and self.dy_median <= 6.0:
            return "通过：文字重合 ≥98%、页数一致、行位中位漂移 ≤6pt"
        bad = []
        if self.text_overlap < 0.98:
            bad.append(f"文字重合 {self.text_overlap * 100:.1f}%（<98%）")
        if self.src_pages != self.dst_pages:
            bad.append(f"页数 {self.src_pages}→{self.dst_pages}")
        if self.drifts and self.dy_median > 6.0:
            bad.append(f"行位中位漂移 {self.dy_median:.1f}pt（>6pt）")
        return "未达标：" + "；".join(bad) if bad else "通过"

    def text(self) -> str:
        L = [f"转换验证：{Path(self.src).name} → {Path(self.dst).name}",
             f"  页数 {self.src_pages} → {self.dst_pages}"]
        if self.src_no_text:
            L.append("  源文件无文字层（图片型），文字比对不适用")
        else:
            L.append(f"  文字重合 {self.text_overlap * 100:.2f}%"
                     f"（{self.src_chars} 字 → {self.dst_chars} 字，"
                     f"缺 {self.missing_chars}、多 {self.extra_chars}）")
        if self.drifts:
            dy = [d.dy for d in self.drifts]
            dx = [d.dx for d in self.drifts]
            dw = [d.dw for d in self.drifts]
            L.append(f"  配对上 {len(self.drifts)} 行")
            for name, v in (("行顶漂移", dy), ("行左漂移", dx), ("行宽变化", dw)):
                L.append(f"    {name}：中位 {statistics.median(v):6.2f}pt  "
                         f"90分位 {sorted(v)[int(len(v) * 0.9) - 1]:6.2f}pt  "
                         f"最大 {max(v):6.2f}pt")
        L.append(f"  结论：{self.verdict()}")
        for n in self.notes:
            L.append(f"  · {n}")
        return "\n".join(L)


def _norm(s: str) -> str:
    return "".join(c for c in s if not c.isspace())


def _lines(path: str | Path) -> list[list[tuple[float, float, float, str]]]:
    """逐页取文本行 (top, left, width, text)，按 y 再 x 排序。"""
    out = []
    doc = pymupdf.open(str(path))
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            # rawdict 才能拿到行级切分；get_text("dict") 的 line 就是行
            d = page.get_text("dict")
            rows = []
            for blk in d.get("blocks", []):
                if blk.get("type") != 0:
                    continue
                for line in blk.get("lines", []):
                    txt = "".join(sp.get("text", "")
                                  for sp in line.get("spans", []))
                    if not txt.strip():
                        continue
                    x0, y0, x1, y1 = line["bbox"]
                    rows.append((round(y0, 2), round(x0, 2),
                                 round(x1 - x0, 2), txt))
            rows.sort(key=lambda r: (r[0], r[1]))
            out.append(rows)
    finally:
        doc.close()
    return out


def compare_text_layout(src: str | Path, dst: str | Path,
                        dy_tol: float = 12.0) -> CompareReport:
    """比较两份 PDF 的文字内容与行级版面。

    `dy_tol` 是配对的容差：同一页里顶端相差 ≤ tol 的行才认为是"同一行"。
    配不上的行不计入漂移统计，但会在缺字/多字里体现。
    """
    rep = CompareReport(src=str(src), dst=str(dst))
    a = _lines(src)
    b = _lines(dst)
    rep.src_pages, rep.dst_pages = len(a), len(b)

    ta = _norm("".join(r[3] for pg in a for r in pg))
    tb = _norm("".join(r[3] for pg in b for r in pg))
    rep.src_chars, rep.dst_chars = len(ta), len(tb)
    rep.src_no_text = len(ta) < 20
    if not rep.src_no_text:
        sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
        rep.text_overlap = sm.ratio()
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("delete", "replace"):
                rep.missing_chars += i2 - i1
            if tag in ("insert", "replace"):
                rep.extra_chars += j2 - j1

    # 行位漂移：同页内按 (行顶 + 0.25×行左) 的代价贪心配对。
    # ★ 只按行顶配对不行：同一 y 上左右两栏、表格相邻单元格会互相错配，
    #   把"配错对象"算成"版面崩了"（实测行左漂移 90 分位虚报成 173pt）。
    for pno in range(min(len(a), len(b))):
        used = [False] * len(b[pno])
        best_for_line: list[tuple[float, object, int] | None] = []
        for top, left, width, txt in a[pno]:
            cand = []
            for j, (t2, l2, w2, txt2) in enumerate(b[pno]):
                if used[j]:
                    continue
                dy = abs(t2 - top)
                if dy > dy_tol:
                    continue
                cand.append((dy + 0.25 * abs(l2 - left), j))
            cand.sort()
            best_for_line.append((cand[0][0], (top, left, width, txt),
                                  cand[0][1]) if cand else None)
        # 按代价从小到大落定，避免"先配先得"抢走别人最近的行
        for item in sorted([c for c in best_for_line if c],
                           key=lambda c: c[0]):
            _, src_line, bj = item
            if used[bj]:
                continue
            used[bj] = True
            top, left, width, txt = src_line
            t2, l2, w2, _ = b[pno][bj]
            rep.drifts.append(LineDrift(pno + 1, txt[:30], l2 - left,
                                        t2 - top, w2 - width))
    return rep


def report(path: str | Path, out: str | Path | None = None) -> CompareReport:
    """便捷入口：比对并（可选）把文字报告写到文件。"""
    src = Path(path)
    dst = Path(out) if out else None
    if dst is None:
        raise ValueError("需要给出比对目标")
    r = compare_text_layout(src, dst)
    return r
