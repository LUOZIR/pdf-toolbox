"""格式转换：PDF <-> Word / Excel / PPT / 图片。

★ 转换方向决定用哪条路，因为两个方向的"最佳做法"完全不同：

**PDF → Office（往外走）**：用本工程自己的结构模型重建，不用黑盒转换器。
  原因是我们已经有一份经过实测校准的版面理解 —— 块、行、字号、字体、
  对齐、表格。拿它生成 docx/xlsx，每一处属性都能追到源 PDF 的哪个元素，
  出了问题能定位、能改；而黑盒转换器一旦某段版式错了，无从下手。
  图片型 PDF 走 OCR 文本，保证扫描件也能转出字来。

**Office → PDF（往回收）**：交给本机装的 Office 自己渲染（COM 自动化）。
  这一条没什么可争的：Office 的排版引擎就是这些格式的权威实现，
  任何第三方重现都不可能比它更像"原样"。而且它天然支持字体替换、
  自动编号、SmartArt 这些自己实现等于重写一个 Office 的东西。
  没有 Office 时装不了，那时明确报错，不假装能转。

保真度的诚实说明写在每个函数的返回里 —— 用户要的是"尽可能降低损失"，
那就得知道损失发生在哪里。
"""

from __future__ import annotations

import io
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
              ".gif", ".jfif"}
OFFICE_EXTS = {".doc", ".docx", ".rtf", ".xls", ".xlsx", ".csv", ".ppt",
               ".pptx"}

# PDF 里字体名 → Windows 常见字体名。PDF 存的是 PostScript 名，
# 直接塞进 docx 的话 Word 找不到字体就退回默认，字号对了字形也不对。
_FONT_ALIAS = {
    "simsun": "宋体", "songti": "宋体", "nsimsun": "新宋体",
    "simhei": "黑体", "heiti": "黑体",
    "simkai": "楷体", "kaiti": "楷体",
    "simfang": "仿宋", "fangsong": "仿宋",
    "microsoftyahei": "微软雅黑", "microsoftyaheiui": "微软雅黑",
    "msyh": "微软雅黑",
    "arial": "Arial", "arialmt": "Arial", "helvetica": "Arial",
    "timesnewroman": "Times New Roman", "times": "Times New Roman",
    "calibri": "Calibri", "cambria": "Cambria", "couriernew": "Courier New",
    "courier": "Courier New", "verdana": "Verdana", "tahoma": "Tahoma",
    "georgia": "Georgia", "consolas": "Consolas",
    "droidsansfallback": "宋体", "notosanssc": "微软雅黑",
}


@dataclass
class ConvertReport:
    src: str
    out: list[str] = field(default_factory=list)
    kind: str = ""
    detail: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def text(self) -> str:
        L = [f"{self.kind}：{Path(self.src).name} → "
             + "、".join(Path(o).name for o in self.out)]
        for d in self.detail:
            L.append(f"  {d}")
        for w in self.warnings:
            L.append(f"  ! {w}")
        return "\n".join(L)


def _docx_font(name: str) -> str:
    """把 PDF 的字体名映射成 Word 能认的字体名。"""
    if not name:
        return "宋体"
    key = re.sub(r"[^a-z]", "", name.split("+")[-1].lower())
    for pat, cn in _FONT_ALIAS.items():
        if pat in key:
            return cn
    # 中文名直接用（PDF 里偶尔直接存了中文字体名）
    if any("\u4e00" <= c <= "\u9fff" for c in name):
        return name
    return name


def _span_bold(flags: int) -> bool:
    return bool(flags & 16)


def _span_italic(flags: int) -> bool:
    return bool(flags & 2)


# ============================================================ PDF → 图片


def pdf_to_images(src: str | Path, out_dir: str | Path, *,
                  fmt: str = "png", dpi: float = 150.0,
                  pages: list[int] | None = None,
                  jpeg_quality: int = 90,
                  progress=None) -> ConvertReport:
    """把每页渲染成图片文件。"""
    src_p = Path(src)
    outd = Path(out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        fmt = "jpeg"

    rep = ConvertReport(src=str(src_p), kind="导出图片")
    doc = pymupdf.open(str(src_p))
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"{src_p.name} 是加密的，需要密码")
        idx = ([_ck(i, doc.page_count) for i in pages] if pages
               else list(range(doc.page_count)))
        stem = src_p.stem
        zoom = dpi / 72.0
        for i in idx:
            page = doc[i]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                                 alpha=False)
            out = outd / f"{stem}_p{i + 1:03d}.{'jpg' if fmt == 'jpeg' else fmt}"
            if fmt == "jpeg":
                pix.pil_save(str(out), format="JPEG", quality=jpeg_quality,
                             optimize=True)
            else:
                pix.save(str(out))
            rep.out.append(str(out))
            if progress:
                progress(f"第 {i + 1} 页 → {out.name} "
                         f"({pix.width}×{pix.height})")
            pix = None
        rep.detail.append(f"{len(idx)} 页，{dpi:.0f} DPI，格式 {fmt}")
    finally:
        doc.close()
    return rep


# ============================================================ PDF → Word


def _align_of(block, page_rect) -> int:
    """按块相对页宽的位置判对齐：0 左 1 居中 2 右 3 两端对齐。

    判据用"两侧留白之差"而不是"绝对位置"：不同文档页边距差别很大，
    而在正文区里居中的块，其左右留白本来就应该接近相等。
    """
    x0, _, x1, _ = block.bbox
    left = x0
    right = page_rect.width - x1
    if left > 60.0 and right > 60.0 and abs(left - right) <= 0.35 * (
            block.bbox[2] - block.bbox[0]):
        return 1
    if left > 60.0 and right < 0.5 * left and right < 60.0:
        return 2
    return 0


def _prepare_document(w, page_rect=None, margin: float = 72.0,
                      scale: float = 1.0):
    """把 docx 的默认样式与页面尺寸对齐到源 PDF。

    ★ 这两处不设，转出来的页数会凭空多出好几页（实测 3 页 → 5 页、
      5 页的扫描件 → 17 页）：

    1. **python-docx 默认模板的 Normal 样式自带段后间距与 1.08 倍行距**
       （Calibri 11 + spacing after 160 twips）。每个段落都吃一遍，
       几十段累积下来就溢出好几页。必须显式归零。
    2. **默认页面是 Letter（612×792pt）**，而源 PDF 多是 A4（595×842）。
       宽高都差几个百分点，折行位置和每页行数全变。
    """
    from docx.shared import Pt

    st = w.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10.5 * scale)
    pf = st.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    # 东亚字体要单独设，Word 只认西文名时中文会退回默认字体
    try:
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(f"{ns}rFonts")
        if rf is None:
            from lxml import etree
            rf = etree.SubElement(rpr, f"{ns}rFonts")
        rf.set(f"{ns}eastAsia", "宋体")
        rf.set(f"{ns}ascii", "宋体")
        rf.set(f"{ns}hAnsi", "宋体")
    except Exception:
        pass

    if page_rect is not None:
        for sec in w.sections:
            sec.page_width = Pt(page_rect.width)
            sec.page_height = Pt(page_rect.height)
            sec.left_margin = Pt(margin)
            sec.right_margin = Pt(margin)
            sec.top_margin = Pt(margin)
            sec.bottom_margin = Pt(margin)


def pdf_to_word(src: str | Path, out: str | Path, *,
                pages: list[int] | None = None,
                include_images: bool = True,
                skip_page_scans: bool = True,
                fit_pages: bool = True,
                progress=None) -> ConvertReport:
    """PDF → Word（.docx）。

    按阅读顺序重建"流式段落"，不去模仿绝对定位的文字框 ——
    绝对定位还原度高但转出来的 Word 没法改，而"转 Word"这件事的意义
    就在于要改。为了把版式偏差压到最小，这里逐段带上：
      · 字体 / 字号 / 粗体 / 斜体（尽量贴到原值）
      · 段落对齐、缩进、行距（按原文的**绝对**行距设固定值）
      · 同一视觉行上的并排块合并成一段（否则表单会被竖起来排）
    图片按纵向位置插入，整页扫描图跳过（其文字已由 OCR 转成段落）。

    ★ fit_pages：本机装了 Office 时，会自动"构建 → 用 Word 实际渲染量页数
      → 按比例微调字号重建"，把页数对齐到原文。为什么必须实测：
      流式排版的段高受字体上下缘、行距取整、段后间距共同影响，
      任何估算都会累积偏差（实测估算说"装得下 698pt"，Word 却多出一页）。
      与其把模型调准到刚好，不如让它自己收敛。

    返回里会说明损失点：分栏会被拉成一栏、绝对定位的图会浮在段落间。
    """
    src_p = Path(src)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(src_p))
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"{src_p.name} 是加密的，需要密码")
        page_count = doc.page_count
        idx = ([_ck(i, page_count) for i in pages] if pages
               else list(range(page_count)))
    finally:
        doc.close()
    n_src = len(idx)

    # 先按原字号转一次
    cur_scale = 1.0
    rep = _pdf_to_word_once(src_p, out_p, idx, cur_scale, include_images,
                            skip_page_scans, progress)

    if fit_pages and n_src and _office_ready():
        # ★ 收缩阶梯到 0.88 为止：这是"页数对齐"与"字号保真"之间的取舍点。
        #   流式排版天然比绝对定位占地方（同一行里并排的块要合并、
        #   放不下的行要折行），不缩一点通常就多出一页。
        #   但缩得更多就开始改变观感了，那就该老实报告做不到，
        #   而不是交一份字号明显变小的文档说"页数对上了"。
        #   需要严格保持字号时传 fit_pages=False。
        tried: list[tuple[int, float]] = []
        for step, scale in enumerate((1.0, 0.95, 0.90, 0.88)):
            if step:
                rep = _pdf_to_word_once(src_p, out_p, idx, scale,
                                        include_images, skip_page_scans,
                                        progress)
                cur_scale = scale
            got = _count_pages_via_office(out_p)
            if got is None:
                rep.warnings.append("Office 未能渲染出页数，跳过页数自校正")
                break
            tried.append((got, scale))
            if got <= n_src:
                break
            # 连续两次都没改善才停（图片撑起来的页数缩字号是没用的），
            # 只停一次会把"差一点点就能压回去"的文档误判成压不动
            if len(tried) >= 3 and tried[-1][0] >= tried[-2][0]:
                break
        if tried:
            best = min(tried, key=lambda t: (t[0], -t[1]))
            if abs(best[1] - cur_scale) > 1e-6:
                # 最后一次构建不是最优的那次 —— 重建，保证磁盘上的文件
                # 与报告里说的字号一致
                rep = _pdf_to_word_once(src_p, out_p, idx, best[1],
                                        include_images, skip_page_scans,
                                        progress)
            rep.detail.append(
                f"页数自校正：原文 {n_src} 页，实测 "
                + "、".join(f"字号×{s:.2f}→{g}页" for g, s in tried)
                + f"；采用 ×{best[1]:.2f}")
            if best[0] != n_src:
                rep.warnings.append(
                    f"页数仍为 {best[0]}（原文 {n_src} 页）。"
                    "流式排版无法复现绝对定位的版面 —— "
                    "图文混排的页面尤其明显（图片在原文里可以叠放，"
                    "在 Word 里只能顺序堆叠）。"
                    "需要逐页一致请改用《编辑 PDF》直接改字")
    return rep


def _office_ready() -> bool:
    """本机有没有可用的 Office（决定能不能做实测自校正）。"""
    if os.name != "nt":
        return False
    try:
        import win32com.client  # noqa: F401
    except Exception:
        return False
    return any(Path(p).exists() for p in (
        r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
    ))


def _count_pages_via_office(docx_path: Path) -> int | None:
    """用 Word 把 docx 渲染成 PDF 并数页数。"""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "probe.pdf"
            office_to_pdf(docx_path, tmp)
            d = pymupdf.open(str(tmp))
            try:
                return d.page_count
            finally:
                d.close()
    except Exception:
        return None


def _pdf_to_word_once(src_p: Path, out_p: Path, idx: list[int],
                      scale: float, include_images: bool,
                      skip_page_scans: bool, progress) -> ConvertReport:
    """按给定字号缩放构建一次 docx。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    rep = ConvertReport(src=str(src_p), kind="转换为 Word")
    doc = pymupdf.open(str(src_p))
    w = Document()
    try:
        _prepare_document(w, doc[idx[0]].rect, scale=scale)
        from .session import Session
        sess = Session(str(src_p))
        sess.load(progress=progress)
        try:
            n_para = n_tab = n_img = n_skip = 0
            for k, pno in enumerate(idx):
                page = doc[pno]
                blocks = [b for b in sess.page_blocks(pno)
                          if b.kind not in ("watermark",)]
                blocks.sort(key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
                # 把块整理成"版面项"序列：一行文字 / 一张表 / 一张图，
                # 按纵向位置排序 —— 流式输出必须按真实阅读顺序，
                # 否则表格和图会跑到不相干的位置。
                items_all = _page_items(blocks)
                # 每项的绝对高度 = 本项顶到下一项顶的实测距离。
                # ★ 为什么不按"行距 × 行数"估：
                #   那样每项都会多出一点（字号上下缘、行距取整、段后间距的叠加），
                #   几十项累积就把页面撑破 —— 实测 5 页扫描件膨胀成 7 页。
                para_h = _paragraph_heights(items_all, page.rect)
                if k > 0:
                    w.add_page_break()
                for ii, it in enumerate(items_all):
                    kind = it[0]
                    if kind == "table":
                        n_tab += _add_table(w, page, it[1])
                        continue
                    if kind == "image":
                        if not include_images:
                            continue
                        if skip_page_scans and _is_page_scan(it[1], page.rect):
                            n_skip += 1
                            continue
                        if _add_image(w, page, it[1]):
                            n_img += 1
                        continue
                    row = it[1]
                    head = row[0]
                    size = (max((b.size for b in row), default=0.0)
                            or 10.5) * scale
                    nlines = 1 if len(row) > 1 else max(1, len(head.lines))
                    total = para_h.get(ii)
                    if total:
                        # 段高必须跟着缩放，否则"行距×缩放 + 段后(未缩放)"
                        # 之和仍是原高度，缩放完全被抵消
                        total = total * scale

                    if total and total > 0:
                        # 行内高度取"自然行高"，剩下的用段后间距补足。
                        # ★ 不能把总高平均摊到行距上：一个只有 1 行的段落在原文里
                        #   可能离下一段 34pt，把它撑成 34pt 的行距就成了"超大行距"，
                        #   正确做法是行距保持自然、多出来的算段后留白。
                        #   这样"行距 + 段后"之和恰好等于原文的段高，
                        #   页数能对齐，观感也不失真。
                        per = min(total, size * 1.5)
                        per = max(size * 0.98, per / max(1, nlines)
                                  if nlines > 1 else per)
                        extra = max(0.0, total - per * nlines)
                    else:
                        per = max(size * 0.98, head.line_gap or size * 1.12)
                        extra = 0.0
                    p = w.add_paragraph()
                    p.paragraph_format.line_spacing = Pt(round(per, 1))
                    if extra > 0.5:
                        p.paragraph_format.space_after = Pt(
                            round(min(extra, 36.0), 1))

                    al = _align_of(head, page.rect) if len(row) == 1 else 0
                    p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if al == 1
                                   else WD_ALIGN_PARAGRAPH.RIGHT if al == 2
                                   else WD_ALIGN_PARAGRAPH.LEFT)
                    if len(row) == 1 and head.bbox[0] - 72.0 > 4.0:
                        # ★ 缩进必须克制，两个限制都是实测换来的：
                        #   1. 居中/靠右的块**不能**再给左缩进 ——
                        #      居中是相对正文宽度算的，加了缩进就变成
                        #      "在一个右移的窄框里居中"，既偏又挤。
                        #   2. 缩进封顶 6 个字宽。超出的说明这块根本是
                        #      "定位"而非"缩进"（幻灯片、表格外的零散文字），
                        #      照抄成缩进会把正文挤窄、换行数暴增。
                        ind = max(0.0, head.bbox[0] - 72.0)
                        if al == 0 and ind <= 6.0 * size:
                            p.paragraph_format.left_indent = Pt(ind * scale)

                    for bi, b in enumerate(row):
                        if bi:
                            # 同一视觉行上的多个块：用两个全角空格相连 ——
                            # 这才是它们在原版面上的关系（并排），
                            # 竖排成两段既膨胀又不像原文
                            rr = p.add_run("　　")
                            rr.font.size = Pt(round(size, 1))
                        for li, line in enumerate(b.lines):
                            if li and b.kind != "list":
                                p.add_run().add_break()
                            for sp in line.spans:
                                if not sp.text:
                                    continue
                                r = p.add_run(sp.text)
                                r.font.name = _docx_font(sp.font)
                                r.font.size = Pt(round(sp.size * scale, 1)
                                                 or round(size, 1))
                                r.bold = _span_bold(sp.flags)
                                r.italic = _span_italic(sp.flags)
                                try:
                                    r.font.color.rgb = RGBColor(
                                        *(int(round(c * 255))
                                          for c in sp.color))
                                except Exception:
                                    pass
                    n_para += 1
            w.save(str(out_p))
        finally:
            sess.close()
    finally:
        doc.close()

    rep.out.append(str(out_p))
    bits = f"{n_para} 段、{n_tab} 个表格、{n_img} 张图片"
    if n_skip:
        bits += f"（另有 {n_skip} 张整页扫描图未嵌入 —— 其文字已由 OCR 转成段落）"
    rep.detail.append(bits)
    rep.warnings.append(
        "段落属性（字体/字号/加粗/对齐/行距/缩进）逐段还原；"
        "绝对定位的文本框、分栏、页眉页脚不在还原范围 —— "
        "这类版面需要保留时请改用本工具的《编辑 PDF》功能，不要中转 Word")
    return rep


def _group_rows(blocks, page_rect):
    """把版面里的块按"视觉行"分组：同一水平行上的块归为一组。

    ★ 这一步是流式转换能否对齐页数的关键。
    PDF 是绝对定位的，表单类页面上一行里并排放着好几个独立文本块
    （"姓名：___  性别：___  年龄：___"）。流式段落只能纵向堆叠，
    照块逐段输出等于把它们竖起来排 ——
    实测某扫描件第 1 页 48 个块实际只有 49 行，内容跨度 677pt，
    竖排后变成 874pt，直接溢出 176pt、5 页文档胀成 7 页。
    合并成一行既解决了膨胀，也更接近人在 Word 里会写成的样子。

    判据：与本行纵向带的重叠占自身高度的比例 > 0.5 就算同一行。
    用重叠比例而不是"顶端距离"：同一行里不同块的顶端本来就参差不齐
    （上下标、不同字号），顶端距离会把它们判成两行。
    """
    items = [b for b in blocks if b.lines and b.text.strip()]
    if not items:
        return []
    items.sort(key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    rows: list[list] = []
    for b in items:
        top, bot = b.bbox[1], b.bbox[3]
        h = max(1.0, bot - top)
        placed = False
        for row in rows:
            rtop = min(x.bbox[1] for x in row)
            rbot = max(x.bbox[3] for x in row)
            ov = min(bot, rbot) - max(top, rtop)
            if ov > 0.5 * h:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b.bbox[0])
    rows.sort(key=lambda r: min(b.bbox[1] for b in r))
    return rows


def _page_items(blocks):
    """把一页的块整理成有顺序的"版面项"。

    每项是三元组：
      ("row", [块, ...], 段高 pt)      一行文字（可能由并排的多个块合并）
      ("table", 块, 段高)              表格
      ("image", 块, 段高)              图片

    顺序按纵向位置 —— 流式输出必须按真实阅读顺序走，
    否则表格和图会落到不相干的位置。
    """
    texts = [b for b in blocks if b.lines and b.text.strip()]
    rows = _group_rows(texts, None)

    specials: list[tuple[float, tuple]] = []
    for b in blocks:
        if b.kind == "table":
            specials.append((b.bbox[1], ("table", b, 0.0)))
        elif b.kind == "image" and not b.lines:
            specials.append((b.bbox[1], ("image", b, 0.0)))

    merged: list[tuple[float, tuple]] = []
    for row in rows:
        top = min(b.bbox[1] for b in row)
        merged.append((top, ("row", row, 0.0)))
    merged.extend(specials)
    merged.sort(key=lambda t: t[0])
    return [t[1] for t in merged]


def _paragraph_heights(items, page_rect) -> dict[int, float]:
    """每项的绝对段高（pt）= 本项顶 → 下一项顶的实测距离。

    只给"行"项算：表格与图片自带高度，不该被这个值覆盖。
    返回 {items 里的下标: 段高}。
    """
    out: dict[int, float] = {}
    for j, (kind, payload, _) in enumerate(items):
        if kind != "row":
            continue
        head = payload[0]
        top = min(b.bbox[1] for b in payload)
        nxt = None
        for k in range(j + 1, len(items)):
            k_kind, k_pay, _ = items[k]
            nxt = min(b.bbox[1] for b in k_pay) if k_kind == "row" \
                else k_pay.bbox[1]
            break
        if nxt is not None and nxt > top:
            h = nxt - top
        else:
            gap = head.line_gap or (head.size or 10.5) * 1.12
            h = (1 if len(payload) > 1 else max(1, len(head.lines))) * gap
        size = max((b.size for b in payload), default=0.0) or 10.5
        nlines = 1 if len(payload) > 1 else max(1, len(head.lines))
        # 上限 1.5 倍字号/行：块之间的大片留白不该被算成段高，
        # 否则一个孤立标题会把整页撑开
        if h >= nlines * size * 0.98:
            out[j] = min(h, nlines * size * 1.5 + 36.0)
    return out





def _is_page_scan(block, page_rect, ratio: float = 0.7) -> bool:
    """判断这张图是不是"整页扫描图"。"""
    aw = max(0.0, block.bbox[2] - block.bbox[0])
    ah = max(0.0, block.bbox[3] - block.bbox[1])
    return aw * ah >= ratio * page_rect.width * page_rect.height


def _add_table(w, page, block) -> int:
    """把表格块写成真的 Word 表格。"""
    from docx.shared import Pt

    try:
        tabs = page.find_tables()
    except Exception:
        return 0
    target = None
    for t in tabs.tables:
        tb = tuple(t.bbox)
        if (abs(tb[0] - block.bbox[0]) < 8 and abs(tb[1] - block.bbox[1]) < 8):
            target = t
            break
    if target is None:
        return 0
    rows = target.extract() or []
    if not rows:
        return 0
    cols = max(len(r) for r in rows)
    table = w.add_table(rows=0, cols=cols)
    table.style = "Table Grid"
    for r in rows:
        cells = table.add_row().cells
        for i in range(cols):
            txt = (r[i] if i < len(r) else "") or ""
            cells[i].text = txt.replace("\n", " ")
            for para in cells[i].paragraphs:
                for run in para.runs:
                    run.font.size = Pt(round(block.size, 1) or 10.5)
    return 1


def _add_image(w, page, block) -> bool:
    """从 PDF 里抽出该块对应的图片并插进 Word。"""
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return False
    for info in infos:
        ib = tuple(info["bbox"])
        if abs(ib[0] - block.bbox[0]) < 4 and abs(ib[1] - block.bbox[1]) < 4:
            xref = int(info.get("xref") or 0)
            if xref <= 0:
                return False
            try:
                raw = page.parent.extract_image(xref)
            except Exception:
                return False
            data = raw.get("image")
            if not data:
                return False
            # python-docx 只接受流；尺寸按原显示大小（pt → 英寸×72）
            from docx.shared import Pt
            p = w.add_paragraph()
            p.alignment = 1
            run = p.add_run()
            try:
                run.add_picture(io.BytesIO(data),
                               width=Pt(min(460.0, ib[2] - ib[0])))
            except Exception:
                return False
            return True
    return False


# ============================================================ PDF → Excel


def _split_row_by_gaps(line, gap_ratio: float = 1.6) -> list[str]:
    """把一行文字按大间隙切成若干单元格。

    阈值取"字高 × 1.6"：小间隙是字距、大间隙才是列分隔。
    不这么分的话，整行会挤进一个单元格，Excel 里没法用。
    """
    spans = sorted(line.spans, key=lambda s: s.bbox[0])
    if not spans:
        return []
    h = max(1.0, line.bbox[3] - line.bbox[1])
    out: list[str] = []
    cur = spans[0].text
    for prev, sp in zip(spans, spans[1:]):
        if sp.bbox[0] - prev.bbox[2] > gap_ratio * h:
            out.append(cur)
            cur = sp.text
        else:
            cur += sp.text
    out.append(cur)
    return [c.strip() for c in out]


def pdf_to_excel(src: str | Path, out: str | Path, *,
                 pages: list[int] | None = None,
                 progress=None) -> ConvertReport:
    """PDF → Excel（.xlsx）。

    优先用 `find_tables()` 的真表格；没有表格的页退化成"按大间隙切列"，
    至少保证内容是结构化的行×列而不是一堆文本。
    每页一个工作表，表名带页码，多张表纵向排布并留空行分隔。
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    src_p = Path(src)
    out_p = Path(out)
    rep = ConvertReport(src=str(src_p), kind="转换为 Excel")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(src_p))
    wb = Workbook()
    wb.remove(wb.active)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"{src_p.name} 是加密的，需要密码")
        idx = ([_ck(i, doc.page_count) for i in pages] if pages
               else list(range(doc.page_count)))

        from .session import Session
        sess = Session(str(src_p))
        sess.load(progress=progress)
        n_tab = n_rows = 0
        try:
            for pno in idx:
                ws = wb.create_sheet(title=f"第{pno + 1}页")
                page = doc[pno]
                tabs = []
                try:
                    tabs = list(page.find_tables().tables)
                except Exception:
                    tabs = []
                row_cursor = 1
                for t in tabs:
                    rows = t.extract() or []
                    if not rows:
                        continue
                    cols = max(len(r) for r in rows)
                    for ri, r in enumerate(rows):
                        for ci in range(cols):
                            v = (r[ci] if ci < len(r) else "") or ""
                            cell = ws.cell(row=row_cursor + ri,
                                           column=ci + 1,
                                           value=v.replace("\n", " ").strip())
                            cell.alignment = Alignment(
                                wrap_text=True, vertical="center")
                            if ri == 0:
                                cell.font = Font(bold=True)
                    row_cursor += len(rows) + 2
                    n_tab += 1
                    n_rows += len(rows)
                if not tabs:
                    # 没有表格：按行 + 大间隙切列
                    blocks = [b for b in sess.page_blocks(pno) if b.lines]
                    blocks.sort(key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
                    for b in blocks:
                        for line in b.lines:
                            cells = _split_row_by_gaps(line)
                            if not cells:
                                continue
                            for ci, v in enumerate(cells):
                                ws.cell(row=row_cursor, column=ci + 1,
                                        value=v).alignment = Alignment(
                                    wrap_text=True, vertical="center")
                            row_cursor += 1
                            n_rows += 1
                        row_cursor += 1
                for col in range(1, 9):
                    ws.column_dimensions[
                        ws.cell(row=1, column=col).column_letter
                    ].width = 22
        finally:
            sess.close()
        wb.save(str(out_p))
    finally:
        doc.close()

    rep.out.append(str(out_p))
    rep.detail.append(f"{n_tab} 个表格、{n_rows} 行")
    rep.warnings.append(
        "有表格线/明显列间隙的页会还原成真正的行×列；"
        "纯段落文字按'大间隙切列'近似处理 —— "
        "需要精确对位请改用本工具的《编辑 PDF》功能")
    return rep


# ============================================================ PDF → PPT


def pdf_to_ppt(src: str | Path, out: str | Path, *,
               dpi: float = 150.0, pages: list[int] | None = None,
               put_text_in_notes: bool = True,
               progress=None) -> ConvertReport:
    """PDF → PowerPoint（.pptx）：一页一张幻灯片，页面图铺满。

    "保持原样"是 PDF 转 PPT 唯一不撒谎的做法 —— 把 PDF 的绝对定位版面
    硬拆成 PPT 的可编辑形状，几乎必然把版面拆散（免费工具都这么干，
    结果就是每页都歪）。所以这里把版面整个保留成图片，
    同时把该页的文字放进**备注**里，需要改内容时能直接复制，
    也便于做讲稿。
    """
    from pptx import Presentation
    from pptx.util import Emu, Pt

    src_p = Path(src)
    out_p = Path(out)
    rep = ConvertReport(src=str(src_p), kind="转换为 PPT")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(src_p))
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"{src_p.name} 是加密的，需要密码")
        idx = ([_ck(i, doc.page_count) for i in pages] if pages
               else list(range(doc.page_count)))

        first = doc[idx[0]]
        prs = Presentation()
        # 幻灯片尺寸跟随原页：不跟随的话宽高比一变，图就被拉伸
        prs.slide_width = Emu(int(first.rect.width * 12700))
        prs.slide_height = Emu(int(first.rect.height * 12700))
        blank = prs.slide_layouts[6]

        zoom = dpi / 72.0
        for pno in idx:
            page = doc[pno]
            slide = prs.slides.add_slide(blank)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                                 alpha=False)
            # 等比放进幻灯片，避免非 A4 页被拉伸
            pw, ph = page.rect.width, page.rect.height
            sw = prs.slide_width
            sh = prs.slide_height
            scale = min(sw / (pw * 12700), sh / (ph * 12700))
            iw = int(pw * 12700 * scale)
            ih = int(ph * 12700 * scale)
            slide.shapes.add_picture(io.BytesIO(pix.tobytes("png")),
                                    int((sw - iw) / 2), int((sh - ih) / 2),
                                    width=iw, height=ih)
            if put_text_in_notes:
                txt = page.get_text("text").strip()
                if txt:
                    slide.notes_slide.notes_text_frame.text = txt[:5000]
            if progress:
                progress(f"第 {pno + 1} 页 → 幻灯片")
        prs.save(str(out_p))
    finally:
        doc.close()

    rep.out.append(str(out_p))
    rep.detail.append(f"{len(idx)} 张幻灯片，{dpi:.0f} DPI 页面图")
    rep.warnings.append(
        "版面以整页图片保留（这是唯一不失真的转换方式），"
        "文字已放进取每页的备注，方便复制取用")
    return rep


# ============================================================ 图片 → PDF


def images_to_pdf(paths: list[str | Path], out: str | Path, *,
                  page_size: str = "fit", dpi: float = 150.0,
                  margin: float = 0.0, progress=None) -> ConvertReport:
    """把多张图片合成一份 PDF。

    page_size:
      "fit"  每页尺寸 = 图片尺寸（按 dpi 换算），最适合图片集
      "a4"   每页 A4 竖版，图片等比缩放居中，适合打印
    """
    from PIL import Image

    rep = ConvertReport(src=str(paths[0]) if paths else "", kind="图片合成 PDF")
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    A4 = (595.276, 841.89)

    doc = pymupdf.open()
    try:
        for p in paths:
            p = Path(p)
            if not p.exists():
                rep.warnings.append(f"{p.name} 不存在，已跳过")
                continue
            with Image.open(p) as im:
                im.load()
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                buf = io.BytesIO()
                if p.suffix.lower() in (".jpg", ".jpeg", ".jfif"):
                    im.save(buf, format="JPEG", quality=92)
                    img_bytes, ext = buf.getvalue(), "jpeg"
                else:
                    im.save(buf, format="PNG")
                    img_bytes, ext = buf.getvalue(), "png"
                w_pt = im.width / dpi * 72.0
                h_pt = im.height / dpi * 72.0
            if page_size.lower() == "a4":
                pw, ph = A4
                if w_pt > h_pt:
                    pw, ph = A4[1], A4[0]
                page = doc.new_page(width=pw, height=ph)
                s = min((pw - 2 * margin) / w_pt, (ph - 2 * margin) / h_pt)
                dw, dh = w_pt * s, h_pt * s
                rect = pymupdf.Rect((pw - dw) / 2, (ph - dh) / 2,
                                    (pw + dw) / 2, (ph + dh) / 2)
            else:
                page = doc.new_page(width=w_pt + 2 * margin,
                                    height=h_pt + 2 * margin)
                rect = pymupdf.Rect(margin, margin, margin + w_pt,
                                    margin + h_pt)
            page.insert_image(rect, stream=img_bytes)
            if progress:
                progress(f"{p.name} → {rect.width:.0f}×{rect.height:.0f}pt")
    finally:
        pass
    doc.save(str(out_p), garbage=3, deflate=True)
    doc.close()
    rep.out.append(str(out_p))
    rep.detail.append(f"{len(paths)} 张图片，页面 {page_size}，{dpi:.0f} DPI 基准")
    return rep


# ============================================================ Office → PDF


def _office_app_for(ext: str) -> tuple[str, str]:
    """返回 (COM ProgID, 应用名)。"""
    if ext in (".doc", ".docx", ".rtf", ".odt"):
        return "Word.Application", "Word"
    if ext in (".xls", ".xlsx", ".csv", ".ods"):
        return "Excel.Application", "Excel"
    if ext in (".ppt", ".pptx", ".odp"):
        return "PowerPoint.Application", "PowerPoint"
    raise ValueError(f"不支持的格式 {ext}")


def office_to_pdf(src: str | Path, out: str | Path, *,
                  progress=None) -> ConvertReport:
    """Word / Excel / PPT → PDF：调用本机 Office 渲染。

    ★ 为什么必须用 Office 本体：这些格式的排版规则（字体替换、
    自动编号、表格跨页、SmartArt、图表）在 Office 内部是唯一权威实现，
    第三方库重现出来总会有偏差 —— 而"转换的损失"正是用户最在意的。
    没装 Office 时直接报错，不假装有别的办法。

    三个应用都要显式设 Visible=False、DisplayAlerts=0，
    否则会在用户桌面弹出窗口、遇到保存提示时卡死。
    """
    if os.name != "nt":
        raise RuntimeError("Office 转换依赖 Windows COM，本平台不可用")
    src_p = Path(src)
    out_p = Path(out).with_suffix(".pdf")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ext = src_p.suffix.lower()
    progid, name = _office_app_for(ext)

    try:
        import pythoncom
        import win32com.client
    except Exception as exc:
        raise RuntimeError(
            "缺少 pywin32，无法调用本机 Office："
            "pip install pywin32") from exc

    rep = ConvertReport(src=str(src_p), kind=f"{name} → PDF")
    pythoncom.CoInitialize()
    app = None
    try:
        app = win32com.client.DispatchEx(progid)
        try:
            app.Visible = False
        except Exception:
            pass
        try:
            app.DisplayAlerts = 0
        except Exception:
            pass
        s, o = str(src_p.resolve()), str(out_p.resolve())
        if name == "Word":
            d = app.Documents.Open(s, ReadOnly=True, AddToRecentFiles=False,
                                   Visible=False)
            try:
                d.SaveAs(o, FileFormat=17)       # 17 = wdFormatPDF
            finally:
                d.Close(False)
        elif name == "Excel":
            d = app.Workbooks.Open(s, ReadOnly=True, UpdateLinks=0)
            try:
                # 0 = xlTypePDF
                d.ExportAsFixedFormat(0, o)
            finally:
                d.Close(False)
        else:
            d = app.Presentations.Open(s, ReadOnly=True,
                                      WithWindow=False)
            try:
                d.SaveAs(o, 32)                  # 32 = ppSaveAsPDF
            finally:
                d.Close()
        if progress:
            progress(f"{name} 渲染完成 → {out_p.name}")
    finally:
        try:
            if app is not None:
                app.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()

    if not out_p.exists():
        raise RuntimeError(f"{name} 没有产出文件，转换失败")
    rep.out.append(str(out_p))
    sz = out_p.stat().st_size / 1024
    rep.detail.append(f"由本机 {name} 渲染，{sz:.0f} KB")
    return rep


# ============================================================ 统一入口


def convert(src: str | Path, out: str | Path | None = None, *,
            to: str = "", dpi: float = 150.0, fmt: str = "png",
            pages: list[int] | None = None, progress=None) -> ConvertReport:
    """按源格式与目标格式自动选路。

    `to` 可给 "docx"/"xlsx"/"pptx"/"pdf"/"png"/"jpeg" 等后缀；
    源是图片且目标是 pdf 时自动走 images_to_pdf。
    """
    src_p = Path(src)
    sext = src_p.suffix.lower()
    target = (to or "").lower().lstrip(".")
    if not target:
        target = "pdf" if sext in IMAGE_EXTS else "docx"
    if target in ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"):
        outd = Path(out) if out else Path("out") / f"{src_p.stem}_图片"
        return pdf_to_images(src_p, outd, fmt=target, dpi=dpi, pages=pages,
                            progress=progress)
    out_p = Path(out) if out else src_p.with_suffix("." + target)

    if sext in IMAGE_EXTS and target == "pdf":
        return images_to_pdf([src_p], out_p, dpi=dpi, progress=progress)

    if sext == ".pdf":
        if target == "docx":
            return pdf_to_word(src_p, out_p, pages=pages, progress=progress)
        if target in ("xlsx", "xls"):
            return pdf_to_excel(src_p, out_p.with_suffix(".xlsx"),
                                pages=pages, progress=progress)
        if target in ("pptx", "ppt"):
            return pdf_to_ppt(src_p, out_p.with_suffix(".pptx"), dpi=dpi,
                              pages=pages, progress=progress)
        raise ValueError(f"PDF 转 {target} 暂不支持"
                         f"（支持 docx / xlsx / pptx / 图片）")

    if sext in OFFICE_EXTS and target == "pdf":
        return office_to_pdf(src_p, out_p, progress=progress)

    raise ValueError(f"不支持从 {sext} 转到 .{target}")


def _ck(pno: int, total: int) -> int:
    i = pno + total if pno < 0 else pno
    if not (0 <= i < total):
        raise ValueError(f"页码 {pno} 越界（共 {total} 页）")
    return i
