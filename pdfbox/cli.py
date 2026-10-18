"""命令行工具：不打开图形界面也能完成查看与编辑。

用法：
    python -m pdfbox.cli info    <pdf>
    python -m pdfbox.cli blocks  <pdf> [页码]
    python -m pdfbox.cli find    <pdf> "关键词"
    python -m pdfbox.cli edit    <pdf> --page 1 --match "原文片段" --to "新文字" -o out.pdf
    python -m pdfbox.cli render  <pdf> --page 1 -o page.png
    python -m pdfbox.cli fonts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接执行本文件
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from pdfbox.core import describe, summarize
from pdfbox.core.fonts import FAMILY_CANDIDATES, FONT_DIRS, _find_file, match_font
from pdfbox.core.model import KIND_IMAGE, LAYER_FIXED
from pdfbox.core.session import DOC_IMAGE, DOC_MIXED, DOC_TEXT, Session


def _open(args) -> Session:
    s = Session(args.pdf)
    force = None
    if getattr(args, "ocr", False):
        force = True
    elif getattr(args, "no_ocr", False):
        force = False
    s.load(force_ocr=force, progress=lambda m: print(f"  · {m}", file=sys.stderr))
    return s


# --------------------------------------------------------------- info


def cmd_info(args) -> int:
    s = _open(args)
    type_name = {DOC_TEXT: "文本型（有文字层）",
                 DOC_IMAGE: "图片型（需 OCR）",
                 DOC_MIXED: "混合型"}[s.doc_type]
    print(f"文件    : {s.path}")
    print(f"页数    : {s.page_count}")
    print(f"文档类型: {type_name}")
    print(f"页面类型: {', '.join(f'p{i+1}={k}' for i, k in enumerate(s.page_kinds))}")
    if s.structure and s.structure.font_xrefs:
        print(f"嵌入字体: {', '.join(f'{k}(xref={v})' for k, v in s.structure.font_xrefs.items())}")
    if s.structure:
        print(f"正文字号: {s.structure.body_size}pt")
    print()
    for i in range(s.page_count):
        blocks = s.page_blocks(i)
        editable = s.editable_blocks(i)
        fixed = sum(1 for b in blocks if b.layer == LAYER_FIXED)
        print(f"  第 {i+1} 页: {len(blocks)} 块（可编辑 {len(editable)}、固定层 {fixed}）")
    s.close()
    return 0


# --------------------------------------------------------------- blocks


def cmd_blocks(args) -> int:
    s = _open(args)
    pages = [args.page - 1] if args.page else range(s.page_count)
    for pno in pages:
        if not (0 <= pno < s.page_count):
            print(f"页码超出范围: {args.page}")
            return 2
        editable = s.editable_blocks(pno)
        print(f"===== 第 {pno+1} 页  可编辑块 {len(editable)} 个 =====")
        for i, b in enumerate(editable):
            tag = "固定" if b.layer == LAYER_FIXED else "流式"
            conf = f"{b.confidence:.2f}"
            text = b.text.replace("\n", "[换行]")
            print(f"  #{i:<3} [{tag}][{b.kind:<9}] 置信 {conf}  "
                  f"y={b.bbox[1]:6.1f}  {b.size:5.1f}pt  {text[:56]}")
            if args.verbose:
                for r in b.reasons:
                    print(f"        · {r}")
    s.close()
    return 0


# --------------------------------------------------------------- find


def cmd_find(args) -> int:
    s = _open(args)
    hits = 0
    needle = args.keyword
    for pno in range(s.page_count):
        for i, b in enumerate(s.editable_blocks(pno)):
            if needle in b.text:
                hits += 1
                print(f"p{pno+1} #{i}  [{b.kind}] {b.text[:70]!r}")
    print(f"\n共 {hits} 处匹配")
    s.close()
    return 0 if hits else 1


# --------------------------------------------------------------- edit


def _pick_font(args, s: Session, block):
    """选择字体：命令行指定 > 自动匹配。"""
    if args.font:
        p = Path(args.font)
        if not p.exists():
            raise SystemExit(f"字体文件不存在：{p}")
        m = match_font(p.name, need_chars=args.to)
        if m.ok:
            return m
        m.font = pymupdf.Font(fontfile=str(p))
        m.file = str(p)
        m.source = "cli"
        return m

    if block.font:
        return s.match_for(block, args.to)

    # 图片型文档没有字体信息，按语言猜一个合适的
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (args.to or ""))
    if has_cjk:
        for cand in ("NotoSansSC-VF.ttf", "msyh.ttc", "simhei.ttf", "simsun.ttc"):
            p = _find_file(cand)
            if p:
                m = match_font(cand, need_chars=args.to)
                if m.ok:
                    m.source = "auto"
                    return m
    m = match_font("arial", need_chars=args.to)
    return m


def cmd_edit(args) -> int:
    s = _open(args)

    # 定位目标块
    target = None
    if args.index is not None:
        blocks = s.editable_blocks(args.page - 1)
        if not (0 <= args.index < len(blocks)):
            print(f"块序号超出范围：{args.index}（该页共 {len(blocks)} 块）")
            return 2
        target = blocks[args.index]
    elif args.match:
        for pno in range(s.page_count):
            for b in s.editable_blocks(pno):
                if args.match in b.text:
                    target = b
                    break
            if target:
                break
    if target is None:
        print("没有找到目标块。请用 blocks 命令查看可用块，或用 --match 指定原文片段。")
        return 2

    print(f"目标块 p{target.page_no+1}  [{target.kind}]  {target.size:.1f}pt")
    print(f"  原文: {target.text[:100]!r}")

    # 替换：整块替换，或只把 match 片段换成 to
    if args.append:
        new_text = target.text + args.to
    elif args.match and args.replace_only:
        new_text = target.text.replace(args.match, args.to)
    else:
        new_text = args.to
    print(f"  新文: {new_text[:100]!r}")

    font_file = None
    if not target.font:
        m = _pick_font(args, s, target)
        if not m.ok:
            print(f"找不到可用字体：{m.warnings}")
            return 2
        font_file = m.file
        print(f"  字体: {describe(m)}")
    else:
        m = s.match_for(target, new_text)
        print(f"  字体: {describe(m)}")

    cover = None
    if getattr(args, "cover_rect", None):
        try:
            parts = [float(v) for v in args.cover_rect.split(",")]
            if len(parts) != 4:
                raise ValueError
            cover = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            print("--cover-rect 格式应为 x0,y0,x1,y1（四个数字）")
            return 2

    tcolor = None
    if getattr(args, "color", None):
        try:
            parts = [float(v) / 255.0 for v in args.color.split(",")]
            if len(parts) != 3:
                raise ValueError
            tcolor = (parts[0], parts[1], parts[2])
        except ValueError:
            print("--color 格式应为 R,G,B（三个 0-255 的数字）")
            return 2

    try:
        rec = s.replace_text(target, new_text, font_file=font_file,
                             fontsize=args.size, reflow=not args.no_reflow,
                             cover_rect=cover, text_color=tcolor,
                             weight=getattr(args, "weight", "auto"))
    except RuntimeError as e:
        print(f"编辑失败：{e}")
        return 3

    print()
    print(f"√ {rec.summary()}")
    if rec.method:
        print(f"    {rec.method}")
    for w in rec.warned:
        print(f"    ! {w}")

    out = Path(args.out) if args.out else s.path.with_name(
        s.path.stem + "_edited" + s.path.suffix)
    saved = s.save(out)
    print(f"\n已保存: {saved}")
    s.close()
    return 0


# --------------------------------------------------------------- render


def cmd_render(args) -> int:
    s = _open(args)
    pno = (args.page or 1) - 1
    if not (0 <= pno < s.page_count):
        print(f"页码超出范围: {args.page}")
        return 2
    out = Path(args.out) if args.out else Path(f"page{pno+1}.png")
    zoom = args.zoom
    pix = s.doc[pno].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pix.save(str(out))
    print(f"已渲染: {out}  ({pix.width}×{pix.height})")
    if args.annotate:
        from pdfbox.core.viz import render_page
        ann = out.with_name(out.stem + "_annotated" + out.suffix)
        render_page(str(s.path), s.structure, pno, ann)
        print(f"分类标注图: {ann}")
    s.close()
    return 0


# --------------------------------------------------------------- fonts


def cmd_fonts(args) -> int:
    print("系统可用字体（按字体族）：")
    for family, cands in FAMILY_CANDIDATES.items():
        found = []
        for c in cands:
            p = _find_file(c)
            if p:
                found.append(Path(p).name)
        if found:
            print(f"  {family:<18} → {', '.join(found)}")
    print(f"\n搜索目录：")
    for d in FONT_DIRS:
        print(f"  {d}  {'√' if Path(d).exists() else '× 不存在'}")
    return 0


# --------------------------------------------------------------- pages


def _page_spec(spec: str | None, total: int) -> list[int]:
    from pdfbox.core.pages import parse_page_spec
    return parse_page_spec(spec, total) if spec else []


def cmd_pages(args) -> int:
    from pdfbox.core import pages as P

    src = Path(args.pdf)
    if not src.exists():
        raise FileNotFoundError(str(src))

    # 合并是独立的一条路：没有主文档
    if args.merge:
        paths = [Path(p.strip()) for p in args.merge.split(",") if p.strip()]
        out = Path(args.out) if args.out else src.with_name(
            f"{src.stem}_合并.pdf")
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(str(p))
        P.merge_pdfs([src] + paths, out)
        print(f"已合并 {len(paths) + 1} 个文件 → {out}")
        return 0

    # 提取也是独立的一条路：不改动原文档
    if args.extract and not (args.delete or args.keep or args.order
                             or args.rotate or args.insert):
        d = pymupdf.open(str(src))
        n = d.page_count
        d.close()
        idx = P.parse_page_spec(args.extract, n)
        out = Path(args.out) if args.out else src.with_name(
            f"{src.stem}_提取.pdf")
        P.extract_pages(src, idx, out)
        print(f"已提取第 {', '.join(str(i + 1) for i in idx)} 页 → {out}")
        return 0

    if not (args.delete or args.keep or args.order or args.rotate
            or args.insert):
        return _list_pages(src)

    doc = pymupdf.open(str(src))
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise RuntimeError(f"{src.name} 是加密的，需要密码")
        n0 = doc.page_count
        reports = []
        if args.delete or args.keep:
            spec = args.keep or args.delete
            picked = P.parse_page_spec(spec, n0)
            idx = ([p for p in range(n0) if p not in set(picked)]
                   if args.keep else picked)
            reports.append(P.delete_pages(doc, idx))
        if args.order:
            after = doc.page_count
            order = P.parse_page_spec(args.order, after)
            reports.append(P.reorder_pages(doc, order))
        if args.rotate:
            reports.append(P.rotate_pages(
                doc, P.parse_page_spec(args.rotate, doc.page_count),
                args.degrees))
        if args.insert:
            at = (args.at - 1) if args.at > 0 else -1
            ins = Path(args.insert)
            if not ins.exists():
                raise FileNotFoundError(str(ins))
            d2 = pymupdf.open(str(ins))
            n2 = d2.page_count
            d2.close()
            pgs = P.parse_page_spec(args.insert_pages, n2) \
                if args.insert_pages else None
            reports.append(P.insert_pdf(doc, ins, at, pgs))

        for r in reports:
            print(r.text())
        out = Path(args.out) if args.out else src.with_name(
            f"{src.stem}_已改页.pdf")
        doc.save(str(out), garbage=3, deflate=True)
        print(f"\n已保存 → {out}（{out.stat().st_size / 1024:.0f} KB）")
    finally:
        doc.close()
    return 0


def _list_pages(src: Path) -> int:
    doc = pymupdf.open(str(src))
    try:
        print(f"{src.name}：{doc.page_count} 页，"
              f"{src.stat().st_size / 1024:.0f} KB")
        print(f"{'页':>4}  {'尺寸 pt':>14}  {'旋转':>4}  {'文字':>6}  "
              f"{'图':>3}  {'首行文字'}")
        print("-" * 78)
        for i in range(doc.page_count):
            page = doc[i]
            r = page.rect
            txt = page.get_text("text").strip()
            first = next((l for l in txt.splitlines() if l.strip()), "")
            n_img = len(page.get_images(full=True))
            print(f"{i + 1:>4}  {r.width:>6.0f}×{r.height:<6.0f}  "
                  f"{page.rotation:>4}  {len(txt):>6}  {n_img:>3}  "
                  f"{first[:34]}")
    finally:
        doc.close()
    return 0


# --------------------------------------------------------------- compress


def cmd_compress(args) -> int:
    from pdfbox.core import compress as C

    src = Path(args.pdf)
    if not src.exists():
        raise FileNotFoundError(str(src))

    if args.analyze:
        doc = pymupdf.open(str(src))
        try:
            stats = C.analyze_images(doc)
        finally:
            doc.close()
        total = sum(s.nbytes for s in stats)
        print(f"{src.name}：{src.stat().st_size / 1024:.0f} KB，"
              f"{len(stats)} 张图像，占 {total / 1024:.0f} KB")
        print(f"{'xref':>6}  {'页面':>14}  {'像素':>12}  {'显示 pt':>12}  "
              f"{'有效DPI':>7}  {'体积':>9}  色彩空间")
        print("-" * 92)
        for s in sorted(stats, key=lambda x: -x.nbytes):
            pg = ",".join(str(p + 1) for p in sorted(set(s.pages))[:4])
            print(f"{s.xref:>6}  {pg:>14}  {s.width:>5}×{s.height:<6}  "
                  f"{s.disp_w:>5.0f}×{s.disp_h:<6.0f}  "
                  f"{s.effective_dpi:>7.0f}  {s.nbytes / 1024:>6.0f} KB  "
                  f"{s.colorspace}")
        return 0

    out = Path(args.out) if args.out else src.with_name(
        f"{src.stem}_压缩.pdf")
    rep = C.compress_document(
        src, out, args.preset,
        max_dpi=args.max_dpi, quality=args.quality,
        gray=True if args.gray else None,
        psnr_floor=args.psnr_floor,
        verify=not args.no_verify,
        progress=lambda m: print(f"  · {m}", file=sys.stderr))
    print(rep.text())
    return 0


# --------------------------------------------------------------- convert


def cmd_convert(args) -> int:
    from pdfbox.core import convert as C

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(str(src))
    pages = None
    if args.pages:
        doc = pymupdf.open(str(src))
        n = doc.page_count
        doc.close()
        from pdfbox.core.pages import parse_page_spec
        pages = parse_page_spec(args.pages, n)

    target = args.to
    if not target:
        target = "pdf" if src.suffix.lower() in C.IMAGE_EXTS else "docx"

    rep = C.convert(src, args.out, to=target, dpi=args.dpi,
                    pages=pages,
                    progress=lambda m: print(f"  · {m}", file=sys.stderr))
    print(rep.text())

    if args.verify and rep.out:
        from pdfbox.core.verify_convert import compare_text_layout
        for o in rep.out:
            if Path(o).suffix.lower() == ".pdf":
                print()
                print(compare_text_layout(src, o).text())
    return 0


# --------------------------------------------------------------- 入口


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfbox",
        description="PDF 工具箱 —— 结构模型驱动的 PDF 编辑器",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, with_page=True):
        sp.add_argument("pdf", help="PDF 文件路径")
        sp.add_argument("--ocr", action="store_true", help="强制按图片型文档处理（走 OCR）")
        sp.add_argument("--no-ocr", action="store_true", help="强制按文本型文档处理")

    sp = sub.add_parser("info", help="显示文档信息与页面类型")
    add_common(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("blocks", help="列出可编辑的文本块")
    add_common(sp)
    sp.add_argument("--page", type=int, help="只看指定页（从 1 开始）")
    sp.add_argument("-v", "--verbose", action="store_true", help="显示判定理由")
    sp.set_defaults(func=cmd_blocks)

    sp = sub.add_parser("find", help="按关键词查找块")
    add_common(sp)
    sp.add_argument("keyword", help="要查找的文字")
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("edit", help="替换某一块的文字")
    add_common(sp)
    sp.add_argument("--page", type=int, default=1, help="目标页（从 1 开始）")
    sp.add_argument("--index", type=int, help="目标块的序号（见 blocks 命令）")
    sp.add_argument("--match", help="按原文片段定位目标块")
    sp.add_argument("--to", required=True, help="新文字")
    sp.add_argument("--replace-only", action="store_true",
                    help="只把 --match 片段替换为 --to，保留块内其余文字")
    sp.add_argument("--append", action="store_true", help="把 --to 追加到原文末尾")
    sp.add_argument("--font", help="字体文件路径（图片型文档必须指定）")
    sp.add_argument("--size", type=float, help="覆盖字号")
    sp.add_argument("--no-reflow", action="store_true",
                    help="不做重排，只原位替换（不推动下游内容）")
    sp.add_argument("--cover-rect", metavar="x0,y0,x1,y1",
                    help="图片型文档：手工指定覆盖区域（坐标单位 pt），"
                         "OCR 框不准时用它接管")
    sp.add_argument("--weight", choices=["auto", "regular", "bold"], default="auto",
                    help="图片型文档：字重。auto=按原文笔画自动判断（保守）")
    sp.add_argument("--color", metavar="R,G,B",
                    help="图片型文档：新文字颜色，0-255。默认取原文字实测色")
    sp.add_argument("-o", "--out", help="输出文件路径")
    sp.set_defaults(func=cmd_edit)

    sp = sub.add_parser("render", help="把某页渲染成 PNG")
    add_common(sp)
    sp.add_argument("--page", type=int, default=1)
    sp.add_argument("--zoom", type=float, default=2.0, help="渲染倍率，默认 2.0")
    sp.add_argument("-a", "--annotate", action="store_true", help="同时输出结构分类标注图")
    sp.add_argument("-o", "--out")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("fonts", help="列出可用的系统字体")
    sp.set_defaults(func=cmd_fonts)

    # ---------------------------------------------------------- 页面管理
    sp = sub.add_parser("pages", help="列出页面概况，或对页面做增删与重排")
    sp.add_argument("pdf", help="PDF 文件路径")
    sp.add_argument("--delete", metavar="SPEC",
                    help="删除页面，如 2,5,7-9（页码从 1 开始）")
    sp.add_argument("--keep", metavar="SPEC",
                    help="只保留这些页面（与 --delete 二选一）")
    sp.add_argument("--order", metavar="SPEC", help="重排为新顺序，如 3,1,2,4-6")
    sp.add_argument("--rotate", metavar="SPEC", help="要旋转的页面范围")
    sp.add_argument("--degrees", type=int, default=90,
                    choices=[90, 180, 270, -90], help="旋转角度")
    sp.add_argument("--extract", metavar="SPEC", help="把指定页另存为新文件")
    sp.add_argument("--insert", metavar="FILE", help="把另一个 PDF 的页面插入进来")
    sp.add_argument("--insert-pages", metavar="SPEC", help="--insert 时要插入的页")
    sp.add_argument("--at", type=int, default=0,
                    help="插入位置：插到第 N 页之前（1 开始），0 表示末尾")
    sp.add_argument("--merge", metavar="FILES",
                    help="合并多个 PDF（逗号分隔，按给定顺序）")
    sp.add_argument("-o", "--out", help="输出文件路径")
    sp.set_defaults(func=cmd_pages)

    # ---------------------------------------------------------- 压缩
    sp = sub.add_parser("compress", help="压缩文档（重点处理图像型 PDF）")
    sp.add_argument("pdf", help="PDF 文件路径")
    sp.add_argument("--preset", default="balanced",
                    choices=["lossless", "light", "balanced", "strong", "tiny"],
                    help="压缩档位，默认 balanced")
    sp.add_argument("--max-dpi", type=float,
                    help="图像有效分辨率上限，超过则降采样（0 表示不降）")
    sp.add_argument("--quality", type=float, help="JPEG 质量，0 表示只用无损编码")
    sp.add_argument("--gray", action="store_true",
                    help="允许把近似灰度的图转成灰度")
    sp.add_argument("--psnr-floor", type=float,
                    help="质量下限 dB，低于它的方案一律不采用")
    sp.add_argument("--no-verify", action="store_true",
                    help="跳过整页渲染逐像素对比（更快，但失去最终裁判）")
    sp.add_argument("--analyze", action="store_true",
                    help="只分析图像分布，不做压缩")
    sp.add_argument("-o", "--out", help="输出文件路径")
    sp.set_defaults(func=cmd_compress)

    # ---------------------------------------------------------- 格式转换
    sp = sub.add_parser("convert", help="格式转换 PDF<->Word/Excel/PPT/图片")
    sp.add_argument("src", help="源文件（PDF / Office 文档 / 图片）")
    sp.add_argument("--to", help="目标格式：docx / xlsx / pptx / png / jpeg / pdf")
    sp.add_argument("--pages", metavar="SPEC", help="只处理指定页，如 1,3-5")
    sp.add_argument("--dpi", type=float, default=150.0,
                    help="导出图片或 PPT 页面图的分辨率，默认 150")
    sp.add_argument("--no-fit", action="store_true",
                    help="PDF→Word：不做页数自校正（保持原字号）")
    sp.add_argument("--no-images", action="store_true",
                    help="PDF→Word：不嵌入图片，只要文字")
    sp.add_argument("--page-size", default="fit", choices=["fit", "a4"],
                    help="图片合成 PDF 时的页尺寸模式")
    sp.add_argument("--verify", action="store_true",
                    help="转换后自动比对文字与版面偏差")
    sp.add_argument("-o", "--out", help="输出文件或目录")
    sp.set_defaults(func=cmd_convert)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"文件不存在：{e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
