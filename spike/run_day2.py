"""Day 2 驱动：擦除 + 重画闭环验证。

四组实验，从易到难，最后一个是关键的对照实验：

  0. 基准 —— 把一段文字替换成完全相同的文字。
     测的不是"改得对不对"，而是"重绘这个动作本身会引入多少漂移"。
     这个数字决定了最小位移原则能省掉多少损失。

  1. 等长改字 —— 服务器 → 工作站。
     未改动区域必须零差异。

  2. 变长改字 —— 段落被撑开，行数增加。
     检查行数、高度变化、溢出判定是否正确。

  3. 对照实验 —— 用 apply_redactions 的默认 graphics 参数擦除表格文字。
     预期：表格框线被误删。这验证了 graphics=LINE_ART_NONE 不是可选项。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfbox.core.classify import classify                    # noqa: E402
from pdfbox.core.verify import compare                    # noqa: E402
from pdfbox.core.editor import replace_block_text              # noqa: E402
from pdfbox.core.structure import extract_document             # noqa: E402
from pdfbox.core.fonts import describe, resolve_for_block    # noqa: E402

SRC = ROOT / "tests" / "fixtures" / "contract_sample.pdf"
OUT = ROOT / "out" / "day2"


def find_block(structure, predicate):
    for page in structure.pages:
        for b in page.blocks:
            if predicate(b):
                return b
    return None


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_case(
    name: str,
    block,
    new_text: str,
    *,
    out_name: str,
    expect_zero_outside: bool = True,
) -> None:
    banner(name)
    doc = pymupdf.open(SRC)
    match = resolve_for_block(block.font, block, new_text)

    print(f"目标块 p{block.page_no+1} [{block.kind}] {block.size:.1f}pt  字体 {block.font}")
    print(f"    原文: {block.text[:60]!r}")
    print(f"    新文: {new_text[:60]!r}")
    print(f"    字体: {describe(match)}")
    print()

    result = replace_block_text(doc, block, new_text, match)
    out = OUT / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out), garbage=4, deflate=True)
    doc.close()

    print(f"  {result.summary()}")
    for w in result.warnings:
        print(f"    ⚠ {w}")

    region = block.bbox
    diff = compare(str(SRC), str(out), block.page_no, expect_region=region,
                   diff_image=OUT / out_name.replace(".pdf", "_diff.png"))
    print()
    print(diff.report())
    if expect_zero_outside and not diff.clean:
        print("\n  ⚠ 该用例要求期望区域外零差异，但未达成")
    return result, diff


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    structure = classify(extract_document(str(SRC)))

    para = find_block(structure, lambda b: b.kind == "paragraph" and "甲方向乙方采购" in b.text)
    if para is None:
        raise SystemExit("找不到测试段落")

    # ---------------- 实验 0：基准，替换成完全相同的文字
    banner("实验 0：基准 —— 替换为完全相同的文字（测重绘自身的漂移）")
    doc = pymupdf.open(SRC)
    match = resolve_for_block(para.font, para, para.text)
    print(f"目标块 p{para.page_no+1}  行数 {len(para.lines)}  字体 {match.family}")
    result = replace_block_text(doc, para, para.text, match)
    out0 = OUT / "t0_identity.pdf"
    doc.save(str(out0), garbage=4, deflate=True)
    doc.close()
    print(f"  {result.summary()}")
    diff0 = compare(str(SRC), str(out0), para.page_no,
                    expect_region=para.bbox, diff_image=OUT / "t0_identity_diff.png")
    print()
    print(diff0.report())
    print(f"\n  ★ 重绘自身漂移：{diff0.changed_pixels} 个像素"
          f"（占该块区域的比例可用来判断重绘保真度）")

    # ---------------- 实验 1：等长改字
    new1 = para.text.replace("服务器设备", "工作站设备")
    run_case("实验 1：等长改字（服务器 → 工作站）", para, new1,
             out_name="t1_same_len.pdf")

    # ---------------- 实验 2：变长改字
    new2 = para.text.replace(
        "甲方向乙方采购服务器设备一批",
        "甲方向乙方采购服务器设备、存储阵列、网络交换机及配套不间断电源共四类产品各一批",
    )
    res2, diff2 = run_case("实验 2：变长改字（段落被撑开）", para, new2,
                           out_name="t2_longer.pdf", expect_zero_outside=False)

    # ---------------- 实验 3：对照实验 —— 默认 graphics 参数会误删矢量图形
    banner("实验 3：对照 —— apply_redactions 的 graphics 默认值会误删表格框线")
    table = find_block(structure, lambda b: b.kind == "table")
    if table is None:
        print("  样本里没找到表格块，跳过")
    else:
        print(f"目标 p{table.page_no+1} 表格块 bbox={tuple(round(v,1) for v in table.bbox)}")
        print("擦除整个表格区域，观察框线（矢量图形）是否存活\n")

        for label, gfx in (
            ("默认 graphics = REMOVE_IF_COVERED", pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED),
            ("显式 graphics = LINE_ART_NONE", pymupdf.PDF_REDACT_LINE_ART_NONE),
        ):
            doc = pymupdf.open(SRC)
            page = doc[table.page_no]
            before_drawings = len(page.get_drawings())
            r = pymupdf.Rect(table.bbox)
            page.add_redact_annot(pymupdf.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1), fill=False)
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=gfx,
                text=pymupdf.PDF_REDACT_TEXT_REMOVE,
            )
            after_drawings = len(page.get_drawings())
            remaining_text = page.get_text().strip()
            tag = "default" if gfx else "lineartnone"
            p = OUT / f"t3_{tag}.pdf"
            doc.save(str(p), garbage=4, deflate=True)
            doc.close()

            d = compare(str(SRC), str(p), table.page_no, expect_region=None,
                        diff_image=OUT / f"t3_{tag}_diff.png")
            print(f"  [{label}]")
            print(f"      矢量图形对象: {before_drawings} → {after_drawings}"
                  + ("   ← 框线被删光" if after_drawings < before_drawings else "   ← 框线完整保留"))
            print(f"      页面剩余文字: {remaining_text[:40]!r}" if remaining_text else "      页面剩余文字: （空）")
            print(f"      差异像素: {d.changed_pixels}   差异区域: {len(d.regions)} 处")
            print()


if __name__ == "__main__":
    main()
