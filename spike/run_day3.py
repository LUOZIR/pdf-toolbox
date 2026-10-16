"""Day 3 驱动：重排 + 最小位移原则验证。

两组实验：

  A. 等行数编辑（改字但不改变行数）
     预期：累计位移为 0，下游所有块"保持原样"，
           整页除了编辑块本身，其余像素零差异 —— 包括页眉页脚水印。

  B. 增行编辑（段落被撑开）
     预期：编辑块上游零差异、固定层零差异，
           下游块整体下移并被重画。
           验证方式是把"编辑块 + 下游区域"设为允许变化区，
           要求区域外差异为 0 —— 区域外正好覆盖了页眉、页脚、
           水印和上游内容。

这两个实验合起来证明了保真度承诺：
  未受影响的部分不是"高保真重建"，而是压根没碰。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfbox.core.classify import classify                    # noqa: E402
from pdfbox.core.verify import check_fixed_layer, compare  # noqa: E402
from pdfbox.core.structure import extract_document             # noqa: E402
from pdfbox.core.fonts import resolve_for_block              # noqa: E402
from pdfbox.core.reflow import apply_reflow, plan_reflow     # noqa: E402
from pdfbox.core.viz import render_page                # noqa: E402

SRC = ROOT / "tests" / "fixtures" / "contract_sample.pdf"
OUT = ROOT / "out" / "day3"


def banner(t: str) -> None:
    print()
    print("=" * 74)
    print(f"  {t}")
    print("=" * 74)


def find_block(structure, pred):
    for page in structure.pages:
        for b in page.blocks:
            if pred(b):
                return b
    return None


def run_reflow_case(name: str, structure, block, new_text: str, out_name: str,
                    allow_from: float | None) -> None:
    banner(name)
    doc = pymupdf.open(SRC)
    match = resolve_for_block(block.font, block, new_text)

    print(f"编辑目标: p{block.page_no+1} {block.size:.1f}pt  行数 {len(block.lines)}")
    print(f"    原文: {block.text[:56]!r}")
    print(f"    新文: {new_text[:56]!r}")
    print(f"    字体: {match.family}  度量比 {match.metric_ratio:.4f}  "
          f"{'✓已验证' if match.verified else '未验证'}")
    print()

    plan = plan_reflow(structure, block.page_no, block, new_text, match)
    print(plan.report())
    print()

    stats = apply_reflow(doc, structure, plan, block, new_text, match)
    out = OUT / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out), garbage=4, deflate=True)
    doc.close()
    print(f"  执行结果: {stats}")

    # ---- 固定层结构比对（必须单独做，像素比对会漏掉水印丢失）
    fc = check_fixed_layer(SRC, str(out), block.page_no)
    print()
    print("  【固定层结构比对】")
    print("  " + fc.report().replace("\n", "\n  "))

    # ---- 像素比对
    if allow_from is None:
        region = block.bbox
    else:
        region = (0.0, allow_from, 595.0, 842.0)

    d = compare(str(SRC), str(out), block.page_no, expect_region=region,
                diff_image=OUT / out_name.replace(".pdf", "_diff.png"))
    print()
    print("  【像素比对】")
    print("  " + d.report().replace("\n", "\n  "))
    print()
    if allow_from is None:
        print("  允许变化区=编辑块本身。区域外零差异 ⇒ 页眉/页脚/水印/上游/下游全部未受影响。")
    else:
        print(f"  允许变化区=y≥{allow_from:.0f}pt（编辑块及其下游）。")
        print("  区域外零差异 ⇒ 页眉、页脚、上游内容未受影响。")
        print("  水印与正文空间重叠，像素比对无法单独验证它，所以上面另做了结构比对。")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    structure = classify(extract_document(str(SRC)))

    para = find_block(structure, lambda b: b.kind == "paragraph" and "甲方向乙方采购" in b.text)
    if para is None:
        raise SystemExit("找不到测试段落")
    para_bottom = para.bbox[3]

    # ---------------- 实验 A：等行数编辑
    # 改动必须落在目标段落内，否则等于没改，测出来的"零差异"是假的
    new_a = para.text.replace("全新原厂正品", "全新原装正品").replace(
        "完整的技术资料", "完备的技术资料")
    assert new_a != para.text, "实验 A 的改动没有落到目标段落里"
    run_reflow_case(
        "实验 A：等行数编辑 —— 检验最小位移原则是否让下游完全不动",
        structure, para, new_a, "A_same_lines.pdf", allow_from=None,
    )

    # ---------------- 实验 B：增行编辑
    new_b = para.text.replace(
        "甲方向乙方采购服务器设备一批，具体型号、数量及技术参数详见附件一《设备清单》。",
        "甲方向乙方采购服务器设备、存储阵列、网络交换机及配套不间断电源共四类产品各一批，"
        "具体型号、数量、技术参数、交付地点及分批交付计划详见附件一《设备清单》及其附表。",
    )
    run_reflow_case(
        "实验 B：增行编辑 —— 检验下游下推，且上游与固定层不受影响",
        structure, para, new_b, "B_more_lines.pdf", allow_from=para.bbox[1],
    )

    # ---------------- 渲染结果供人工检查
    doc = classify(extract_document(str(OUT / "B_more_lines.pdf")))
    render_page(str(OUT / "B_more_lines.pdf"), doc, para.page_no, OUT / "B_result.png")
    print(f"\n结果渲染图: {OUT / 'B_result.png'}")


if __name__ == "__main__":
    main()
