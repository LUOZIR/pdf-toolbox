"""Day 1 驱动：提取结构 → 规则分类 → 可视化 → 输出 JSON 与摘要。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfbox.core.classify import classify, summarize          # noqa: E402
from pdfbox.core.structure import extract_document              # noqa: E402
from pdfbox.core.viz import render_all, render_plain    # noqa: E402


def main() -> None:
    pdf = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "tests" / "fixtures" / "contract_sample.pdf"
    )
    out_dir = ROOT / "out" / "day1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("提取结构…")
    doc = extract_document(pdf)
    print("规则分类…")
    doc = classify(doc)

    # JSON 结构导出
    payload = {
        "path": doc.path,
        "body_size": doc.body_size,
        "font_xrefs": doc.font_xrefs,
        "pages": [
            {
                "page_no": p.page_no + 1,
                "size": [p.width, p.height],
                "blocks": [b.to_dict() for b in p.blocks],
                "drawings": p.drawings,
            }
            for p in doc.pages
        ],
    }
    json_path = out_dir / "structure.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可视化
    print("渲染分类图…")
    paths = render_all(pdf, doc, out_dir)
    for i, p in enumerate(doc.pages):
        render_plain(pdf, p.page_no, out_dir / f"plain_p{p.page_no + 1}.png")

    print()
    print(summarize(doc))
    print()
    print(f"结构 JSON : {json_path}")
    for p in paths:
        print(f"分类图    : {p}")


if __name__ == "__main__":
    main()
