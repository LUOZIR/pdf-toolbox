"""生成技术验证用的测试 PDF 样本。

覆盖：中文段落、标题、表格、图片、页眉页脚、水印、多页。

关键设计：最后调用 doc.subset_fonts() 把字体裁成子集。
这复现了真实 PDF 的核心难题——嵌入字体只含原文用到的字形，
想输入新字就会缺字形。Day 2 的字体引擎必须处理这个条件，
所以样本必须带着这个条件生成，否则验证是假的。
"""

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
ASSETS = ROOT / "assets"

W, H = 595.0, 842.0  # A4
MARGIN_L = 72.0
MARGIN_R = 72.0

FONT_DIR = Path("C:/Windows/Fonts")
FONT_BODY = FONT_DIR / "simsun.ttc"   # 宋体 —— 正文
FONT_HEAD = FONT_DIR / "simhei.ttf"   # 黑体 —— 标题
FONT_UI = FONT_DIR / "Deng.ttf"       # 等线 —— 页眉页脚

BODY_TEXT = [
    "第一条  合同标的",
    "甲方向乙方采购服务器设备一批，具体型号、数量及技术参数详见附件一《设备清单》。"
    "乙方应保证所供设备为全新原厂正品，符合国家相关质量标准及行业规范，"
    "并提供完整的技术资料、合格证及保修凭证。",
    "第二条  合同价款",
    "本合同总价款为人民币贰佰叁拾伍万陆仟捌佰元整（¥2,356,800.00），"
    "该价款已包含设备价款、运输费、安装调试费、税费及其他一切相关费用。"
    "除双方另有书面约定外，本合同价款不因任何原因调整。",
    "第三条  交付与验收",
    "乙方应于本合同生效之日起三十个日历日内，将全部设备运送至甲方指定地点并完成安装调试。"
    "甲方应在设备到货后十个工作日内组织验收，验收标准以附件一约定的技术参数为准。"
    "验收不合格的，乙方应在收到书面通知后五个工作日内完成整改并重新提交验收。",
    "第四条  付款方式",
    "合同生效后七个工作日内，甲方向乙方支付合同总价款的百分之三十作为预付款；"
    "设备验收合格后十五个工作日内，甲方支付合同总价款的百分之六十；"
    "剩余百分之十作为质量保证金，于质保期满且无质量问题后三十日内一次性无息支付。",
    "第五条  质量保证与售后服务",
    "乙方对所供设备提供自验收合格之日起三十六个月的免费质保服务。"
    "质保期内因设备本身质量问题导致的故障，乙方应在接到通知后四小时内响应，"
    "二十四小时内到达现场处理，所需费用由乙方承担。",
    "第六条  违约责任",
    "任何一方违反本合同约定，应向守约方支付合同总价款百分之五的违约金。"
    "因违约造成守约方损失超过违约金数额的，违约方还应就超出部分予以赔偿。"
    "乙方逾期交付的，每逾期一日应按合同总价款的万分之五向甲方支付逾期违约金。",
]

TABLE_DATA = [
    ["序号", "设备名称", "型号规格", "数量", "单价（元）"],
    ["1", "机架式服务器", "PowerEdge R750", "12", "128,000.00"],
    ["2", "存储阵列", "Unity XT 480", "2", "386,400.00"],
    ["3", "万兆交换机", "S5731-H48T4XC", "8", "21,600.00"],
    ["4", "不间断电源", "UPS-20KVA", "4", "35,200.00"],
]

SIGN_BLOCK = [
    "甲方（盖章）：____________________          乙方（盖章）：____________________",
    "法定代表人：______________________          法定代表人：______________________",
    "日期：______年______月______日              日期：______年______月______日",
]


def make_chart_image(path: Path) -> None:
    """生成一张简单的柱状图当作正文插图。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (900, 420), "white")
    d = ImageDraw.Draw(img)
    bars = [0.42, 0.68, 0.55, 0.81, 0.73, 0.91]
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    base_y, max_h = 360, 300
    d.line([(60, base_y), (860, base_y)], fill="#333333", width=2)
    for i, (v, lab) in enumerate(zip(bars, labels)):
        x = 100 + i * 125
        h = int(v * max_h)
        d.rectangle([x, base_y - h, x + 70, base_y], fill="#4a7ebb")
        d.text((x + 22, base_y + 12), lab, fill="#333333")
    d.text((60, 20), "Equipment Delivery Progress", fill="#333333")
    img.save(path)


def draw_header_footer(page: pymupdf.Page, page_no: int, total: int) -> None:
    """页眉页脚 —— 属于固定层，永不参与重排。

    故意画了两条分隔线：它们用于验证擦除文字时不会误删矢量图形。
    """
    page.insert_text(
        (MARGIN_L, 52), "恒远科技有限公司", fontname="F0", fontfile=str(FONT_BODY), fontsize=9,
        color=(0.35, 0.35, 0.35),
    )
    page.insert_text(
        (W - MARGIN_R - 60, 52), "采购合同", fontname="F0", fontfile=str(FONT_BODY), fontsize=9,
        color=(0.35, 0.35, 0.35),
    )
    page.draw_line(
        pymupdf.Point(MARGIN_L, 60), pymupdf.Point(W - MARGIN_R, 60),
        color=(0.6, 0.6, 0.6), width=0.6,
    )
    page.draw_line(
        pymupdf.Point(MARGIN_L, H - 60), pymupdf.Point(W - MARGIN_R, H - 60),
        color=(0.6, 0.6, 0.6), width=0.6,
    )
    label = f"第 {page_no} 页  共 {total} 页"
    tw = pymupdf.Font(fontfile=str(FONT_BODY)).text_length(label, fontsize=9)
    page.insert_text(
        ((W - tw) / 2, H - 44), label, fontname="F0", fontfile=str(FONT_BODY), fontsize=9,
        color=(0.35, 0.35, 0.35),
    )


def draw_watermark(page: pymupdf.Page) -> None:
    """斜向水印 —— 固定层，同时是"旋转 + 低饱和"的判定样本。

    insert_text 的 rotate 只接受 90 的倍数，任意角度得用 morph 绕轴旋转。
    """
    text = "内 部 资 料"
    fs = 58.0
    tw = pymupdf.Font(fontfile=str(FONT_BODY)).text_length(text, fontsize=fs)
    pivot = pymupdf.Point(W / 2, H / 2)
    morph = (pivot, pymupdf.Matrix(32))
    page.insert_text(
        (W / 2 - tw / 2, H / 2 + fs * 0.35), text,
        fontname="F0", fontfile=str(FONT_BODY), fontsize=fs,
        color=(0.85, 0.85, 0.85), morph=morph, overlay=False,
    )


def draw_table(page: pymupdf.Page, top: float) -> float:
    """画一个带框线的表格，返回表格底部 y。"""
    col_w = [50, 130, 150, 55, 115]
    row_h = 26.0
    x0 = MARGIN_L
    y = top
    body_font = pymupdf.Font(fontfile=str(FONT_BODY))

    for row in TABLE_DATA:
        x = x0
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x, y, x + col_w[c], y + row_h)
            page.draw_rect(rect, color=(0.4, 0.4, 0.4), width=0.7)
            fs = 9.5
            if c in (0, 3):
                tw = body_font.text_length(cell, fontsize=fs)
                tx = rect.x1 - 6 - tw
            else:
                tx = rect.x0 + 6
            page.insert_text(
                (tx, rect.y0 + 17), cell, fontname="F0", fontfile=str(FONT_BODY), fontsize=fs, color=(0, 0, 0),
            )
            x += col_w[c]
        y += row_h
    return y


def build_pages(doc: pymupdf.Document) -> None:
    total = 3
    for pno in range(total):
        page = doc.new_page(width=W, height=H)

        draw_watermark(page)                     # 下层
        draw_header_footer(page, pno + 1, total)

        y = 110.0
        if pno == 0:
            title = "设备采购合同"
            tw = pymupdf.Font(fontfile=str(FONT_HEAD)).text_length(title, fontsize=18)
            page.insert_text(
                ((W - tw) / 2, y), title, fontname="F1", fontfile=str(FONT_HEAD), fontsize=18, color=(0, 0, 0),
            )
            y += 40

        chunk = BODY_TEXT if pno < 2 else BODY_TEXT[:3]
        for para in chunk:
            if y > H - 210:
                break
            is_clause = para.startswith("第") and "条" in para[:6]
            # 中文文档的段落缩进用行首两个全角空格实现（真实文档就是这么做的）
            text = para if is_clause else "　　" + para
            rect = pymupdf.Rect(MARGIN_L, y, W - MARGIN_R, y + 300)
            used = page.insert_textbox(
                rect, text, fontname="F0", fontfile=str(FONT_BODY), fontsize=10.5,
                lineheight=1.5, align=pymupdf.TEXT_ALIGN_JUSTIFY,
            )
            y += (300 - used) + 8

        if pno == 1:
            img_path = ASSETS / "chart.png"
            if not img_path.exists():
                make_chart_image(img_path)
            rect = pymupdf.Rect(MARGIN_L + 60, y + 10, W - MARGIN_R - 60, y + 10 + 160)
            page.insert_image(rect, filename=str(img_path))
            y = rect.y1 + 20

        if pno == 2:
            y = draw_table(page, y + 6) + 26
            for line in SIGN_BLOCK:
                page.insert_text(
                    (MARGIN_L, y), line, fontname="F0", fontfile=str(FONT_BODY), fontsize=10.5, color=(0, 0, 0),
                )
                y += 30


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()
    build_pages(doc)

    raw = len(doc.tobytes())

    # ★ 关键一步：裁成子集，复现真实 PDF 的字体条件
    doc.subset_fonts()
    out = FIXTURES / "contract_sample.pdf"
    doc.save(str(out), garbage=4, deflate=True)
    doc.close()

    print(f"生成 {out}")
    print(f"  未子集化 {raw/1024:.0f} KB  →  子集化后 {out.stat().st_size/1024:.0f} KB")
    print("  字体清单：")
    d = pymupdf.open(out)
    for pno in range(len(d)):
        for f in d[pno].get_fonts(full=True):
            xref, ext, ftype, basefont, name, encoding = f[0], f[1], f[2], f[3], f[4], f[5]
            print(f"    p{pno}  {basefont:<28} 类型={ftype:<8} 编码={encoding:<10} 嵌入={ext}")
    d.close()


if __name__ == "__main__":
    main()
