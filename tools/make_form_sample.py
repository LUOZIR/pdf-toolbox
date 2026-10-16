"""生成合成测试样本：扫描件风格的表单 + 幻灯片风格的图册。

★ 为什么需要它：真实的测试文档含个人信息，不能随源码分发。
  但回归测试依赖那些文档的**技术特征** ——
    · 表单填写栏两侧的下划线（会撑宽墨迹跨度，是逐字框的主要误差源）
    · 中西混排（汉字 1em、数字 0.5em、'@' 约 1em）
    · 低分辨率扫描（1.55 像素/pt，一个源像素 = 0.645pt）
    · 密集数字串（字间空隙不足一个源像素，墨迹糊成一团）
    · 全角标点（（）：，、）
    · 锐利位图（用于验证"降采样损失"判据：锐利图不该降采样）
  所以替代样本必须**带着这些特征**生成，否则回归就变成了空跑。

★ 全部内容均为虚构：示例省/示例市、张示例、13800000000、example@example.com、
  110101200405160000。任何与真实人物或单位的相似都属巧合。

用法：python -m tools.make_form_sample

★ 文字一律用 MuPDF 内置中文字体绘制，**不嵌入系统字体**：
  这两份样本都会被画成位图，文字层只为渲染服务；
  嵌入完整宋体会让文件从几百 KB 膨胀到 5 MB（实测）。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

W, H = 595.0, 842.0
SCAN_PPP = 1.55            # 模拟扫描件的分辨率（像素/pt）
FONT_SONG = Path("C:/Windows/Fonts/simsun.ttc")
FONT_HEI = Path("C:/Windows/Fonts/simhei.ttf")

# ---------------------------------------------------------------- 虚构内容

TITLE = "示例大学学生赴国（境）外交流学习及参与线上项目安全承诺书"

# 表单行：(整行文字, 字号, 左沿 x)。含下划线的行会复现"非文字墨迹"问题。
#
# ★ 左沿必须**逐行错开**：真实表单的标签起始位置各不相同
#   （实测 x=96 / 126 / 302），而段落合并规则要求左右边界一致 ——
#   左沿一致时 OCR 会把相邻几行并成一个大块，样本就失去了
#   "逐行独立成块"这个特征（真实文档第 1 页 48 行对应 48 个块）。
FORM_LINES: list[tuple[str, float, float]] = [
    ("甲方：示例大学国际合作部（港澳台事务办公室）", 13.0, 110.0),
    ("邮编：100000", 13.0, 126.0),
    ("地址：示例省示例市示例区示例路1号", 13.0, 88.0),
    ("传真：010-00000000", 13.0, 88.0),
    ("电话：010-00000001", 13.0, 96.0),
    ("乙方（承诺方）：姓名（参与学生）", 13.0, 150.0),
    ("姓名：张示例", 13.0, 150.0),
    ("学号：2023117550", 12.0, 126.0),
    ("院（系）示例学院", 13.0, 126.0),
    ("联系电话：13800000000", 12.0, 126.0),
    ("性别：男", 12.0, 88.0),
    ("出生日期_2004.05.16", 12.0, 88.0),
    ("家庭住址示例省示例市示例区示例街道示例村123号", 11.5, 88.0),
    ("中国护照号码：E12345678", 12.0, 126.0),
    ("国籍：中国", 12.0, 110.0),
    ("Email：example@example.com", 12.0, 88.0),
    ("身份证号：110101200405160000", 11.5, 88.0),
    ("丙方：（乙方暨承诺方的父母或法定监护人）", 12.5, 150.0),
    ("父亲姓名：张示例    母亲姓名：李示例", 12.0, 96.0),
    ("手机号码：13900000000    职业：学生", 12.0, 88.0),
    ("家庭电话：010-00000002", 12.0, 88.0),
    ("紧急联系人姓名：王示例    紧急电话：13700000000", 12.0, 110.0),
    ("电子邮箱：zhangsan@example.com", 11.5, 88.0),
    ("邮政编码_100000_电话13900000000传真010-00000000", 11.0, 88.0),
    ("身份证号码_110101200405160000_身份证号码3306", 10.5, 88.0),
    ("zhangsan_268@example.com", 10.0, 88.0),
]

BODY = [
    "一、本人已知悉并自愿遵守国家及学校关于学生赴国（境）外交流学习的各项规定，"
    "保证所提交的全部材料真实、完整、有效。",
    "二、本人承诺在国（境）外期间遵守当地法律法规，尊重当地风俗习惯，"
    "不从事任何与交流学习无关的活动，不参与任何可能危害国家安全的活动。",
    "三、本人承诺按期返校，不擅自延期或变更行程。如遇特殊情况需变更，"
    "将提前向国际合作部书面报告并获得批准。",
    "四、本人已充分了解国（境）外期间可能存在的风险，"
    "并已就本承诺书全部内容向父母或法定监护人如实告知，取得其同意。",
    "五、如违反上述承诺，本人愿意承担由此产生的一切后果，"
    "并接受学校依据相关规定作出的处理。",
]

TABLE = [
    ["项目", "起止时间", "目的地", "负责人"],
    ["暑期交流", "2026.07.01-2026.08.15", "示例国示例市", "张示例"],
    ["学期交换", "2026.09.01-2027.01.15", "示例国示例大学", "李示例"],
]

# ---------------------------------------------------------------- 生成


def _draw_form_page(page: pymupdf.Page, pno: int, total: int) -> None:
    """画一页表单：标题 + 若干行 + 填写栏下划线。"""
    page.insert_text((72, 60), f"示例大学 · 安全承诺书", fontname="china-ss", fontsize=9,
                     color=(0.35, 0.35, 0.35))
    page.insert_text((470, 60), f"第 {pno} 页 / 共 {total} 页", fontname="china-ss", fontsize=9,
                     color=(0.35, 0.35, 0.35))

    if pno == 1:
        # 标题分两行，复现"标题被识别成两行"的场景
        page.insert_text((88, 96), TITLE[:22], fontname="china-ss", fontsize=15)
        page.insert_text((88, 118), TITLE[22:], fontname="china-ss", fontsize=15)
        y = 150.0
        for text, size, lx in FORM_LINES[:12]:
            page.insert_text((lx, y), text, fontname="F0",
                             fontfile=str(FONT_SONG), fontsize=size)
            y += 20.0
            if text.startswith(("家庭住址", "身份证号", "Email")):
                # 填写栏下划线：非文字墨迹，会撑宽墨迹跨度
                page.draw_line(pymupdf.Point(300, y - 14),
                               pymupdf.Point(min(540.0, 300 + len(text) * 9), y - 14),
                               color=(0.55, 0.55, 0.55), width=0.8)
    elif pno == 2:
        y = 90.0
        for text, size, lx in FORM_LINES[12:]:
            page.insert_text((lx, y), text, fontname="F0",
                             fontfile=str(FONT_SONG), fontsize=size)
            y += 20.0
            if text.startswith(("家庭住址", "身份证号码_", "邮政编码_")):
                page.draw_line(pymupdf.Point(300, y - 14),
                               pymupdf.Point(505, y - 14),
                               color=(0.55, 0.55, 0.55), width=0.8)
    else:
        y = 90.0
        for para in BODY:
            rect = pymupdf.Rect(88, y, 512, y + 90)
            page.insert_textbox(rect, para, fontname="F0",
                                fontfile=str(FONT_SONG), fontsize=12,
                                align=pymupdf.TEXT_ALIGN_LEFT)
            y += 96.0
        if pno == 3:
            # 表格：用线画出来，供 find_tables 检出
            top = y + 6
            colx = [88, 160, 300, 420, 512]
            for r, row in enumerate(TABLE):
                yy = top + r * 22
                page.draw_line(pymupdf.Point(colx[0], yy),
                               pymupdf.Point(colx[-1], yy),
                               color=(0.4, 0.4, 0.4), width=0.6)
                for c, cell in enumerate(row):
                    page.insert_text((colx[c] + 4, yy + 15), cell,
                                     fontname="china-s",
                                     fontsize=10)
            yy = top + len(TABLE) * 22
            page.draw_line(pymupdf.Point(colx[0], yy),
                           pymupdf.Point(colx[-1], yy),
                           color=(0.4, 0.4, 0.4), width=0.6)
            for x in colx:
                page.draw_line(pymupdf.Point(x, top), pymupdf.Point(x, yy),
                               color=(0.4, 0.4, 0.4), width=0.6)

    # 页脚签名区：带下划线
    page.insert_text((88, 790), "本人签名：____________    日期：______年___月___日",
                     fontname="china-s", fontsize=11)


def _add_scan_noise(pix, sigma: float = 4.0, seed: int = 20261015) -> bytes:
    """给渲染结果加轻度高斯噪声 + 轻微模糊，模拟扫描件的传感器噪声。

    噪声幅度刻意控制在"看起来仍是干净的文档"的程度：太大就不是测试样本
    而是压力测试了。目的只是让无损压缩压不动，从而真正走到有损分支。
    """
    import numpy as np

    h, w, n = pix.height, pix.width, pix.n
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, n).astype(
        np.float32)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0.0, sigma, size=(h, w, 1))
    # 3x3 均值模糊：模拟扫描的光学扩散
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    out = np.empty_like(arr)
    for ch in range(n):
        plane = np.pad(arr[:, :, ch], 1, mode="edge")
        acc = np.zeros((h, w), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                acc += plane[dy:dy + h, dx:dx + w] * k[dy, dx]
        out[:, :, ch] = acc
    out = np.clip(out, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(out[:, :, :3] if n >= 3 else out[:, :, 0]).save(
        buf, format="PNG")
    return buf.getvalue()


def build_form_sample(path: Path, pages: int = 4) -> None:
    """生成扫描件风格的表单样本（1.55 像素/pt 的位图 PDF）。"""
    doc = pymupdf.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=W, height=H)
        _draw_form_page(page, i, pages)
    # 先存成矢量，再整体光栅化 —— 得到"图片型 PDF"
    buf = io.BytesIO()
    doc.save(buf, garbage=3, deflate=True)
    doc.close()
    src = pymupdf.open("pdf", buf.getvalue())

    out = pymupdf.open()
    for i in range(pages):
        pix = src[i].get_pixmap(matrix=pymupdf.Matrix(SCAN_PPP, SCAN_PPP),
                                alpha=False)
        # ★ 加扫描噪声，让它更像真实扫描件。
        #   不加噪声时样本"太干净"：无损压缩就能压到一半，
        #   压缩功能的有损分支（JPEG 降采样/量化）根本不会被走到 ——
        #   回归看着全绿，实际漏测。真实扫描件有传感器噪声与纸张纹理，
        #   无损压缩压不动，才会真正考验"降采样损失 vs 编码损失"那套判据。
        png = _add_scan_noise(pix)
        page = out.new_page(width=W, height=H)
        page.insert_image(pymupdf.Rect(0, 0, W, H), stream=png)
        pix = None
    out.save(str(path), garbage=4, deflate=True)
    out.close()
    src.close()
    print(f"  已生成 {path.name}（{pages} 页，{SCAN_PPP} 像素/pt 的扫描风格）")


def build_slides_sample(path: Path) -> None:
    """生成幻灯片风格的图册：几张大而锐利的位图。

    它的用途是验证压缩里的"降采样损失"判据 ——
    锐利位图降采样代价大（实测 31~34dB），不该降；
    扫描件降采样代价小（39.6dB），该降。
    没有这样一份样本，那条判据就没法回归。
    """
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page(width=W, height=H)
        page.insert_text((60, 70), f"示例项目 · 阶段 {i + 1}", fontname="china-ss", fontsize=20)
        page.insert_text((60, 100), "University of Example — CE000 Example Project",
                         fontname="china-s", fontsize=11)
        # 三张锐利位图：细线网格 + 文字，模拟截图/图表
        for k in range(3):
            # 显示尺寸刻意取小：样本要小，但**有效分辨率**要够高（200 DPI）
            # 才触发"降采样"分支 —— 这是它存在的唯一目的
            rect = pymupdf.Rect(60, 130 + k * 130, 300, 240 + k * 130)
            sub = pymupdf.open()
            sp = sub.new_page(width=rect.width, height=rect.height)
            for gx in range(0, int(rect.width) + 1, 18):
                sp.draw_line(pymupdf.Point(gx, 0), pymupdf.Point(gx, rect.height),
                             color=(0.75, 0.78, 0.82), width=0.5)
            for gy in range(0, int(rect.height) + 1, 18):
                sp.draw_line(pymupdf.Point(0, gy), pymupdf.Point(rect.width, gy),
                             color=(0.75, 0.78, 0.82), width=0.5)
            for j in range(3):
                sp.insert_text((12, 24 + j * 30),
                               f"阶段 {i + 1}.{k + 1}  里程碑 {j + 1}  "
                               f"权重 {(j + 1) * 5}%",
                               fontname="china-s",
                               fontsize=10)
            # 200 DPI 的锐利位图（与真实样本同一量级）
            png = sp.get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72),
                                alpha=False).tobytes("png")
            sub.close()
            page.insert_image(rect, stream=png)
    doc.save(str(path), garbage=4, deflate=True)
    doc.close()
    print(f"  已生成 {path.name}（2 页，6 张 200 DPI 锐利位图）")


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print("生成合成测试样本（内容全部虚构）")
    build_form_sample(FIXTURES / "form_sample.pdf")
    build_slides_sample(FIXTURES / "slides_sample.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
