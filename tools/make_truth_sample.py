"""生成带「标准答案」的测试样本。

为什么需要它：逐字框的精度一直无法客观判断 —— 视觉模型在密集小字上的
判断前后不一致，而"与 OCR 估算位置的偏差""框宽均匀性"这类指标又都被混淆了
（前者量的是和估算值的接近度，而估算值正是要修正的对象；后者因为紧致框宽度
本来就随字形变化而不成立）。

做法：先用已知字体在**已知坐标**写字并记录每个字的精确包围盒（真值），
再把整页光栅化成低分辨率位图并重新装进 PDF —— 这样它就变成了一份
"扫描件"，而我们知道每个字的真实位置。

真实位置 vs 算法输出的框 = 可精确测量的误差。有了它，
每次改动是变好还是变坏就有据可依了。
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
FONT_SONG = Path("C:/Windows/Fonts/simsun.ttc")
FONT_KAI = Path("C:/Windows/Fonts/simkai.ttf")

W, H = 595.0, 842.0
# 模拟真实扫描件的分辨率（真样本是 1.55 px/pt）
RASTER_PPP = 1.55

# 测试行：覆盖纯中文、中文混数字、密集数字、带倾斜
TEXT_LINES = [
    # ★ 内容全部虚构。真值样本当初直接抄了真实文档的文字（含住址、身份证号、
    #   电话），那份文档不能随源码分发，所以换成虚构数据 ——
    #   但**字数、脚本混排、数字位数分布保持不变**：这些特征才是被测对象。
    #   （数字要各异，不能出现长串重复的同一个数字 —— 那会变成另一种
    #     "字间空隙全部消失"的极端情形，与被替代的原样本不是一回事。）
    ("示例住址示例省示例市示例区示例镇示例村示例路一号", 12.0, 80.0, 0.0),
    ("邮政编码100100电话13812340000传真01087650000", 11.0, 110.0, 0.0),
    ("身份证号码110101200405160123身份证号码3306", 10.5, 140.0, 0.0),
    # 倾斜行：验证倾斜文字的逐字框（内容为通用法律用语，不含个人信息）
    ("甲方乙方丙方三方在平等自愿的基础上就出国事", 12.0, 170.0, 1.8),
    ("紧急联系人姓名张示例联系电话13812340000", 11.5, 200.0, 0.0),
]


def glyph_ink_box(ch: str, size: float, font_file: Path) -> tuple | None:
    """把单个字单独渲染出来，量它的墨迹包围盒（相对基线原点）。

    PyMuPDF 的 Font.glyph_bbox() 实测不可用（抛底层异常），
    所以改用"单独渲染 + 量墨迹"的办法 —— 结果精确，且不受相邻字干扰。
    返回 (x0, y0, x1, y1)，y 以基线为 0、向下为正。
    """
    import numpy as np
    zoom = 24.0
    doc = pymupdf.open()
    page = doc.new_page(width=size * 4, height=size * 4)
    ox, oy = size * 2, size * 2
    page.insert_text((ox, oy), ch, fontname="F0", fontfile=str(font_file),
                     fontsize=size, color=(0, 0, 0))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3]
    doc.close()
    mask = arr.mean(axis=2) < 128
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (xs.min() / zoom - ox, ys.min() / zoom - oy,
            (xs.max() + 1) / zoom - ox, (ys.max() + 1) / zoom - oy)


def measure_truth_from_render(pdf: Path, lines_meta) -> list[list[dict]]:
    """从高分辨率渲染里**实测**每个字的墨迹框。

    不再用旋转公式推导 —— 实测发现 MuPDF 的 morph 旋转方向与推导相反，
    导致倾斜行的真值整体是镜像的，把算法正确的输出误判成错。
    这里改成：把文字 PDF 渲染到 24 像素/pt，按空白列分段，
    按顺序把段分配给字符，取每段的墨迹包围盒作真值。完全从像素来。
    """
    import numpy as np
    Z = 24.0
    d = pymupdf.open(str(pdf))
    out = []
    for meta in lines_meta:
        # 裁剪要收窄：行距只有 30pt，±3 倍字号（±36pt）会把上下相邻行也框进来，
        # 分段数就永远对不上字符数。±1.6 倍字号足够覆盖字形（含升降部）。
        y0 = meta["baseline_y"] - meta["size"] * 1.6
        y1 = meta["baseline_y"] + meta["size"] * 1.6
        clip = pymupdf.Rect(60, y0, W - 60, y1)
        pix = d[0].get_pixmap(matrix=pymupdf.Matrix(Z, Z), clip=clip, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3]
        g = arr.mean(axis=2)
        mask = g < (np.median(g) - 60)
        colhas = mask.any(axis=0)
        # 按空白列分段（间隙 > 0.35pt 才断开）
        min_gap = int(0.35 * Z)
        runs = []
        st = None
        gap = 0
        for i, v in enumerate(colhas):
            if v:
                if st is None:
                    st = i
                gap = 0
            else:
                if st is not None:
                    gap += 1
                    if gap > min_gap:
                        runs.append([st, i - gap + 1])
                        st = None
        if st is not None:
            runs.append([st, len(colhas)])

        chars = meta["chars"]
        if len(runs) != len(chars):
            # 段数对不上就退回按等分格切（这一步只为造真值，宁可保守）
            out.append([{"c": ch["c"], "ink_abs": None} for ch in chars])
            continue
        row = []
        for ch, (a, b) in zip(chars, runs):
            sub = mask[:, a:b]
            ys = np.where(sub.any(axis=1))[0]
            xs = np.where(sub.any(axis=0))[0]
            if len(ys) == 0:
                row.append({"c": ch["c"], "ink_abs": None})
                continue
            row.append({
                "c": ch["c"],
                "ink_abs": (clip.x0 + (a + xs.min()) / Z,
                            clip.y0 + ys.min() / Z,
                            clip.x0 + (a + xs.max() + 1) / Z,
                            clip.y0 + (ys.max() + 1) / Z),
            })
        out.append(row)
    d.close()
    return out


def build_text_pdf(path: Path) -> dict:
    """用已知字体在已知坐标写字，同时记录每个字的精确包围盒（真值）。"""
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    font = pymupdf.Font(fontfile=str(FONT_SONG))
    truth = {"ppp": RASTER_PPP, "lines": []}

    for text, size, baseline_y, tilt_deg in TEXT_LINES:
        x = 80.0
        morph = None
        if tilt_deg:
            morph = (pymupdf.Point(x, baseline_y), pymupdf.Matrix(tilt_deg))
        page.insert_text(
            (x, baseline_y), text, fontname="F0", fontfile=str(FONT_SONG),
            fontsize=size, color=(0, 0, 0), morph=morph,
        )
        chars = []
        cx = x
        for ch in text:
            adv = font.text_length(ch, fontsize=size)
            if tilt_deg:
                # 倾斜后的位置：绕锚点旋转
                import math
                rad = math.radians(tilt_deg)
                dx, dy = cx - x, 0.0
                rx = x + dx * math.cos(rad) - dy * math.sin(rad)
                ry = baseline_y + dx * math.sin(rad) + dy * math.cos(rad)
                dx2, dy2 = cx + adv - x, 0.0
                rx2 = x + dx2 * math.cos(rad) - dy2 * math.sin(rad)
                ry2 = baseline_y + dx2 * math.sin(rad) + dy2 * math.cos(rad)
                ink = glyph_ink_box(ch, size, FONT_SONG)
                chars.append({"c": ch, "x0": min(rx, rx2), "x1": max(rx, rx2),
                              "baseline": (ry + ry2) / 2, "ink": ink})
            else:
                ink = glyph_ink_box(ch, size, FONT_SONG)
                chars.append({"c": ch, "x0": cx, "x1": cx + adv,
                              "baseline": baseline_y, "ink": ink})
            cx += adv
        truth["lines"].append({
            "text": text, "size": size, "baseline_y": baseline_y,
            "tilt": tilt_deg, "chars": chars,
        })

    doc.save(str(path), garbage=4, deflate=True)
    doc.close()
    return truth


def rasterize(src: Path, dst: Path, ppp: float) -> None:
    """把文字 PDF 光栅化成低分辨率位图 PDF —— 模拟扫描件。"""
    d = pymupdf.open(str(src))
    pix = d[0].get_pixmap(matrix=pymupdf.Matrix(ppp, ppp), alpha=False)
    d.close()

    out = pymupdf.open()
    page = out.new_page(width=W, height=H)
    img_path = dst.with_suffix(".png")
    pix.save(str(img_path))
    page.insert_image(pymupdf.Rect(0, 0, W, H), filename=str(img_path))
    out.save(str(dst), garbage=4, deflate=True)
    out.close()
    img_path.unlink(missing_ok=True)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    text_pdf = FIXTURES / "_truth_text.pdf"
    raster_pdf = FIXTURES / "truth_sample.pdf"

    truth = build_text_pdf(text_pdf)
    measured = measure_truth_from_render(text_pdf, truth["lines"])
    for ln, rows in zip(truth["lines"], measured):
        for ch, m in zip(ln["chars"], rows):
            ch["ink_abs"] = m["ink_abs"]
    rasterize(text_pdf, raster_pdf, RASTER_PPP)

    truth["source"] = str(raster_pdf.name)
    (FIXTURES / "truth_sample.truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"生成 {raster_pdf}  ({RASTER_PPP} 像素/pt)")
    print(f"真值 {FIXTURES / 'truth_sample.truth.json'}")
    for ln in truth["lines"]:
        print(f"  y={ln['baseline_y']:6.1f} {ln['size']:5.1f}pt 倾斜{ln['tilt']:4.1f}°  "
              f"{len(ln['chars'])} 字  {ln['text'][:24]!r}")
    text_pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
