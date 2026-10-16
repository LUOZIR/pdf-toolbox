"""扫描件模板匹配：用参考字形在低分辨率扫描件上定位字符。

这是**独立于切分**的一种测量手段：它不假设"怎么把墨迹切成 N 份"，
而是直接问"这个字放在哪里最像"。所以它既能用来定位，也能用来
**检验**切分结果 —— 切分是被检验的对象，不能同时当参照。

精度已用真值样本标定（tools/locate_chars.py --calibrate）：
  横向误差 中位 0.30pt、95% 0.91pt、最大 1.54pt
  纵向误差 中位 0.33pt、95% 0.98pt、最大 2.00pt

低分辨率下仍然可行：112 DPI 时一个 10.5pt 的汉字约 16×16 个源像素、
几十个墨迹像素，二维灰度互相关用的是全部这些像素 ——
而"列剖面相关性"只用一维投影，实测两簇完全重叠、判不出东西（已废弃）。

★ 与 pdfbox.core.raster 的关系：本模块是独立的一份实现，
  raster 的切分流程**不依赖**它（试过用它替代切分，实测在真值样本上
  反而从 99% 退化到 87%）。它只服务于验证工具，
  这样"验证"与"被验证"才是两套独立的东西。
"""

from __future__ import annotations

import numpy as np
import pymupdf

_TEMPLATE_CACHE: dict = {}


def glyph_template(font_path: str, ch: str, size_pt: float, ppp: float):
    """渲染字形模板：返回 (去均值模板, 墨迹质心x, 墨迹质心y, 墨迹宽, 墨迹高)。

    ★ 三个坑，都是实测撞出来的：

    1. **模板必须裁到字形墨迹范围再加余量。** 整页渲染出来是 62×86 像素，
       而字形只占 16×16 —— 背景占 95% 面积，互相关被背景稀释，区分度近零。
    2. **质心要用原始墨迹算，不能用去均值后的数组。** `t -= t.mean()`
       之后总和**恰好为 0**，拿它当分母会得到 1.4e9 这种垃圾值，
       进而把整个搜索窗口算错、分数恒为 0。
    3. **按 4 倍渲染再降采样，不要直接按目标分辨率渲染。**
       1.55 倍下 14.6pt 的宋体笔画只有 1 个像素，反锯齿后灰度高于墨迹阈值，
       整个模板被判成"没有墨迹"、直接返回 None，匹配全部落空。
    """
    key = (font_path, ch, round(size_pt, 2), round(ppp, 3))
    hit = _TEMPLATE_CACHE.get(key)
    if hit is not None:
        return hit
    pad = max(1.0, size_pt * 0.6)
    try:
        d = pymupdf.open()
        page = d.new_page(width=size_pt * 3 + 2 * pad, height=size_pt * 3)
        page.insert_text((pad, size_pt * 2), ch, fontname="R",
                         fontfile=font_path, fontsize=size_pt)
        ss = 4
        pix = page.get_pixmap(matrix=pymupdf.Matrix(ppp * ss, ppp * ss),
                              alpha=False)
        arr_full = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3].mean(axis=2).astype(
                np.float32)
        d.close()
    except Exception:
        _TEMPLATE_CACHE[key] = (None, 0.0, 0.0, 0.0, 0.0)
        return _TEMPLATE_CACHE[key]
    h = (arr_full.shape[0] // ss) * ss
    w = (arr_full.shape[1] // ss) * ss
    arr = arr_full[:h, :w].reshape(h // ss, ss, w // ss, ss).mean(axis=(1, 3))

    raw = 255.0 - arr
    ys, xs = np.where(raw > 25.0)
    if xs.size == 0:
        _TEMPLATE_CACHE[key] = (None, 0.0, 0.0, 0.0, 0.0)
        return _TEMPLATE_CACHE[key]
    m = max(1, int(round(size_pt * 0.15 * ppp)))
    y0 = max(0, int(ys.min()) - m)
    y1 = min(raw.shape[0], int(ys.max()) + 1 + m)
    x0 = max(0, int(xs.min()) - m)
    x1 = min(raw.shape[1], int(xs.max()) + 1 + m)
    sub = raw[y0:y1, x0:x1]
    wgt = sub.copy()
    wgt[wgt < 20.0] = 0.0
    tot = float(wgt.sum())
    if tot <= 1e-6:
        _TEMPLATE_CACHE[key] = (None, 0.0, 0.0, 0.0, 0.0)
        return _TEMPLATE_CACHE[key]
    gy, gx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    cx = float((wgt * gx).sum() / tot)
    cy = float((wgt * gy).sum() / tot)
    ink_w = (int(xs.max()) - int(xs.min()) + 1) / ppp
    ink_h = (int(ys.max()) - int(ys.min()) + 1) / ppp
    _TEMPLATE_CACHE[key] = (sub - sub.mean(), cx, cy, ink_w, ink_h)
    return _TEMPLATE_CACHE[key]


def ncc_peak(img: np.ndarray, tmpl: np.ndarray, cx: float, cy: float,
             win: int):
    """在 (cx,cy) 附近 win 像素内找归一化互相关峰值。

    返回 (最佳 dx, dy, 分数)，坐标单位是像素。

    ★ 必须向量化：逐位置循环是 361 个位置 × 每次 400 次乘加，
      纯 Python 下每个字 0.1 秒、一页 500 字就是 50 秒。
      对每个 dy 取一整行窗口，用滑动窗口视图一次算出所有 dx 的相关，
      19 次迭代覆盖整个搜索窗。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    th, tw = tmpl.shape
    ih, iw = img.shape
    tn = float(np.sqrt((tmpl * tmpl).sum()))
    if tn < 1e-6 or th > ih or tw > iw:
        return 0.0, 0.0, 0.0
    tflat = tmpl.reshape(-1)
    best = (-2.0, 0, 0)
    cx0 = int(round(cx))
    for dy in range(-win, win + 1):
        y0 = int(round(cy)) + dy
        if y0 < 0 or y0 + th > ih:
            continue
        x0 = max(0, cx0 - win)
        x1 = min(iw, cx0 + win + tw)
        if x1 - x0 < tw:
            continue
        row = img[y0:y0 + th, x0:x1]
        wv = sliding_window_view(row, (th, tw))[0]
        flat = wv.reshape(wv.shape[0], -1)
        num = flat @ tflat
        den = np.sqrt((flat * flat).sum(axis=1)) * tn
        den[den < 1e-9] = 1e-9
        sc = num / den
        k = int(np.argmax(sc))
        if sc[k] > best[0]:
            best = (float(sc[k]), x0 + k - cx0, dy)
    return float(best[1]), float(best[2]), best[0]


def scan_image(page: pymupdf.Page, rect, ppp: float):
    """取一块区域的灰度图（墨迹为正、去均值），供互相关使用。"""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ppp, ppp),
                          clip=pymupdf.Rect(*rect), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].mean(axis=2).astype(np.float32)
    img = 255.0 - arr
    img -= img.mean()
    return img


def extract_patch(page: pymupdf.Page, rect, ppp: float, pad_pt: float = 0.5):
    """从扫描件上截取一个字形的墨迹补丁，供自模板使用。

    返回 (去均值补丁, 墨迹质心x, 墨迹质心y, 墨迹宽, 墨迹高)；无墨迹时返回 None 系列。
    与 glyph_template 的返回格式一致，所以两者可以直接互换使用。

    ★ 自模板为什么比系统字体强：
    它来自**同一份扫描件**——同字体、同字号、同模糊程度、同笔画粗细。
    系统字体去凑时，参考字体与真实字体的度量差异（字宽比例、边距）
    和低分辨率下的形状差异都会变成定位误差；自模板把这两个误差源一起消掉。
    """
    import numpy as np

    x0, y0, x1, y1 = rect
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(ppp, ppp),
                              clip=pymupdf.Rect(x0 - pad_pt, y0 - pad_pt,
                                                x1 + pad_pt, y1 + pad_pt),
                              alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :3].mean(axis=2).astype(
                np.float32)
    except Exception:
        return None, 0.0, 0.0, 0.0, 0.0
    if arr.size == 0:
        return None, 0.0, 0.0, 0.0, 0.0
    raw = 255.0 - arr
    raw -= float(np.median(raw))          # 以背景为基准
    ys, xs = np.where(raw > 25.0)
    if xs.size == 0:
        return None, 0.0, 0.0, 0.0, 0.0
    y0i, y1i = int(ys.min()), int(ys.max()) + 1
    x0i, x1i = int(xs.min()), int(xs.max()) + 1
    sub = raw[y0i:y1i, x0i:x1i]
    wgt = sub.copy()
    wgt[wgt < 20.0] = 0.0
    tot = float(wgt.sum())
    if tot <= 1e-6:
        return None, 0.0, 0.0, 0.0, 0.0
    gy, gx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    cx = float((wgt * gx).sum() / tot)
    cy = float((wgt * gy).sum() / tot)
    return (sub - sub.mean(), cx, cy,
            sub.shape[1] / ppp, sub.shape[0] / ppp)


def rescale_template(tmpl, tcx: float, tcy: float, scale: float):
    """把模板按比例缩放（自模板可能取自另一个字号）。"""
    import numpy as np

    if abs(scale - 1.0) < 0.03:
        return tmpl, tcx, tcy
    h, w = tmpl.shape
    nh = max(3, int(round(h * scale)))
    nw = max(3, int(round(w * scale)))
    yi = np.clip((np.arange(nh) + 0.5) / scale - 0.5, 0, h - 1)
    xi = np.clip((np.arange(nw) + 0.5) / scale - 0.5, 0, w - 1)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    top = tmpl[y0][:, x0] * (1 - wx) + tmpl[y0][:, x1] * wx
    bot = tmpl[y1][:, x0] * (1 - wx) + tmpl[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    out = out - out.mean()
    return out, tcx * scale, tcy * scale


class GlyphBank:
    """从扫描件自身收集的字形模板库。

    只收录"已证明正确"的字的墨迹：那些切分走了**墨迹簇一对一**路径的行
    （游程数与字数相等、且逐字宽度与参考字形墨迹宽吻合），
    它们的框就是墨迹游程的边界 —— 精确，不需要任何字宽推算。
    """

    def __init__(self):
        self._by_char: dict = {}

    def add(self, ch: str, size_pt: float, patch) -> None:
        tmpl, cx, cy, w, h = patch
        if tmpl is None or w <= 0.4 or h <= 0.4:
            return
        cur = self._by_char.get(ch)
        # 同一个字多次出现时留"墨迹最多"的那份（笔画最清楚）
        score = float((tmpl + abs(tmpl.mean())).sum())
        if cur is None or score > cur[3]:
            self._by_char[ch] = (tmpl, cx, cy, score, size_pt)

    def get(self, ch: str, size_pt: float):
        """取模板并按字号缩放。返回 (补丁, 质心x, 质心y, 墨迹宽px, 墨迹高px) 或 None。"""
        rec = self._by_char.get(ch)
        if rec is None:
            return None
        tmpl, cx, cy, _score, tmpl_size = rec
        if tmpl_size <= 0.5 or size_pt <= 0.5:
            return None
        ratio = size_pt / tmpl_size
        if not (0.6 <= ratio <= 1.7):
            return None                    # 字号差太多，缩放不可靠
        t, tcx, tcy = rescale_template(tmpl, cx, cy, ratio)
        h, w = t.shape
        return t, tcx, tcy, float(w), float(h)

    def __len__(self) -> int:
        return len(self._by_char)

    def chars(self) -> str:
        return "".join(sorted(self._by_char))
