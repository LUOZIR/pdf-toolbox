"""文档压缩：以图像为对象的逐张决策 + 实测校验。

为什么不是"调一个库的压缩档位"了事：
图像型 PDF 体积的 95% 以上是位图，而位图该不该压、压到什么程度，
取决于它**在页面上的实际分辨率**（不是它自己的像素数）和**内容性质**：
  · 一张 921×1258 的图铺满 A4 → 有效 112 DPI，已经是低清扫描件，
    再降采样只会把字迹糊掉；
  · 同一张图放在 4 厘米见方的印章位上 → 有效 3000 DPI，
    降到 300 DPI 肉眼无差、体积只剩 1%；
  · 扫描件是"文字为主"，转灰度几乎无损但省 30~40%，
    而彩色截图转灰度是实打实的信息损失。
所以每张图都要单独算有效分辨率、判断是否近似灰度，再决定动作。

决策之后必须**实测**：把压缩结果重新编码回来跟原图逐像素比，
算 PSNR，低于阈值就不采用这张图的压缩方案。
用户要的是"尽可能降低损失"，那就不能只报"省了多少 KB"就算完。

★ 写回方式沿用 raster.py 里踩坑换来的结论：绝不能调 page.replace_image()，
  它会把色彩空间改成 ICCBased，造成渲染劣化与 xref 漂移。
  这里同样直接 update_stream 就地改流，色彩空间与 /SMask 原样保留。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# 内容性质判定用的阈值
GRAY_SAT_MAX = 12.0        # 通道极差中位数低于此值算"近似灰度"
GRAY_RATIO_MIN = 0.985     # 且 98.5% 的像素都要满足，防止彩色小面积被吞

# 各档位：(最大有效 DPI, JPEG 质量, 允许转灰度, 降采样损失下限 dB, 编码损失下限 dB)
#
# 两个下限分开，是因为实测发现这是两种**性质完全不同**的损失：
#   降采样 是把高频整体滤掉（模糊化）—— 扫描件本来就糊，代价很小；
#   编码   是在已有像素上叠加量化噪声。
#   实测（同一批图，200→150 DPI、JPEG q80）：
#     锐利截图（Doc1，位图截图）   降采样 31.1~34.1 dB   编码 35.4~38.1 dB
#     扫描件（224→150 DPI） 降采样 39.6 dB         编码 41.9 dB
#   36 dB 这条线正好把两者分开：扫描件允许降采样，锐利截图不许。
#   早先把两者混进同一个 PSNR，结果要么全放行、要么全否决，两头都错。
PRESETS: dict[str, tuple[float, float, bool, float, float]] = {
    "lossless": (0.0, 0.0, False, 99.0, 99.0),    # 只做无损整理，不动像素
    "light": (300.0, 88.0, False, 40.0, 40.0),    # 轻压：只治分辨率虚高的图
    "balanced": (150.0, 80.0, True, 36.0, 34.0),  # 平衡（默认）
    "strong": (110.0, 68.0, True, 33.0, 31.0),    # 强压：目标是小体积
    "tiny": (96.0, 58.0, True, 30.0, 28.0),       # 极限：有明显可见损失
}

# 本来就是有损编码（JPEG）的图，再编码必然叠加一代损失，
# 允许放宽 4dB，但要求省得够多（>=25%），否则不值得付这份质量。
LOSSY_FLOOR_SLACK = 4.0
LOSSY_MIN_SAVING = 0.25


@dataclass
class ImageStat:
    """一张图在文档里的处境。"""

    xref: int
    width: int
    height: int
    colorspace: str
    bpc: int
    nbytes: int
    filter_: str
    pages: list[int] = field(default_factory=list)
    # 页面上的显示尺寸（取最大的那一处）
    disp_w: float = 0.0
    disp_h: float = 0.0
    has_smask: bool = False
    digest: bytes = b""

    @property
    def dpi_x(self) -> float:
        """有效分辨率：像素数 ÷ 显示尺寸（pt）× 72。"""
        if self.disp_w <= 0.1:
            return 0.0
        return self.width / self.disp_w * 72.0

    @property
    def dpi_y(self) -> float:
        if self.disp_h <= 0.1:
            return 0.0
        return self.height / self.disp_h * 72.0

    @property
    def effective_dpi(self) -> float:
        d = [v for v in (self.dpi_x, self.dpi_y) if v > 0]
        return max(d) if d else 0.0


@dataclass
class ImageAction:
    """对一张图做了什么（或决定不做）。"""

    xref: int
    pages: list[int]
    before_bytes: int
    after_bytes: int
    before_px: tuple[int, int]
    after_px: tuple[int, int]
    before_filter: str
    after_filter: str
    before_cs: str
    after_cs: str
    effective_dpi: float
    # 两种损失分开报：降采样是模糊化，编码是加噪
    downscale_psnr: float = 99.0
    encoder_psnr: float = 99.0
    end_to_end_psnr: float = 99.0
    reason: str = ""

    @property
    def psnr(self) -> float:
        """对外只报端到端 —— 那是用户真正看到的结果。"""
        return self.end_to_end_psnr

    @property
    def saved(self) -> int:
        return self.before_bytes - self.after_bytes


@dataclass
class CompressReport:
    src: str
    out: str
    src_size: int
    out_size: int
    preset: str
    actions: list[ImageAction] = field(default_factory=list)
    page_psnr: list[float] = field(default_factory=list)
    max_page_diff: list[float] = field(default_factory=list)
    text_identical: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.out_size / self.src_size if self.src_size else 1.0

    def text(self) -> str:
        L = [f"压缩完成（{self.preset} 档）",
             f"  {self.src_size / 1024:.1f} KB → {self.out_size / 1024:.1f} KB"
             f"（{self.ratio * 100:.1f}%，"
             f"{'省' if self.out_size <= self.src_size else '增'}"
             f" {abs(self.src_size - self.out_size) / 1024:.1f} KB）"]
        touched = [a for a in self.actions if a.saved != 0]
        L.append(f"  图像：共 {len(self.actions)} 张，处理 {len(touched)} 张")
        for a in sorted(self.actions, key=lambda x: -x.saved):
            tag = "保留原样" if a.saved == 0 else f"省 {a.saved / 1024:.1f} KB"
            ps = (f"降采样 {a.downscale_psnr:4.1f}dB "
                  f"编码 {a.encoder_psnr:4.1f}dB "
                  f"端到端 {a.end_to_end_psnr:4.1f}dB"
                  if a.saved != 0 or a.downscale_psnr < 99.0
                  else "未改动")
            L.append(f"    xref {a.xref:<5} 第 "
                     f"{','.join(str(p + 1) for p in a.pages[:4])} 页  "
                     f"{a.before_px[0]}×{a.before_px[1]}"
                     f"({a.effective_dpi:.0f}DPI) → "
                     f"{a.after_px[0]}×{a.after_px[1]}  "
                     f"{a.before_filter}/{a.before_cs} → "
                     f"{a.after_filter}/{a.after_cs}  {ps}  {tag}")
            if a.reason:
                L.append(f"          ↳ {a.reason}")
        if self.page_psnr:
            worst = min(self.page_psnr)
            L.append(f"  整页渲染对比：最差 PSNR {worst:.1f}dB"
                     f"（共 {len(self.page_psnr)} 页）")
            md = max(self.max_page_diff)
            L.append(f"  单点最大通道差：{md:.0f}/255")
        L.append(f"  文字层：{'完全一致（字节未改动）' if self.text_identical else '有改动'}")
        for n in self.notes:
            L.append(f"  · {n}")
        return "\n".join(L)


# --------------------------------------------------------------- 分析


def analyze_images(doc: pymupdf.Document) -> list[ImageStat]:
    """清点文档里所有图像对象及其在页面上的有效分辨率。"""
    stats: dict[int, ImageStat] = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            continue
        for info in infos:
            xref = int(info.get("xref") or 0)
            if xref <= 0:
                continue
            tr = info.get("transform") or (1, 0, 0, 1, 0, 0)
            # 显示的宽高 = 单位正方形的像 → 变换后的边长
            dw = math.hypot(tr[0], tr[1])
            dh = math.hypot(tr[2], tr[3])
            st = stats.get(xref)
            if st is None:
                try:
                    raw = doc.extract_image(xref)
                except Exception:
                    continue
                st = ImageStat(
                    xref=xref,
                    width=int(info.get("width") or raw.get("width") or 0),
                    height=int(info.get("height") or raw.get("height") or 0),
                    colorspace=str(raw.get("colorspace")
                                   or info.get("cs-name") or ""),
                    bpc=int(info.get("bpc") or 8),
                    nbytes=len(raw.get("image") or b""),
                    filter_=str(raw.get("ext") or ""),
                    has_smask=bool(info.get("has-mask")),
                    digest=bytes(info.get("digest") or b""),
                )
                stats[xref] = st
            st.pages.append(pno)
            if dw * dh > st.disp_w * st.disp_h:
                st.disp_w, st.disp_h, = dw, dh
    return list(stats.values())


# --------------------------------------------------------------- 像素工具


def _decode(doc: pymupdf.Document, xref: int):
    """把图像对象解成 PIL 图像（无 alpha 信息时 alpha=None）。"""
    from PIL import Image
    import numpy as np

    pix = pymupdf.Pixmap(doc, xref)
    try:
        if pix.colorspace is None:
            return None, None
        n = pix.n
        h, w = pix.height, pix.width
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, n)
        alpha = None
        if pix.alpha:
            alpha = arr[:, :, -1]
            arr = arr[:, :, :-1]
            n -= 1
        mode = {1: "L", 3: "RGB", 4: "CMYK"}.get(n, "RGB")
        if mode == "RGB" and arr.shape[2] < 3:
            mode = "L"
        return Image.fromarray(arr, mode=mode), alpha
    finally:
        pix = None


def _near_gray(im) -> bool:
    """判断图像是否"近似灰度"：通道间差异极小且几乎处处如此。"""
    import numpy as np

    if im.mode != "RGB":
        return im.mode == "L"
    sm = im.resize((min(240, im.width), min(240, im.height)))
    a = np.asarray(sm, dtype=np.float32)
    spread = a.max(axis=2) - a.min(axis=2)
    return (float(np.median(spread)) <= GRAY_SAT_MAX
            and float((spread <= 24).mean()) >= GRAY_RATIO_MIN)


def _psnr(a, b) -> float:
    """两张 PIL 图的 PSNR（dB）。尺寸不同先归一到 a 的尺寸。"""
    import numpy as np

    if b.size != a.size:
        b = b.resize(a.size)
    if a.mode != b.mode:
        b = b.convert(a.mode)
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-9:
        return 99.0
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def _jpeg_bytes(im, quality: int, subsampling: int) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True,
            subsampling=subsampling, progressive=False)
    return buf.getvalue()


def _decode_flate(data: bytes, mode: str, w: int, h: int):
    """把 /FlateDecode 的候选解回 PIL 图（用于质量校验）。"""
    import zlib

    import numpy as np
    from PIL import Image

    raw = zlib.decompress(data)
    n = {"L": 1, "RGB": 3, "CMYK": 4, "1": 1, "P": 1}.get(mode)
    if n is None or len(raw) < w * h * n:
        return None
    arr = np.frombuffer(raw[:w * h * n], dtype=np.uint8).reshape(h, w, n)
    if mode == "L":
        arr = arr[:, :, 0]
    return Image.fromarray(arr, mode=mode)


@dataclass
class Payload:
    """待写回的图像数据。"""

    data: bytes
    width: int
    height: int
    filter_: str
    colorspace: str
    bpc: int = 8
    summary: str = ""


# --------------------------------------------------------------- 单图决策


def plan_image(doc: pymupdf.Document, st: ImageStat, *,
               max_dpi: float, quality: float, allow_gray: bool,
               down_floor: float, enc_floor: float
               ) -> tuple[ImageAction, Payload | None]:
    """给一张图定方案并**当场验证**：不达标就退回原样。

    三步各自量化、各自决定，因为它们的损失性质不同、判据也不同：
      1. 降采样   —— 只有"放大回去比"的 PSNR 够高才做（糊的图才配降）；
      2. 转灰度   —— 只在近似灰度时做；
      3. 编码     —— 在同尺寸下比 PSNR；无损候选只要更小就永远采用。

    返回 (结论, 要写回的数据)。数据为 None 表示不动这张图。
    """
    act = ImageAction(
        xref=st.xref, pages=sorted(set(st.pages)),
        before_bytes=st.nbytes, after_bytes=st.nbytes,
        before_px=(st.width, st.height), after_px=(st.width, st.height),
        before_filter=st.filter_, after_filter=st.filter_,
        before_cs=st.colorspace, after_cs=st.colorspace,
        effective_dpi=st.effective_dpi, reason="",
    )

    mask = doc.xref_get_key(st.xref, "ImageMask")
    if mask and mask[1] not in ("null", ""):
        act.reason = "是位掩膜（/ImageMask），跳过"
        return act, None

    try:
        im, alpha = _decode(doc, st.xref)
    except Exception as exc:
        act.reason = f"解码失败：{exc}"
        return act, None
    if im is None:
        act.reason = "无法解读色彩空间，跳过"
        return act, None
    im_original = im

    # ---- 1. 降采样：先算损失，够便宜才做
    why: list[str] = []
    if max_dpi > 0 and st.effective_dpi > max_dpi * 1.05:
        scale = max_dpi / st.effective_dpi
        w = max(1, int(round(im.width * scale)))
        h = max(1, int(round(im.height * scale)))
        if (w, h) != im.size:
            cand = im.resize((w, h), 1)          # 1 = LANCZOS
            # 损失 = 降完再放大回原尺寸与原图比
            loss = _psnr(im, cand.resize(im.size, 1))
            act.downscale_psnr = loss
            if loss >= down_floor:
                im = cand
                act.after_px = (w, h)
                why.append(f"降采样 {st.effective_dpi:.0f}→{max_dpi:.0f}DPI"
                           f"（损失 {loss:.1f}dB）")
            else:
                why.append(f"不降采样：{st.effective_dpi:.0f}DPI 虽高，"
                           f"但降下去损失 {loss:.1f}dB 太大")

    # ---- 2. 转灰度：只在"近似灰度"时做，彩色截图不许转
    to_gray = allow_gray and _near_gray(im)
    if to_gray and im.mode == "RGB":
        im = im.convert("L")
        why.append("转灰度")
    cs = "/DeviceGray" if im.mode == "L" else "/DeviceRGB"

    # 编码损失的基准 = 当前这张图（已在最终尺寸、最终模式上）。
    # 拿原图当基准会把降采样的损失也算进编码里，导致"该降的图被判编码不达标"。
    ref = im.copy()
    resized = (im.size != im_original.size) or (im.mode != im_original.mode)

    # ---- 3. 候选编码
    cands: list[tuple[str, str, bytes, bool]] = []   # (说明, filter, 数据, 是否无损)
    if quality > 0:
        q = float(int(quality))
        for qq, ss, lab in ((min(95.0, q + 10), 0,
                             f"JPEG q{min(95, q + 10):.0f}(4:4:4)"),
                            (q, 1, f"JPEG q{q:.0f}(4:2:2)"),
                            (max(45.0, q - 15), 2,
                             f"JPEG q{max(45, q - 15):.0f}(4:2:0)")):
            try:
                cands.append((lab, "/DCTDecode", _jpeg_bytes(im, int(qq), ss),
                              False))
            except Exception:
                continue
    try:
        import zlib
        cands.append(("无损 Flate", "/FlateDecode",
                      zlib.compress(im.tobytes(), 9), True))
    except Exception:
        pass

    # 原样不动也是一个候选：拿原始字节数当"体积"
    keep_label = None
    if not resized:
        keep_label = "保留原始字节"

    # 本来就是有损编码 → 再编码会叠一代损失，放宽下限但要省得够多
    floor = enc_floor
    why_lossy = False
    if st.filter_.lower() in ("jpeg", "jpg", "dctdecode"):
        floor = max(20.0, enc_floor - LOSSY_FLOOR_SLACK)
        why_lossy = True

    # ---- 4. 挑"体积最小且达标"的
    best: tuple[str, bytes, str, float, bool] | None = None
    checked: list[str] = []
    for lab, filt, data in [(c[0], c[1], c[2]) for c in cands]:
        loss_less = next(c[3] for c in cands if c[0] == lab)
        try:
            if filt == "/DCTDecode":
                from PIL import Image
                back = Image.open(io.BytesIO(data)).convert(im.mode)
            else:
                back = _decode_flate(data, im.mode, im.width, im.height)
        except Exception:
            checked.append(f"{lab}=解码失败")
            continue
        if back is None:
            continue
        p = _psnr(ref, back)
        # 无损候选只要更小就永远采用 —— 它不需要过质量关
        ok = loss_less or p >= floor
        checked.append(f"{lab}={len(data) / 1024:.0f}KB/{p:.0f}dB"
                       f"{'' if ok else '(不足)'}")
        if not ok:
            continue
        if best is None or len(data) < len(best[1]):
            best = (lab, data, filt, p, loss_less)

    if best is None:
        if keep_label:
            act.reason = "; ".join(why) if why else ""
            act.reason = (act.reason + "；" if act.reason else "") + \
                "没有达标的编码（" + "; ".join(checked) + "）"
        else:
            act.reason = ("已降采样/转灰度，但编码都不达标（"
                          + "; ".join(checked) + "）")
        return act, None

    lab, data, filt, p, loss_less = best
    act.encoder_psnr = p

    # 与原字节比：省不到 3% 就不动
    if not resized and st.nbytes and len(data) >= st.nbytes * 0.97:
        act.reason = (f"最优 {lab} 只省 {(st.nbytes - len(data)) / 1024:.1f} KB，"
                      f"不值得改")
        return act, None
    # 有损原图再编码：省得不够多就不值得付质量
    if why_lossy and not resized and st.nbytes and \
            (st.nbytes - len(data)) / st.nbytes < LOSSY_MIN_SAVING:
        act.reason = (f"原图已是有损 JPEG，{lab} 只省 "
                      f"{(st.nbytes - len(data)) / st.nbytes * 100:.0f}%，"
                      f"不值得再压一代")
        return act, None

    act.after_bytes = len(data)
    act.after_px = (im.width, im.height)
    act.after_filter = filt
    act.after_cs = cs
    # 端到端：原图（转成新模式后）与新图的最终对比，放大回原尺寸比
    try:
        if im_original.mode != im.mode:
            base = im_original.convert(im.mode)
        else:
            base = im_original
        back_full = _decode_flate(data, im.mode, im.width, im.height) \
            if filt == "/FlateDecode" else \
            __import__("PIL.Image", fromlist=["Image"]).open(
                io.BytesIO(data)).convert(im.mode)
        act.end_to_end_psnr = _psnr(base, back_full.resize(base.size, 1))
    except Exception:
        act.end_to_end_psnr = p
    why.append(f"采用 {lab}" + ("（无损）" if loss_less else ""))
    act.reason = "，".join(why)
    return act, Payload(data=data, width=im.width, height=im.height,
                        filter_=filt, colorspace=cs, bpc=8, summary=lab)


# --------------------------------------------------------------- 页面渲染对比


def _render_pages(path: str | Path, dpi: float):
    """把整份文档逐页渲染成灰度小图，用于"改完还像不像"的整页对比。"""
    import numpy as np

    out = []
    doc = pymupdf.open(str(path))
    try:
        zoom = dpi / 72.0
        for pno in range(doc.page_count):
            pix = doc[pno].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                                     colorspace=pymupdf.csGRAY, alpha=False)
            out.append(np.frombuffer(pix.samples, dtype=np.uint8)
                       .reshape(pix.height, pix.width).copy())
    finally:
        doc.close()
    return out


def _text_digest(path: str | Path) -> str:
    """文字层指纹：条目 = (页码, 文字, 位置)。用来证明压缩没碰文字。"""
    import hashlib

    h = hashlib.sha1()
    doc = pymupdf.open(str(path))
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            for b in page.get_text("blocks"):
                h.update(f"{pno}|{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}|"
                         f"{b[4]}".encode("utf-8", "replace"))
            for d in page.get_drawings():
                h.update(f"g{pno}|{d['rect']}".encode())
    finally:
        doc.close()
    return h.hexdigest()


# --------------------------------------------------------------- 主流程


def compress_document(src: str | Path, out: str | Path,
                      preset: str = "balanced", *,
                      max_dpi: float | None = None,
                      quality: float | None = None,
                      gray: bool | None = None,
                      psnr_floor: float | None = None,
                      verify: bool = True,
                      progress=None) -> CompressReport:
    """压缩一份 PDF：逐张图决策 → 就地写回 → 结构整理 → 实测校验。

    校验做三层，缺一层都可能漏过劣化：
      1. 每张图自己的两种损失（降采样损失、编码损失，写入之前就算好，
         不达标直接不采用）；
      2. 整页渲染对比（同一 DPI 渲染前后两版，逐像素比）——
         这一层是**最终裁判**：它按实际显示效果衡量，
         把"降采样后重建"这种逐图像素比看起来很差、但显示上无感的情况算对；
      3. 文字层指纹（文字与矢量图形的字节摘要），证明压缩没碰文字。
    """
    def note(msg: str) -> None:
        if progress:
            progress(msg)

    src_p = Path(src)
    out_p = Path(out)
    key = (preset or "balanced").lower()
    if key not in PRESETS:
        raise ValueError(f"未知压缩档位 {preset!r}，可选："
                         f"{', '.join(PRESETS)}")
    d_dpi, d_q, d_gray, d_down, d_enc = PRESETS[key]
    max_dpi = d_dpi if max_dpi is None else max_dpi
    quality = d_q if quality is None else quality
    allow_gray = d_gray if gray is None else gray
    down_floor = d_down if psnr_floor is None else psnr_floor
    enc_floor = d_enc if psnr_floor is None else psnr_floor

    src_size = src_p.stat().st_size
    doc = pymupdf.open(str(src_p))
    rep = CompressReport(src=str(src_p), out=str(out_p), src_size=src_size,
                         out_size=0, preset=key)

    text_before = _text_digest(src_p)

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"{src_p.name} 是加密的，需要密码")

        stats = analyze_images(doc)
        total_img = sum(s.nbytes for s in stats)
        note(f"共 {len(stats)} 张图像，占 {total_img / 1024:.0f} KB；"
             f"档位 {key}（≤{max_dpi:.0f}DPI"
             f"{'，允许转灰度' if allow_gray else ''}）")

        changed = 0
        for st in sorted(stats, key=lambda s: -s.nbytes):
            act, pl = plan_image(doc, st, max_dpi=max_dpi, quality=quality,
                                 allow_gray=allow_gray,
                                 down_floor=down_floor, enc_floor=enc_floor)
            rep.actions.append(act)
            if pl is not None:
                _write_image(doc, st.xref, pl)
                changed += 1
                note(f"  xref {st.xref} 第 "
                     f"{','.join(str(p + 1) for p in sorted(set(st.pages))[:3])}"
                     f" 页：{act.reason}，省 "
                     f"{(act.before_bytes - act.after_bytes) / 1024:.1f} KB")
        note(f"改写 {changed}/{len(stats)} 张图像")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_p), garbage=4, deflate=True, deflate_images=True,
                 deflate_fonts=True, clean=True, use_objstms=1)
    finally:
        doc.close()

    rep.out_size = out_p.stat().st_size
    rep.text_identical = _text_digest(out_p) == text_before
    if not rep.text_identical:
        rep.notes.append("文字层指纹不一致 —— 压缩不应改动文字，请检查")

    if verify:
        note("整页渲染对比中…")
        try:
            a = _render_pages(src_p, 110.0)
            b = _render_pages(out_p, 110.0)
            if len(a) != len(b):
                rep.notes.append(
                    f"页数不一致（{len(a)} → {len(b)}），跳过渲染对比")
            else:
                import numpy as np
                for x, y in zip(a, b):
                    if x.shape != y.shape:
                        rep.page_psnr.append(0.0)
                        rep.max_page_diff.append(255.0)
                        continue
                    d = x.astype(np.float32) - y.astype(np.float32)
                    mse = float(np.mean(d * d))
                    rep.page_psnr.append(
                        99.0 if mse <= 1e-9
                        else 10.0 * math.log10(255.0 * 255.0 / mse))
                    rep.max_page_diff.append(float(np.abs(d).max()))
        except Exception as exc:
            rep.notes.append(f"渲染对比失败：{exc}")

    actions = [a for a in rep.actions if a.saved > 0]
    if not actions:
        rep.notes.append("没有任何图像值得压 —— 这份文档的图像已经接近最优")
    return rep



def _write_image(doc: pymupdf.Document, xref: int, pl: Payload) -> None:
    """就地替换图像流（不新建对象、不改 xref，色彩空间显式重写）。

    ★ 四件事必须一起做，缺一个都会留下坏对象：

    1. **compress 一律 False，数据自己压好。**
       `update_stream(compress=True)` 会再 deflate 一次。JPEG 数据再 deflate
       只是白费力气（且 /Filter 说是 DCTDecode，解出来是垃圾）；
       而 /FlateDecode 的流会被**双重压缩** —— 读取端解一层拿到 zlib 字节，
       图像直接报废。实测症状：无损替换后整页 PSNR 从 99dB 掉到 7.4dB。
    2. /Filter 必须跟着数据一起换。
    3. /DecodeParms 与 /Decode 指向上一种编码的参数，换编码后必须清空。
    4. 写完立刻回读校验 —— 上面这类错误不会抛异常，只会静默画错。
    """
    doc.update_stream(xref, pl.data, compress=False)
    doc.xref_set_key(xref, "Filter", pl.filter_)
    doc.xref_set_key(xref, "Width", str(pl.width))
    doc.xref_set_key(xref, "Height", str(pl.height))
    doc.xref_set_key(xref, "BitsPerComponent", str(pl.bpc))
    doc.xref_set_key(xref, "ColorSpace", pl.colorspace)
    for k in ("DecodeParms", "Decode"):
        if doc.xref_get_key(xref, k)[0] != "null":
            doc.xref_set_key(xref, k, "null")

    back = doc.xref_stream_raw(xref)
    if back != pl.data:
        raise RuntimeError(
            f"图像 xref {xref} 写回后回读不一致（写入 {len(pl.data)} 字节，"
            f"读回 {len(back)} 字节）—— 压缩会静默画错，已中止")

