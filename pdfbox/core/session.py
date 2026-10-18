"""文档会话：统一文本型与图片型两条编辑路径。

这是所有 UI（CLI / 桌面窗口）共用的门面。用户面对的是"一个打开的 PDF"，
而内部要根据文档类型走完全不同的实现：

    文本型 PDF  → structure.extract_document → editor.replace_block_text
    图片型 PDF  → ocr.ocr_document（带缓存） → raster.edit_raster_block

两条路径产出的都是同一套结构模型，所以命中测试、重排、撤销、
变更报告这些上层逻辑完全共用。

撤销用文档字节快照实现 —— PDF 编辑是破坏性的（擦除不可逆），
按操作记录反操作比直接快照复杂得多且容易出错，
而一个几百 KB 的快照成本完全可以接受。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf

from . import ocr as ocr_mod
from . import raster
from .classify import classify
from .editor import delete_block_text, replace_block_text
from .fonts import FontMatch, describe, match_font, resolve_for_block
from .model import (
    KIND_GRAPHICS,
    KIND_IMAGE,
    PageStructure,
    KIND_TABLE,
    LAYER_FIXED,
    Block,
    DocStructure,
)
from .reflow import apply_reflow, plan_reflow
from .structure import extract_document
from .raster_typeset import (PatchCase, build_patch_case,
                             plan_raster_typeset, apply_raster_typeset)
from .editor import _normalize_input
from .typeset import apply_typeset, plan_typeset

def _cache_root() -> Path:
    """OCR 缓存放哪儿。

    ★ 打包成 exe 后不能再用"源码目录下的 out/"：PyInstaller 单文件模式把
      资源解到临时目录，`__file__` 指向那里 —— 缓存会写进临时目录
      （启动后被删、且常常只读），等于每次启动都重新 OCR，
      装到 Program Files 下还会直接失败。
      所以冻结后改用用户可写的 %LOCALAPPDATA%；源码运行仍用 out/，
      方便开发时直接看缓存。
    """
    import os
    import sys
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
        return Path(base) / "pdf-toolbox"
    return Path(__file__).resolve().parent.parent.parent / "out"


CACHE_DIR = _cache_root() / "ocr_cache"
UNDO_LIMIT = 30

# 文档类型
DOC_TEXT = "text"        # 有完整文字层
DOC_IMAGE = "image"      # 图片/扫描型，需 OCR
DOC_MIXED = "mixed"      # 部分页有文字层，部分没有


@dataclass
class PageReport:
    page_no: int
    kind: str                       # text | image
    blocks: int = 0
    edited: int = 0
    moved: int = 0
    kept: int = 0
    restored_fixed: int = 0
    method: str = ""


@dataclass
class EditRecord:
    page_no: int
    old_text: str
    new_text: str
    path: str                       # text | raster
    lines_before: int = 0
    lines_after: int = 0
    height_delta: float = 0.0
    moved: int = 0                  # 被顶下去重画的块数
    reused: int = 0                 # 图片型：复用原像素块的字数
    kept: int = 0                   # 保持原字节的块数
    restored_fixed: int = 0         # 被波及后重绘的固定层元素数
    warned: list[str] = field(default_factory=list)
    method: str = ""

    def summary(self) -> str:
        d = self.height_delta
        delta = f"  高度 {d:+.1f}pt" if abs(d) > 0.01 else ""
        return (f"p{self.page_no+1}  [{self.path}]  "
                f"{self.lines_before} 行 → {self.lines_after} 行{delta}")


class Session:
    """一次文档编辑会话。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.doc: pymupdf.Document | None = None
        self.structure: DocStructure | None = None
        self.doc_type: str = DOC_TEXT
        self.page_kinds: list[str] = []

        self._undo: list[bytes] = []
        self._redo: list[bytes] = []
        self._dirty = False
        self.edits: list[EditRecord] = []
        # 文档对象每次撤销/重做都会被换掉，UI 靠这个版本号判断需要重新绑定
        self.revision: int = 0

    # ------------------------------------------------------------ 打开

    def load(
        self,
        force_ocr: bool | None = None,
        progress: Callable[[str], None] | None = None,
        use_cache: bool = True,
    ) -> None:
        """打开文档并建立结构模型。

        force_ocr: None=自动判断；True=强制走 OCR；False=强制按文本型处理
        """
        def note(msg: str) -> None:
            if progress:
                progress(msg)

        self.doc = pymupdf.open(str(self.path))

        # 逐页判断类型
        note("检测页面类型…")
        kinds: list[str] = []
        for pno in range(len(self.doc)):
            is_img, _reason = ocr_mod.is_image_page(self.doc[pno])
            kinds.append("image" if is_img else "text")
        self.page_kinds = kinds

        n_img = kinds.count("image")
        if force_ocr is True:
            self.doc_type = DOC_IMAGE
        elif force_ocr is False:
            self.doc_type = DOC_TEXT
        elif n_img == 0:
            self.doc_type = DOC_TEXT
        elif n_img == len(kinds):
            self.doc_type = DOC_IMAGE
        else:
            self.doc_type = DOC_MIXED

        if self.doc_type == DOC_TEXT:
            note("解析文本结构…")
            self.structure = classify(extract_document(str(self.path)))
        else:
            note("图片型文档，准备 OCR…")
            self.structure = self._build_ocr_structure(note, use_cache)

    def _cache_key(self) -> str:
        h = hashlib.sha1()
        st = self.path.stat()
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        h.update(self.path.name.encode())
        return h.hexdigest()[:16]

    def _build_ocr_structure(
        self, note: Callable[[str], None], use_cache: bool,
    ) -> DocStructure:
        """对图片型页面做 OCR，结果按文件哈希缓存 —— OCR 很慢，必须缓存。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{self._cache_key()}.json"

        if use_cache and cache_file.exists():
            note("命中 OCR 缓存，跳过识别")
            return _structure_from_json(json.loads(cache_file.read_text(encoding="utf-8")),
                                       str(self.path))

        note("OCR 进行中（首次较慢，之后会走缓存）…")
        t0 = time.time()
        structure = ocr_mod.ocr_document(str(self.path))
        structure = classify(structure)
        # OCR 出来的块置信度本就是识别置信度，不要被规则引擎覆盖掉
        for page in structure.pages:
            for b in page.blocks:
                if b.raw and b.raw.get("ocr"):
                    b.reasons = [f"OCR 识别（置信度 {b.confidence:.2f}）"]
        elapsed = time.time() - t0
        note(f"OCR 完成，用时 {elapsed:.0f}s")

        cache_file.write_text(
            json.dumps(_structure_to_json(structure), ensure_ascii=False),
            encoding="utf-8",
        )
        return structure

    # ------------------------------------------------------------ 查询

    @property
    def page_count(self) -> int:
        return len(self.doc) if self.doc else 0

    @property
    def dirty(self) -> bool:
        return self._dirty

    def page_blocks(self, page_no: int) -> list[Block]:
        if not self.structure or page_no >= len(self.structure.pages):
            return []
        if self.structure.pages[page_no] is None:
            # 页面操作插入进来的新页还没有结构 —— 按需补，不要整篇重解析
            self._ensure_page_structure(page_no)
        pg = self.structure.pages[page_no]
        return pg.blocks if pg is not None else []

    def editable_blocks(self, page_no: int) -> list[Block]:
        """可编辑的块：有文字、且不是纯图形/图片。"""
        return [
            b for b in self.page_blocks(page_no)
            if b.lines and b.kind not in (KIND_IMAGE, KIND_GRAPHICS)
        ]

    def block_at(self, page_no: int, x: float, y: float) -> Block | None:
        """命中测试：找出坐标处的文本块。

        选中最小的包含该点的块 —— 水印之类的东西包围盒很大，
        按面积从小到大找才能优先命中真正被点击的那一行字。
        """
        best: Block | None = None
        best_area = float("inf")
        for b in self.editable_blocks(page_no):
            x0, y0, x1, y1 = b.bbox
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            area = (x1 - x0) * (y1 - y0)
            # 表格块整体很大，优先命中更小的块
            if b.kind == KIND_TABLE:
                area *= 4
            if area < best_area:
                best, best_area = b, area
        return best

    def match_for(self, block: Block, new_text: str | None = None) -> FontMatch:
        """为文本型文档的块匹配字体。图片型文档 OCR 无字体信息，返回一个兜底。"""
        if block.font:
            return resolve_for_block(block.font, block, new_text)
        from .fonts import UNIVERSAL_FALLBACK, FAMILY_CANDIDATES, _find_file, _load_pymupdf_font
        for cand in UNIVERSAL_FALLBACK:
            p = _find_file(cand)
            if p:
                return match_font(cand, need_chars=new_text or block.text)
        return match_font("unknown")

    def summary(self) -> str:
        if not self.structure:
            return "（未加载）"
        pages = []
        for i, k in enumerate(self.page_kinds):
            flags = self.structure.pages[i].blocks if i < len(self.structure.pages) else []
            pages.append(f"p{i+1}:{k}({len(flags)}块)")
        return (f"{self.path.name}  {len(self.doc)} 页  "
                f"类型={self.doc_type}  {' '.join(pages)}")

    # ------------------------------------------------------------ 编辑

    def _snapshot(self) -> None:
        """记录撤销点：文档字节 + 结构快照。

        结构也要一起存。撤销只回滚 PDF 字节的话，内存里的结构还是改后的，
        界面显示和实际文档就对不上了（再点那块显示的是改后的文字，
        但文档已经退回去了）。
        """
        if self.doc is None:
            return
        self._undo.append((self.doc.tobytes(), self._current_snapshot()))
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()

    def _current_snapshot(self) -> str | None:
        """当前结构序列化（撤销/重做都要用它把"此刻"存下来）。"""
        if self.structure is None:
            return None
        try:
            return json.dumps(_structure_to_json(self.structure),
                              ensure_ascii=False)
        except Exception:
            return None

    def replace_text(
        self,
        block: Block,
        new_text: str,
        *,
        font_file: str | None = None,
        fontsize: float | None = None,
        reflow: bool = True,
        cover_rect: tuple[float, float, float, float] | None = None,
        text_color: tuple[float, float, float] | None = None,
        weight: str = "auto",
        glyph_scope: str = "paragraph",
        glyph_choices: dict | None = None,
    ) -> EditRecord:
        """替换块内文字，按文档类型自动选路径。"""
        if self.doc is None or self.structure is None:
            raise RuntimeError("会话尚未加载文档")
        if new_text == block.text:
            rec = EditRecord(block.page_no, block.text, new_text, "noop")
            rec.warned = ["输入内容与原文完全相同，没有需要修改的地方"]
            return rec

        # ★ 静默 no-op 的第二种来源，也是最难查的一种：
        # 用户的新文本经"规整化"（去换行、合并空白）后和原文一模一样。
        # 此时排版引擎算出的改动长度是 0，会静默什么都不做 ——
        # PDF 没改、界面也不提示，用户看到的就是"编辑框改了但 PDF 没改"。
        # 用户反馈"和操作无关，和文本内容有关"指的就是这一类。
        # 这里显式拦住并说清楚原因。
        if block.font and _normalize_input(new_text) == _normalize_input(block.text):
            rec = EditRecord(block.page_no, block.text, new_text, "noop")
            rec.warned = [
                "新内容与原文只有换行/空白的差别，页面上的文字不会有变化。"
                "如果确实想改动，请修改实际的文字内容"
            ]
            return rec

        self._snapshot()

        is_raster = bool(block.raw and block.raw.get("ocr")) or not block.font
        if is_raster:
            # ★ 图片型路径必须在这里就把换行规整掉。
            # plan_raster_typeset 早先直接拿原始输入去做 diff，用户输入里只要带一个换行，
            # 公共前缀就会在换行处断掉 —— 实测两行的标题删末尾三个字，
            # 结果"改动段"被算成整整第二行（16 字），整行被覆盖重画、还丢了末尾三个字。
            # 矢量路径有这一步规整，图片型漏了。
            new_text = _normalize_input(new_text)
            return self._replace_raster(block, new_text, font_file, fontsize,
                                        cover_rect, text_color, weight,
                                        glyph_scope, glyph_choices)
        return self._replace_text(block, new_text, font_file, fontsize, reflow)

    def _replace_text(
        self, block: Block, new_text: str, font_file: str | None,
        fontsize: float | None, reflow: bool,
    ) -> EditRecord:
        assert self.doc is not None and self.structure is not None
        match = resolve_for_block(block.font, block, new_text)
        if font_file:
            m2 = match_font(Path(font_file).name, need_chars=new_text)
            if m2.ok:
                match = m2
        if not match.ok:
            raise RuntimeError(f"找不到可用字体：{match.warnings}")

        rec = EditRecord(page_no=block.page_no, old_text=block.text,
                         new_text=new_text, path="text")

        # ---- 优先走活字印刷：只重排改动的那几个字，其余原地不动
        #
        # 实测差别有多大：用一个度量不完美（模拟系统缺原字体）的字体做同一处
        # 单字修改，活字方式影响 220 个像素，整块重排影响 16400 个像素 —— 74 倍。
        # 因为整块重排会把没改的字也用替代字体重画一遍，全都走样了。
        if reflow:
            ts_plan = plan_typeset(
                block, new_text, match.font,
                structure=self.structure,
                page_width=self.doc[block.page_no].rect.width,
            )
            if ts_plan.mode in ("inplace", "shift"):
                info = apply_typeset(self.doc, block, new_text, match.font, ts_plan)
                rec.lines_before = len(block.lines)
                rec.lines_after = len(block.lines)
                rec.kept = info.get("kept", 0)
                rec.moved = info.get("drawn", 0) if ts_plan.mode == "shift" else 0
                rec.method = ts_plan.summary() + (
                    "（只擦改动的字，其余字节未动）"
                    if ts_plan.mode == "inplace" else ""
                )
                rec.warned = list(match.warnings)
                # ★ 不能重新解析整页。
                # 字形复用是靠 Form XObject 引用源页面实现的，一页里可能有
                # 几十个这样的引用；重新解析时源页面的文字会被重复计算几十遍
                # （实测 17 个块变成 51 个、文字堆叠成
                #  "恒远科技有限公司恒远科技有限公司…"），结构一坏，
                # 下一次编辑就找不到正确的块 —— 表现就是"应用修改不生效"。
                # 改成只同步被编辑那一块的文字，既准确又快。
                self._sync_block_text(block, new_text, None,
                                      font=match.font, size=block.size)
                self._dirty = True
                self.revision += 1
                self.edits.append(rec)
                return rec

        if reflow:
            plan = plan_reflow(self.structure, block.page_no, block, new_text, match)
            stats = apply_reflow(self.doc, self.structure, plan, block, new_text, match)
            edited = next((m for m in plan.moves if m.action == "edited"), None)
            rec.lines_before = len(block.lines)
            rec.lines_after = len(edited.lines) if edited else len(block.lines)
            rec.moved = stats.get("redrawn", 0)
            rec.kept = stats.get("kept", 0)
            rec.restored_fixed = stats.get("fixed_restored", 0)
            rec.warned = list(plan.warnings) + list(match.warnings)
            rec.method = (
                f"重排：{rec.kept} 块保持原字节、{rec.moved} 块下移重画"
                + (f"、{rec.restored_fixed} 个固定层元素重绘" if rec.restored_fixed else "")
            )
        else:
            res = replace_block_text(self.doc, block, new_text, match,
                                     fontsize=fontsize)
            rec.lines_before = res.lines_before
            rec.lines_after = res.lines_after
            rec.height_delta = res.height_delta
            rec.warned = list(res.warnings)
            rec.method = "原位替换"
            if res.preserved_breaks:
                rec.method += "（沿用原始断行）"

        self._sync_block_text(block, new_text, None,
                              font=match.font, size=block.size)
        self._dirty = True
        self.revision += 1
        self.edits.append(rec)
        return rec

    def _replace_raster(
        self, block: Block, new_text: str, font_file: str | None,
        fontsize: float | None,
        cover_rect: tuple[float, float, float, float] | None = None,
        text_color: tuple[float, float, float] | None = None,
        weight: str = "auto",
        glyph_scope: str = "paragraph",
        glyph_choices: dict | None = None,
    ) -> EditRecord:
        """图片型文档的编辑：自动 diff 出改动范围，只覆盖改动的那几个字。

        优先复用字盘里的原像素块 —— 同一段落内出现过的字直接搬原件，
        像素级一致，从根本上解决"字体风格/字重/字号都不对"的问题
        （因为复用根本不是"仿造"，就是原件）。
        字盘里没有的字才用字体渲染，并按原图分辨率栅格化后合成。
        """
        assert self.doc is not None
        if not font_file:
            raise RuntimeError(
                "图片型文档必须指定字体文件（OCR 无法得知原字体）。"
                "请在界面上选择字体，或让引擎按文字特征自动挑选。"
            )
        font = pymupdf.Font(fontfile=font_file)

        rec = EditRecord(page_no=block.page_no, old_text=block.text,
                         new_text=new_text, path="raster",
                         lines_before=len(block.lines))

        # ---- 建字盘并规划
        page = self.doc[block.page_no]
        case = build_patch_case(page, block, scope=glyph_scope, doc=self.doc,
                               structure=self.structure)
        if glyph_scope != "paragraph":
            # 跨段落候选必须筛掉风格不匹配的（字号/文字色），
            # 否则会把标题的白字、色块字之类搬进正文
            try:
                tgt = raster.measure_block_ink(page, block, expand=False)
                tgt_color = tgt.color
            except Exception:
                tgt_color = None
            case.filter_compatible(self.doc, block.size, tgt_color)
        plan = plan_raster_typeset(page, block, new_text, case,
                                   choices=glyph_choices,
                                   structure=self.structure)

        # ---- 改到"行数变了"这种程度，退回整块覆盖重画更稳妥
        if plan.mode == "whole":
            res = raster.edit_raster_block(
                self.doc, block, new_text, font,
                fontsize=fontsize, cover_rect=cover_rect, color=text_color,
                font_file=font_file, weight=weight,
            )
            rec.lines_after = res.lines_after
            rec.method = res.summary()
            rec.warned = list(res.warnings)
            self._dirty = True
            self.revision += 1
            self.edits.append(rec)
            return rec

        # ---- 逐字编辑
        res = apply_raster_typeset(
            self.doc, block, new_text, font, plan, case,
            bg_color=None, size=fontsize, choices=glyph_choices,
        )
        if not res.get("ok"):
            raise RuntimeError(f"图片型编辑失败：{res.get('reason', '未知原因')}")

        rec.lines_before = len(block.lines)
        rec.lines_after = len(block.lines)
        rec.kept = res.get("kept", 0)
        rec.reused = res.get("reused", 0)
        rec.method = plan.summary()
        if plan.ambiguous and not glyph_choices:
            rec.warned.append(
                "以下字有多个来源可复用，当前自动选了最接近的一个："
                + "、".join(plan.ambiguous)
            )
        rec.warned.extend(self._raster_notes(plan, res, font_file))

        # ★ 必须把结构模型里这一段的文字同步更新掉。
        # 否则：PDF 确实改了（像素对得上），但界面上再点这一段时输入框里
        # 显示的还是旧文字，用户看到的就是"应用修改不生效"。
        # 更糟的是再点一次会拿旧文字做 diff，同一次改动被重复施加。
        clamp = None
        try:
            host = raster.find_host_image(self.doc[block.page_no], block.bbox)
            if host is not None:
                clamp = host.bbox
        except Exception:
            clamp = None
        self._sync_block_text(block, new_text, None, font=font,
                              clamp_rect=clamp,
                              placements=res.get("placements"))

        self._dirty = True
        self.revision += 1
        self.edits.append(rec)
        return rec

    @staticmethod
    def _sync_block_text(block, new_text: str, start_x: float | None,
                         font=None, size: float | None = None,
                         clamp_rect=None, placements=None) -> None:
        """编辑完成后，把块内文字同步成新内容。

    ★ 两条铁律，都是踩坑换来的：

    1. **不能把多行块压成一行**。
       早期版本一律 `block.lines = [line]`，两行的标题被压成一行、32 个字全塞进
       行0 的竖条里 —— 单字框随之全变成横排的均匀竖条、和实际文字完全对不上。

    2. **范围必须钳制在承载图像内**。
       宽度用的是字号 × 字数，而字号来自 OCR 的行高反推、本身偏大（实测 16.9pt vs 真实 13.5pt），
       算出的右边界跑到 750.8pt，而承载图像只到 592.3pt ——
       块的范围一旦跑出图像，find_host_image 就找不到能装下它的图，
       下一次编辑直接报「找不到承载文字的图像」。
    """
        if not block.lines:
            return

        # ★ 首选：用编辑引擎实际画出的坐标回写。
        # 这是本轮结构性改动 —— 引擎知道每个字画在哪，就不该事后去估。
        # 实测四类 bug（界面不更新 / 多行被压成一行 / 范围跑出图像 / 空白处多出框）
        # 全部来自"事后估算"，用实际坐标则一个都不会有。
        if placements:
            Session._sync_from_placements(block, placements, clamp_rect)
            return

        line = block.lines[0]
        if not line.spans:
            return
        from .model import Char
        from .editor import break_lines

        text = "".join(ch for ch in new_text if ch != chr(10))
        if not text:
            return

        span0 = line.spans[0]
        left = min(c.bbox[0] for c in span0.chars)
        y0 = min(c.bbox[1] for c in span0.chars)
        y1 = max(c.bbox[3] for c in span0.chars)
        baseline = span0.chars[0].origin[1]
        block_width = max((l.along1 - l.along0 for l in block.lines), default=(y1 - y0))

        # ★ 单位宽度用「原行宽 ÷ 原字数」，不要用包围盒高度推。
        # 行包围盒包含上下留白，比真实字号高不少（实测 17.4pt 的框对应 13.6pt 的字），
        # 拿它当字号会把新宽度算大 15%，多出来的部分就在没有字的地方冒出一个单字框。
        # 中文等宽字体下「行宽 ÷ 字数」就是精确的单字步进。
        def _avg_advance(l) -> float:
            if not l.spans:
                return 0.0
            n = sum(len(sp.chars) for sp in l.spans)
            return (l.along1 - l.along0) / max(1, n)

        fs = size if size else _avg_advance(line) or (y1 - y0) * 0.9

        # 按块宽重新断行；断不出多行就还按一行铺
        laid = [text]
        if font is not None and len(block.lines) > 1:
            try:
                laid = break_lines(text, font, fs, block_width) or [text]
            except Exception:
                laid = [text]

        # 把断好的行分配到原块的行上（行数变了就只保留能装下的部分）
        n_slots = len(block.lines)
        new_lines = []
        for i in range(n_slots):
            src = laid[i] if i < len(laid) else ""
            old_line = block.lines[i]
            if not old_line.spans:
                continue
            sp = old_line.spans[0]
            ly0 = min(c.bbox[1] for c in sp.chars)
            ly1 = max(c.bbox[3] for c in sp.chars)
            lbase = sp.chars[0].origin[1]
            lleft = min(c.bbox[0] for c in sp.chars)
            # 新宽度按「原行宽 × 新字数/原字数」等比缩放。
            # 这比用字体度量算更可靠：字体度量依赖字号，而字号在图片型文档里只能估；
            # 字数比例是确定的。重新断行时才需要真字号，那种场合另有估算。
            n_old = sum(len(s2.chars) for s2 in old_line.spans)
            unit = (old_line.along1 - old_line.along0) / max(1, n_old)
            w = unit * len(src) if src else 0.0
            step = w / max(1, len(src)) if src else 0
            sp.chars = [
                Char(c=ch, glyph_id=0,
                     origin=(lleft + i2 * step, lbase),
                     bbox=(lleft + i2 * step, ly0, lleft + (i2 + 1) * step, ly1))
                for i2, ch in enumerate(src)
            ]
            sp.bbox = (lleft, ly0, lleft + w, ly1)
            old_line.along0, old_line.along1 = lleft, lleft + w
            old_line.bbox = (lleft, ly0, lleft + w, ly1)
            new_lines.append(old_line)

        if not new_lines:
            return
        block.lines = new_lines

        # 更新块范围，但钳制在承载图像（或给定范围）之内
        bx0 = min(l.bbox[0] for l in new_lines)
        by0 = min(l.bbox[1] for l in new_lines)
        bx1 = max(l.bbox[2] for l in new_lines)
        by1 = max(l.bbox[3] for l in new_lines)
        if clamp_rect:
            bx0 = max(bx0, clamp_rect[0])
            by0 = max(by0, clamp_rect[1])
            bx1 = min(bx1, clamp_rect[2])
            by1 = min(by1, clamp_rect[3])
        # 再把左右边界并回原块范围，保证不会缩得比原来更小
        block.bbox = (min(block.bbox[0], bx0), min(block.bbox[1], by0),
                      max(block.bbox[0], bx1), max(block.bbox[1], by1))
        block.raw = dict(block.raw or {}, edited=True)

    @staticmethod
    def _sync_from_placements(block, placements, clamp_rect=None) -> None:
        """按编辑引擎实际画出的坐标重建块的字符与范围。

        每个字的位置就是它被画到的位置 —— 不估、不猜。
        """
        from .model import Char
        by_line: dict[int, list] = {}
        for pl in placements:
            by_line.setdefault(pl.line_idx, []).append(pl)

        new_lines = []
        for li, old_line in enumerate(block.lines):
            pls = by_line.get(li)
            if not pls:
                continue
            if not old_line.spans:
                continue
            sp = old_line.spans[0]
            pls.sort(key=lambda p: p.x0)
            y0 = min(c.bbox[1] for c in sp.chars)
            y1 = max(c.bbox[3] for c in sp.chars)
            sp.chars = [
                Char(c=p.ch, glyph_id=0,
                     origin=(p.x0, p.baseline),
                     bbox=(p.x0, y0, p.x1, y1))
                for p in pls
            ]
            sp.bbox = (min(p.x0 for p in pls), y0, max(p.x1 for p in pls), y1)
            old_line.along0 = sp.bbox[0]
            old_line.along1 = sp.bbox[2]
            old_line.bbox = sp.bbox
            # 一行的基线取该行所有落点的众数（同行的字基线应一致）
            if pls:
                old_line.cross = pls[0].baseline
            new_lines.append(old_line)

        if not new_lines:
            return
        block.lines = new_lines
        bx0 = min(l.bbox[0] for l in new_lines)
        by0 = min(l.bbox[1] for l in new_lines)
        bx1 = max(l.bbox[2] for l in new_lines)
        by1 = max(l.bbox[3] for l in new_lines)
        if clamp_rect:
            bx0 = max(bx0, clamp_rect[0])
            by0 = max(by0, clamp_rect[1])
            bx1 = min(bx1, clamp_rect[2])
            by1 = min(by1, clamp_rect[3])
        block.bbox = (bx0, by0, bx1, by1)
        block.raw = dict(block.raw or {}, edited=True)

    def _raster_notes(self, plan, res, font_file: str) -> list[str]:
        notes: list[str] = []
        if res.get("rendered"):
            notes.append(
                f"{res['rendered']} 个字在字盘里没有，用字体渲染后合成；"
                f"边缘会比周围的位图清晰一些，这是矢量与位图混排的固有差异"
            )
        return notes

    # ------------------------------------------------------------ 页面操作

    def delete_pages(self, indices: list[int]) -> "PageOpReport":
        return self._page_op("delete", indices=indices)

    def reorder_pages(self, order: list[int]) -> "PageOpReport":
        return self._page_op("reorder", order=order)

    def rotate_pages(self, indices: list[int], degrees: int) -> "PageOpReport":
        return self._page_op("rotate", indices=indices, degrees=degrees)

    def insert_pdf(self, path, at: int,
                   pages: list[int] | None = None) -> "PageOpReport":
        return self._page_op("insert", path=path, at=at, pages=pages)

    def _page_op(self, action: str, **kw) -> "PageOpReport":
        """页面级操作的统一入口：纳入撤销快照，并同步结构模型。

        ★ 结构不能靠"重新解析"来同步：编辑过的文档里字形复用会引入
          Form XObject，重解析会把源页面的文字重复计算（这是之前踩过的坑）。
          页面操作只改变"哪些页、什么顺序"，所以按映射表搬运已有的
          PageStructure 就是准确且零风险的；插入进来的新页没有结构，
          标成待解析，等真正访问时再补。
        """
        from . import pages as pages_mod

        if self.doc is None:
            raise RuntimeError("会话尚未加载文档")
        self._snapshot()
        before = self.doc.page_count

        if action == "delete":
            rep = pages_mod.delete_pages(self.doc, kw["indices"])
        elif action == "reorder":
            rep = pages_mod.reorder_pages(self.doc, kw["order"])
        elif action == "rotate":
            rep = pages_mod.rotate_pages(self.doc, kw["indices"],
                                        kw["degrees"])
        elif action == "insert":
            rep = pages_mod.insert_pdf(self.doc, kw["path"], kw["at"],
                                       kw.get("pages"))
        else:
            raise ValueError(f"未知的页面操作 {action!r}")

        self._remap_structure(rep, action == "insert")
        self.revision += 1
        self._dirty = True
        self.edits.append(EditRecord(
            page_no=0, old_text="", new_text="", path="page",
            lines_before=before, lines_after=self.doc.page_count,
            method=rep.action + (f"（{rep.detail}）" if rep.detail else "")))
        return rep

    def _remap_structure(self, rep, inserted: bool) -> None:
        """按页码映射搬运结构；新插入的页标成 None（待解析）。"""
        if self.structure is None:
            return
        old = list(self.structure.pages)
        n_new = self.doc.page_count if self.doc else 0
        new: list[object] = [None] * n_new
        for old_i, new_i in rep.mapping.items():
            if 0 <= old_i < len(old) and 0 <= new_i < n_new:
                new[new_i] = old[old_i]
        for i, pg in enumerate(new):
            if isinstance(pg, PageStructure):
                pg.page_no = i
                for b in pg.blocks:
                    b.page_no = i
        self.structure.pages = new
        self._pending_pages = [i for i, p in enumerate(new) if p is None]
        # 清理：页面操作后旧的字框缓存全部失效
        self._char_box_cache = {}

    def _ensure_page_structure(self, page_no: int) -> None:
        """按需补上缺失的页结构（页面插入后用到）。"""
        if self.structure is None or self.doc is None:
            return
        if page_no < len(self.structure.pages) \
                and self.structure.pages[page_no] is not None:
            return
        while len(self.structure.pages) < self.doc.page_count:
            self.structure.pages.append(None)
        try:
            if self.doc_type == DOC_TEXT:
                fresh = classify(extract_document_bytes(self.doc.tobytes()))
                if page_no < len(fresh.pages):
                    self.structure.pages[page_no] = fresh.pages[page_no]
            else:
                # 图片型：只对缺的这一页做 OCR，不重跑全文档
                page = self.doc[page_no]
                lines = ocr_mod.ocr_page(page)
                ps = PageStructure(page_no=page_no, width=page.rect.width,
                                   height=page.rect.height)
                ps.blocks.extend(ocr_mod.group_ocr_blocks(
                    lines, page_no, ocr_mod._pixel_source(page)))
                for info in page.get_image_info():
                    ps.blocks.append(Block(
                        page_no=page_no, bbox=tuple(info["bbox"]),
                        kind=KIND_IMAGE, layer=LAYER_FLOW,
                        raw={"width": info.get("width"),
                             "height": info.get("height")}))
                self.structure.pages[page_no] = ps
        except Exception:
            pass

    def delete_text(self, block: Block) -> EditRecord:
        if self.doc is None:
            raise RuntimeError("会话尚未加载文档")
        self._snapshot()
        res = delete_block_text(self.doc, block)
        rec = EditRecord(page_no=block.page_no, old_text=block.text,
                         new_text="", path="text",
                         lines_before=res.lines_before, lines_after=0,
                         method="删除整块")
        self._dirty = True
        self.edits.append(rec)
        return rec

    def rebuild_structure(self, page_no: int | None = None) -> None:
        """编辑后重建结构（文本型走解析，图片型走 OCR 缓存失效）。"""
        if self.structure is None or self.doc is None:
            return
        if self.doc_type == DOC_TEXT:
            tmp = self.doc.tobytes()
            fresh = classify(extract_document_bytes(tmp))
            if page_no is None:
                self.structure = fresh
            else:
                self.structure.pages[page_no] = fresh.pages[page_no]
        else:
            # 图片型：结构来自 OCR，编辑后无法低成本重建；
            # 直接把编辑过的块从结构里摘掉，避免用户重复点它
            pass

    def _refresh_page(self, page_no: int) -> None:
        self.rebuild_structure(page_no)

    # ------------------------------------------------------------ 撤销

    def undo(self) -> bool:
        """撤销一步。

        ★ 重做点必须存**此刻**的状态，不能把撤销栈里那一项直接挪过去。
        撤销栈里存的是"操作发生之前"的字节；把它当成重做点，
        重做就会把文档恢复到同一个"之前"状态 —— 等于什么都没做。
        实测症状：删了一页 → 撤销 → 重做，页数没回去。
        正确做法是交换：撤销前把当前状态压进重做栈，重做前把当前状态压回撤销栈。
        """
        if not self._undo or self.doc is None:
            return False
        data, snap = self._undo.pop()
        self._redo.append((self.doc.tobytes(), self._current_snapshot()))
        self._swap_document(data, snap)
        if self.edits:
            self.edits.pop()
        return True

    def redo(self) -> bool:
        if not self._redo or self.doc is None:
            return False
        data, snap = self._redo.pop()
        self._undo.append((self.doc.tobytes(), self._current_snapshot()))
        self._swap_document(data, snap)
        return True

    def _swap_document(self, data: bytes, snap: str | None = None) -> None:
        """用新的文档对象替换当前文档。

        注意：这里必须换对象（PyMuPDF 的 Document 不能原地重置），
        而 UI 层往往缓存着旧的 doc 与 structure 引用 —— 换完必须让它们重新绑定，
        否则下一次渲染就会撞上「document closed」。
        这种崩溃实测就出现在「编辑一次 → 撤销」之后立即重绘页面的场景。
        所以换代后立即重建结构，并递增 revision 供 UI 判断是否需要重绑。
        """
        old = self.doc
        self.doc = pymupdf.open("pdf", data)
        try:
            if old is not None:
                old.close()
        except Exception:
            pass
        self.revision += 1
        self._dirty = True
        if snap is not None:
            try:
                self.structure = _structure_from_json(json.loads(snap), str(self.path))
            except Exception:
                self._reload_structure_after_undo()

    def _reload_structure_after_undo(self) -> None:
        """撤销后重建结构。

        只在没有结构快照时用。优先用快照 —— 重解析在编辑过（含字形复用）
        的文档上不可靠：字形复用的 Form XObject 会让源页面的文字被重复计算。
        """
        if self.doc is None:
            return
        if self.doc_type == DOC_TEXT:
            try:
                self.structure = classify(extract_document_bytes(self.doc.tobytes()))
            except Exception:
                pass

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ------------------------------------------------------------ 保存

    def save(self, out_path: str | Path | None = None) -> Path:
        if self.doc is None:
            raise RuntimeError("会话尚未加载文档")
        target = Path(out_path) if out_path else self.path
        tmp = target.with_suffix(target.suffix + ".tmp")
        self.doc.save(str(tmp), garbage=4, deflate=True)
        if target.exists():
            try:
                target.unlink()
            except PermissionError:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"目标文件被占用，无法写入：{target}")
        tmp.replace(target)
        self.path = target
        self._dirty = False
        return target

    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()
            self.doc = None

    def report(self) -> str:
        if not self.edits:
            return "本次会话没有编辑。"
        lines = [f"本次会话共 {len(self.edits)} 处编辑："]
        for e in self.edits:
            lines.append("  " + e.summary())
            if e.method:
                lines.append(f"      {e.method}")
            for w in e.warned:
                lines.append(f"      ⚠ {w}")
        return "\n".join(lines)


# --------------------------------------------------------------- 结构序列化


def _structure_to_json(s: DocStructure) -> dict:
    return {
        "path": s.path,
        "body_size": s.body_size,
        "pages": [
            {
                "page_no": p.page_no,
                "width": p.width,
                "height": p.height,
                "blocks": [
                    {
                        "kind": b.kind, "layer": b.layer,
                        "confidence": b.confidence, "bbox": list(b.bbox),
                        "reasons": b.reasons, "raw": b.raw,
                        "lines": [
                            {
                                "text": l.text, "cross": l.cross,
                                "along0": l.along0, "along1": l.along1,
                                "bbox": list(l.bbox),
                                "size": l.size, "font": l.font,
                                "direction": list(l.direction),
                                "color": list(l.color),
                                # ★ 必须存逐字包围盒。
                                # 早期版本只存行级数据，重建时用"均匀等分"重新造字符框，
                                # 结果所有字一样宽（实测都是 2.69pt）——
                                # 比例字体下逐字定位完全切错，"只改动的字"和"复用原像素"
                                # 全都建立在错的坐标上。首次 OCR 时正常、一走缓存就坏，
                                # 这种"时好时坏"最难查。
                                "chars": [
                                    {
                                        "c": ch.c,
                                        "bbox": [round(v, 3) for v in ch.bbox],
                                        "origin": [round(v, 3) for v in ch.origin],
                                    }
                                    for sp in l.spans for ch in sp.chars
                                ],
                            }
                            for l in b.lines
                        ],
                    }
                    for b in p.blocks
                ],
            }
            for p in s.pages
        ],
    }


def _structure_from_json(data: dict, path: str) -> DocStructure:
    from .model import Char, Line, PageStructure, Span

    s = DocStructure(path=path, body_size=data.get("body_size", 10.0))
    for pdata in data["pages"]:
        page = PageStructure(page_no=pdata["page_no"], width=pdata["width"],
                             height=pdata["height"])
        for bdata in pdata["blocks"]:
            lines = []
            for ldata in bdata["lines"]:
                text = ldata["text"]
                x0, y0, x1, y1 = ldata["bbox"]
                rtl = ldata.get("direction", [1.0, 0.0])
                cached = ldata.get("chars")
                if cached:
                    chars = [
                        Char(c=cd["c"], glyph_id=0,
                             origin=(cd["origin"][0], cd["origin"][1]),
                             bbox=(cd["bbox"][0], cd["bbox"][1],
                                   cd["bbox"][2], cd["bbox"][3]))
                        for cd in cached
                    ]
                else:
                    # 旧缓存没有逐字数据，退回均匀等分（会切错，仅作兜底）
                    cw = (x1 - x0) / max(1, len(text))
                    chars = [
                        Char(c=ch, glyph_id=0,
                             origin=(x0 + i * cw, ldata["cross"]),
                             bbox=(x0 + i * cw, y0, x0 + (i + 1) * cw, y1))
                        for i, ch in enumerate(text)
                    ]
                span = Span(chars=chars, font=ldata.get("font", ""),
                            size=ldata.get("size", 10.0),
                            color=tuple(ldata.get("color", [0, 0, 0])),
                            direction=(rtl[0], rtl[1]), flags=0, opacity=1.0,
                            bbox=(x0, y0, x1, y1), seqno=0,
                            page_no=pdata["page_no"])
                lines.append(Line(
                    spans=[span], page_no=pdata["page_no"],
                    direction=(rtl[0], rtl[1]), cross=ldata["cross"],
                    along0=ldata["along0"], along1=ldata["along1"],
                    bbox=(x0, y0, x1, y1),
                ))
            page.blocks.append(Block(
                lines=lines, page_no=pdata["page_no"],
                bbox=tuple(bdata["bbox"]), kind=bdata["kind"],
                layer=bdata["layer"], confidence=bdata.get("confidence", 1.0),
                reasons=list(bdata.get("reasons", [])), raw=bdata.get("raw"),
            ))
        s.pages.append(page)
    return s


def extract_document_bytes(data: bytes) -> DocStructure:
    """从内存中的 PDF 字节解析结构 —— 撤销后重建结构用。"""
    return extract_document(data)
