"""页面级操作：删除、重排、提取、插入、旋转、合并。

为什么这些操作要跟「文字编辑」放在同一套模型里做：
用户在窗口里改完字、又删了两页、再另存 —— 如果页面操作另开一条
"把文件读进来再拼一遍"的路径，前面编辑得到的内容流改动就会丢。
所以这里一律就地操作同一个 `pymupdf.Document`，并且每一步都返回
一份"操作前 → 操作后"的页码映射，让上层能同步它的结构模型与撤销快照。

销毁性操作用 `pymupdf` 的 `delete_page` / `select` 直接做，
它们会正确维护书签、链接、命名目标、页面标签这些附属结构 ——
自己拼对象树很容易漏掉其中一半。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


class PageOpError(ValueError):
    """页面操作不合法（页码越界、结果会变成零页等）。"""


@dataclass
class PageOpReport:
    """一次页面操作的说明。"""

    action: str
    before: int
    after: int
    detail: str = ""
    # 旧页码 → 新页码；被删除的页不出现在里面
    mapping: dict[int, int] = field(default_factory=dict)

    def text(self) -> str:
        lines = [f"{self.action}：{self.before} 页 → {self.after} 页"]
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


def _check_index(doc: pymupdf.Document, pno: int) -> int:
    n = doc.page_count
    if n == 0:
        raise PageOpError("文档没有页面")
    i = pno + n if pno < 0 else pno
    if not (0 <= i < n):
        raise PageOpError(f"页码 {pno} 越界（共 {n} 页，合法范围 1~{n}）")
    return i


def parse_page_spec(spec: str, total: int) -> list[int]:
    """把 "1,3,5-8,12-" 这类页码范围解析成 0 基页码列表（保序、去重）。

    单独抽出来是因为这是唯一面向用户的输入格式，
    要能容忍全角逗号、中文顿号、空格 —— 用户从 Word 里复制页码过来时
    经常带着这些字符。
    """
    if total <= 0:
        raise PageOpError("文档没有页面")
    text = spec.strip()
    if not text:
        raise PageOpError("页码范围为空")
    for ch in "，、;；":
        text = text.replace(ch, ",")
    text = text.replace("—", "-").replace("－", "-").replace("~", "-")
    out: list[int] = []
    seen: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            if "-" in b:
                raise PageOpError(f"页码范围 {part!r} 写法不对，"
                                  f"只支持 a-b 这种形式")
            start = _to_int(a, part) if a.strip() else 1
            end = _to_int(b, part) if b.strip() else total
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                i = _check_index_range(p, total)
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        else:
            i = _check_index_range(_to_int(part, part), total)
            if i not in seen:
                seen.add(i)
                out.append(i)
    if not out:
        raise PageOpError(f"页码范围 {spec!r} 没有解析出任何页面")
    return out


def _to_int(s: str, whole: str) -> int:
    try:
        return int(s.strip())
    except ValueError:
        raise PageOpError(
            f"页码范围 {whole!r} 里有非数字内容 {s!r}；"
            f"页码要写成 1,3,5-8 这样") from None


def _check_index_range(pno: int, total: int) -> int:
    """页码范围里用的是**人类页码（从 1 开始）**，这里换算成 0 基。"""
    if not (1 <= pno <= total):
        raise PageOpError(f"页码 {pno} 越界（共 {total} 页，合法范围 1~{total}）")
    return pno - 1


def delete_pages(doc: pymupdf.Document, indices: list[int]) -> PageOpReport:
    """删除若干页（就地）。"""
    if not indices:
        raise PageOpError("没有指定要删除的页面")
    before = doc.page_count
    idx = sorted({_check_index(doc, i) for i in indices})
    if len(idx) >= before:
        raise PageOpError(
            f"要删除 {len(idx)} 页，而文档只有 {before} 页 —— "
            "PDF 不允许零页，至少保留一页")
    # 倒序删，避免前面的删除让后面的页码失效
    for i in reversed(idx):
        doc.delete_page(i)
    keep = [p for p in range(before) if p not in set(idx)]
    mapping = {p: k for k, p in enumerate(keep)}
    return PageOpReport("删除页面", before, doc.page_count,
                        f"已删除第 {', '.join(str(i + 1) for i in idx)} 页",
                        mapping)


def reorder_pages(doc: pymupdf.Document, order: list[int]) -> PageOpReport:
    """按 `order`（旧页码的排列）重排页面（就地）。

    `order` 必须是 0..n-1 的一个排列。用 `select()` 做，
    它会一并处理书签与命名目标的重定向。
    """
    before = doc.page_count
    if not order:
        raise PageOpError("页面顺序为空")
    idx = [_check_index(doc, i) for i in order]
    if len(set(idx)) != len(idx):
        dup = [i + 1 for i in idx if idx.count(i) > 1]
        raise PageOpError(f"页面顺序里有重复页：{sorted(set(dup))}")
    if set(idx) != set(range(before)):
        raise PageOpError(
            f"页面顺序必须包含全部 {before} 页且不重复"
            f"（当前给了 {len(idx)} 项）")
    if idx == list(range(before)):
        return PageOpReport("调整页面顺序", before, before, "顺序未变",
                            {p: p for p in range(before)})
    doc.select(idx)
    mapping = {old: new for new, old in enumerate(idx)}
    return PageOpReport(
        "调整页面顺序", before, doc.page_count,
        "新顺序：" + ", ".join(str(i + 1) for i in idx), mapping)


def rotate_pages(doc: pymupdf.Document, indices: list[int],
                 degrees: int) -> PageOpReport:
    """旋转页面（就地）。只接受 90 的整数倍。"""
    if degrees % 90 != 0:
        raise PageOpError(f"旋转角必须是 90 的整数倍，收到 {degrees}")
    before = doc.page_count
    idx = sorted({_check_index(doc, i) for i in indices})
    for i in idx:
        page = doc[i]
        page.set_rotation((page.rotation + degrees) % 360)
    return PageOpReport("旋转页面", before, doc.page_count,
                        f"第 {', '.join(str(i + 1) for i in idx)} 页"
                        f"旋转 {degrees}°",
                        {p: p for p in range(before)})


def insert_pdf(doc: pymupdf.Document, src_path: str | Path,
               at: int, pages: list[int] | None = None) -> PageOpReport:
    """把另一个 PDF 的页面插入到 `at` 位置之前（就地）。"""
    before = doc.page_count
    src = pymupdf.open(str(src_path))
    try:
        if src.is_encrypted and not src.authenticate(""):
            raise PageOpError(f"{Path(src_path).name} 是加密的，需要密码")
        if pages is None:
            pick = list(range(src.page_count))
        else:
            pick = [_check_index(src, p) for p in pages]
        if not pick:
            raise PageOpError("没有指定要插入的页面")
        pos = before if at < 0 else max(0, min(at, before))
        # 逐页插入而不是一次 insert_pdf(from,to)：后者只能连续区段，
        # 选非连续页时会静默把所有中间页也带进来。
        for k, p in enumerate(pick):
            doc.insert_pdf(src, from_page=p, to_page=p, start_at=pos + k)
    finally:
        src.close()
    n = doc.page_count - before
    mapping = {p: (p if p < pos else p + n) for p in range(before)}
    return PageOpReport(
        "插入页面", before, doc.page_count,
        f"{Path(src_path).name} 的 {n} 页插入到第 {pos + 1} 页之前", mapping)


def extract_pages(src_path: str | Path, indices: list[int],
                  out_path: str | Path) -> Path:
    """把若干页另存成一个新 PDF。"""
    doc = pymupdf.open(str(src_path))
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PageOpError(f"{Path(src_path).name} 是加密的，需要密码")
        idx = sorted({_check_index(doc, i) for i in indices})
        if not idx:
            raise PageOpError("没有指定要提取的页面")
        out = pymupdf.open()
        # 同样逐页插入：提取 "1,3,5" 时不该把 2、4 页也带出来
        for i in idx:
            out.insert_pdf(doc, from_page=i, to_page=i)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(str(out_path), garbage=3, deflate=True)
        out.close()
    finally:
        doc.close()
    return Path(out_path)


def merge_pdfs(paths: list[str | Path], out_path: str | Path) -> Path:
    """按给定顺序合并多个 PDF。"""
    if not paths:
        raise PageOpError("没有要合并的文件")
    out = pymupdf.open()
    try:
        for p in paths:
            src = pymupdf.open(str(p))
            try:
                if src.is_encrypted and not src.authenticate(""):
                    raise PageOpError(f"{Path(p).name} 是加密的，需要密码")
                out.insert_pdf(src)
            finally:
                src.close()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(str(out_path), garbage=3, deflate=True)
    finally:
        out.close()
    return Path(out_path)
