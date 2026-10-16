"""L0→L2 结构提取：PDF 页面 → 字符表 → 行 → 块。

核心是 get_texttrace()，它是 PyMuPDF 最低层的文本接口，
给出每个字符的字形 ID、基线原点、包围盒，以及所在 span 的字体、
字号、颜色、方向向量和内容流绘制序号（= z 序）。

这些信息是后续一切的基础：
  - 字形 ID + 字体 xref → Day 2 判断嵌入字体能否输出目标字符
  - 方向向量 → 识别旋转文本（水印）
  - seqno → 判断元素在内容流中的层级
"""

from __future__ import annotations

import math
import re

import pymupdf

from .model import (
    KIND_GRAPHICS,
    KIND_IMAGE,
    KIND_PARAGRAPH,
    KIND_TABLE,
    LAYER_FIXED,
    LAYER_FLOW,
    Block,
    Char,
    DocStructure,
    Line,
    PageStructure,
    Span,
)

# 行聚合容差
CROSS_TOL_RATIO = 0.6      # 跨行方向容差 = 比例 × 字号
ADJACENT_GAP_RATIO = 3.0   # 同行内相邻 span 的最大间隙 = 比例 × 字号

# 段落聚合阈值
SIZE_TOL = 0.5             # 字号差
COLOR_TOL = 0.06           # 颜色差
OVERLAP_MIN = 0.5          # 水平重叠比例下限
GAP_MAX_RATIO = 2.2        # 行间距上限 = 比例 × 字号
GAP_JITTER = 0.35          # 行间距允许的抖动比例
SHORT_LINE_RATIO = 0.75    # 判定"短行"的比例
LEFT_TOL = 2.5             # 左边界对齐容差

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


def _strip_subset(name: str) -> str:
    """去掉子集前缀，如 'LIOJMF+SimSun Regular' → 'SimSun Regular'。"""
    return _SUBSET_PREFIX.sub("", name)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]


# --------------------------------------------------------------- span 提取


def _build_bbox_index(page: pymupdf.Page) -> dict[tuple[int, int], tuple]:
    """从 rawdict 建立「字符原点 → 字符包围盒」索引。

    为什么需要这个：get_texttrace() 返回的字符 bbox 宽度恒为 0
    （形如 (x, y0, x, y1) 的退化竖线），只有 origin 是对的。
    而 rawdict 的 bbox 是准确的，但又不提供字形 ID。
    两者必须按原点配对，才能同时拿到准确坐标和字形信息。
    """
    index: dict[tuple[int, int], tuple] = {}
    for b in page.get_text("rawdict")["blocks"]:
        if b.get("type", 0) != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    ox, oy = ch["origin"]
                    index[(round(ox * 10), round(oy * 10))] = tuple(ch["bbox"])
    return index


def extract_spans(page: pymupdf.Page, page_no: int) -> tuple[list[Span], int]:
    """提取 span，返回 (spans, 包围盒配对失败数)。"""
    bbox_index = _build_bbox_index(page)
    spans: list[Span] = []
    unmatched = 0

    for raw in page.get_texttrace():
        if raw.get("type", 0) != 0:
            continue
        raw_chars = raw["chars"]
        if not raw_chars:
            continue

        chars: list[Char] = []
        for i, (uni, gid, origin, _bad_bbox) in enumerate(raw_chars):
            key = (round(origin[0] * 10), round(origin[1] * 10))
            bbox = bbox_index.get(key)
            if bbox is None:
                unmatched += 1
                # 退化兜底：用下一个字符的原点推断宽度，末字符退化为零宽
                if i + 1 < len(raw_chars):
                    nx = raw_chars[i + 1][2][0]
                    ny = raw_chars[i + 1][2][1]
                    if abs(ny - origin[1]) < 0.5:
                        bbox = (origin[0], origin[1] - raw["size"],
                                nx, origin[1] + raw["size"] * 0.2)
                    else:
                        bbox = (origin[0], origin[1] - raw["size"],
                                origin[0] + raw["size"], origin[1] + raw["size"] * 0.2)
                else:
                    bbox = (origin[0], origin[1] - raw["size"],
                            raw["bbox"][2], origin[1] + raw["size"] * 0.2)
            chars.append(Char(
                c=chr(uni) if isinstance(uni, int) else uni,
                glyph_id=gid, origin=(origin[0], origin[1]), bbox=bbox,
            ))

        # span 级 bbox 同样不可信（由退化字符框并集而来），按字符框重算
        xs0 = [c.bbox[0] for c in chars]
        ys0 = [c.bbox[1] for c in chars]
        xs1 = [c.bbox[2] for c in chars]
        ys1 = [c.bbox[3] for c in chars]

        spans.append(Span(
            chars=chars,
            font=_strip_subset(raw["font"]),
            size=raw["size"],
            color=tuple(raw["color"]),
            direction=(raw["dir"][0], raw["dir"][1]),
            flags=raw.get("flags", 0),
            opacity=raw.get("opacity", 1.0),
            bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
            seqno=raw.get("seqno", 0),
            page_no=page_no,
        ))
    return spans, unmatched


# --------------------------------------------------------------- L1 行聚合


def _proj(span: Span) -> tuple[float, float]:
    """把 span 投影到 (沿行方向 along, 跨行方向 cross)。"""
    cos, sin = span.direction
    ox, oy = span.chars[0].origin
    along = ox * cos + oy * sin
    cross = -ox * sin + oy * cos
    return along, cross


def _split_span_by_baseline(span: Span) -> list[Span]:
    """把一个 span 按基线拆成多个。

    为什么必须拆：get_texttrace() 是按「内容流里的文本对象」分组的，
    而 insert_textbox 之类会把整个段落写成一个文本对象 ——
    于是一个 span 里混着好几行、基线各不相同的字符。
    span ≠ 视觉行，不拆开的话整段会被当成一行处理，
    后续的段落分析、字体度量、重排全部失准。
    """
    cos, sin = span.direction
    tol = 0.25 * span.size

    runs: list[list[Char]] = []
    cur: list[Char] = [span.chars[0]]
    cur_cross = -span.chars[0].origin[0] * sin + span.chars[0].origin[1] * cos

    for ch in span.chars[1:]:
        c = -ch.origin[0] * sin + ch.origin[1] * cos
        if abs(c - cur_cross) <= tol:
            cur.append(ch)
        else:
            runs.append(cur)
            cur = [ch]
            cur_cross = c
    runs.append(cur)

    if len(runs) == 1:
        return [span]

    out: list[Span] = []
    for run in runs:
        xs0 = [c.bbox[0] for c in run]
        ys0 = [c.bbox[1] for c in run]
        xs1 = [c.bbox[2] for c in run]
        ys1 = [c.bbox[3] for c in run]
        out.append(Span(
            chars=run, font=span.font, size=span.size, color=span.color,
            direction=span.direction, flags=span.flags, opacity=span.opacity,
            bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
            seqno=span.seqno, page_no=span.page_no,
        ))
    return out


def group_lines(spans: list[Span], page_no: int) -> list[Line]:
    """把 span 聚合成行。

    用旋转不变的方式：每个 span 投影到自身的 (along, cross) 坐标系，
    cross 相近且 along 相邻的 span 属于同一行。
    这样水平文本和任意角度的旋转文本用同一套逻辑处理。
    """
    # 先按基线拆开，确保一个单元只对应一条视觉行
    units: list[Span] = []
    for s in spans:
        units.extend(_split_span_by_baseline(s))

    remaining = list(units)
    lines: list[Line] = []

    while remaining:
        seed = remaining.pop(0)
        cos, sin = seed.direction
        cross0 = -seed.chars[0].origin[0] * sin + seed.chars[0].origin[1] * cos
        tol = CROSS_TOL_RATIO * seed.size

        group = [seed]
        rest: list[Span] = []
        for s in remaining:
            if _dot(s.direction, seed.direction) < 0.9995:
                rest.append(s)
                continue
            s_cross = -s.chars[0].origin[0] * sin + s.chars[0].origin[1] * cos
            if abs(s_cross - cross0) <= max(tol, CROSS_TOL_RATIO * s.size):
                group.append(s)
            else:
                rest.append(s)
        remaining = rest

        # 组内按 along 排序
        group.sort(key=lambda s: _proj(s)[0])

        # 再把 along 方向断开的（间隙过大）拆成多行
        for chunk in _split_by_gap(group):
            lines.append(_make_line(chunk, page_no))

    lines.sort(key=lambda l: (round(l.cross, 1), l.along0))
    return lines


def _split_by_gap(group: list[Span]) -> list[list[Span]]:
    if len(group) <= 1:
        return [group]
    chunks: list[list[Span]] = [[group[0]]]
    prev = group[0]
    for s in group[1:]:
        prev_end = max(
            c.origin[0] * prev.direction[0] + c.origin[1] * prev.direction[1] + c.width
            for c in prev.chars
        )
        s_start = _proj(s)[0]
        gap = s_start - prev_end
        if gap > ADJACENT_GAP_RATIO * max(prev.size, s.size):
            chunks.append([s])
        else:
            chunks[-1].append(s)
        prev = s
    return chunks


def _char_along_range(ch, cos: float, sin: float) -> tuple[float, float]:
    """字符包围盒四角投影到行方向后的范围。

    用四角投影而非"原点 + 宽度"，这样旋转文本也能得到正确的沿行长度。
    """
    x0, y0, x1, y1 = ch.bbox
    ts = [x0 * cos + y0 * sin, x1 * cos + y0 * sin,
          x0 * cos + y1 * sin, x1 * cos + y1 * sin]
    return min(ts), max(ts)


def _make_line(spans: list[Span], page_no: int) -> Line:
    cos, sin = spans[0].direction
    starts: list[float] = []
    ends: list[float] = []
    for s in spans:
        for c in s.chars:
            a, b = _char_along_range(c, cos, sin)
            starts.append(a)
            ends.append(b)

    xs0 = [s.bbox[0] for s in spans]
    ys0 = [s.bbox[1] for s in spans]
    xs1 = [s.bbox[2] for s in spans]
    ys1 = [s.bbox[3] for s in spans]
    cross = -spans[0].chars[0].origin[0] * sin + spans[0].chars[0].origin[1] * cos
    return Line(
        spans=spans,
        page_no=page_no,
        direction=(cos, sin),
        cross=cross,
        along0=min(starts),
        along1=max(ends),
        bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
    )


# --------------------------------------------------------------- L2 块聚合


def _overlap_ratio(a: Line, b: Line) -> float:
    lo = max(a.along0, b.along0)
    hi = min(a.along1, b.along1)
    if hi <= lo:
        return 0.0
    span = hi - lo
    shorter = min(a.width, b.width)
    return span / shorter if shorter > 0 else 0.0


def _same_style(a: Line, b: Line) -> bool:
    if abs(a.size - b.size) > SIZE_TOL:
        return False
    if a.font != b.font:
        return False
    if any(abs(x - y) > COLOR_TOL for x, y in zip(a.color, b.color)):
        return False
    return True


def group_paragraphs(lines: list[Line], page_no: int, text_width: float) -> list[Block]:
    """把行聚合成块。

    规则：字号/字体/颜色一致 + 水平重叠足够 + 行间距在正常范围内且保持稳定。
    另外单独识别"短行 + 紧跟整行"的模式 —— 这是"第一条 合同标的"这类
    小标题的特征（它们与正文同字号同字体，只能靠版面特征区分）。
    """
    if not lines:
        return []

    # 先按方向分组，水平文本和旋转文本不混在一起
    horizontal = [l for l in lines if not l.is_rotated]
    rotated = [l for l in lines if l.is_rotated]

    blocks: list[Block] = []
    blocks.extend(_group_horizontal(horizontal, page_no, text_width))
    for l in rotated:
        b = Block(lines=[l], page_no=page_no, bbox=l.bbox, z=l.seqno)
        blocks.append(b)
    return blocks


def _group_horizontal(lines: list[Line], page_no: int, text_width: float) -> list[Block]:
    lines = sorted(lines, key=lambda l: l.cross)
    blocks: list[Block] = []
    cur: list[Line] = []
    expected_gap: float | None = None

    def flush():
        nonlocal cur, expected_gap
        if cur:
            xs0 = [l.bbox[0] for l in cur]
            ys0 = [l.bbox[1] for l in cur]
            xs1 = [l.bbox[2] for l in cur]
            ys1 = [l.bbox[3] for l in cur]
            blocks.append(Block(
                lines=list(cur), page_no=page_no,
                bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
                z=min(l.seqno for l in cur),
            ))
        cur = []
        expected_gap = None

    for line in lines:
        if not cur:
            cur = [line]
            continue

        prev = cur[-1]
        gap = abs(line.cross - prev.cross)

        # 短行 + 紧跟整行 → 上一行是独立的小标题，断开
        is_short = prev.width < SHORT_LINE_RATIO * text_width
        starts_full = line.width >= SHORT_LINE_RATIO * text_width
        if is_short and starts_full and _overlap_ratio(prev, line) > OVERLAP_MIN:
            flush()
            cur = [line]
            continue

        same = _same_style(prev, line)
        overlap_ok = _overlap_ratio(prev, line) >= OVERLAP_MIN
        gap_ok = gap <= GAP_MAX_RATIO * max(prev.size, line.size)
        if expected_gap is not None:
            gap_ok = gap_ok and abs(gap - expected_gap) <= GAP_JITTER * expected_gap

        if same and overlap_ok and gap_ok:
            cur.append(line)
            if len(cur) >= 2:
                gaps = [abs(cur[i + 1].cross - cur[i].cross) for i in range(len(cur) - 1)]
                expected_gap = sum(gaps) / len(gaps)
        else:
            flush()
            cur = [line]

    flush()
    return blocks


# --------------------------------------------------------------- 非文本元素


def extract_images(page: pymupdf.Page, page_no: int) -> list[Block]:
    blocks = []
    for info in page.get_image_info():
        bbox = tuple(info["bbox"])
        blocks.append(Block(
            page_no=page_no, bbox=bbox, kind=KIND_IMAGE, layer=LAYER_FLOW,
            raw={"width": info.get("width"), "height": info.get("height")},
            confidence=1.0, reasons=["页面图像对象"],
        ))
    return blocks


def _cluster_rects(rects: list[pymupdf.Rect], tol: float = 3.0) -> list[pymupdf.Rect]:
    """把相邻/相交的矩形合并成簇。

    表格框线会拆成几十个独立矩形，必须聚合成一个图形块，
    否则结构树会被噪声淹没。
    """
    clusters: list[pymupdf.Rect] = []
    for r in rects:
        merged = None
        grown = pymupdf.Rect(r.x0 - tol, r.y0 - tol, r.x1 + tol, r.y1 + tol)
        for c in clusters:
            if grown.intersects(c):
                merged = c
                break
        if merged is None:
            clusters.append(pymupdf.Rect(r))
            continue
        merged.include_rect(r)
        # 合并后可能又和别的簇相交，收敛处理
        changed = True
        while changed:
            changed = False
            for other in list(clusters):
                if other is merged:
                    continue
                g = pymupdf.Rect(merged.x0 - tol, merged.y0 - tol,
                                 merged.x1 + tol, merged.y1 + tol)
                if g.intersects(other):
                    merged.include_rect(other)
                    clusters.remove(other)
                    changed = True
                    break
    return clusters


def extract_graphics(
    page: pymupdf.Page,
    page_no: int,
    top_band: float,
    bot_band: float,
    exclude: list[pymupdf.Rect] | None = None,
) -> list[Block]:
    """矢量图形，按连通性聚合成块。

    位于页眉/页脚条带内的标记为固定层（分隔线）；
    落在表格区域内的直接丢弃（那是表格的框线，归表格块管）。
    """
    rects = [d["rect"] for d in page.get_drawings() if d.get("rect") is not None]
    exclude = exclude or []

    blocks = []
    for cluster in _cluster_rects(rects):
        if any(cluster.intersects(e) and (cluster & e).get_area() > 0.6 * cluster.get_area()
               for e in exclude):
            continue
        bbox = (cluster.x0, cluster.y0, cluster.x1, cluster.y1)
        in_band = cluster.y1 <= top_band or cluster.y0 >= bot_band
        blocks.append(Block(
            page_no=page_no, bbox=bbox, kind=KIND_GRAPHICS,
            layer=LAYER_FIXED if in_band else LAYER_FLOW,
            raw={"n_rects": sum(1 for r in rects if cluster.contains(r))},
            confidence=0.9,
            reasons=["页眉/页脚条带内的矢量图形" if in_band else "正文区矢量图形"],
        ))
    return blocks


def extract_tables(page: pymupdf.Page, page_no: int) -> list[Block]:
    blocks = []
    try:
        finder = page.find_tables()
    except Exception:
        return blocks
    for t in finder.tables:
        bbox = (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
        blocks.append(Block(
            page_no=page_no, bbox=bbox, kind=KIND_TABLE, layer=LAYER_FLOW,
            raw={"rows": t.row_count, "cols": t.col_count},
            confidence=0.85, reasons=[f"find_tables 检出 {t.row_count}×{t.col_count}"],
        ))
    return blocks


# --------------------------------------------------------------- 主入口


def extract_document(source) -> DocStructure:
    """解析文档结构。

    source 可以是文件路径、内存中的 PDF 字节，或已打开的 Document。
    支持字节是为了撤销后重建结构 —— 早期版本把字节先落盘成临时文件再解析，
    多了一个「文件被占用/写不进去」的失败点，撤销时就可能报错。
    """
    if isinstance(source, bytes):
        doc = pymupdf.open("pdf", source)
        label = "<memory>"
    elif isinstance(source, pymupdf.Document):
        doc = pymupdf.open("pdf", source.tobytes())
        label = "<document>"
    else:
        doc = pymupdf.open(str(source))
        label = str(source)

    structure = DocStructure(path=label)

    # 字体名 → xref 映射（Day 2 提取嵌入字体要用）
    for pno in range(len(doc)):
        for f in doc[pno].get_fonts(full=True):
            xref, basefont = f[0], f[3]
            structure.font_xrefs[_strip_subset(basefont)] = xref

    for pno in range(len(doc)):
        page = doc[pno]
        rect = page.rect
        top_band = 70.0
        bot_band = rect.height - 70.0

        spans, unmatched = extract_spans(page, pno)
        if unmatched:
            print(f"  [警告] 第 {pno+1} 页有 {unmatched} 个字符的包围盒未能从 rawdict 配对")
        lines = group_lines(spans, pno)
        text_width = rect.width - 144.0  # 左右各 72pt 页边距

        text_blocks = group_paragraphs(lines, pno, text_width)

        # 表格区域内的文本行应归入表格，从段落块中剔除
        page_struct = PageStructure(
            page_no=pno, width=rect.width, height=rect.height,
        )
        table_blocks = extract_tables(page, pno)
        table_rects = [pymupdf.Rect(b.bbox) for b in table_blocks]

        for b in text_blocks:
            brect = pymupdf.Rect(b.bbox)
            if any(brect.intersects(tr) and brect.get_area() > 0
                   and (brect & tr).get_area() > 0.5 * brect.get_area()
                   for tr in table_rects):
                continue  # 归表格管
            page_struct.blocks.append(b)

        page_struct.blocks.extend(table_blocks)
        page_struct.blocks.extend(extract_images(page, pno))
        page_struct.blocks.extend(
            extract_graphics(page, pno, top_band, bot_band, exclude=table_rects)
        )

        page_struct.drawings = [
            {"bbox": tuple(d["rect"]), "type": d.get("type"), "width": d.get("width")}
            for d in page.get_drawings() if d.get("rect") is not None
        ]

        structure.pages.append(page_struct)

    doc.close()

    # 文档级统计：正文主导字号（按字符数加权的中位数近似）
    from collections import Counter
    counter: Counter[float] = Counter()
    for p in structure.pages:
        for b in p.blocks:
            for l in b.lines:
                counter[round(l.size, 1)] += len(l.text)
    if counter:
        structure.body_size = counter.most_common(1)[0][0]

    return structure
