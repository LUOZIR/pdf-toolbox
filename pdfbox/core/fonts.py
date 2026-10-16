"""字体引擎：识别原字体 → 匹配可用字体 → 验证度量一致性。

Day 2 的核心发现决定了这里的设计：

  1. PyMuPDF 无法使用 PDF 里的嵌入子集字体。给它字体 buffer 后，
     它会静默替换成内置的 Droid Sans Fallback —— 不报错、文字也能正确提取，
     但渲染出来是另一个字体（黑体而非宋体）。这是个静默陷阱。
  2. 嵌入字体的 /ToUnicode 提供了 CID 映射，且 CID 与系统字体的 GID 一致，
     但字形轮廓只保留了原文用到的那些（样本中 289/28905），新字符确实无轮廓。
  3. 因此可行路径是「系统字体匹配」：原字体是 SimSun，系统也有 SimSun，
     匹配后渲染外观完全一致。

匹配是否成功不能靠感觉，必须量化：用原文档中该字串的实际渲染宽度
与候选字体的计算宽度比对，比值接近 1.0 才算匹配。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pymupdf

FONT_DIRS = [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts"]

# PDF 字体名关键字 → 系统字体候选文件（按优先级）
FAMILY_CANDIDATES: dict[str, list[str]] = {
    "simsun":        ["simsun.ttc", "NotoSerifSC-VF.ttf"],
    "宋体":           ["simsun.ttc", "NotoSerifSC-VF.ttf"],
    "nsimsun":       ["simsun.ttc", "NotoSerifSC-VF.ttf"],
    "simhei":        ["simhei.ttf", "NotoSansSC-VF.ttf"],
    "黑体":           ["simhei.ttf", "NotoSansSC-VF.ttf"],
    "microsoftyahei": ["msyh.ttc", "NotoSansSC-VF.ttf"],
    "微软雅黑":        ["msyh.ttc", "NotoSansSC-VF.ttf"],
    "yahei":         ["msyh.ttc", "NotoSansSC-VF.ttf"],
    "kaiti":         ["simkai.ttf", "NotoSerifSC-VF.ttf"],
    "楷体":           ["simkai.ttf", "NotoSerifSC-VF.ttf"],
    "fangsong":      ["simfang.ttf", "NotoSerifSC-VF.ttf"],
    "仿宋":           ["simfang.ttf", "NotoSerifSC-VF.ttf"],
    "dengxian":      ["Deng.ttf", "NotoSansSC-VF.ttf"],
    "等线":           ["Deng.ttf", "NotoSansSC-VF.ttf"],
    "notosanssc":    ["NotoSansSC-VF.ttf"],
    "notoserifsc":   ["NotoSerifSC-VF.ttf"],
    "sourcehans":    ["NotoSansSC-VF.ttf"],
    "arial":         ["arial.ttf"],
    "helvetica":     ["arial.ttf"],
    "times":         ["times.ttf", "NotoSerifSC-VF.ttf"],
    "courier":       ["cour.ttf"],
    "cambria":       ["cambria.ttc"],
    "calibri":       ["calibri.ttf"],
}

# 最终兜底：保证任何字符都能渲染（外观可能不同，必须提示用户）
UNIVERSAL_FALLBACK = ["NotoSansSC-VF.ttf", "simsun.ttc"]

# 度量一致性阈值：候选字体的计算宽度 / 原文档实际宽度
METRIC_TOLERANCE = 0.03   # 3% 以内视为同字体


@dataclass
class FontMatch:
    """一次字体匹配的结果。"""

    pdf_font: str                       # PDF 里声明的字体名
    family: str                         # 识别出的字体族
    font: pymupdf.Font | None           # 可用于绘制的字体对象
    source: str                         # system | fallback | universal | none
    file: str | None = None             # 字体文件路径
    metric_ratio: float = 0.0           # 计算宽度 / 原宽度
    verified: bool = False              # 度量是否通过验证
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.font is not None


@lru_cache(maxsize=64)
def _load_pymupdf_font(path: str) -> pymupdf.Font:
    return pymupdf.Font(fontfile=path)


@lru_cache(maxsize=256)
def _find_file(filename: str) -> str | None:
    for d in FONT_DIRS:
        p = d / filename
        if p.exists():
            return str(p)
    return None


def identify_family(pdf_font_name: str) -> str:
    """从 PDF 字体名识别字体族。

    PDF 字体名形态多样：'SimSun Regular'、'LIOJMF+SimSun'、
    'MicrosoftYaHei-Bold'、'SimSun,Bold'。统一压成小写去分隔符后做关键字匹配。
    """
    name = re.sub(r"^[A-Z]{6}\+", "", pdf_font_name)
    # 去掉粗细/斜体等样式后缀，保留主名
    name = re.split(r"[-,]", name)[0]
    key = re.sub(r"[\s_]+", "", name).lower()

    # 先做精确前缀匹配，再做包含匹配
    for fam in FAMILY_CANDIDATES:
        if key.startswith(fam) or fam in key:
            return fam
    return key


def find_system_font(pdf_font_name: str) -> tuple[str | None, str]:
    """按字体族找系统字体，返回 (文件路径, 来源标记)。"""
    family = identify_family(pdf_font_name)
    for cand in FAMILY_CANDIDATES.get(family, []):
        path = _find_file(cand)
        if path:
            return path, "system"
    # 没找到同族字体，退到通用兜底
    for cand in UNIVERSAL_FALLBACK:
        path = _find_file(cand)
        if path:
            return path, "universal"
    return None, "none"


def measure_ratio(font: pymupdf.Font, text: str, fontsize: float, actual_width: float) -> float:
    """候选字体的计算宽度 / 原文档实际宽度。

    这是判断"是不是同一个字体"的硬指标：
    同字体比值应≈1.0；不同字体因字宽不同会明显偏离。
    """
    if actual_width <= 0 or not text:
        return 0.0
    return font.text_length(text, fontsize=fontsize) / actual_width


def match_font(
    pdf_font_name: str,
    sample_text: str = "",
    fontsize: float = 10.0,
    actual_width: float = 0.0,
    need_chars: str = "",
) -> FontMatch:
    """为一个文本块挑选可用字体。

    sample_text/actual_width 来自原文档的实测值，用于验证度量一致性；
    need_chars 是新文本中出现的字符，用于检查字形覆盖。
    """
    warnings: list[str] = []
    family = identify_family(pdf_font_name)
    path, source = find_system_font(pdf_font_name)

    if path is None:
        return FontMatch(
            pdf_font=pdf_font_name, family=family, font=None, source="none",
            warnings=["系统中找不到任何可用字体"],
        )

    font = _load_pymupdf_font(path)
    ratio = measure_ratio(font, sample_text, fontsize, actual_width)
    verified = actual_width > 0 and abs(ratio - 1.0) <= METRIC_TOLERANCE

    if actual_width > 0 and not verified:
        warnings.append(
            f"度量偏差 {abs(ratio - 1.0) * 100:.1f}%（计算 {ratio:.3f}× 原宽），"
            f"可能不是同一字体"
        )
    if source == "universal":
        warnings.append(f"未找到 {family} 同族字体，已退到通用兜底字体，外观可能有差异")

    # 字形覆盖检查（跳过控制字符 —— 换行制表符在字体里本来就没有字形）
    if need_chars:
        probe = {c for c in need_chars if ord(c) >= 32 and c not in "\u007f"}
        missing = [c for c in probe if font.has_glyph(ord(c)) == 0]
        if missing:
            warnings.append(f"字体缺少 {len(missing)} 个字形：{''.join(sorted(missing))[:30]}")

    return FontMatch(
        pdf_font=pdf_font_name, family=family, font=font, source=source,
        file=path, metric_ratio=ratio, verified=verified, warnings=warnings,
    )


def span_actual_width(span) -> float:
    """从结构模型里的 span 实测渲染宽度。"""
    if not span.chars:
        return 0.0
    return max(c.bbox[2] for c in span.chars) - min(c.bbox[0] for c in span.chars)


def resolve_for_block(pdf_font_name: str, block, new_text: str | None = None) -> FontMatch:
    """为整个块解析字体。

    字体必须按块统一 —— 一段文字里混用两种字体视觉上会很突兀，
    所以只要有一个字符缺字形，就整块换用兜底字体。

    度量样本的选择很关键：两端对齐的段落会把每一行拉伸到满宽，
    用这种行去比对必然得到偏大的比值。所以要挑块内最短的一行
    （通常是段落末行，不会被拉伸）作为样本。
    """
    if not block.lines:
        return match_font(pdf_font_name)

    widths = [l.width for l in block.lines if l.width > 0]
    if not widths:
        return match_font(pdf_font_name)
    max_w = max(widths)

    # 挑最短的一行做样本 —— 两端对齐不会拉伸段落末行，
    # 所以最短行最能反映字体的自然宽度。
    sample_line = min((l for l in block.lines if l.width > 0), key=lambda l: l.width)

    # 只有「多行块里宽度接近满宽的行」才有被两端对齐拉伸的嫌疑。
    # 单行块不存在"末行不拉伸"的规律可用，但它若是短行，同样未被拉伸，
    # 所以照样验证；只有多行且每行都满宽时才无法判断。
    ambiguous = len(widths) >= 2 and sample_line.width >= 0.95 * max_w

    sample = sample_line.text
    fontsize = sample_line.size
    width = sample_line.width

    need = new_text if new_text is not None else block.text
    m = match_font(pdf_font_name, sample, fontsize, width, need_chars=need)

    if ambiguous:
        m.metric_ratio = 0.0
        m.verified = False
        m.warnings.append("块内各行宽度一致且均接近满宽，无法判断是否被两端对齐拉伸")
    return m


def describe(m: FontMatch) -> str:
    status = "✓ 已验证同字体" if m.verified else ("· 未验证" if m.metric_ratio == 0 else "⚠ 度量不符")
    parts = [
        f"{m.pdf_font!r} → {m.family}",
        f"来源={m.source}",
        f"文件={Path(m.file).name if m.file else '无'}",
        f"度量比={m.metric_ratio:.4f}" if m.metric_ratio else "",
        status,
    ]
    line = "  ".join(p for p in parts if p)
    if m.warnings:
        line += "\n      " + "\n      ".join(m.warnings)
    return line
