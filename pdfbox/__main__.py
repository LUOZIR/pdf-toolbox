"""python -m pdfbox —— 统一入口。

默认启动桌面界面；给了 --cli 就走命令行，这样不用记两个模块名：

    python -m pdfbox                      启动界面
    python -m pdfbox 文件.pdf             启动界面并打开该文件
    python -m pdfbox --cli info 文件.pdf  命令行模式
    python -m pdfbox --help               看用法

★ 早先这个入口把 `--help` 当成文件路径去打开，直接抛
  `FileNotFoundError: --help` —— 报错信息完全看不出是"参数写错了"。
  凡是用户可能随手敲的参数，都要在这里先接住。
"""

from __future__ import annotations

import sys

USAGE = """PDF 工具箱 —— 本地运行的 PDF 编辑器

用法：
  python -m pdfbox [文件.pdf]              启动桌面界面（可带一个文件）
  python -m pdfbox --cli <子命令> [参数]    命令行模式
  python -m pdfbox --help                  显示本帮助

桌面界面：
  菜单「页面」  删除 / 重排 / 旋转 / 提取 / 插入 / 合并页面
  菜单「工具」  分析图像占用、压缩文档、转换为 Word/Excel/PPT/图片

命令行子命令（python -m pdfbox --cli <子命令> --help 看详情）：
  info      文档信息与页面类型
  blocks    列出可编辑的文本块
  find      按关键词查找块
  edit      替换某一块的文字
  render    把某页渲染成 PNG
  fonts     列出可用的系统字体
  pages     页面列表 / 删除 / 重排 / 旋转 / 提取 / 插入 / 合并
  compress  压缩文档（重点处理图像型 PDF）
  convert   格式转换 PDF 与 Word/Excel/PPT/图片 互转

例：
  python -m pdfbox --cli edit 合同.pdf --match "金额" --to "￥5000" -o 改后.pdf
  python -m pdfbox --cli pages 文件.pdf --delete 2,5-7 -o 输出.pdf
  python -m pdfbox --cli compress 扫描件.pdf --preset balanced
  python -m pdfbox --cli convert 文件.pdf --to docx --verify

详细说明见 使用说明.md。
"""


def _ensure_stdio() -> None:
    """把标准输出/错误流准备好。

    ★ 两个平台问题，都会让程序"看起来能跑、一输出就崩"：
      1. Windows 控制台默认是 GBK(cp936)。本工具的输出里有中文标点与
         箭头符号，其中一部分 GBK 编码不了 —— 实测 `python -m pdfbox --help`
         在 cmd/PowerShell 下直接抛 UnicodeEncodeError（Git Bash 是 UTF-8
         所以从没暴露过）。所以统一改成 UTF-8 并加 errors="replace"：
         宁可个别字符显示成 `?`，也不能让帮助信息本身把程序打崩。
      2. 打包成窗口模式 exe 后 sys.stdout/stderr 是 None。

    ★ 两个后果，都要处理：
      1. 从终端跑 `pdf-toolbox.exe --cli info 文件.pdf` 时看不到任何输出 ——
         所以先尝试 AttachConsole 接回父进程的控制台；
      2. 接不上时（双击启动）任何 print() 都会炸，所以给个丢弃用的兜底流。
    """
    for name in ("stdout", "stderr"):
        f = getattr(sys, name, None)
        if f is None:
            continue
        try:
            f.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):      # ATTACH_PARENT_PROCESS
            sys.stdout = open("CONOUT$", "w", encoding="utf-8",
                              errors="replace", buffering=1)
            sys.stderr = open("CONOUT$", "w", encoding="utf-8",
                              errors="replace", buffering=1)
            return
    except Exception:
        pass
    try:
        import os
        null = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        null = None
    if sys.stdout is None:
        sys.stdout = null
    if sys.stderr is None:
        sys.stderr = null


def main() -> int:
    _ensure_stdio()
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help", "help", "/?"):
        print(USAGE)
        return 0
    if args and args[0] in ("-V", "--version"):
        print("PDF 工具箱 —— 结构模型驱动的 PDF 编辑器")
        return 0
    if args and args[0] == "--cli":
        from pdfbox.cli import main as cli_main
        return cli_main(args[1:])

    path = args[0] if args else None
    if path and not path.startswith("-"):
        from pathlib import Path
        if not Path(path).exists():
            print(f"文件不存在：{path}\n\n{USAGE}", file=sys.stderr)
            return 2
    elif path:
        print(f"未知参数：{path}\n\n{USAGE}", file=sys.stderr)
        return 2

    from pdfbox.ui.main_window import run
    return run(path)


if __name__ == "__main__":
    raise SystemExit(main())
