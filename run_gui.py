"""打包入口：双击启动桌面界面；带参数时按参数走（--cli / 文件路径 / --help）。

单独一个入口文件，是因为 PyInstaller 需要一个"脚本"作为起点；
真正的逻辑仍在 pdfbox/__main__.py，保证 exe 与 `python -m pdfbox`
两条入口的行为完全一致。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdfbox.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
