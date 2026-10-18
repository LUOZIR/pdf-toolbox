# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：单文件 exe。

为什么用 .spec 而不是一串命令行参数：需要带的东西不是"一个入口"那么简单，
把决策写进文件才能被复查和复现 ——

  · **OCR 模型**（rapidocr 的 3 个 onnx + config.yaml，共 15.6MB）
    必须整包带上，否则打包出来的程序在别人的机器上 OCR 直接不可用。
  · **PySide6 的平台插件**（platforms/qwindows.dll 等）
    少一个就是"双击没反应"这类最难排查的故障。
  · **win32com / pythoncom**：Office→PDF 走 COM，是延迟导入的，
    PyInstaller 的静态分析看不到，必须显式声明。
  · **排除项**：matplotlib / scipy / IPython / tkinter 等本工程完全不用，
    留着会让体积翻倍。

跑：python -m PyInstaller --clean --noconfirm pdfbox.spec
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)

# ---- OCR 模型与配置（缺了它扫描件功能在别人机器上直接不可用）
datas = collect_data_files("rapidocr_onnxruntime")

# ---- 显式声明的隐藏导入：都是运行时才导入的，静态分析看不到
hiddenimports = [
    "pythoncom",            # COM 初始化（Office → PDF）
    "win32com", "win32com.client",
    "pymupdf", "pymupdf.utils",
    "PIL", "PIL.Image",
    "docx", "openpyxl", "pptx",
    "fontTools", "fontTools.ttLib",
    "rapidocr_onnxruntime",
] + collect_submodules("onnxruntime.capi")

# ---- 本工程完全用不到的大件，排掉能省一半体积
excludes = [
    "matplotlib", "scipy", "pandas", "IPython", "jupyter", "notebook",
    "tkinter", "test", "unittest", "pydoc_data", "setuptools", "pip",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia", "PySide6.QtQuick", "PySide6.QtQml",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtBluetooth",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtNetworkAuth",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtPositioning",
    "PySide6.QtRemoteObjects", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio", "PySide6.QtStateMachine", "PySide6.QtSvgWidgets",
    "PySide6.QtTextToSpeech", "PySide6.QtWebChannel", "PySide6.QtWebSockets",
]

a = Analysis(
    ["run_gui.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PDF工具箱",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX 常被杀软误报，收益也不大，不开
    runtime_tmpdir=None,
    console=False,          # 双击直接开界面；从终端跑 --cli 时自动接回控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico"),
)
