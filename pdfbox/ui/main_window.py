"""主窗口：打开文档、浏览结构、编辑文字、保存。

两条编辑路径在界面上是统一的，差异只体现在右侧面板：
  · 文本型文档：显示原字体、匹配置信度
  · 图片型文档：显示覆盖框（可拖动）、实测字号与文字色，并提示保真度限制
"""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)

from ..core.fonts import FAMILY_CANDIDATES, _find_file
from ..core.model import KIND_IMAGE, KIND_TABLE, LAYER_FIXED
from ..core.raster import measure_block_ink
from ..core.session import DOC_IMAGE, DOC_MIXED, DOC_TEXT, Session
from .page_view import PageView


class LoadWorker(QThread):
    """后台加载文档 —— OCR 可能要几十秒，不能卡住界面。"""

    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

    def run(self) -> None:
        try:
            self.session.load(progress=lambda m: self.progress.emit(m))
            self.done.emit(self.session)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF 工具箱")
        self.resize(1480, 940)

        self.session: Session | None = None
        self.view = PageView()
        self._selected = None
        self._cover_user_edited = False
        self._bound_revision = -1
        self._last_font: str | None = None

        self._build_menu()
        self._build_toolbar()
        self._build_center()
        self._build_dock()
        self._build_status()

    # ---------------------------------------------------------- 界面

    def _build_menu(self) -> None:
        """菜单栏：页面管理、压缩、格式转换。

        这三类操作跟"改字"是两种不同性质的事 —— 改字是增量的、
        可撤销的编辑；页面操作与压缩/转换是"对整份文档另存一份"。
        放在菜单里而不是工具栏，是为了让这个区别在界面上就看得见：
        工具栏那一排是编辑动作，菜单里的是文档级动作。
        """
        mb = self.menuBar()

        m = mb.addMenu("文件(&F)")
        m.addAction("打开…", self.open_file, QKeySequence("Ctrl+O"))
        m.addAction("保存", self.save_file, QKeySequence("Ctrl+S"))
        m.addAction("另存为…", self.save_as, QKeySequence("Ctrl+Shift+S"))
        m.addSeparator()
        m.addAction("退出", self.close)

        m = mb.addMenu("页面(&P)")
        m.addAction("页面列表…", self.show_pages_dialog)
        m.addSeparator()
        self.act_del_page = m.addAction("删除当前页", self.delete_current_page,
                                       QKeySequence("Ctrl+Delete"))
        m.addAction("删除指定页…", self.delete_pages_dialog)
        m.addSeparator()
        m.addAction("上移当前页", lambda: self.move_current_page(-1),
                    QKeySequence("Ctrl+Up"))
        m.addAction("下移当前页", lambda: self.move_current_page(1),
                    QKeySequence("Ctrl+Down"))
        m.addAction("调整页面顺序…", self.reorder_pages_dialog)
        m.addSeparator()
        m.addAction("旋转当前页 90°", lambda: self.rotate_current_page(90))
        m.addAction("旋转当前页 -90°", lambda: self.rotate_current_page(-90))
        m.addSeparator()
        m.addAction("提取指定页…", self.extract_pages_dialog)
        m.addAction("插入其他 PDF…", self.insert_pdf_dialog)
        m.addAction("合并 PDF…", self.merge_pdfs_dialog)

        m = mb.addMenu("工具(&T)")
        m.addAction("分析图像占用…", self.analyze_images_dialog)
        m.addAction("压缩文档…", self.compress_dialog)
        m.addSeparator()
        m.addAction("转换为 Word…", lambda: self.convert_dialog("docx"))
        m.addAction("转换为 Excel…", lambda: self.convert_dialog("xlsx"))
        m.addAction("转换为 PPT…", lambda: self.convert_dialog("pptx"))
        m.addAction("导出为图片…", lambda: self.convert_dialog("png"))
        m.addAction("其他文档转 PDF…", self.office_to_pdf_dialog)

    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            tb.addAction(a)
            return a

        act("打开", self.open_file, "Ctrl+O", "打开 PDF")
        act("保存", self.save_file, "Ctrl+S", "保存到原文件")
        act("另存为", self.save_as, "Ctrl+Shift+S", "另存为新文件")
        tb.addSeparator()
        act("撤销", self.do_undo, "Ctrl+Z")
        act("重做", self.do_redo, "Ctrl+Y")
        tb.addSeparator()
        act("缩小", lambda: self._zoom_by(1 / 1.25), "Ctrl+-")
        act("放大", lambda: self._zoom_by(1.25), "Ctrl+=")
        act("适合宽度", self.fit_width, "Ctrl+0")
        tb.addSeparator()

        self.chk_overlay = QCheckBox("显示结构框")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setToolTip("显示/隐藏块分类叠加层")
        self.chk_overlay.toggled.connect(self._toggle_overlay)
        tb.addWidget(self.chk_overlay)

        self.chk_charbox = QCheckBox("显示单字框(调试)")
        self.chk_charbox.setToolTip("把每个字各自的紧致框画出来 —— 用来看倾斜文字是否贴合")
        self.chk_charbox.toggled.connect(self._toggle_charbox)
        tb.addWidget(self.chk_charbox)

        self.chk_fixed = QCheckBox("显示固定层")
        self.chk_fixed.setChecked(False)
        self.chk_fixed.setToolTip("页眉/页脚/水印 —— 这些元素永不参与重排")
        self.chk_fixed.toggled.connect(self._toggle_fixed)
        tb.addWidget(self.chk_fixed)

        tb.addSeparator()
        tb.addWidget(QLabel("  页 "))
        self.sp_page = QSpinBox()
        self.sp_page.setMinimum(1)
        self.sp_page.setMaximum(1)
        self.sp_page.valueChanged.connect(self._page_changed)
        tb.addWidget(self.sp_page)
        self.lbl_pages = QLabel(" / 0")
        tb.addWidget(self.lbl_pages)

    def _build_center(self) -> None:
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.view)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidgetResizable(False)
        self.scroll.setBackgroundRole(self.scroll.backgroundRole())
        self.setCentralWidget(self.scroll)

    def _build_dock(self) -> None:
        dock = QDockWidget("编辑", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        dock.setMinimumWidth(340)

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # ---- 块信息
        gb_info = QGroupBox("选中的块")
        fi = QFormLayout(gb_info)
        fi.setLabelAlignment(Qt.AlignRight)
        self.lbl_kind = QLabel("—")
        self.lbl_layer = QLabel("—")
        self.lbl_size = QLabel("—")
        self.lbl_font = QLabel("—")
        self.lbl_conf = QLabel("—")
        self.lbl_bbox = QLabel("—")
        for k, w in (("类型", self.lbl_kind), ("图层", self.lbl_layer),
                     ("字号", self.lbl_size), ("字体", self.lbl_font),
                     ("置信度", self.lbl_conf), ("位置", self.lbl_bbox)):
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            w.setWordWrap(True)
            fi.addRow(k, w)
        outer.addWidget(gb_info)

        # ---- 编辑
        gb_edit = QGroupBox("修改为")
        ev = QVBoxLayout(gb_edit)


        self.ed_text = QTextEdit()
        self.ed_text.setPlaceholderText("在页面上单击选中一段文字，然后在这里输入新内容。")
        self.ed_text.setMinimumHeight(110)
        f = QFont("Microsoft YaHei", 10)
        self.ed_text.setFont(f)
        ev.addWidget(self.ed_text)


        row = QHBoxLayout()
        row.addWidget(QLabel("字体"))
        self.cb_font = QComboBox()
        self._fill_fonts()
        row.addWidget(self.cb_font, 1)
        btn_font = QPushButton("…")
        btn_font.setFixedWidth(30)
        btn_font.clicked.connect(self._pick_font_file)
        row.addWidget(btn_font)
        ev.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("字号"))
        self.sp_size = QDoubleSpinBox()
        self.sp_size.setRange(3.0, 300.0)
        self.sp_size.setDecimals(1)
        self.sp_size.setSingleStep(0.5)
        self.sp_size.setValue(10.5)
        self.sp_size.setToolTip("留空/勾选自动则按原文字号")
        row2.addWidget(self.sp_size)
        self.chk_auto_size = QCheckBox("自动")
        self.chk_auto_size.setChecked(True)
        self.chk_auto_size.toggled.connect(lambda v: self.sp_size.setEnabled(not v))
        self.sp_size.setEnabled(False)
        row2.addWidget(self.chk_auto_size)
        ev.addLayout(row2)

        row_w = QHBoxLayout()
        row_w.addWidget(QLabel("字重"))
        self.cb_weight = QComboBox()
        self.cb_weight.addItem("常规（默认）", "regular")
        self.cb_weight.addItem("粗体", "bold")
        self.cb_weight.setToolTip(
            "图片型文档：原文若是粗体，重绘时选粗体才像。"
            "自动判定在真实文档上不可靠，所以由你决定。")
        row_w.addWidget(self.cb_weight, 1)
        ev.addLayout(row_w)

        row3 = QHBoxLayout()
        self.lbl_color = QLabel("文字色：自动")
        row3.addWidget(self.lbl_color)
        self.chk_custom_color = QCheckBox("自定义")
        self.chk_custom_color.setChecked(False)
        row3.addWidget(self.chk_custom_color)
        row3.addStretch(1)
        ev.addLayout(row3)

        self.lbl_extend = QLabel("")
        self.lbl_extend.setWordWrap(True)
        self.lbl_extend.setStyleSheet(
            "color:#0a5c66; background:#e3f6f8; padding:5px; border-radius:4px;")
        self.lbl_extend.setVisible(False)
        ev.addWidget(self.lbl_extend)

        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet(
            "color:#155724; background:#eaf7ee; padding:5px; border-radius:4px;")
        ev.addWidget(self.lbl_plan)

        self.btn_apply = QPushButton("应用修改")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setMinimumHeight(32)
        self.btn_apply.clicked.connect(self.apply_edit)
        ev.addWidget(self.btn_apply)

        outer.addWidget(gb_edit)

        # ---- 覆盖范围（图片型）
        self.gb_cover = QGroupBox("覆盖范围（图片型文档）")
        cv = QVBoxLayout(self.gb_cover)
        self.lbl_cover = QLabel("—")
        self.lbl_cover.setWordWrap(True)
        cv.addWidget(self.lbl_cover)
        self.chk_show_cover = QCheckBox("在页面上显示覆盖框（可拖动调整）")
        self.chk_show_cover.setChecked(True)
        self.chk_show_cover.toggled.connect(self._toggle_cover)
        cv.addWidget(self.chk_show_cover)
        rowc = QHBoxLayout()
        btn_reset_cover = QPushButton("按实测墨迹重算")
        btn_reset_cover.clicked.connect(self._reset_cover)
        rowc.addWidget(btn_reset_cover)
        cv.addLayout(rowc)
        hint = QLabel(
            "提示：OCR 的检测框在真实文档上可能严重失真。"
            "如果覆盖框没框正要改的那行字，请直接拖动橙色框调整。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8a6d00; background:#fff8e1; padding:6px; border-radius:4px;")
        cv.addWidget(hint)
        self.gb_cover.setVisible(False)
        outer.addWidget(self.gb_cover)

        # ---- 报告
        gb_rep = QGroupBox("变更报告")
        rv = QVBoxLayout(gb_rep)
        self.tx_report = QTextEdit()
        self.tx_report.setReadOnly(True)
        self.tx_report.setMinimumHeight(110)
        self.tx_report.setFont(QFont("Consolas", 9))
        rv.addWidget(self.tx_report)
        outer.addWidget(gb_rep)

        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_status(self) -> None:
        self.lbl_status = QLabel("就绪 —— 用「打开」载入一个 PDF")
        self.statusBar().addWidget(self.lbl_status, 1)

    def _fill_fonts(self) -> None:
        self.cb_font.clear()
        for fam in ("simsun", "simhei", "msyh", "dengxian", "kaiti",
                    "notosanssc", "notoserifsc", "arial", "times"):
            for cand in FAMILY_CANDIDATES.get(fam, []):
                p = _find_file(cand)
                if p:
                    self.cb_font.addItem(f"{fam}  ({Path(p).name})", p)
                    break

    # ---------------------------------------------------------- 打开/保存

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 PDF", "", "PDF 文件 (*.pdf)")
        if not path:
            return
        self.load_path(path)

    def load_path(self, path: str) -> None:
        sess = Session(path)
        dlg = QProgressDialog("正在打开…", "取消", 0, 0, self)
        dlg.setWindowTitle("载入文档")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        worker = LoadWorker(sess)
        worker.progress.connect(dlg.setLabelText)
        worker.done.connect(lambda s: (dlg.close(), self._loaded(s)))
        worker.failed.connect(lambda msg: (dlg.close(), self._load_failed(msg)))
        self._worker = worker
        worker.start()

    def _load_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "打开失败", msg)
        self.lbl_status.setText("打开失败")

    def _loaded(self, sess: Session) -> None:
        self.session = sess
        self._bound_revision = -1
        self.view.set_document(sess.doc, sess.structure)
        self.sp_page.setMaximum(max(1, sess.page_count))
        self.sp_page.setValue(1)
        self.lbl_pages.setText(f" / {sess.page_count}")
        self._page_changed(1)

        kind_name = {DOC_TEXT: "文本型（有文字层）",
                     DOC_IMAGE: "图片型（OCR 模式）",
                     DOC_MIXED: "混合型"}[sess.doc_type]
        self.setWindowTitle(f"PDF 工具箱 —— {sess.path.name}")
        self.lbl_status.setText(f"{sess.summary()}   文档类型：{kind_name}")

        if sess.doc_type in (DOC_IMAGE, DOC_MIXED):
            QMessageBox.information(
                self, "图片型文档",
                "这个 PDF 没有文字层，内容是位图（扫描件/截图导出）。\n\n"
                "已用 OCR 建立可编辑的文本层。请注意两点：\n"
                "  1. OCR 识别结果可能有错字，编辑时请以页面上看到的原文为准；\n"
                "  2. 修改方式是用背景覆盖原像素后重绘，原像素无法真正移除，\n"
                "     因此本类文档做不到字节级无损。\n\n"
                "单击选中一段文字后，右侧会出现可拖动的覆盖框。",
            )
        self.fit_width()

    def save_file(self) -> None:
        if not self.session:
            return
        try:
            p = self.session.save()
        except RuntimeError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.lbl_status.setText(f"已保存：{p}")

    def save_as(self) -> None:
        if not self.session:
            return
        default = str(self.session.path.with_name(
            self.session.path.stem + "_edited.pdf"))
        path, _ = QFileDialog.getSaveFileName(self, "另存为", default, "PDF 文件 (*.pdf)")
        if not path:
            return
        try:
            self.session.save(path)
        except RuntimeError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.setWindowTitle(f"PDF 工具箱 —— {Path(path).name}")
        self.lbl_status.setText(f"已另存为：{path}")

    # ---------------------------------------------------------- 视图

    def _page_changed(self, value: int) -> None:
        if not self.session:
            return
        self._rebind()
        pno = value - 1
        blocks = self.session.page_blocks(pno)
        self.view.set_page(pno, blocks)
        self._selected = None
        self._clear_panel()

    def _rebind(self) -> None:
        """把界面持有的文档与结构引用换成会话当前的那一份。

        撤销/重做会整体换掉 Document 对象（PyMuPDF 不能原地重置），
        编辑也会重建结构。界面若继续用旧引用，下一次渲染就会撞上
        「document closed」—— 这就是之前「编辑后撤销就报错」的根因。
        所以每次刷新页面前都无条件重绑一次。
        """
        if not self.session or self.session.doc is None:
            return
        if self._bound_revision != self.session.revision:
            self._bound_revision = self.session.revision
            self.view.set_document(self.session.doc, self.session.structure)
        else:
            self.view.set_document(self.session.doc, self.session.structure)

    def _zoom_by(self, factor: float) -> None:
        self.view.set_zoom(self.view.zoom * factor)
        self.lbl_status.setText(f"缩放 {self.view.zoom*100:.0f}%")

    def fit_width(self) -> None:
        if not self.session:
            return
        w = self.scroll.viewport().width() - 28
        page_w = self.session.doc[self.view.page_no].rect.width
        self.view.set_zoom(max(0.25, w / page_w))

    def _toggle_overlay(self, on: bool) -> None:
        self.view.show_overlays = on
        self.view.update()

    def _toggle_fixed(self, on: bool) -> None:
        self.view.show_fixed_layer = on
        self.view.update()

    def _toggle_charbox(self, on: bool) -> None:
        self.view.show_char_boxes = on
        if on:
            self._refresh_char_boxes()
        else:
            self.view.set_char_boxes(None)

    def _refresh_char_boxes(self) -> None:
        if not self.session or self._selected is None:
            self.view.set_char_boxes(None)
            return
        try:
            from ..core.raster import per_char_boxes
            b = self._selected
            if not b.font:      # 图片型才需要逐字抠（倾斜问题主要出在扫描件）
                boxes = per_char_boxes(self.session.doc[b.page_no], b)
            else:
                boxes = [(c, tuple(c.bbox))
                         for line in b.lines for sp in line.spans for c in sp.chars]
            self.view.set_char_boxes(boxes)
        except Exception:
            self.view.set_char_boxes(None)

    def _toggle_cover(self, on: bool) -> None:
        self.view.set_cover(self._pending_cover(), enabled=on)

    # ---------------------------------------------------------- 选中

    def _clear_panel(self) -> None:
        for w in (self.lbl_kind, self.lbl_layer, self.lbl_size,
                  self.lbl_font, self.lbl_conf, self.lbl_bbox):
            w.setText("—")
        self.ed_text.clear()
        self.btn_apply.setEnabled(False)
        self.gb_cover.setVisible(False)
        self.view.set_cover(None, enabled=False)

    def on_block_clicked(self, block) -> None:
        self._selected = block
        self._cover_user_edited = False
        if block is None:
            self._clear_panel()
            return

        self.lbl_kind.setText(block.kind)
        self.lbl_layer.setText(
            "固定层（不参与重排）" if block.layer == LAYER_FIXED else "流式层（参与重排）")
        self.lbl_size.setText(f"{block.size:.2f} pt")
        self.lbl_font.setText(block.font or "（OCR 未知）")
        self.lbl_conf.setText(f"{block.confidence:.2f}")
        self.lbl_bbox.setText(
            f"x {block.bbox[0]:.0f}..{block.bbox[2]:.0f}   "
            f"y {block.bbox[1]:.0f}..{block.bbox[3]:.0f}")

        self.ed_text.setPlainText(block.text)
        self.ed_text.textChanged.connect(self._preview_plan)
        if getattr(self, "chk_charbox", None) is not None and self.chk_charbox.isChecked():
            self._refresh_char_boxes()

        is_raster = bool(block.raw and block.raw.get("ocr")) or not block.font
        self.btn_apply.setEnabled(bool(block.lines) and block.kind not in
                                  (KIND_IMAGE, KIND_TABLE))
        self.gb_cover.setVisible(is_raster)
        if is_raster:
            self._refresh_cover_info(block)
        else:
            self.view.set_cover(None, enabled=False)
            self.chk_auto_size.setChecked(True)

    def _pending_cover(self):
        return getattr(self, "_cover_rect", None)

    def _refresh_cover_info(self, block) -> None:
        if not self.session or self.session.doc is None:
            self.lbl_cover.setText("（文档未就绪）")
            return
        try:
            ink = measure_block_ink(self.session.doc[block.page_no], block)
        except Exception as e:
            self.lbl_cover.setText(f"测量失败：{e}")
            return
        self._cover_rect = ink.rect
        self.view.set_cover(ink.rect, enabled=self.chk_show_cover.isChecked())

        bw = block.bbox[2] - block.bbox[0]
        iw = ink.rect[2] - ink.rect[0]
        # 字重：把实测笔画比摆出来，并给出建议 —— 不再让工具猜
        wgt = ""
        try:
            from ..core.raster import REGULAR_STROKE, BOLD_STROKE
            sr = ink.stroke_ratio
            if sr > 0:
                mid = (REGULAR_STROKE + BOLD_STROKE) / 2
                suggest = "粗体" if sr >= mid else "常规"
                wgt = (f"笔画粗细：{sr:.3f}"
                       f"（常规≈{REGULAR_STROKE:.2f}，粗体≈{BOLD_STROKE:.2f}）"
                       f" → 建议用{suggest}")
                idx = self.cb_weight.findData("bold" if sr >= mid else "regular")
                if idx >= 0:
                    self.cb_weight.setCurrentIndex(idx)
        except Exception:
            pass

        lines = [f"实测墨迹：宽 {iw:.1f}pt  高 {ink.height:.1f}pt",
                 f"推算字号：{ink.size_for(self.ed_text.toPlainText() or block.text):.2f}pt",
                 f"文字色：{tuple(round(v*255) for v in ink.color)}",
                 f"OCR 框宽：{bw:.1f}pt"]
        if ink.rule_span > 0:
            lines.append(f"检测到横线（下划线）厚 {ink.rule_span:.1f}pt，已排除在文字测量之外")
        if wgt:
            lines.append(wgt)
        if iw > bw * 2:
            lines.append("⚠ OCR 框与实测墨迹相差悬殊，请确认覆盖框是否框正")
        self.lbl_cover.setText("\n".join(lines))

    def on_cover_changed(self, rect) -> None:
        self._cover_rect = rect
        self._cover_user_edited = True
        self.lbl_cover.setText(
            f"已手工调整覆盖框：x {rect[0]:.1f}..{rect[2]:.1f}  "
            f"y {rect[1]:.1f}..{rect[3]:.1f}")

    def _reset_cover(self) -> None:
        if self._selected is None:
            return
        self._cover_user_edited = False
        self._refresh_cover_info(self._selected)


    def _compose_new_text(self) -> str | None:
        """输入框里的内容就是这一段的全部新文字。

        不需要用户指定"改了哪几个字"——工具自己对比新旧文字算出改动范围：
        文本型走 typeset 的逐字 diff，图片型走 raster_typeset 的逐字 diff。
        """
        if self._selected is None:
            return None
        return self.ed_text.toPlainText()

    def _preview_plan(self) -> None:
        """预览这次改动会影响多少文字，并实时算出加长后的框。"""
        if not self.session or self._selected is None:
            return
        block = self._selected
        new = self._compose_new_text()
        if not new or new == block.text:
            self.lbl_plan.setText("")
            self.view.set_preview(None)
            return

        self._update_extend_preview(block, new)
        try:
            from ..core.typeset import plan_typeset
            m = self.session.match_for(block, new)
            if not block.font:
                # 图片型：走像素块的活字方案
                from ..core.raster_typeset import (build_patch_case,
                                                   plan_raster_typeset)
                case = build_patch_case(self.session.doc[block.page_no], block)
                self.lbl_plan.setText(plan_raster_typeset(
                    self.session.doc[block.page_no], block, new, case).summary())
            else:
                self.lbl_plan.setText(plan_typeset(block, new, m.font).summary())
        except Exception:
            self.lbl_plan.setText("")

    def _update_extend_preview(self, block, new_text: str) -> None:
        """算出内容加长后占多宽，画成青色预览框。

        用户要求：装不下时优先加长文本框，而且要在界面上先看到。
        这里就用同一套 available_width 算——保证预览和实际执行一致。
        """
        if not block.lines or not self.session:
            self.view.set_preview(None)
            return
        try:
            from ..core.typeset import available_width
            m = self.session.match_for(block, new_text)
            size = block.size
            flat = "".join(ch for ch in new_text if ch != chr(10))
            need = m.font.text_length(flat, fontsize=size) if m.ok else 0.0
            cur = max(l.along1 - l.along0 for l in block.lines)
            room, note = available_width(
                block, self.session.structure,
                self.session.doc[block.page_no].rect.width)
            want = min(max(cur, need + 4), room)
            rect = (block.bbox[0], block.bbox[1],
                    block.bbox[0] + want, block.bbox[3])
            self.view.set_preview(rect)
            if need > room + 0.5:
                self.lbl_extend.setVisible(True)
                self.lbl_extend.setText(
                    f"内容需 {need:.0f}pt，右侧最多只能加到 {room:.0f}pt"
                    f"（{note}）→ 需要折行")
            elif need > cur + 0.5:
                self.lbl_extend.setVisible(True)
                self.lbl_extend.setText(
                    f"内容需 {need:.0f}pt，原宽 {cur:.0f}pt → 加长到 {want:.0f}pt（{note}）")
            else:
                self.lbl_extend.setVisible(False)
                self.lbl_extend.setText("")
        except Exception:
            self.view.set_preview(None)
            self.lbl_extend.setText("")

    def _pick_font_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择字体文件", "C:/Windows/Fonts",
            "字体文件 (*.ttf *.ttc *.otf)")
        if path:
            self.cb_font.addItem(Path(path).name, path)
            self.cb_font.setCurrentIndex(self.cb_font.count() - 1)

    # ---------------------------------------------------------- 编辑

    def apply_edit(self) -> None:
        if not self.session or self._selected is None:
            return
        block = self._selected
        new_text = self._compose_new_text()
        if new_text is None or not new_text.strip():
            QMessageBox.information(self, "内容为空", "新内容不能为空。要删除整段请用删除功能。")
            return
        if new_text == block.text:
            self.lbl_status.setText("内容没有变化")
            return

        is_raster = bool(block.raw and block.raw.get("ocr")) or not block.font
        font_file = None
        if is_raster:
            # 字体自动匹配：优先沿用本会话上一次用过的字体
            # （同一份文档通常通篇一个字体，改第二处时不用再选一遍）
            font_file = self.cb_font.currentData() or self._last_font
            if not font_file:
                font_file = self._auto_pick_font(new_text)
            if not font_file:
                QMessageBox.warning(self, "需要字体",
                                    "图片型文档无法得知原字体，请先指定一个字体文件。")
                return
            if font_file != self.cb_font.currentData():
                idx = self.cb_font.findData(font_file)
                if idx >= 0:
                    self.cb_font.setCurrentIndex(idx)
            self._last_font = font_file

        size = None if self.chk_auto_size.isChecked() else self.sp_size.value()

        cover = None
        if is_raster and self._cover_user_edited:
            cover = getattr(self, "_cover_rect", None)

        tcolor = None
        if is_raster and self.chk_custom_color.isChecked():
            tcolor = (0.0, 0.0, 0.0)

        try:
            rec = self.session.replace_text(
                block, new_text, font_file=font_file,
                fontsize=size, cover_rect=cover, text_color=tcolor,
                weight=self.cb_weight.currentData() if is_raster else "auto",
            )
        except RuntimeError as e:
            QMessageBox.critical(self, "编辑失败", str(e))
            return
        except Exception:
            QMessageBox.critical(self, "编辑失败", traceback.format_exc())
            return

        if rec.path == "noop":
            # 没有实际改动：把原因直接显示出来，不要静默返回
            msg = rec.warned[0] if rec.warned else "内容没有变化"
            self.lbl_status.setText("未修改：" + msg)
            QMessageBox.information(self, "没有改动", msg)
            self._append_report(rec)
            return

        self._page_changed(self.sp_page.value())
        self._append_report(rec)
        self.lbl_status.setText("✓ " + rec.summary())

    def _auto_pick_font(self, text: str) -> str | None:
        """按文字脚本自动挑一个合适的字体文件（中文用宋体，西文用 Arial）。"""
        from ..core.fonts import _find_file
        has_cjk = any("一" <= ch <= "鿿" for ch in text)
        cands = (["simsun.ttc", "NotoSansSC-VF.ttf", "msyh.ttc"]
                 if has_cjk else ["arial.ttf", "calibri.ttf"])
        for c in cands:
            p = _find_file(c)
            if p:
                return p
        return None

    def _append_report(self, rec) -> None:
        lines = [f"● {rec.summary()}"]
        if rec.method:
            lines.append(f"    {rec.method}")
        for w in rec.warned:
            lines.append(f"    ⚠ {w}")
        self.tx_report.append("\n".join(lines))

    def do_undo(self) -> None:
        if not self.session:
            return
        try:
            ok = self.session.undo()
        except Exception:
            QMessageBox.critical(self, "撤销失败", traceback.format_exc())
            return
        if ok:
            self._bound_revision = -1        # 强制重绑新文档
            self._page_changed(self.sp_page.value())
            self.lbl_status.setText("已撤销")
        else:
            self.lbl_status.setText("没有可撤销的操作")

    def do_redo(self) -> None:
        if not self.session:
            return
        try:
            ok = self.session.redo()
        except Exception:
            QMessageBox.critical(self, "重做失败", traceback.format_exc())
            return
        if ok:
            self._bound_revision = -1
            self._page_changed(self.sp_page.value())
            self.lbl_status.setText("已重做")
        else:
            self.lbl_status.setText("没有可重做的操作")

    # ------------------------------------------------- 页面管理（菜单动作）

    def _need_session(self) -> bool:
        if self.session is None:
            QMessageBox.information(self, "还没有文档", "请先打开一个 PDF。")
            return False
        return True

    def _ask(self, title: str, label: str, default: str = "") -> str | None:
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, title, label, text=default)
        if not ok or not text.strip():
            return None
        return text.strip()

    def _after_page_op(self, rep) -> None:
        """页面操作后统一刷新界面。

        ★ 必须整篇重绑：页面增删会改变页码，视图缓存的 doc/structure 引用
          和"当前是第几页"全部失效，只重绘当前页会看到别页的内容。
        """
        s = self.session
        self._bound_revision = -1
        n = s.page_count
        self.sp_page.blockSignals(True)
        self.sp_page.setMaximum(max(1, n))
        cur = min(self.sp_page.value(), n)
        self.sp_page.setValue(max(1, cur))
        self.sp_page.blockSignals(False)
        self.lbl_pages.setText(f" / {n}")
        self._page_changed(self.sp_page.value())
        self.lbl_status.setText("✓ " + rep.text().replace("\n", "   "))
        lines = [f"● {rep.text()}"]
        self.tx_report.append("\n".join(lines))

    def _page_op(self, fn, *a, **kw) -> None:
        if not self._need_session():
            return
        try:
            rep = fn(*a, **kw)
        except Exception as e:
            QMessageBox.critical(self, "页面操作失败",
                                 f"{type(e).__name__}: {e}")
            return
        self._after_page_op(rep)

    def show_pages_dialog(self) -> None:
        """页面总览：每页尺寸、旋转、文字量、图片数、首行文字。"""
        if not self._need_session():
            return
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox,
                                       QTableWidget, QTableWidgetItem)
        s = self.session
        dlg = QDialog(self)
        dlg.setWindowTitle(f"页面列表 —— {s.path.name}")
        dlg.resize(880, 560)
        lay = QVBoxLayout(dlg)
        tb = QTableWidget(s.page_count, 6, dlg)
        tb.setHorizontalHeaderLabels(
            ["页", "尺寸 pt", "旋转", "文字", "图", "首行文字"])
        tb.setSelectionBehavior(QTableWidget.SelectRows)
        tb.setEditTriggers(QTableWidget.NoEditTriggers)
        for i in range(s.page_count):
            page = s.doc[i]
            r = page.rect
            txt = page.get_text("text").strip()
            first = next((l for l in txt.splitlines() if l.strip()), "")
            cells = [str(i + 1), f"{r.width:.0f}×{r.height:.0f}",
                     f"{page.rotation}°", str(len(txt)),
                     str(len(page.get_images(full=True))), first[:44]]
            for c, v in enumerate(cells):
                tb.setItem(i, c, QTableWidgetItem(v))
        tb.resizeColumnsToContents()
        lay.addWidget(tb)

        act = QHBoxLayout()
        b_del = QPushButton("删除所选页")
        b_up = QPushButton("上移")
        b_down = QPushButton("下移")
        b_ext = QPushButton("提取所选页…")
        b_rot = QPushButton("旋转 90°")
        for b in (b_del, b_up, b_down, b_ext, b_rot):
            act.addWidget(b)
        act.addStretch(1)
        lay.addLayout(act)
        bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        lay.addWidget(bb)

        def picked() -> list[int]:
            return sorted({ix.row() for ix in tb.selectedIndexes()})

        def do_delete() -> None:
            idx = picked()
            if not idx:
                QMessageBox.information(dlg, "未选择", "请先选中要删除的页。")
                return
            ok = QMessageBox.question(
                dlg, "确认删除",
                f"将删除第 {', '.join(str(i + 1) for i in idx)} 页"
                f"（共 {s.page_count} 页 → {s.page_count - len(idx)} 页）。\n"
                "可以撤销。")
            if ok != QMessageBox.Yes:
                return
            self._page_op(s.delete_pages, idx)
            dlg.accept()
            self.show_pages_dialog()

        def do_move(step: int) -> None:
            idx = picked()
            if len(idx) != 1:
                QMessageBox.information(dlg, "只能移动一页",
                                        "请只选中一页再移动。")
                return
            i = idx[0]
            j = i + step
            if not (0 <= j < s.page_count):
                return
            order = list(range(s.page_count))
            order[i], order[j] = order[j], order[i]
            self._page_op(s.reorder_pages, order)
            dlg.accept()
            self.show_pages_dialog()

        def do_extract() -> None:
            idx = picked()
            if not idx:
                return
            out, _ = QFileDialog.getSaveFileName(
                dlg, "提取为新 PDF",
                str(s.path.with_name(f"{s.path.stem}_提取.pdf")), "PDF (*.pdf)")
            if not out:
                return
            from ..core.pages import extract_pages
            extract_pages(s.path, idx, out)
            QMessageBox.information(dlg, "已提取",
                                    f"第 {', '.join(str(i + 1) for i in idx)}"
                                    f" 页已另存为\n{out}")

        def do_rotate() -> None:
            idx = picked()
            if not idx:
                return
            self._page_op(s.rotate_pages, idx, 90)
            dlg.accept()
            self.show_pages_dialog()

        b_del.clicked.connect(do_delete)
        b_up.clicked.connect(lambda: do_move(-1))
        b_down.clicked.connect(lambda: do_move(1))
        b_ext.clicked.connect(do_extract)
        b_rot.clicked.connect(do_rotate)
        bb.rejected.connect(dlg.reject)
        dlg.exec()

    def delete_current_page(self) -> None:
        if not self._need_session():
            return
        s = self.session
        i = self.sp_page.value() - 1
        if s.page_count <= 1:
            QMessageBox.information(self, "不能删除", "PDF 至少要保留一页。")
            return
        if QMessageBox.question(
                self, "确认删除",
                f"删除第 {i + 1} 页（共 {s.page_count} 页）？可以撤销。"
        ) != QMessageBox.Yes:
            return
        self._page_op(s.delete_pages, [i])

    def delete_pages_dialog(self) -> None:
        spec = self._ask("删除页面", "要删除的页码（如 2,5,7-9）：")
        if not spec:
            return
        if not self._need_session():
            return
        from ..core.pages import PageOpError, parse_page_spec
        s = self.session
        try:
            idx = parse_page_spec(spec, s.page_count)
        except PageOpError as e:
            QMessageBox.warning(self, "页码不对", str(e))
            return
        if len(idx) >= s.page_count:
            QMessageBox.warning(self, "不能全部删除", "PDF 至少要保留一页。")
            return
        self._page_op(s.delete_pages, idx)

    def move_current_page(self, step: int) -> None:
        if not self._need_session():
            return
        s = self.session
        i = self.sp_page.value() - 1
        j = i + step
        if not (0 <= j < s.page_count):
            self.lbl_status.setText("已经到头了")
            return
        order = list(range(s.page_count))
        order[i], order[j] = order[j], order[i]
        self._page_op(s.reorder_pages, order)
        self.sp_page.setValue(j + 1)

    def reorder_pages_dialog(self) -> None:
        if not self._need_session():
            return
        s = self.session
        cur = ",".join(str(i + 1) for i in range(s.page_count))
        spec = self._ask("调整页面顺序",
                         f"按新顺序输入页码（当前共 {s.page_count} 页）。\n"
                         f"例如把最后一页挪到最前：{s.page_count},1-"
                         f"{max(1, s.page_count - 1)}\n"
                         "也可以用分号分段，如 1-3;6;4-5", cur)
        if not spec:
            return
        from ..core.pages import PageOpError, parse_page_spec
        try:
            order = parse_page_spec(spec, s.page_count)
        except PageOpError as e:
            QMessageBox.warning(self, "页码不对", str(e))
            return
        if sorted(order) != list(range(s.page_count)):
            QMessageBox.warning(
                self, "页码不完整",
                f"新顺序必须把 {s.page_count} 页各列一次。\n"
                f"当前给了 {len(order)} 页："
                f"{', '.join(str(i + 1) for i in order)}")
            return
        self._page_op(s.reorder_pages, order)

    def rotate_current_page(self, degrees: int) -> None:
        if not self._need_session():
            return
        self._page_op(self.session.rotate_pages,
                      [self.sp_page.value() - 1], degrees)

    def extract_pages_dialog(self) -> None:
        if not self._need_session():
            return
        s = self.session
        spec = self._ask("提取页面",
                         f"要提取的页码（共 {s.page_count} 页，如 1,3-5）：")
        if not spec:
            return
        from ..core.pages import PageOpError, extract_pages, parse_page_spec
        try:
            idx = parse_page_spec(spec, s.page_count)
        except PageOpError as e:
            QMessageBox.warning(self, "页码不对", str(e))
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "提取为新 PDF",
            str(s.path.with_name(f"{s.path.stem}_提取.pdf")), "PDF (*.pdf)")
        if not out:
            return
        try:
            extract_pages(s.path, idx, out)
        except Exception as e:
            QMessageBox.critical(self, "提取失败", f"{type(e).__name__}: {e}")
            return
        QMessageBox.information(
            self, "已提取",
            f"第 {', '.join(str(i + 1) for i in idx)} 页 →\n{out}")

    def insert_pdf_dialog(self) -> None:
        if not self._need_session():
            return
        s = self.session
        src, _ = QFileDialog.getOpenFileName(self, "选择要插入的 PDF", "",
                                             "PDF (*.pdf)")
        if not src:
            return
        import pymupdf
        try:
            d = pymupdf.open(src)
            n = d.page_count
            d.close()
        except Exception as e:
            QMessageBox.critical(self, "打不开", f"{type(e).__name__}: {e}")
            return
        spec = self._ask("插入页面",
                         f"要插入的页码（共 {n} 页，留空表示全部；"
                         f"如 1,3-5）：", "")
        from ..core.pages import PageOpError, parse_page_spec
        idx = None
        if spec:
            try:
                idx = parse_page_spec(spec, n)
            except PageOpError as e:
                QMessageBox.warning(self, "页码不对", str(e))
                return
        at = self.sp_page.value() - 1        # 插到当前页之前
        self._page_op(s.insert_pdf, src, at, idx)

    def merge_pdfs_dialog(self) -> None:
        if not self._need_session():
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要合并进来的 PDF（按选择顺序追加）", "", "PDF (*.pdf)")
        if not paths:
            return
        from ..core.pages import PageOpError, merge_pdfs
        s = self.session
        out, _ = QFileDialog.getSaveFileName(
            self, "合并结果另存为",
            str(s.path.with_name(f"{s.path.stem}_合并.pdf")), "PDF (*.pdf)")
        if not out:
            return
        try:
            merge_pdfs([s.path, *paths], out)
        except Exception as e:
            QMessageBox.critical(self, "合并失败", f"{type(e).__name__}: {e}")
            return
        QMessageBox.information(
            self, "已合并",
            f"{1 + len(paths)} 个文件 → {out}\n"
            "（合并结果另存为新文件，当前编辑会话未受影响）")

    # ------------------------------------------------- 工具（压缩 / 转换）

    def analyze_images_dialog(self) -> None:
        if not self._need_session():
            return
        from ..core.compress import analyze_images
        s = self.session
        try:
            stats = analyze_images(s.doc)
        except Exception as e:
            QMessageBox.critical(self, "分析失败", f"{type(e).__name__}: {e}")
            return
        total = sum(x.nbytes for x in stats)
        size = Path(s.path).stat().st_size
        lines = [f"文档 {size / 1024:.0f} KB，{s.page_count} 页；"
                 f"{len(stats)} 张图像共 {total / 1024:.0f} KB"
                 f"（占 {total / max(1, size) * 100:.0f}%）", ""]
        for x in sorted(stats, key=lambda v: -v.nbytes):
            pg = ",".join(str(p + 1) for p in sorted(set(x.pages))[:4])
            lines.append(
                f"xref {x.xref:<5} 第 {pg:<12} 页  "
                f"{x.width}×{x.height} 显示 {x.disp_w:.0f}×{x.disp_h:.0f}pt  "
                f"有效 {x.effective_dpi:.0f} DPI  {x.nbytes / 1024:.0f} KB  "
                f"{x.colorspace}")
        lines += ["", "有效 DPI 高的图才有压缩空间；"
                      "低于 150 DPI 的扫描件已经接近最优，"
                      "再压只会糊掉字迹。"]
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("图像占用分析")
        dlg.resize(880, 560)
        lay = QVBoxLayout(dlg)
        te = QPlainTextEdit("\n".join(lines), dlg)
        te.setReadOnly(True)
        te.setFont(QFont("Consolas", 9))
        lay.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec()

    def compress_dialog(self) -> None:
        if not self._need_session():
            return
        from ..core.compress import PRESETS, compress_document
        s = self.session
        chosen = self._choose(
            "压缩文档",
            f"当前文档：{Path(s.path).name}  "
            f"{Path(s.path).stat().st_size / 1024:.0f} KB",
            [("lossless", "无损整理 —— 只做结构优化，图像像素一个不动"),
             ("light", "轻压 —— 只治分辨率虚高的图（≤300 DPI）"),
             ("balanced", "平衡（推荐）—— ≤150 DPI、近似灰度转灰、JPEG q80"),
             ("strong", "强压 —— ≤110 DPI、JPEG q68，体积优先"),
             ("tiny", "极限 —— ≤96 DPI、JPEG q58，有明显可见损失")],
            "balanced")
        if not chosen:
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "压缩结果另存为",
            str(Path(s.path).with_name(f"{Path(s.path).stem}_压缩.pdf")),
            "PDF (*.pdf)")
        if not out:
            return
        self._run_long(
            f"正在压缩（{chosen} 档）…",
            lambda pr: compress_document(
                s.path, out, chosen,
                verify=True,
                progress=lambda m: pr.setLabelText(m)),
            f"压缩完成 → {out}")

    def convert_dialog(self, target: str) -> None:
        if not self._need_session():
            return
        from ..core.convert import convert
        s = self.session
        names = {"docx": "Word", "xlsx": "Excel", "pptx": "PPT",
                 "png": "图片"}
        if target == "png":
            out = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if not out:
                return
        else:
            out, _ = QFileDialog.getSaveFileName(
                self, f"转换为 {names.get(target, target)}",
                str(Path(s.path).with_name(f"{Path(s.path).stem}.{target}")),
                f"{target} (*.{target})")
            if not out:
                return
        self._run_long(
            f"正在转换为 {names.get(target, target)}…",
            lambda pr: convert(s.path, out, to=target, dpi=150.0,
                               progress=lambda m: pr.setLabelText(m)),
            f"转换完成 → {out}")

    def office_to_pdf_dialog(self) -> None:
        from ..core.convert import office_to_pdf
        src, _ = QFileDialog.getOpenFileName(
            self, "选择要转成 PDF 的文件", "",
            "Office 文档 (*.docx *.doc *.rtf *.xlsx *.xls *.csv "
            "*.pptx *.ppt);;所有文件 (*)")
        if not src:
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "输出 PDF", str(Path(src).with_suffix(".pdf")), "PDF (*.pdf)")
        if not out:
            return
        self._run_long("正在调用本机 Office 渲染…",
                       lambda pr: office_to_pdf(
                           src, out,
                           progress=lambda m: pr.setLabelText(m)),
                       f"已生成 → {out}")

    def _choose(self, title: str, label: str,
                options: list[tuple[str, str]], default: str) -> str | None:
        """带说明的单选对话框 —— QInputDialog 的列表项显示不了这么长的说明。"""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox,
                                       QListWidget, QListWidgetItem)
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(620, 400)
        lay = QVBoxLayout(dlg)
        lb = QLabel(label, dlg)
        lb.setWordWrap(True)
        lay.addWidget(lb)
        lw = QListWidget(dlg)
        for key, desc in options:
            it = QListWidgetItem(f"{key}　—　{desc}")
            it.setData(Qt.UserRole, key)
            lw.addItem(it)
            if key == default:
                lw.setCurrentItem(it)
        lw.setWordWrap(True)
        lay.addWidget(lw)
        bb = QDialogButtonBox(QDialogButtonBox.Ok
                              | QDialogButtonBox.Cancel, dlg)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return None
        cur = lw.currentItem()
        return cur.data(Qt.UserRole) if cur else None

    def _run_long(self, title: str, fn, done_msg: str) -> None:
        """长任务统一走进度对话框，并保证异常被显示出来。

        压缩与转换都要跑几秒到几十秒（Word 渲染更久），
        必须在界面上有反馈，否则用户会以为程序卡死。
        """
        pr = QProgressDialog(title, "取消", 0, 0, self)
        pr.setWindowModality(Qt.WindowModal)
        pr.setMinimumDuration(0)
        pr.show()
        QApplication.processEvents()

        class _W(QThread):
            done = Signal(object)
            fail = Signal(str)
            prog = Signal(str)

            def run(self):
                try:
                    self.done.emit(fn(pr))
                except Exception:
                    self.fail.emit(traceback.format_exc())

        self._worker = _W()
        result: dict = {}

        def on_done(rep) -> None:
            result["rep"] = rep
            pr.close()

        def on_fail(tb: str) -> None:
            pr.close()
            QMessageBox.critical(self, "失败", tb)

        self._worker.done.connect(on_done)
        self._worker.fail.connect(on_fail)
        self._worker.start()
        while self._worker.isRunning():
            QApplication.processEvents()
            self._worker.wait(60)
            if pr.wasCanceled():
                self._worker.terminate()
                self.lbl_status.setText("已取消")
                return
        pr.close()

        rep = result.get("rep")
        if rep is None:
            return
        txt = rep.text() if hasattr(rep, "text") else str(rep)
        self.tx_report.append("● " + txt.replace("\n", "\n    "))
        self.lbl_status.setText("✓ " + done_msg)
        if hasattr(rep, "warnings") and rep.warnings:
            txt += "\n\n注意：\n" + "\n".join("· " + w for w in rep.warnings)
        QMessageBox.information(self, "完成", txt[:4000])

    def _reload_after_external(self) -> None:
        """外部工具（压缩/转换）只写新文件，不动当前会话 —— 无需重载。"""
        return


def run(path: str | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("PDF 工具箱")
    w = MainWindow()
    w.view.block_clicked.connect(w.on_block_clicked)
    w.view.block_activated.connect(w.on_block_clicked)
    w.view.cover_rect_changed.connect(w.on_cover_changed)
    w.show()
    if path:
        w.load_path(path)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
