"""页面视图：渲染 PDF 页 + 叠加交互层。

设计要点：
  · 页面位图由 PyMuPDF 按当前缩放渲染，叠加层用 QPainter 画在同一个 widget 上，
    避免 QGraphicsScene 在大量图元时的卡顿。
  · 点击命中用结构模型的块包围盒做测试，优先命中面积最小的块
    （水印之类包围盒很大，按面积排序才能选中真正被点的字）。
  · 图片型文档额外画一个**覆盖框**，可以拖动和拉伸 —— 这是关键：
    OCR 的检测框在真实文档上经常严重失真，自动推断不可靠，
    必须让用户能一眼看到覆盖范围并手工修正。
"""

from __future__ import annotations

import pymupdf
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..core.model import KIND_IMAGE, KIND_TABLE, LAYER_FIXED, Block

# 分类配色（与 core/viz.py 保持一致，便于和命令行输出对照）
KIND_COLORS = {
    "heading":   QColor(230, 57, 70),
    "paragraph": QColor(29, 53, 87),
    "list":      QColor(69, 123, 157),
    "header":    QColor(244, 162, 97),
    "footer":    QColor(244, 162, 97),
    "watermark": QColor(155, 93, 229),
    "image":     QColor(42, 157, 143),
    "table":     QColor(0, 180, 216),
    "graphics":  QColor(150, 150, 150),
    "unknown":   QColor(0, 0, 0),
}

HANDLE_SIZE = 7


class PageView(QWidget):
    """单页视图：页面位图 + 块叠加层 + 覆盖框交互。"""

    block_clicked = Signal(object)          # Block | None
    block_activated = Signal(object)        # 双击，触发编辑
    cover_rect_changed = Signal(object)     # (x0,y0,x1,y1) | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.page_no = 0
        self.zoom = 1.6
        self.show_overlays = True
        self.show_fixed_layer = True

        self._pixmap: QPixmap | None = None
        self._blocks: list[Block] = []
        self._selected: Block | None = None
        self._hover: Block | None = None
        self._cover: tuple[float, float, float, float] | None = None
        self._cover_enabled = False
        self._preview: tuple[float, float, float, float] | None = None
        self._char_boxes: list | None = None      # 单字框（调试用）
        self.show_char_boxes = False
        self._drag_mode: str | None = None
        self._drag_start: QPoint | None = None
        self._cover_at_drag_start = None
        self._doc = None

    # ---------------------------------------------------------- 数据

    def set_document(self, doc, structure) -> None:
        self._doc = doc
        self._structure = structure

    def set_page(self, page_no: int, blocks: list[Block]) -> None:
        self.page_no = page_no
        self._blocks = blocks
        self._selected = None
        self._hover = None
        self._cover = None
        self._render()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = max(0.25, min(8.0, zoom))
        self._render()

    def set_selected(self, block: Block | None) -> None:
        self._selected = block
        self.update()

    def set_cover(self, rect, enabled: bool = True) -> None:
        self._cover = rect
        self._cover_enabled = enabled
        self.update()

    @property
    def cover(self):
        return self._cover

    def set_char_boxes(self, boxes) -> None:
        """设置单字框（调试用）。boxes 是 [(Char, rect), ...]。"""
        self._char_boxes = boxes
        self.update()

    def set_preview(self, rect) -> None:
        """设置"内容加长后"的预览框 —— 打字时实时跟着变。"""
        self._preview = rect
        self.update()

    @property
    def preview(self):
        return self._preview

    def _render(self) -> None:
        if self._doc is None or self.page_no >= len(self._doc):
            self._pixmap = None
            self.update()
            return
        mat = pymupdf.Matrix(self.zoom, self.zoom)
        pix = self._doc[self.page_no].get_pixmap(matrix=mat, alpha=False)

        # 直接用 Pixmap 的原始像素构造 QImage —— 比先编码成 PPM/PNG 再解码快得多。
        # 踩过的坑：走 QPixmap.loadFromData(ppm_bytes) 时格式参数不被识别，
        # 位图静默变成空图，视图尺寸变成 0×0，界面上什么都看不到。
        n = pix.n
        fmt = {1: QImage.Format_Grayscale8,
               3: QImage.Format_RGB888,
               4: QImage.Format_RGBA8888}.get(n, QImage.Format_RGB888)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
        self._pixmap = QPixmap.fromImage(img.copy())
        self.setFixedSize(pix.width, pix.height)

    def refresh(self) -> None:
        self._render()

    # ---------------------------------------------------------- 坐标

    def to_pdf(self, pos: QPoint) -> tuple[float, float]:
        return pos.x() / self.zoom, pos.y() / self.zoom

    def to_widget(self, rect) -> QRectF:
        x0, y0, x1, y1 = rect
        return QRectF(x0 * self.zoom, y0 * self.zoom,
                      (x1 - x0) * self.zoom, (y1 - y0) * self.zoom)

    def _hit(self, pos: QPoint) -> Block | None:
        x, y = self.to_pdf(pos)
        best, best_area = None, float("inf")
        for b in self._blocks:
            if not b.lines and b.kind not in (KIND_IMAGE, KIND_TABLE):
                continue
            x0, y0, x1, y1 = b.bbox
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            area = (x1 - x0) * (y1 - y0)
            if b.kind == KIND_TABLE:
                area *= 4          # 表格块很大，降低优先级
            if b.kind == KIND_IMAGE:
                area *= 8
            if not self.show_fixed_layer and b.layer == LAYER_FIXED:
                continue
            if area < best_area:
                best, best_area = b, area
        return best

    # ---------------------------------------------------------- 交互

    def _cover_handles(self) -> dict[str, QRectF]:
        if self._cover is None:
            return {}
        r = self.to_widget(self._cover)
        s = HANDLE_SIZE
        return {
            "tl": QRectF(r.left() - s / 2, r.top() - s / 2, s, s),
            "tr": QRectF(r.right() - s / 2, r.top() - s / 2, s, s),
            "bl": QRectF(r.left() - s / 2, r.bottom() - s / 2, s, s),
            "br": QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
            "left": QRectF(r.left() - s / 2, r.center().y() - s / 2, s, s),
            "right": QRectF(r.right() - s / 2, r.center().y() - s / 2, s, s),
            "top": QRectF(r.center().x() - s / 2, r.top() - s / 2, s, s),
            "bottom": QRectF(r.center().x() - s / 2, r.bottom() - s / 2, s, s),
        }

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.LeftButton:
            return
        pos = ev.position().toPoint()

        # 先判断是否点在覆盖框的把手或边缘
        if self._cover_enabled and self._cover is not None:
            for name, hr in self._cover_handles().items():
                if hr.adjusted(-3, -3, 3, 3).contains(pos):
                    self._drag_mode = name
                    self._drag_start = pos
                    self._cover_at_drag_start = self._cover
                    return
            r = self.to_widget(self._cover)
            if r.adjusted(-2, -2, 2, 2).contains(pos):
                self._drag_mode = "move"
                self._drag_start = pos
                self._cover_at_drag_start = self._cover
                return

        blk = self._hit(pos)
        self._selected = blk
        self.block_clicked.emit(blk)
        self.update()

    def mouseMoveEvent(self, ev) -> None:
        pos = ev.position().toPoint()

        if self._drag_mode and self._drag_start is not None:
            dx = (pos.x() - self._drag_start.x()) / self.zoom
            dy = (pos.y() - self._drag_start.y()) / self.zoom
            x0, y0, x1, y1 = self._cover_at_drag_start
            m = self._drag_mode
            if m == "move":
                x0, x1 = x0 + dx, x1 + dx
                y0, y1 = y0 + dy, y1 + dy
            else:
                if "l" in m or m == "left":
                    x0 += dx
                if "r" in m or m == "right":
                    x1 += dx
                if "t" in m or m == "top":
                    y0 += dy
                if "b" in m or m == "bottom":
                    y1 += dy
            if x1 - x0 < 4:
                x1 = x0 + 4
            if y1 - y0 < 3:
                y1 = y0 + 3
            self._cover = (x0, y0, x1, y1)
            self.cover_rect_changed.emit(self._cover)
            self.update()
            return

        hov = self._hit(pos)
        if hov is not self._hover:
            self._hover = hov
            self.update()
        self._update_cursor(pos)

    def _update_cursor(self, pos) -> None:
        if self._cover_enabled and self._cover is not None:
            for name, hr in self._cover_handles().items():
                if hr.adjusted(-3, -3, 3, 3).contains(pos):
                    cursors = {
                        "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                        "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                        "left": Qt.SizeHorCursor, "right": Qt.SizeHorCursor,
                        "top": Qt.SizeVerCursor, "bottom": Qt.SizeVerCursor,
                    }
                    self.setCursor(cursors.get(name, Qt.ArrowCursor))
                    return
        self.setCursor(Qt.PointingHandCursor if self._hit(pos) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev) -> None:
        self._drag_mode = None
        self._drag_start = None

    def mouseDoubleClickEvent(self, ev) -> None:
        blk = self._hit(ev.position().toPoint())
        if blk is not None:
            self._selected = blk
            self.block_clicked.emit(blk)
            self.block_activated.emit(blk)

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.ControlModifier:
            delta = 1 if ev.angleDelta().y() > 0 else -1
            self.set_zoom(self.zoom * (1.12 if delta > 0 else 1 / 1.12))
            ev.accept()
        else:
            ev.ignore()

    # ---------------------------------------------------------- 绘制

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(70, 70, 76))

        if self._pixmap is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, "未打开文档")
            return

        p.drawPixmap(0, 0, self._pixmap)

        if not self.show_overlays:
            return

        p.setRenderHint(QPainter.Antialiasing, False)

        for b in self._blocks:
            if b.kind == KIND_IMAGE and not self.show_fixed_layer:
                continue
            if b.layer == LAYER_FIXED and not self.show_fixed_layer:
                continue
            if not b.lines and b.kind not in (KIND_IMAGE, KIND_TABLE):
                continue

            base = KIND_COLORS.get(b.kind, KIND_COLORS["unknown"])
            r = self.to_widget(b.bbox)
            is_fixed = b.layer == LAYER_FIXED

            if b is self._selected:
                pen = QPen(QColor(0, 200, 0), 2.5)
            elif b is self._hover:
                pen = QPen(base, 2.0)
            else:
                col = QColor(base)
                col.setAlpha(110)
                pen = QPen(col, 1.0)
            if is_fixed and b not in (self._selected, self._hover):
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)

            if b is self._selected:
                tag = f"{b.kind}  {b.size:.1f}pt"
                if is_fixed:
                    tag += "  [固定层]"
                p.setFont(QFont("Microsoft YaHei", 8))
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(tag) + 8
                label = QRectF(r.left(), max(0, r.top() - 17), tw, 15)
                p.fillRect(label, QColor(0, 160, 0, 220))
                p.setPen(QColor(255, 255, 255))
                p.drawText(label, Qt.AlignCenter, tag)

        # 单字框（调试）：每个字各自一个紧致框，倾斜文字能看出来贴合效果
        if self.show_char_boxes and self._char_boxes:
            pen = QPen(QColor(220, 0, 160), 1.4)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for _ch, r in self._char_boxes:
                p.drawRect(self.to_widget(r))

        # 加长预览框：青色虚线，表示"内容再长这么多会占到这里"
        if self._preview is not None:
            r = self.to_widget(self._preview)
            pen = QPen(QColor(0, 180, 200), 2)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QBrush(QColor(0, 180, 200, 35)))
            p.drawRect(r)
            p.setPen(QColor(0, 130, 150))
            p.setFont(QFont("Microsoft YaHei", 8))
            p.drawText(r.left(), max(0, r.top() - 3), "加长预览")

        # 覆盖框（图片型文档）
        if self._cover_enabled and self._cover is not None:
            r = self.to_widget(self._cover)
            pen = QPen(QColor(255, 140, 0), 2)
            pen.setStyle(Qt.DashDotLine)
            p.setPen(pen)
            p.setBrush(QBrush(QColor(255, 200, 0, 45)))
            p.drawRect(r)
            p.setBrush(QColor(255, 140, 0))
            for hr in self._cover_handles().values():
                p.fillRect(hr, QColor(255, 140, 0))
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawRect(hr)
