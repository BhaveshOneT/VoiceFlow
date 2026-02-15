"""Real-time audio waveform visualization using QPainter."""
from __future__ import annotations

import time
from collections import deque

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.ui_qt.styles.theme import Colors

_BUFFER_SIZE = 200  # Number of samples to display
_BAR_WIDTH = 3
_BAR_GAP = 1
_MIN_FRAME_INTERVAL = 1.0 / 30  # cap repaints at 30 fps


class WaveformWidget(QWidget):
    """Rolling audio level bar display.

    Call ``update_level(float)`` from any thread to push a new RMS sample.
    Repaints are throttled to 30 fps to avoid excessive CPU usage.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._levels: deque[float] = deque(maxlen=_BUFFER_SIZE)
        for _ in range(_BUFFER_SIZE):
            self._levels.append(0.0)
        self._last_paint_time: float = 0.0
        self.setMinimumHeight(60)
        self.setMaximumHeight(120)

    @Slot(float)
    def update_level(self, level: float) -> None:
        """Push a new audio level (0.0 - 1.0) and repaint (throttled)."""
        self._levels.append(max(0.0, min(1.0, level)))
        now = time.monotonic()
        if now - self._last_paint_time >= _MIN_FRAME_INTERVAL:
            self.update()

    def reset(self) -> None:
        self._levels.clear()
        for _ in range(_BUFFER_SIZE):
            self._levels.append(0.0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        self._last_paint_time = time.monotonic()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center_y = h / 2

        # Draw subtle center line
        center_pen = QPen(QColor(Colors.BORDER), 1, Qt.PenStyle.SolidLine)
        painter.setPen(center_pen)
        painter.drawLine(0, int(center_y), w, int(center_y))

        # Draw subtle grid lines (quarter marks)
        grid_pen = QPen(QColor(Colors.SEPARATOR), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        painter.drawLine(0, int(h * 0.25), w, int(h * 0.25))
        painter.drawLine(0, int(h * 0.75), w, int(h * 0.75))

        # How many bars fit in the widget?
        bar_total = _BAR_WIDTH + _BAR_GAP
        num_bars = min(len(self._levels), w // bar_total)
        if num_bars <= 0:
            painter.end()
            return

        # Use the most recent num_bars levels
        levels = list(self._levels)[-num_bars:]

        # Start from the right edge
        x_start = w - (num_bars * bar_total)

        accent = QColor(Colors.ACCENT)
        recording_color = QColor(Colors.STATUS_RECORDING)

        for i, level in enumerate(levels):
            x = x_start + i * bar_total
            bar_height = max(2, level * (h - 4))

            # Gradient from blue to red based on level
            if level > 0.7:
                color = QColor(recording_color)
            else:
                color = QColor(accent)
            color.setAlphaF(0.5 + level * 0.5)

            pen = QPen(color, _BAR_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            half_h = bar_height / 2
            painter.drawLine(
                int(x + _BAR_WIDTH / 2), int(center_y - half_h),
                int(x + _BAR_WIDTH / 2), int(center_y + half_h),
            )

        painter.end()
