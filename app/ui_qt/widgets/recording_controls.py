"""Start/stop/pause controls for meeting recording."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from app.ui_qt.styles.theme import Spacing


class RecordingControls(QWidget):
    """Compact recording control bar: start, pause/resume, stop, elapsed time.

    Button state management:
    - Idle: Start enabled, Stop/Pause disabled
    - Recording: Start hidden, Stop+Pause shown
    - Paused: Resume shown, Stop shown, Start hidden
    """

    start_clicked = Signal()
    stop_clicked = Signal()
    pause_clicked = Signal()
    resume_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        # Start button (styled via QSS #primary_button)
        self._start_btn = QPushButton("\u23FA  Start Recording")
        self._start_btn.setObjectName("primary_button")
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self._start_btn)

        # Pause / Resume button (styled via QSS default QPushButton)
        self._pause_btn = QPushButton("\u23F8  Pause")
        self._pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause_btn.clicked.connect(self._on_pause_resume)
        self._pause_btn.hide()
        layout.addWidget(self._pause_btn)

        # Stop button (styled via QSS #danger_button)
        self._stop_btn = QPushButton("\u23F9  Stop")
        self._stop_btn.setObjectName("danger_button")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._stop_btn.hide()
        layout.addWidget(self._stop_btn)

        # Recording indicator dot (animated via timer, styled via QSS objectName)
        self._rec_dot = QLabel()
        self._rec_dot.setObjectName("status_dot_recording")
        self._rec_dot.setFixedSize(10, 10)
        self._rec_dot.hide()
        layout.addWidget(self._rec_dot)
        self._dot_visible = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._blink_dot)

        # Elapsed time (styled via QSS #subheading)
        self._elapsed = QLabel("0:00")
        self._elapsed.setObjectName("subheading")
        self._elapsed.hide()
        layout.addWidget(self._elapsed)

        layout.addStretch()

        self._paused = False

    def set_recording(self, recording: bool) -> None:
        """Toggle between idle and recording state."""
        self._start_btn.setVisible(not recording)
        self._stop_btn.setVisible(recording)
        self._pause_btn.setVisible(recording)
        self._elapsed.setVisible(recording)
        self._rec_dot.setVisible(recording)
        self._paused = False
        self._pause_btn.setText("\u23F8  Pause")

        if recording:
            self._rec_dot.setObjectName("status_dot_recording")
            self._rec_dot.style().unpolish(self._rec_dot)
            self._rec_dot.style().polish(self._rec_dot)
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
            self._rec_dot.hide()

    def set_elapsed(self, seconds: float) -> None:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            self._elapsed.setText(f"{h}:{m:02d}:{s:02d}")
        else:
            self._elapsed.setText(f"{m}:{s:02d}")

    def _on_pause_resume(self) -> None:
        if self._paused:
            self._paused = False
            self._pause_btn.setText("\u23F8  Pause")
            self._blink_timer.start()
            self._rec_dot.setObjectName("status_dot_recording")
            self._rec_dot.style().unpolish(self._rec_dot)
            self._rec_dot.style().polish(self._rec_dot)
            self.resume_clicked.emit()
        else:
            self._paused = True
            self._pause_btn.setText("\u25B6  Resume")
            self._blink_timer.stop()
            self._rec_dot.setObjectName("status_dot_processing")
            self._rec_dot.style().unpolish(self._rec_dot)
            self._rec_dot.style().polish(self._rec_dot)
            self.pause_clicked.emit()

    def _blink_dot(self) -> None:
        """Toggle recording dot visibility for a blinking effect."""
        self._dot_visible = not self._dot_visible
        self._rec_dot.setVisible(self._dot_visible)
