"""Model status indicators -- shows download/loading state for each model."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from app.ui_qt.styles.theme import Colors, Spacing


class _ModelRow(QWidget):
    """Single model status row: name, status dot, status label, optional progress bar."""

    # Map status keywords to (objectName, color) for the dot
    _STATE_MAP = {
        "ready": ("status_dot_ready", Colors.STATUS_READY),
        "loading": ("status_dot_processing", Colors.STATUS_PROCESSING),
        "downloading": ("status_dot_processing", Colors.STATUS_PROCESSING),
        "error": ("status_dot_error", Colors.STATUS_ERROR),
    }

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(Spacing.SM)

        # Status dot
        self._dot = QLabel()
        self._dot.setObjectName("status_dot_processing")
        self._dot.setFixedSize(8, 8)
        layout.addWidget(self._dot)

        self._name_label = QLabel(name)
        self._name_label.setObjectName("subheading")
        self._name_label.setMinimumWidth(120)
        layout.addWidget(self._name_label)

        self._status_label = QLabel("Not installed")
        self._status_label.setObjectName("caption")
        layout.addWidget(self._status_label, 1)

        self._progress = QProgressBar()
        self._progress.setMaximumHeight(8)
        self._progress.setTextVisible(False)
        self._progress.hide()
        layout.addWidget(self._progress, 1)

    def set_status(self, status: str) -> None:
        self._progress.hide()
        self._status_label.show()
        self._status_label.setText(status)

        # Update the dot based on status keyword
        lower = status.lower()
        for keyword, (dot_name, _color) in self._STATE_MAP.items():
            if keyword in lower:
                self._dot.setObjectName(dot_name)
                self._dot.style().unpolish(self._dot)
                self._dot.style().polish(self._dot)
                return

        # Default: muted dot
        self._dot.setObjectName("status_dot_processing")
        self._dot.style().unpolish(self._dot)
        self._dot.style().polish(self._dot)

    def set_progress(self, fraction: float) -> None:
        self._status_label.hide()
        self._progress.show()
        self._progress.setValue(int(fraction * 100))


class ModelStatusWidget(QFrame):
    """Panel showing status of all ML models."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        layout.setSpacing(Spacing.SM)

        header = QLabel("Model Status")
        header.setObjectName("subheading")
        layout.addWidget(header)

        self._rows: dict[str, _ModelRow] = {}
        for key, name in [
            ("stt", "STT (Parakeet)"),
            ("vad", "VAD (Silero)"),
            ("refiner", "Refiner (Qwen)"),
            ("diarization", "Diarization (pyannote)"),
        ]:
            row = _ModelRow(name)
            layout.addWidget(row)
            self._rows[key] = row

        # Set initial states
        self._rows["vad"].set_status("Ready")
        self._rows["diarization"].set_status("Not installed")

    @Slot(str)
    def on_model_loading(self, name: str) -> None:
        row = self._find_row(name)
        if row:
            row.set_status("Loading...")

    @Slot(str)
    def on_model_loaded(self, name: str) -> None:
        row = self._find_row(name)
        if row:
            row.set_status("Ready")

    @Slot(str, float)
    def on_download_progress(self, name: str, progress: float) -> None:
        row = self._find_row(name)
        if row:
            row.set_progress(progress)

    def _find_row(self, name: str) -> _ModelRow | None:
        lower = name.lower()
        if any(alias in lower for alias in ("stt", "whisper", "parakeet")):
            return self._rows.get("stt")
        if any(alias in lower for alias in ("refiner", "qwen")):
            return self._rows.get("refiner")
        for key, row in self._rows.items():
            if key in lower or lower in key:
                return row
        return None
