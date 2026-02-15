"""Dashboard / home page -- status overview and quick actions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.signals import AppSignals
from app.ui_qt.styles.theme import Colors, Spacing, Typography

if TYPE_CHECKING:
    from app.storage.database import MeetingDatabase


class _StatusCard(QFrame):
    """Compact card showing dictation status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        layout.setSpacing(Spacing.SM)

        header = QLabel("Dictation")
        header.setObjectName("subheading")
        layout.addWidget(header)

        # Status row
        status_row = QHBoxLayout()
        status_row.setSpacing(Spacing.SM)
        self._status_dot = QLabel()
        self._status_dot.setObjectName("status_dot_ready")
        self._status_dot.setFixedSize(14, 14)
        status_row.addWidget(self._status_dot)

        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("caption")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        hint = QLabel("Hold Right Cmd to dictate")
        hint.setObjectName("caption")
        layout.addWidget(hint)

    @Slot(str)
    def set_status(self, status: str) -> None:
        self._status_label.setText(status)
        if status == "Recording":
            self._status_dot.setObjectName("status_dot_recording")
        elif status in ("Processing", "Transcribing..."):
            self._status_dot.setObjectName("status_dot_processing")
        else:
            self._status_dot.setObjectName("status_dot_ready")
        # Force QSS re-evaluation after objectName change
        self._status_dot.style().unpolish(self._status_dot)
        self._status_dot.style().polish(self._status_dot)


class _MeetingQuickStart(QFrame):
    """Card with a "Start Meeting" button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        layout.setSpacing(Spacing.MD)

        header = QLabel("Meetings")
        header.setObjectName("subheading")
        layout.addWidget(header)

        desc = QLabel("Record, transcribe, and summarize your meetings with speaker labels.")
        desc.setWordWrap(True)
        desc.setObjectName("caption")
        layout.addWidget(desc)

        self.start_button = QPushButton("\u23FA  Start Meeting")
        self.start_button.setObjectName("primary_button")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.start_button)


class _RecentMeetings(QFrame):
    """Shows the last few meeting entries from the database."""

    def __init__(self, db: "MeetingDatabase | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self.setProperty("class", "card")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        self._layout.setSpacing(Spacing.SM)

        header = QLabel("Recent Meetings")
        header.setObjectName("subheading")
        self._layout.addWidget(header)

        self._empty_label = QLabel("No meetings yet. Start recording to see them here.")
        self._empty_label.setObjectName("caption")
        self._empty_label.setWordWrap(True)
        self._layout.addWidget(self._empty_label)

        self._meeting_widgets: list[QWidget] = []
        self._layout.addStretch()
        self.refresh()

    def refresh(self) -> None:
        """Reload recent meetings from the database."""
        for w in self._meeting_widgets:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._meeting_widgets.clear()

        if not self._db:
            self._empty_label.show()
            return

        meetings = self._db.list_meetings(limit=5)
        if not meetings:
            self._empty_label.show()
            return

        self._empty_label.hide()
        insert_idx = self._layout.count() - 1  # before stretch
        for meeting in meetings:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(Spacing.SM)

            title = QLabel(meeting.title)
            title.setObjectName("caption")
            row_layout.addWidget(title, 1)

            # Duration display
            if meeting.duration_seconds > 0:
                dur = QLabel(meeting.duration_display)
                dur.setObjectName("caption")
                row_layout.addWidget(dur)

            # Date
            if meeting.created_at:
                date_label = QLabel(meeting.created_at.strftime("%b %d"))
                date_label.setObjectName("caption")
                row_layout.addWidget(date_label)

            status = QLabel(meeting.status.value.title())
            status.setObjectName("caption")
            row_layout.addWidget(status)

            self._layout.insertWidget(insert_idx, row)
            self._meeting_widgets.append(row)
            insert_idx += 1


class _ModelStatus(QFrame):
    """Shows which models are loaded / available."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        layout.setSpacing(Spacing.SM)

        header = QLabel("Models")
        header.setObjectName("subheading")
        layout.addWidget(header)

        self._stt_label = QLabel("STT (Parakeet): loading...")
        self._stt_label.setObjectName("caption")
        layout.addWidget(self._stt_label)

        self._vad_label = QLabel("VAD (Silero): ready")
        self._vad_label.setObjectName("caption")
        layout.addWidget(self._vad_label)

        self._refiner_label = QLabel("Refiner (Qwen): not loaded")
        self._refiner_label.setObjectName("caption")
        layout.addWidget(self._refiner_label)

    @Slot(str)
    def on_model_loaded(self, name: str) -> None:
        lower = name.lower()
        if any(alias in lower for alias in ("whisper", "parakeet", "stt")):
            self._stt_label.setText("STT (Parakeet): ready")
        elif "qwen" in lower or "refiner" in lower:
            self._refiner_label.setText("Refiner (Qwen): ready")

    @Slot(str)
    def on_model_loading(self, name: str) -> None:
        lower = name.lower()
        if any(alias in lower for alias in ("whisper", "parakeet", "stt")):
            self._stt_label.setText("STT (Parakeet): loading...")
        elif "qwen" in lower or "refiner" in lower:
            self._refiner_label.setText("Refiner (Qwen): loading...")


class DashboardPage(QWidget):
    """Home page: dictation status, meeting quick-start, recent meetings."""

    def __init__(
        self,
        signals: AppSignals,
        db: "MeetingDatabase | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._db = db

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        # Page title
        title = QLabel("Dashboard")
        title.setObjectName("heading")
        layout.addWidget(title)

        # Top row: status + meeting quick-start
        top_row = QHBoxLayout()
        top_row.setSpacing(Spacing.LG)

        self._status_card = _StatusCard()
        top_row.addWidget(self._status_card, 1)

        self._meeting_card = _MeetingQuickStart()
        top_row.addWidget(self._meeting_card, 1)

        layout.addLayout(top_row)

        # Bottom row: recent meetings + model status
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(Spacing.LG)

        self._recent = _RecentMeetings(db)
        bottom_row.addWidget(self._recent, 2)

        self._models = _ModelStatus()
        bottom_row.addWidget(self._models, 1)

        layout.addLayout(bottom_row)

        # Quick actions / keyboard hints
        hints = QLabel("\u2318 Q  Quit  |  Hold hotkey to dictate  |  \u23FA Start Meeting to record")
        hints.setObjectName("caption")
        hints.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hints)

        layout.addStretch()

        # Wire signals
        signals.status_changed.connect(self._status_card.set_status)
        signals.model_loaded.connect(self._models.on_model_loaded)
        signals.model_loading.connect(self._models.on_model_loading)

    def refresh_recent(self) -> None:
        """Refresh the recent meetings list."""
        self._recent.refresh()

    @property
    def start_meeting_button(self) -> QPushButton:
        return self._meeting_card.start_button
