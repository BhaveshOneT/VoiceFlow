"""Meetings library page -- searchable list of recorded meetings."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.storage.models import Meeting, MeetingStatus
from app.ui_qt.styles.theme import Colors, Spacing, Typography
from app.ui_qt.widgets.search_bar import SearchBar

if TYPE_CHECKING:
    from app.core.signals import AppSignals
    from app.storage.database import MeetingDatabase

log = logging.getLogger(__name__)


class _MeetingCard(QFrame):
    """Compact card representing a single meeting in the list."""

    clicked = Signal(str)  # meeting_id

    # Status -> (color, label) for the badge
    _STATUS_STYLE: dict[MeetingStatus, str] = {
        MeetingStatus.RECORDING: "status_dot_recording",
        MeetingStatus.RECORDED: "status_dot_ready",
        MeetingStatus.TRANSCRIBING: "status_dot_processing",
        MeetingStatus.DIARIZING: "status_dot_processing",
        MeetingStatus.SUMMARIZING: "status_dot_processing",
        MeetingStatus.COMPLETE: "status_dot_ready",
        MeetingStatus.ERROR: "status_dot_error",
    }

    def __init__(self, meeting: Meeting, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._meeting = meeting
        self.setProperty("class", "card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.MD,
            Spacing.CARD_PADDING, Spacing.MD,
        )
        layout.setSpacing(4)

        # Title row with status indicator
        top = QHBoxLayout()
        top.setSpacing(Spacing.SM)

        # Status dot
        dot = QLabel()
        dot_name = self._STATUS_STYLE.get(meeting.status, "status_dot_ready")
        dot.setObjectName(dot_name)
        dot.setFixedSize(10, 10)
        top.addWidget(dot)

        title = QLabel(meeting.title)
        title.setObjectName("subheading")
        top.addWidget(title, 1)

        # Status text badge
        badge = QLabel(meeting.status.value.title())
        badge.setObjectName("caption")
        top.addWidget(badge)
        layout.addLayout(top)

        # Meta row: date | duration | device
        meta_parts = []
        if meeting.created_at:
            meta_parts.append(meeting.created_at.strftime("%b %d, %Y %I:%M %p"))
        if meeting.duration_seconds > 0:
            meta_parts.append(meeting.duration_display)
        if meeting.device_name:
            meta_parts.append(meeting.device_name)
        meta = QLabel(" \u2022 ".join(meta_parts))
        meta.setObjectName("caption")
        layout.addWidget(meta)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self._meeting.id)
        super().mousePressEvent(event)


class MeetingsPage(QWidget):
    """Meeting library list with search and clickable meeting cards."""

    meeting_selected = Signal(str)  # meeting_id

    def __init__(
        self,
        signals: "AppSignals",
        db: "MeetingDatabase | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._db = db

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        title = QLabel("Meetings")
        title.setObjectName("heading")
        layout.addWidget(title)

        # Search bar
        self._search = SearchBar("Search meetings...")
        self._search.search_changed.connect(self._on_search)
        layout.addWidget(self._search)

        # Scrollable meeting list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(Spacing.SM)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_container)

        layout.addWidget(scroll, 1)

        # Empty state
        self._empty_label = QLabel(
            "\U0001F399  No meetings yet\n\n"
            "Record a meeting from the Dashboard to get started."
        )
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setObjectName("caption")
        self._empty_label.hide()
        layout.addWidget(self._empty_label)

    def set_database(self, db: "MeetingDatabase") -> None:
        """Set or update the database reference and refresh."""
        self._db = db
        self.refresh()

    def refresh(self, query: str = "") -> None:
        """Reload the meeting list from the database."""
        if not self._db:
            return

        # Clear current cards
        while self._list_layout.count() > 0:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if query:
            meetings = self._db.search_meetings(query)
        else:
            meetings = self._db.list_meetings(limit=50)

        if not meetings:
            self._empty_label.show()
            self._list_layout.addStretch()
            return

        self._empty_label.hide()
        for meeting in meetings:
            card = _MeetingCard(meeting)
            card.clicked.connect(self.meeting_selected.emit)
            self._list_layout.addWidget(card)
        self._list_layout.addStretch()

    def _on_search(self, query: str) -> None:
        self.refresh(query)
