"""Meeting detail page -- transcript viewer with speaker labels, summary, and export."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.storage.models import Meeting, MeetingSummary, Speaker, TranscriptSegment
from app.ui_qt.styles.theme import Spacing, Typography
from app.ui_qt.widgets.transcript_view import TranscriptView

if TYPE_CHECKING:
    from app.core.signals import AppSignals
    from app.storage.database import MeetingDatabase

log = logging.getLogger(__name__)


class MeetingDetailPage(QWidget):
    """Full meeting view: header, transcript, summary, and export."""

    back_requested = Signal()

    def __init__(
        self,
        signals: "AppSignals",
        db: "MeetingDatabase",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._db = db
        self._meeting: Optional[Meeting] = None
        self._segments: list[TranscriptSegment] = []
        self._speakers: dict[str, Speaker] = {}
        self._summary: Optional[MeetingSummary] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        # -- Back + header ------------------------------------------------
        top_row = QHBoxLayout()
        back_btn = QPushButton("\u2190  Back")
        back_btn.setObjectName("ghost_button")
        back_btn.clicked.connect(self.back_requested.emit)
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        top_row.addWidget(back_btn)
        top_row.addStretch()

        # Export buttons
        export_md = QPushButton("Export Markdown")
        export_md.clicked.connect(self._export_markdown)
        top_row.addWidget(export_md)

        export_srt = QPushButton("Export SRT")
        export_srt.clicked.connect(self._export_srt)
        top_row.addWidget(export_srt)

        export_json = QPushButton("Export JSON")
        export_json.clicked.connect(self._export_json)
        top_row.addWidget(export_json)

        copy_btn = QPushButton("Copy Transcript")
        copy_btn.clicked.connect(self._copy_transcript)
        top_row.addWidget(copy_btn)

        layout.addLayout(top_row)

        # Title (editable) -- styled via QSS QLineEdit
        self._title_edit = QLineEdit()
        self._title_edit.setObjectName("meeting_title_edit")
        self._title_edit.editingFinished.connect(self._on_title_changed)
        layout.addWidget(self._title_edit)

        # Meta info
        self._meta_label = QLabel()
        self._meta_label.setObjectName("caption")
        layout.addWidget(self._meta_label)

        # -- Summary panel (hidden until summary exists) ------------------
        self._summary_frame = QFrame()
        self._summary_frame.setProperty("class", "card")
        summary_layout = QVBoxLayout(self._summary_frame)
        summary_layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        summary_layout.setSpacing(Spacing.SM)

        summary_header = QLabel("Summary")
        summary_header.setObjectName("subheading")
        summary_layout.addWidget(summary_header)

        self._summary_text = QLabel()
        self._summary_text.setWordWrap(True)
        self._summary_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        summary_layout.addWidget(self._summary_text)

        self._key_points_label = QLabel()
        self._key_points_label.setWordWrap(True)
        self._key_points_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        summary_layout.addWidget(self._key_points_label)

        self._action_items_label = QLabel()
        self._action_items_label.setWordWrap(True)
        self._action_items_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        summary_layout.addWidget(self._action_items_label)

        self._summary_frame.hide()
        layout.addWidget(self._summary_frame)

        # -- Transcript view ----------------------------------------------
        self._transcript = TranscriptView()
        layout.addWidget(self._transcript, 1)

    def load_meeting(self, meeting_id: str) -> None:
        """Load a meeting and display it."""
        self._meeting = self._db.get_meeting(meeting_id)
        if not self._meeting:
            return

        self._title_edit.setText(self._meeting.title)
        self._meta_label.setText(
            f"{self._meeting.created_at.strftime('%B %d, %Y %I:%M %p')} "
            f" \u2022 {self._meeting.duration_display} "
            f" \u2022 {self._meeting.status.value.title()}"
        )

        # Load transcript
        self._segments = self._db.get_segments(meeting_id)
        speaker_list = self._db.get_speakers(meeting_id)
        self._speakers = {s.id: s for s in speaker_list}
        self._transcript.set_transcript(self._segments, self._speakers)

        # Load summary
        self._summary = self._db.get_summary(meeting_id)
        if self._summary:
            self._summary_frame.show()
            self._summary_text.setText(self._summary.summary_text)
            if self._summary.key_points:
                points = "\n".join(f"  \u2022 {p}" for p in self._summary.key_points)
                self._key_points_label.setText(f"Key Points:\n{points}")
            if self._summary.action_items:
                items = "\n".join(f"  \u2610 {a}" for a in self._summary.action_items)
                self._action_items_label.setText(f"Action Items:\n{items}")
        else:
            self._summary_frame.hide()

    def _on_title_changed(self) -> None:
        if self._meeting and self._title_edit.text().strip():
            new_title = self._title_edit.text().strip()
            self._db.update_meeting(self._meeting.id, title=new_title)
            self._meeting.title = new_title

    # -- Export -----------------------------------------------------------

    def _transcript_as_text(self) -> str:
        lines: list[str] = []
        for seg in self._segments:
            speaker = self._speakers.get(seg.speaker_id)
            name = speaker.name if speaker else seg.speaker_label or "Unknown"
            m, s = divmod(int(seg.start_time), 60)
            lines.append(f"[{m}:{s:02d}] {name}: {seg.text}")
        return "\n".join(lines)

    def _copy_transcript(self) -> None:
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self._transcript_as_text())

    def _export_markdown(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Markdown", f"{self._meeting.title}.md" if self._meeting else "meeting.md",
            "Markdown (*.md)",
        )
        if not path:
            return
        lines = [f"# {self._meeting.title}\n"] if self._meeting else []
        if self._summary:
            lines.append(f"## Summary\n{self._summary.summary_text}\n")
            if self._summary.key_points:
                lines.append("## Key Points")
                lines.extend(f"- {p}" for p in self._summary.key_points)
                lines.append("")
            if self._summary.action_items:
                lines.append("## Action Items")
                lines.extend(f"- [ ] {a}" for a in self._summary.action_items)
                lines.append("")
        lines.append("## Transcript\n")
        lines.append(self._transcript_as_text())
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _export_srt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SRT", f"{self._meeting.title}.srt" if self._meeting else "meeting.srt",
            "SRT (*.srt)",
        )
        if not path:
            return
        lines: list[str] = []
        for i, seg in enumerate(self._segments, 1):
            start = self._format_srt_time(seg.start_time)
            end = self._format_srt_time(seg.end_time)
            speaker = self._speakers.get(seg.speaker_id)
            name = speaker.name if speaker else seg.speaker_label or ""
            prefix = f"[{name}] " if name else ""
            lines.append(f"{i}\n{start} --> {end}\n{prefix}{seg.text}\n")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", f"{self._meeting.title}.json" if self._meeting else "meeting.json",
            "JSON (*.json)",
        )
        if not path:
            return
        data = {
            "meeting": {
                "title": self._meeting.title if self._meeting else "",
                "created_at": self._meeting.created_at.isoformat() if self._meeting else "",
                "duration": self._meeting.duration_seconds if self._meeting else 0,
            },
            "speakers": [
                {"id": s.id, "label": s.label, "name": s.name, "color": s.color}
                for s in self._speakers.values()
            ],
            "segments": [
                {
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker_label,
                    "text": s.text,
                }
                for s in self._segments
            ],
        }
        if self._summary:
            data["summary"] = {
                "text": self._summary.summary_text,
                "key_points": self._summary.key_points,
                "action_items": self._summary.action_items,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
