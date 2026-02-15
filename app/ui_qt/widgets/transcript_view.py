"""Rich transcript viewer with speaker labels, timestamps, and colors."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.storage.models import Speaker, TranscriptSegment
from app.ui_qt.styles.theme import Colors, Spacing, Typography

# Speaker color palette -- assigned by index for consistency
_SPEAKER_COLORS = [
    "#0A84FF", "#FF9500", "#30D158", "#FF375F",
    "#BF5AF2", "#FFD60A", "#64D2FF", "#FF6482",
]


class _SpeakerBlock(QWidget):
    """A block of consecutive segments from the same speaker."""

    speaker_name_clicked = Signal(str)  # speaker_id

    def __init__(
        self,
        speaker: Optional[Speaker],
        segments: list[TranscriptSegment],
        color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._speaker = speaker
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, Spacing.SM, 0, Spacing.SM)
        layout.setSpacing(4)

        # Header row: colored dot + speaker name + timestamp
        header = QHBoxLayout()
        header.setSpacing(Spacing.SM)

        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;"
        )
        header.addWidget(dot)

        name_text = speaker.name if speaker else "Unknown"
        name_label = QLabel(name_text)
        name_label.setStyleSheet(
            f"color: {color}; font-weight: {Typography.WEIGHT_SEMIBOLD}; "
            f"font-size: {Typography.SIZE_CAPTION}px;"
        )
        name_label.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(name_label)

        if segments:
            ts = segments[0].start_time
            m, s = divmod(int(ts), 60)
            time_str = f"{m}:{s:02d}"
            time_label = QLabel(time_str)
            time_label.setObjectName("caption")
            header.addWidget(time_label)

        header.addStretch()
        layout.addLayout(header)

        # Concatenate text for this speaker block
        full_text = " ".join(seg.text for seg in segments if seg.text)
        text_label = QLabel(full_text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # Text color and size inherit from QSS QLabel defaults
        layout.addWidget(text_label)


class TranscriptView(QScrollArea):
    """Scrollable transcript grouped by speaker blocks.

    Consecutive segments from the same speaker are merged into a single
    visual block with speaker name, timestamp, and concatenated text.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addStretch()
        self.setWidget(self._container)

        # Stable color assignment: speaker_id -> color
        self._speaker_colors: dict[str, str] = {}
        self._full_text = ""  # cached for Copy All

    def _color_for_speaker(self, speaker_id: str) -> str:
        """Return a consistent color for a speaker, assigning one if new."""
        if speaker_id not in self._speaker_colors:
            idx = len(self._speaker_colors) % len(_SPEAKER_COLORS)
            self._speaker_colors[speaker_id] = _SPEAKER_COLORS[idx]
        return self._speaker_colors[speaker_id]

    def set_transcript(
        self,
        segments: list[TranscriptSegment],
        speakers: dict[str, Speaker],
    ) -> None:
        """Populate the view with transcript segments grouped by speaker."""
        # Clear existing blocks
        while self._layout.count() > 0:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not segments:
            empty = QLabel("No transcript available.")
            empty.setObjectName("caption")
            self._layout.addWidget(empty)
            self._layout.addStretch()
            self._full_text = ""
            return

        # Group consecutive segments by speaker
        blocks: list[tuple[str, list[TranscriptSegment]]] = []
        current_speaker = ""
        current_segments: list[TranscriptSegment] = []

        for seg in segments:
            if seg.speaker_id != current_speaker and current_segments:
                blocks.append((current_speaker, current_segments))
                current_segments = []
            current_speaker = seg.speaker_id
            current_segments.append(seg)

        if current_segments:
            blocks.append((current_speaker, current_segments))

        # Build full text for Copy All
        text_parts: list[str] = []

        for speaker_id, block_segments in blocks:
            speaker = speakers.get(speaker_id)
            color = self._color_for_speaker(speaker_id)
            block = _SpeakerBlock(speaker, block_segments, color)
            self._layout.addWidget(block)

            name = speaker.name if speaker else "Unknown"
            block_text = " ".join(seg.text for seg in block_segments if seg.text)
            if block_segments:
                ts = block_segments[0].start_time
                m, s = divmod(int(ts), 60)
                text_parts.append(f"[{m}:{s:02d}] {name}: {block_text}")
            else:
                text_parts.append(f"{name}: {block_text}")

        self._full_text = "\n\n".join(text_parts)

        # Copy All button
        copy_row = QHBoxLayout()
        copy_row.addStretch()
        copy_btn = QPushButton("Copy All")
        copy_btn.setToolTip("Copy entire transcript to clipboard")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_all)
        copy_row.addWidget(copy_btn)
        copy_wrapper = QWidget()
        copy_wrapper.setLayout(copy_row)
        self._layout.addWidget(copy_wrapper)

        self._layout.addStretch()

    def _copy_all(self) -> None:
        """Copy the full transcript text to the system clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard and self._full_text:
            clipboard.setText(self._full_text)
