"""Data classes for meeting persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MeetingStatus(str, Enum):
    RECORDING = "recording"
    RECORDED = "recorded"       # Audio captured, not yet transcribed
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Meeting:
    id: str                     # UUID
    title: str
    created_at: datetime
    duration_seconds: float = 0.0
    audio_path: str = ""        # Relative to audio store root
    status: MeetingStatus = MeetingStatus.RECORDING
    device_name: str = ""
    error_message: str = ""

    @property
    def duration_display(self) -> str:
        m, s = divmod(int(self.duration_seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


@dataclass
class Speaker:
    id: str                     # UUID
    meeting_id: str
    label: str                  # e.g. "SPEAKER_00"
    display_name: str = ""      # User-assigned name, e.g. "Alice"
    color: str = ""             # Hex color for UI

    @property
    def name(self) -> str:
        return self.display_name or self.label


@dataclass
class TranscriptSegment:
    id: str                     # UUID
    meeting_id: str
    speaker_id: str = ""
    speaker_label: str = ""     # Denormalized for fast display
    start_time: float = 0.0     # Seconds from start of meeting
    end_time: float = 0.0
    text: str = ""
    confidence: float = 0.0


@dataclass
class MeetingSummary:
    id: str                     # UUID
    meeting_id: str
    summary_text: str = ""
    key_points: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
