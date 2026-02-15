"""Audio file management for meetings.

Stores WAV files at ~/Library/Application Support/VoiceFlow/audio/{meeting_id}/
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "VoiceFlow"
AUDIO_ROOT = _APP_SUPPORT / "audio"


class AudioStore:
    """Manages on-disk audio files for meetings."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or AUDIO_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    def meeting_dir(self, meeting_id: str) -> Path:
        d = self._root / meeting_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def recording_path(self, meeting_id: str) -> Path:
        return self.meeting_dir(meeting_id) / "recording.wav"

    def delete_meeting_audio(self, meeting_id: str) -> None:
        d = self._root / meeting_id
        if d.exists():
            shutil.rmtree(d)
            log.info("Deleted audio for meeting %s", meeting_id)

    @property
    def root(self) -> Path:
        return self._root
