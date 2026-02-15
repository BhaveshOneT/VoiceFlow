"""Application-wide Qt signals for cross-thread communication.

All backend-to-frontend communication goes through these signals so that
no backend module needs to import any UI code.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AppSignals(QObject):
    """Singleton-style signal hub.

    Usage::

        signals = AppSignals()
        signals.recording_started.connect(some_slot)
        signals.recording_started.emit()
    """

    # -- Dictation lifecycle --------------------------------------------------
    recording_started = Signal()
    recording_stopped = Signal()
    transcription_complete = Signal(str)  # transcribed text

    # -- Model lifecycle ------------------------------------------------------
    model_loading = Signal(str)          # model name
    model_loaded = Signal(str)           # model name
    model_download_progress = Signal(str, float)  # model name, 0.0-1.0

    # -- Meeting lifecycle ----------------------------------------------------
    meeting_recording_started = Signal(str)   # meeting id
    meeting_recording_stopped = Signal(str)   # meeting id
    meeting_transcription_progress = Signal(str, float)  # meeting id, 0.0-1.0
    meeting_transcription_complete = Signal(str)  # meeting id

    # -- Settings changes -----------------------------------------------------
    hotkey_changed = Signal(str)          # new key name (e.g. "right_cmd")
    language_changed = Signal(str)        # new language code
    accuracy_changed = Signal(str)        # new cleanup_mode
    transcription_mode_changed = Signal(str)  # "normal" or "programmer"

    # -- General status -------------------------------------------------------
    status_changed = Signal(str)  # human-readable status message
    error_occurred = Signal(str, str)  # title, message
