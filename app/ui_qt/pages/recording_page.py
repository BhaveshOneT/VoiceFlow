"""Active meeting recording page with waveform and device selection."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.audio.meeting_recorder import MeetingRecorder
from app.storage.audio_store import AudioStore
from app.storage.database import MeetingDatabase
from app.storage.models import Meeting, MeetingStatus
from app.ui_qt.styles.theme import Spacing
from app.ui_qt.widgets.device_selector import DeviceSelector
from app.ui_qt.widgets.recording_controls import RecordingControls
from app.ui_qt.widgets.waveform_widget import WaveformWidget

if TYPE_CHECKING:
    from app.core.signals import AppSignals

log = logging.getLogger(__name__)


class RecordingPage(QWidget):
    """Page for active meeting recording."""

    def __init__(
        self,
        signals: "AppSignals",
        db: MeetingDatabase,
        audio_store: AudioStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._db = db
        self._audio_store = audio_store
        self._recorder = MeetingRecorder()
        self._current_meeting: Optional[Meeting] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        title = QLabel("Record Meeting")
        title.setObjectName("heading")
        layout.addWidget(title)

        # Device selector
        device_frame = QFrame()
        device_frame.setProperty("class", "card")
        device_layout = QHBoxLayout(device_frame)
        device_layout.setContentsMargins(
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
            Spacing.CARD_PADDING, Spacing.CARD_PADDING,
        )
        device_layout.setSpacing(Spacing.MD)
        device_label = QLabel("Input Device:")
        device_layout.addWidget(device_label)
        self._device_selector = DeviceSelector()
        device_layout.addWidget(self._device_selector, 1)
        layout.addWidget(device_frame)

        # Waveform (styled via QSS .card)
        waveform_frame = QFrame()
        waveform_frame.setProperty("class", "card")
        waveform_layout = QVBoxLayout(waveform_frame)
        waveform_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        self._waveform = WaveformWidget()
        waveform_layout.addWidget(self._waveform)
        layout.addWidget(waveform_frame)

        # Recording status label (styled via QSS objectName)
        self._status_label = QLabel("Ready to record")
        self._status_label.setObjectName("caption")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Controls
        self._controls = RecordingControls()
        self._controls.start_clicked.connect(self._start_recording)
        self._controls.stop_clicked.connect(self._stop_recording)
        layout.addWidget(self._controls)

        layout.addStretch()

        # Elapsed time timer
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(500)
        self._tick_timer.timeout.connect(self._update_elapsed)

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    @Slot()
    def _start_recording(self) -> None:
        device_id = self._device_selector.selected_device_id()

        # Create meeting in DB
        title = f"Meeting {datetime.now().strftime('%b %d, %Y %I:%M %p')}"
        device_name = self._device_selector.currentText()
        meeting = self._db.create_meeting(title=title, device_name=device_name)
        self._current_meeting = meeting

        # Get audio path
        audio_path = self._audio_store.recording_path(meeting.id)
        self._db.update_meeting(meeting.id, audio_path=str(audio_path))

        try:
            self._recorder.start(
                output_path=audio_path,
                device_id=device_id,
                on_audio_level=self._on_audio_level,
            )
        except Exception as exc:
            log.exception("Failed to start meeting recording")
            self._status_label.setText(f"\u26A0  Error: {exc}")
            self._db.update_meeting(
                meeting.id, status=MeetingStatus.ERROR, error_message=str(exc)
            )
            return

        self._controls.set_recording(True)
        self._status_label.setText("\u23FA  Recording...")
        self._status_label.setObjectName("caption")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)
        self._tick_timer.start()
        self._signals.meeting_recording_started.emit(meeting.id)
        log.info("Meeting recording started: %s", meeting.id)

    @Slot()
    def _stop_recording(self) -> None:
        if not self._recorder.is_recording:
            return

        self._tick_timer.stop()
        audio_path = self._recorder.stop()
        duration = self._recorder.elapsed_seconds

        self._controls.set_recording(False)
        self._waveform.reset()
        self._status_label.setText("\u2713  Recording saved. Ready for transcription.")
        self._status_label.setObjectName("caption")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

        if self._current_meeting:
            self._db.update_meeting(
                self._current_meeting.id,
                status=MeetingStatus.RECORDED,
                duration_seconds=duration,
                audio_path=str(audio_path),
            )
            self._signals.meeting_recording_stopped.emit(self._current_meeting.id)
            log.info(
                "Meeting recording stopped: %s (%.1fs)",
                self._current_meeting.id, duration,
            )
            self._current_meeting = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_audio_level(self, level: float) -> None:
        """Called from audio thread -- use signal or direct call."""
        # QWidget.update() is thread-safe in PySide6
        self._waveform.update_level(level)

    @Slot()
    def _update_elapsed(self) -> None:
        if self._recorder.is_recording:
            self._controls.set_elapsed(self._recorder.elapsed_seconds)
