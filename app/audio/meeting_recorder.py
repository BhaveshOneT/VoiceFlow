"""Long-form meeting audio recorder.

Streams audio to a WAV file on disk with constant memory usage.
A 2-hour recording at 16kHz mono 16-bit produces ~230MB of WAV data.
"""
from __future__ import annotations

import logging
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 1024  # ~64ms per chunk


class MeetingRecorder:
    """Streams microphone audio to WAV file on disk.

    The audio callback converts float32 to int16 and writes directly
    to the WAV file, so memory usage stays constant regardless of
    recording duration.
    """

    def __init__(self) -> None:
        self._stream: Optional[sd.InputStream] = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._output_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._recording = False
        self._start_time: float = 0.0
        self._sample_count: int = 0
        self._on_audio_level: Optional[Callable[[float], None]] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return self._sample_count / SAMPLE_RATE
        return time.monotonic() - self._start_time

    @property
    def output_path(self) -> Optional[Path]:
        return self._output_path

    def start(
        self,
        output_path: Path,
        device_id: Optional[int] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Start recording to *output_path*.

        Args:
            output_path: Path for the output WAV file.
            device_id: sounddevice device index, or None for default.
            on_audio_level: Optional callback receiving RMS level (0.0-1.0)
                for waveform visualization. Called from audio thread.
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path = output_path
            self._on_audio_level = on_audio_level
            self._sample_count = 0

            # Open WAV file for writing
            wf = wave.open(str(output_path), "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            self._wav_file = wf

            # Open audio stream
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=device_id,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._start_time = time.monotonic()
            self._recording = True
            log.info("Meeting recording started -> %s (device=%s)", output_path, device_id)

    def stop(self) -> Path:
        """Stop recording and return the path to the WAV file."""
        with self._lock:
            if not self._recording:
                raise RuntimeError("Not recording")

            self._recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

            duration = self._sample_count / SAMPLE_RATE
            log.info(
                "Meeting recording stopped; %d samples (%.1fs) -> %s",
                self._sample_count, duration, self._output_path,
            )
            return self._output_path

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        if status:
            log.debug("Audio status: %s", status)

        # Convert float32 [-1, 1] to int16
        audio_int16 = np.clip(indata[:, 0] * 32767, -32768, 32767).astype(np.int16)
        raw_bytes = audio_int16.tobytes()

        with self._lock:
            if self._wav_file is not None:
                self._wav_file.writeframes(raw_bytes)
                self._sample_count += frames

        # Notify level callback for waveform visualization
        if self._on_audio_level is not None:
            rms = float(np.sqrt(np.mean(np.square(indata[:, 0]))))
            self._on_audio_level(min(1.0, rms * 5.0))  # Scale up for visibility
