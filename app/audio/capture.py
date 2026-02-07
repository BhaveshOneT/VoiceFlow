"""Real-time microphone capture using sounddevice."""

import logging
import time

import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from typing import Optional

log = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio at 16kHz mono float32.

    Audio callback pushes raw chunks to a thread-safe queue.
    Zero processing in the audio thread.
    """

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 512  # ~32ms at 16kHz
    TRAILING_CAPTURE_MS = 280  # default tail after key-up

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.queue: Queue[np.ndarray] = Queue()
        self._stream: Optional[sd.InputStream] = None
        self._started_at: Optional[float] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self.queue.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._started_at = time.monotonic()

    def stop(self, trailing_capture_ms: int | None = None) -> np.ndarray:
        """Stop recording and return all remaining audio concatenated.

        A short tail capture window helps avoid clipping the final words when
        users release the hotkey while still finishing a phrase.
        """
        if trailing_capture_ms is None:
            trailing_capture_ms = self._default_trailing_capture_ms()
        if self._stream is not None:
            try:
                if trailing_capture_ms > 0:
                    time.sleep(trailing_capture_ms / 1000.0)
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
                self._started_at = None

        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except Empty:
                break

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get a single audio chunk from the queue. Returns None on timeout."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def is_active(self) -> bool:
        if self._stream is None:
            return False
        return self._stream.active

    def _default_trailing_capture_ms(self) -> int:
        if self._started_at is None:
            return self.TRAILING_CAPTURE_MS
        duration_s = max(0.0, time.monotonic() - self._started_at)
        # Long dictation clips more easily at hotkey release, so keep a longer tail.
        if duration_s >= 14.0:
            return 520
        if duration_s >= 8.0:
            return 420
        if duration_s >= 4.0:
            return 340
        return self.TRAILING_CAPTURE_MS

    def drain(self) -> None:
        """Clear all buffered audio from the queue."""
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
