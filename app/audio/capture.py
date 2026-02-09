"""Real-time microphone capture using sounddevice."""

import logging
import time
from collections import deque
from queue import Empty, Queue
from typing import Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio at 16kHz mono float32.

    Audio callback pushes raw chunks to a thread-safe queue.
    Zero processing in the audio thread.
    """

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 512  # ~32ms at 16kHz
    TRAILING_CAPTURE_MS = 280  # default tail after key-up
    MIN_TRAILING_CAPTURE_MS = 130
    QUIET_BLOCKS_TO_STOP = 3

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.queue: Queue[np.ndarray] = Queue()
        self._stream: Optional[sd.InputStream] = None
        self._started_at: Optional[float] = None
        self._recent_rms: deque[float] = deque(maxlen=32)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        if chunk.size:
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            self._recent_rms.append(rms)
        self.queue.put(chunk)

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
        chunks: list[np.ndarray] = []
        chunks.extend(self._drain_queue_nowait())

        if trailing_capture_ms is None:
            trailing_capture_ms = self._default_trailing_capture_ms()
        if self._stream is not None:
            try:
                chunks.extend(
                    self._collect_trailing_chunks(
                        trailing_capture_ms=trailing_capture_ms,
                        min_trailing_capture_ms=self._min_trailing_capture_ms(),
                    )
                )
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
                self._started_at = None
        chunks.extend(self._drain_queue_nowait())

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
        if duration_s >= 180.0:
            return 1100
        if duration_s >= 120.0:
            return 960
        if duration_s >= 60.0:
            return 820
        if duration_s >= 30.0:
            return 700
        if duration_s >= 14.0:
            return 520
        if duration_s >= 8.0:
            return 420
        if duration_s >= 4.0:
            return 340
        return self.TRAILING_CAPTURE_MS

    def _min_trailing_capture_ms(self) -> int:
        if self._started_at is None:
            return self.MIN_TRAILING_CAPTURE_MS
        duration_s = max(0.0, time.monotonic() - self._started_at)
        if duration_s >= 120.0:
            return 420
        if duration_s >= 60.0:
            return 340
        if duration_s >= 20.0:
            return 260
        return self.MIN_TRAILING_CAPTURE_MS

    def _collect_trailing_chunks(
        self,
        trailing_capture_ms: int,
        min_trailing_capture_ms: int,
    ) -> list[np.ndarray]:
        if trailing_capture_ms <= 0:
            return []
        start = time.monotonic()
        deadline = start + (trailing_capture_ms / 1000.0)
        poll_timeout = max(self.block_size / self.sample_rate, 0.01)
        quiet_blocks = 0
        chunks: list[np.ndarray] = []
        quiet_threshold = self._silence_rms_threshold()

        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            timeout = min(poll_timeout, deadline - now)
            if timeout <= 0:
                break
            try:
                chunk = self.queue.get(timeout=timeout)
            except Empty:
                if (now - start) * 1000.0 >= min_trailing_capture_ms:
                    quiet_blocks += 1
                    if quiet_blocks >= self.QUIET_BLOCKS_TO_STOP:
                        break
                continue

            chunks.append(chunk)
            if chunk.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            if rms <= quiet_threshold:
                if (time.monotonic() - start) * 1000.0 >= min_trailing_capture_ms:
                    quiet_blocks += 1
                    if quiet_blocks >= self.QUIET_BLOCKS_TO_STOP:
                        break
            else:
                quiet_blocks = 0
        return chunks

    def _silence_rms_threshold(self) -> float:
        if not self._recent_rms:
            return 0.004
        values = np.asarray(self._recent_rms, dtype=np.float32)
        baseline = float(np.percentile(values, 25))
        return min(max(baseline * 1.8, 0.0032), 0.02)

    def _drain_queue_nowait(self) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except Empty:
                break
        return chunks

    def drain(self) -> None:
        """Clear all buffered audio from the queue."""
        self._drain_queue_nowait()
