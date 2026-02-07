"""Real-time microphone capture using sounddevice."""

import logging

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

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.queue: Queue[np.ndarray] = Queue()
        self._stream: Optional[sd.InputStream] = None

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

    def stop(self) -> np.ndarray:
        """Stop recording and return all remaining audio concatenated."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None

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

    def drain(self) -> None:
        """Clear all buffered audio from the queue."""
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
