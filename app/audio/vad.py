"""Voice Activity Detection using Silero VAD (ONNX)."""

import hashlib
import ssl
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import certifi
import numpy as np
import onnxruntime as ort


class VoiceActivityDetector:
    """Detects speech boundaries using Silero VAD via ONNX Runtime.

    Accumulates audio during speech, returns complete utterance
    when silence is detected. Pre-buffer captures word onsets.
    """

    MODEL_URL = (
        "https://github.com/snakers4/silero-vad/raw/master/"
        "src/silero_vad/data/silero_vad.onnx"
    )
    MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    ALLOWED_DOWNLOAD_HOSTS = {"github.com", "raw.githubusercontent.com"}
    CACHE_DIR = Path.home() / ".cache" / "voiceflow"
    MODEL_PATH = CACHE_DIR / "silero_vad.onnx"

    def __init__(
        self,
        threshold: float = 0.5,
        silence_duration_ms: int = 700,
        pre_buffer_ms: int = 300,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate

        chunk_ms = 32  # ~32ms per chunk at 512 samples / 16kHz
        self.silence_limit = int(silence_duration_ms / chunk_ms)
        pre_buffer_chunks = int(pre_buffer_ms / chunk_ms)

        self.pre_buffer: deque[np.ndarray] = deque(maxlen=pre_buffer_chunks)
        self.speech_chunks: list[np.ndarray] = []
        self.silence_counter = 0

        self._session: Optional[ort.InferenceSession] = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Feed a ~32ms chunk. Returns complete utterance when speech ends."""
        self._ensure_model()
        prob = self._infer(chunk)
        is_speech = prob > self.threshold

        if is_speech:
            if not self.speech_chunks:
                self.speech_chunks.extend(list(self.pre_buffer))
            self.speech_chunks.append(chunk)
            self.silence_counter = 0
        else:
            if self.speech_chunks:
                self.speech_chunks.append(chunk)
                self.silence_counter += 1
                if self.silence_counter >= self.silence_limit:
                    audio = np.concatenate(self.speech_chunks)
                    self.speech_chunks = []
                    self.silence_counter = 0
                    return audio
            else:
                self.pre_buffer.append(chunk)

        return None

    def speech_probability(self, chunk: np.ndarray) -> float:
        """Return speech probability for a single audio chunk (public API)."""
        self._ensure_model()
        return self._infer(chunk)

    def reset(self) -> None:
        """Reset all state for a new utterance."""
        self.speech_chunks = []
        self.silence_counter = 0
        self.pre_buffer.clear()
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _ensure_model(self) -> None:
        """Download ONNX model if not cached, then load session."""
        if self._session is not None:
            return

        if not self._has_valid_model_file():
            self._download_model()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.MODEL_PATH), sess_options=opts
        )

    def _has_valid_model_file(self) -> bool:
        if not self.MODEL_PATH.exists():
            return False
        return self._sha256_file(self.MODEL_PATH) == self.MODEL_SHA256

    def _download_model(self) -> None:
        parsed = urlparse(self.MODEL_URL)
        if parsed.scheme != "https" or parsed.netloc not in self.ALLOWED_DOWNLOAD_HOSTS:
            raise RuntimeError(f"Blocked model URL: {self.MODEL_URL}")

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(self.MODEL_URL)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:  # nosec B310
            content = resp.read()

        digest = hashlib.sha256(content).hexdigest()
        if digest != self.MODEL_SHA256:
            raise RuntimeError(
                "Silero VAD checksum mismatch. "
                "Refusing to use unverified model artifact."
            )

        tmp_path = self.MODEL_PATH.with_suffix(".onnx.tmp")
        tmp_path.write_bytes(content)
        tmp_path.replace(self.MODEL_PATH)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _infer(self, chunk: np.ndarray) -> float:
        """Run VAD inference on a single chunk. Returns speech probability."""
        audio = chunk.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, samples)

        sr = np.array(self.sample_rate, dtype=np.int64)

        ort_inputs = {
            "input": audio,
            "sr": sr,
            "h": self._h,
            "c": self._c,
        }

        output, self._h, self._c = self._session.run(None, ort_inputs)
        return float(output.squeeze())
