"""Speaker diarization using pyannote.audio 3.1.

Wraps the pyannote speaker-diarization pipeline for local inference
on Apple Silicon (MPS backend).  Downloads the model on first use.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Community model that doesn't require auth gating
_DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass
class DiarizationSegment:
    start: float   # seconds
    end: float     # seconds
    speaker: str   # e.g. "SPEAKER_00"


class SpeakerDiarizer:
    """Speaker diarization powered by pyannote.audio.

    The pipeline is heavy (~1.5GB with PyTorch); use ``load()``/``unload()``
    to manage memory explicitly via the ModelManager.
    """

    def __init__(self, auth_token: str = "", model: str = _DEFAULT_MODEL) -> None:
        self._auth_token = auth_token or None
        self._model = model
        self._pipeline = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if pyannote.audio and torch are importable."""
        try:
            import pyannote.audio  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> bool:
        """Load the diarization pipeline onto MPS (Metal).

        Returns True if the pipeline loaded successfully, False if
        dependencies are missing.
        """
        if self._pipeline is not None:
            return True

        log.info("Loading pyannote diarization pipeline: %s", self._model)
        try:
            import torch
            from pyannote.audio import Pipeline  # type: ignore[import-untyped]

            pipeline = Pipeline.from_pretrained(
                self._model,
                use_auth_token=self._auth_token,
            )

            # Move to Metal GPU if available
            if torch.backends.mps.is_available():
                import torch
                pipeline.to(torch.device("mps"))
                log.info("Diarization pipeline running on MPS (Metal)")
            else:
                log.info("Diarization pipeline running on CPU")

            self._pipeline = pipeline
            log.info("Diarization pipeline loaded")
            return True
        except ImportError:
            log.warning(
                "pyannote.audio and/or PyTorch not installed — "
                "diarization unavailable"
            )
            return False
        except Exception:
            log.exception("Failed to load diarization pipeline")
            return False

    def unload(self) -> None:
        """Free the pipeline and reclaim GPU memory."""
        if self._pipeline is None:
            return
        log.info("Unloading diarization pipeline")
        del self._pipeline
        self._pipeline = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    def diarize(self, audio_path: Path | str) -> list[DiarizationSegment]:
        """Run diarization on an audio file.

        Args:
            audio_path: Path to WAV file (16kHz mono recommended).

        Returns:
            List of segments with speaker labels, sorted by start time.
        """
        if self._pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded. Call load() first.")

        log.info("Starting diarization: %s", audio_path)
        diarization = self._pipeline(str(audio_path))

        segments: list[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start)
        log.info(
            "Diarization complete: %d segments, %d speakers",
            len(segments),
            len({s.speaker for s in segments}),
        )
        return segments
