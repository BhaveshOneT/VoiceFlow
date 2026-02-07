"""Speech-to-text transcription using mlx-whisper."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
_TEMPERATURE = (0.0, 0.2)
_COMPRESSION_RATIO_THRESHOLD = 2.4
_LOGPROB_THRESHOLD = -1.0
_NO_SPEECH_THRESHOLD = 0.6


class WhisperEngine:
    """MLX-powered Whisper transcription optimized for dictation."""

    def __init__(self, model_name: str = DEFAULT_MODEL, language: str = "auto") -> None:
        self.model_name = model_name
        self.language = language
        self._warmed_up = False

    def transcribe(self, audio: np.ndarray, tech_context: str = "") -> str:
        """Transcribe float32 audio array to text.

        Uses mlx_whisper.transcribe() with language pre-set (skips
        auto-detection) and an initial_prompt built from the tech context
        to bias the decoder toward programming vocabulary.
        """
        import mlx_whisper  # type: ignore[import-untyped]

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            temperature=_TEMPERATURE,
            compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=_LOGPROB_THRESHOLD,
            no_speech_threshold=_NO_SPEECH_THRESHOLD,
            language=self._resolve_whisper_language(),
            initial_prompt=self._build_prompt(tech_context),
            condition_on_previous_text=False,
        )
        return result["text"].strip()

    def warm_up(self) -> None:
        """Run dummy inference to initialize the Metal/GPU pipeline."""
        if self._warmed_up:
            return
        log.info("Warming up Whisper model %s", self.model_name)
        dummy = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        try:
            self.transcribe(dummy)
        except Exception as exc:
            raise RuntimeError(
                f"Whisper warm-up failed for model '{self.model_name}': {exc}"
            ) from exc
        self._warmed_up = True
        log.info("Whisper warm-up complete")

    def _build_prompt(self, tech_context: str) -> str:
        """Build initial_prompt biasing Whisper toward clean output.

        Max 224 tokens -- Whisper silently truncates beyond this.
        """
        if self.language == "de":
            base = (
                "Die folgende Aufnahme stammt aus einer Softwareentwicklungssitzung. "
                "Bitte klar und korrekt transkribieren."
            )
        elif self.language == "auto":
            base = (
                "This is a software development dictation in English or German. "
                "Transcribe clearly with natural punctuation."
            )
        else:
            base = (
                "The following is a clean, well-punctuated transcription "
                "from a software development session."
            )
        if tech_context:
            return f"{base} {tech_context}"
        return base

    def set_language(self, language: str) -> None:
        self.language = language

    def _resolve_whisper_language(self) -> str | None:
        # Whisper auto-detects language when this argument is omitted.
        if self.language == "auto":
            return None
        return self.language
