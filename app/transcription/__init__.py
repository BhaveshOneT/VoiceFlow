"""Transcription pipeline: Whisper -> Cleanup -> Output."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.dictionary import Dictionary
from .text_cleaner import TextCleaner
from .text_refiner import TextRefiner
from .whisper_engine import WhisperEngine

log = logging.getLogger(__name__)


class TranscriptionPipeline:
    """Coordinates the full transcription pipeline.

    Stages:
        1. Whisper STT (with tech vocabulary bias via initial_prompt)
        2. Regex cleanup + dictionary term replacement  (always, <5 ms)
        3. LLM refinement  (standard & max_accuracy modes only)
    """

    def __init__(self, config: AppConfig, dictionary: Dictionary) -> None:
        self.config = config
        self.dictionary = dictionary

        # Select whisper model based on cleanup mode
        whisper_model = config.whisper_model
        if config.cleanup_mode == "max_accuracy":
            whisper_model = config.max_accuracy_whisper_model

        self.whisper = WhisperEngine(
            model_name=whisper_model,
            language=config.language,
        )
        self.cleaner = TextCleaner
        self.refiner: Optional[TextRefiner] = None

        # Load LLM refiner for standard and max_accuracy modes
        if config.cleanup_mode != "fast":
            self.refiner = TextRefiner(model_name=config.llm_model)

    def process(self, audio: np.ndarray) -> str:
        """Run the full pipeline on audio data. Returns cleaned text."""
        # 1. Whisper transcription
        tech_context = self.dictionary.get_whisper_context()
        raw = self._transcribe_with_fallback(audio, tech_context=tech_context)
        log.info("Raw transcription: %s", raw)

        if not raw.strip():
            return ""

        # 2. Regex cleanup + dictionary replacement (always, <5ms)
        cleaned = self.cleaner.clean(raw, self.dictionary.get_all_terms())
        log.info("After regex cleanup: %s", cleaned)

        # 3. LLM refinement (standard + max_accuracy modes)
        if self.refiner and self.config.cleanup_mode != "fast":
            if len(cleaned.split()) >= 3:
                try:
                    refined = self.refiner.refine(
                        cleaned, self.dictionary.get_all_terms()
                    )
                    if refined.strip():
                        cleaned = refined
                        log.info("After LLM refinement: %s", cleaned)
                except Exception as e:
                    log.warning(
                        "LLM refinement failed, using regex result: %s", e
                    )

        return cleaned

    def _transcribe_with_fallback(self, audio: np.ndarray, tech_context: str) -> str:
        """Transcribe audio and fall back to turbo model if max-accuracy model fails."""
        primary_error: Exception | None = None
        try:
            return self.whisper.transcribe(audio, tech_context=tech_context)
        except Exception as exc:
            primary_error = exc
            log.exception(
                "Primary Whisper transcription failed with model %s",
                self.whisper.model_name,
            )

        fallback_model = self.config.whisper_model
        if self.whisper.model_name == fallback_model:
            raise RuntimeError(
                f"Whisper transcription failed with model '{self.whisper.model_name}'"
            ) from primary_error

        log.warning(
            "Retrying transcription with fallback model %s", fallback_model
        )
        fallback_engine = WhisperEngine(
            model_name=fallback_model,
            language=self.config.language,
        )
        try:
            fallback_engine.warm_up()
            raw = fallback_engine.transcribe(audio, tech_context=tech_context)
            self.whisper = fallback_engine
            log.warning(
                "Fallback transcription succeeded; switched active model to %s",
                fallback_model,
            )
            return raw
        except Exception as fallback_error:
            raise RuntimeError(
                f"Whisper transcription failed on primary model "
                f"'{self.config.max_accuracy_whisper_model}' and fallback "
                f"'{fallback_model}'"
            ) from fallback_error

    def warm_up(self) -> None:
        """Load and warm up all models."""
        self.whisper.warm_up()
        if self.refiner:
            self.refiner.load()
            log.info("LLM loaded and ready")

    def set_cleanup_mode(self, mode: str) -> None:
        """Switch cleanup mode at runtime."""
        old_mode = self.config.cleanup_mode
        self.config.cleanup_mode = mode

        # Reload Whisper model when switching to/from max_accuracy
        if (old_mode == "max_accuracy") != (mode == "max_accuracy"):
            new_model = (
                self.config.max_accuracy_whisper_model
                if mode == "max_accuracy"
                else self.config.whisper_model
            )
            log.info("Switching Whisper model to %s", new_model)
            self.whisper = WhisperEngine(
                model_name=new_model,
                language=self.config.language,
            )
            self.whisper.warm_up()

        # Handle LLM refiner
        if mode == "fast" and self.refiner:
            self.refiner.unload()
        elif mode != "fast" and self.refiner and not self.refiner.loaded:
            self.refiner.load()
        elif mode != "fast" and not self.refiner:
            self.refiner = TextRefiner(model_name=self.config.llm_model)
            self.refiner.load()
