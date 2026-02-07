"""Transcription pipeline: Whisper -> Cleanup -> Output."""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import numpy as np

from app.config import AppConfig
from app.dictionary import Dictionary
from .text_cleaner import TextCleaner
from .text_refiner import TextRefiner
from .whisper_engine import WhisperEngine

log = logging.getLogger(__name__)
_LOG_TRANSCRIPTS = os.getenv("VOICEFLOW_LOG_TRANSCRIPTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

_CORRECTION_CUE_RE = re.compile(
    r"\b(sorry|i mean|i meant|no wait|wait no|no,\s*no|scratch that|"
    r"never mind|let me rephrase|correction|rather)\b",
    re.IGNORECASE,
)
_COMPLEX_TEXT_RE = re.compile(r"[,:;]|(?:\b(and|but|because|then)\b)", re.IGNORECASE)
_FILLER_CUE_RE = re.compile(
    r"\b(um+|uh+|hmm+|hm+|you know|sort of|kind of|basically|literally)\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|is|are|am|was|were|do|does|did|can|"
    r"could|should|would|will|which|whose|whom|what's|whats|isn't|aren't|"
    r"won't|can't|couldn't|shouldn't|wouldn't|wer|was|wann|wo|warum|wie|"
    r"ist|sind|bin|war|waren|kann|kannst|können|soll|sollte|würde|"
    r"hat|haben|gibt|gibt's)\b",
    re.IGNORECASE,
)
_ORPHAN_END_RE = re.compile(
    r"\b(and|or|but|also|so|because|then|if|that|which|who|when|where|while|"
    r"although|however|therefore)$",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"[.!?]")


def _log_transcript(stage: str, text: str) -> None:
    """Log transcript content only when explicitly enabled."""
    if _LOG_TRANSCRIPTS:
        log.info("%s: %s", stage, text)
        return
    log.info(
        "%s (chars=%d, words=%d)",
        stage,
        len(text),
        len(text.split()),
    )


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
        _log_transcript("Raw transcription", raw)

        if not raw.strip():
            return ""

        # 2. Regex cleanup + dictionary replacement (always, <5ms)
        dictionary_terms = self.dictionary.get_all_terms()
        cleaned = self.cleaner.clean(raw, dictionary_terms)
        _log_transcript("After regex cleanup", cleaned)
        needs_refinement = self._should_refine(cleaned, raw_text=raw)

        # 3. LLM refinement (standard + max_accuracy modes)
        if (
            self.refiner
            and self.config.cleanup_mode != "fast"
            and self.refiner.loaded
            and needs_refinement
        ):
            try:
                pre_refine = cleaned
                refined = self.refiner.refine(
                    cleaned, dictionary_terms
                )
                if refined.strip():
                    if self._is_suspiciously_short_refinement(pre_refine, refined):
                        log.warning(
                            "Rejected LLM refinement due to potential truncation "
                            "(source_words=%d, refined_words=%d)",
                            len(pre_refine.split()),
                            len(refined.split()),
                        )
                    else:
                        cleaned = refined
                        _log_transcript("After LLM refinement", cleaned)
                else:
                    log.warning("LLM output rejected as prompt/meta leakage")
            except Exception as e:
                log.warning(
                    "LLM refinement failed, using regex result: %s", e
                )
        elif (
            self.refiner
            and self.config.cleanup_mode != "fast"
            and needs_refinement
            and not self.refiner.loaded
        ):
            # Keep interaction fast while the refiner model downloads/loads.
            log.info("LLM refiner not ready yet; using deterministic cleanup only")

        # 4. Final deterministic cleanup to enforce tag formatting and
        # disfluency rules even after optional LLM rewriting.
        finalized = self.cleaner.clean(cleaned, dictionary_terms)
        if finalized.strip():
            cleaned = finalized

        cleaned = self._preserve_completeness(raw, cleaned, dictionary_terms)
        _log_transcript("Final transcription output", cleaned)
        return cleaned

    def _should_refine(self, text: str, raw_text: str | None = None) -> bool:
        """Heuristic gate to avoid unnecessary LLM calls and reduce latency."""
        stripped = text.strip()
        word_count = len(text.split())
        if word_count < 4:
            return False
        # Keep dictated questions literal; avoid instruct models hallucinating answers.
        if stripped.endswith("?") or _QUESTION_START_RE.match(stripped):
            return False
        if _CORRECTION_CUE_RE.search(text) or (
            raw_text is not None and _CORRECTION_CUE_RE.search(raw_text)
        ):
            return True
        if raw_text is not None and _FILLER_CUE_RE.search(raw_text):
            return True
        # Prefer completeness over rewrite quality for long dictation.
        if word_count >= 24:
            return False
        sentence_count = len(_SENTENCE_END_RE.findall(text))
        if sentence_count >= 2 and word_count >= 16:
            return False
        # Long-form, already-punctuated dictation should stay on deterministic
        # cleanup path for speed and to avoid unnecessary rewrites.
        if word_count >= 40 and text.endswith((".", "!", "?")):
            return False
        if word_count <= 10:
            return False
        if word_count < 14 and text.endswith((".", "!", "?")):
            return False
        has_complexity = bool(_COMPLEX_TEXT_RE.search(text))
        if text.endswith((".", "!", "?")) and word_count < 24 and not has_complexity:
            return False
        if has_complexity and not text.endswith((".", "!", "?")):
            return True
        return word_count >= 22 and not text.endswith((".", "!", "?"))

    @staticmethod
    def _is_suspiciously_short_refinement(source: str, candidate: str) -> bool:
        source_words = len(source.split())
        candidate_words = len(candidate.split())
        if source_words < 10:
            return False
        # Prevent aggressive shortening that can drop meaning.
        if candidate_words <= 3:
            return True
        if source_words >= 40 and candidate_words < int(source_words * 0.88):
            return True
        if source_words >= 30 and candidate_words < int(source_words * 0.82):
            return True
        if source_words >= 18 and candidate_words < int(source_words * 0.65):
            return True
        if len(candidate) < int(len(source) * 0.70) and source_words >= 24:
            return True
        if candidate.strip() and _ORPHAN_END_RE.search(candidate.strip()):
            return True
        return False

    def _preserve_completeness(
        self,
        raw: str,
        cleaned: str,
        dictionary_terms: dict[str, str],
    ) -> str:
        """Fallback to conservative cleanup if aggressive shortening is detected."""
        raw_words = len(raw.split())
        cleaned_words = len(cleaned.split())
        if raw_words < 24 or cleaned_words == 0:
            return cleaned
        if _CORRECTION_CUE_RE.search(raw):
            return cleaned
        if cleaned_words >= int(raw_words * 0.78):
            return cleaned
        if not _ORPHAN_END_RE.search(cleaned.strip()):
            return cleaned

        conservative = self.cleaner.clean_conservative(raw, dictionary_terms)
        conservative_words = len(conservative.split())
        if conservative_words > cleaned_words:
            log.warning(
                "Using conservative cleanup to preserve long dictation "
                "(raw_words=%d, cleaned_words=%d, conservative_words=%d)",
                raw_words,
                cleaned_words,
                conservative_words,
            )
            return conservative
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
        self._warm_up_whisper_with_fallback()
        if self.refiner:
            self.refiner.load()
            log.info("LLM loaded and ready")

    def warm_up_for_realtime(self) -> None:
        """Warm up only the critical real-time path (Whisper)."""
        self._warm_up_whisper_with_fallback()

    def warm_up_refiner(self) -> None:
        """Warm up optional LLM refiner (can run in background)."""
        if self.refiner and not self.refiner.loaded:
            self.refiner.load()
            log.info("LLM loaded and ready")

    def _warm_up_whisper_with_fallback(self) -> None:
        """Warm up the active Whisper model and fall back when max-accuracy is unavailable."""
        primary_error: Exception | None = None
        try:
            self.whisper.warm_up()
            return
        except Exception as exc:
            primary_error = exc
            log.exception(
                "Whisper warm-up failed for model %s", self.whisper.model_name
            )

        fallback_model = self.config.whisper_model
        if self.whisper.model_name == fallback_model:
            raise RuntimeError(
                f"Whisper warm-up failed for model '{self.whisper.model_name}'"
            ) from primary_error

        log.warning("Retrying warm-up with fallback model %s", fallback_model)
        fallback_engine = WhisperEngine(
            model_name=fallback_model,
            language=self.config.language,
        )
        try:
            fallback_engine.warm_up()
            self.whisper = fallback_engine
            log.warning(
                "Warm-up fallback succeeded; using model %s", fallback_model
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"Whisper warm-up failed on primary model "
                f"'{self.config.max_accuracy_whisper_model}' and fallback "
                f"'{fallback_model}'"
            ) from fallback_error

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
            self._warm_up_whisper_with_fallback()

        # Handle LLM refiner
        if mode == "fast" and self.refiner:
            self.refiner.unload()
        elif mode != "fast" and self.refiner and not self.refiner.loaded:
            self.refiner.load()
        elif mode != "fast" and not self.refiner:
            self.refiner = TextRefiner(model_name=self.config.llm_model)
            self.refiner.load()

    def set_language(self, language: str) -> None:
        """Switch transcription language at runtime.

        Supported:
        - "auto": auto-detect language (best for mixed English/German dictation)
        - "en": force English
        - "de": force German
        """
        self.config.language = language
        self.whisper.set_language(language)
