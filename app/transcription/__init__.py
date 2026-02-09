"""Transcription pipeline: Whisper -> Cleanup -> Output."""
from __future__ import annotations

import logging
import os
import re
import time
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
_TRIM_FRAME_SAMPLES = 320  # 20 ms at 16 kHz
_TRIM_PADDING_SAMPLES = 3520  # 220 ms safety pad around speech
_TRIM_MIN_RMS = 0.0025
_TRIM_MAX_RMS = 0.018
_LONG_AUDIO_CHUNK_THRESHOLD_S = 75.0
_LONG_AUDIO_CHUNK_S = 42.0
_LONG_AUDIO_CHUNK_OVERLAP_S = 1.2
_LONG_AUDIO_MIN_FINAL_CHUNK_S = 12.0
_LONG_AUDIO_TAIL_PASS_THRESHOLD_S = 95.0
_LONG_AUDIO_TAIL_WINDOW_S = 24.0
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_TOKEN_SPLIT_RE = re.compile(r"\S+")
_MIN_OVERLAP_WORDS = 4
_MAX_OVERLAP_WORDS = 20


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
        self.transcription_mode = config.transcription_mode

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
        total_started = time.perf_counter()
        input_samples = int(audio.size)

        audio, trimmed = self._trim_silence_for_decode(audio)
        decode_samples = int(audio.size)
        programmer_mode = self._programmer_mode_enabled()

        # 1. Whisper transcription
        stt_started = time.perf_counter()
        tech_context = self.dictionary.get_whisper_context() if programmer_mode else ""
        raw = self._transcribe_adaptive(audio, tech_context=tech_context)
        stt_ms = (time.perf_counter() - stt_started) * 1000.0
        _log_transcript("Raw transcription", raw)

        if not raw.strip():
            total_ms = (time.perf_counter() - total_started) * 1000.0
            log.info(
                "Pipeline timings (ms): total=%.1f stt=%.1f clean=0.0 refine=0.0 "
                "finalize=0.0 input_s=%.2f decode_s=%.2f trimmed=%s",
                total_ms,
                stt_ms,
                input_samples / 16000.0,
                decode_samples / 16000.0,
                trimmed,
            )
            return ""

        # 2. Regex cleanup + dictionary replacement (always, <5ms)
        clean_started = time.perf_counter()
        dictionary_terms = self.dictionary.get_all_terms() if programmer_mode else {}
        cleaned = self.cleaner.clean(
            raw,
            dictionary_terms,
            programmer_mode=programmer_mode,
        )
        clean_ms = (time.perf_counter() - clean_started) * 1000.0
        _log_transcript("After regex cleanup", cleaned)
        needs_refinement = self._should_refine(cleaned, raw_text=raw)

        # 3. LLM refinement (standard + max_accuracy modes)
        refine_ms = 0.0
        if (
            self.refiner
            and self.config.cleanup_mode != "fast"
            and self.refiner.loaded
            and needs_refinement
        ):
            refine_started = time.perf_counter()
            try:
                pre_refine = cleaned
                refined = self.refiner.refine(
                    cleaned,
                    dictionary_terms,
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
            finally:
                refine_ms = (time.perf_counter() - refine_started) * 1000.0
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
        finalize_started = time.perf_counter()
        finalized = self.cleaner.clean(
            cleaned,
            dictionary_terms,
            programmer_mode=programmer_mode,
        )
        if finalized.strip():
            cleaned = finalized

        cleaned = self._preserve_completeness(
            raw,
            cleaned,
            dictionary_terms,
            programmer_mode=programmer_mode,
        )
        finalize_ms = (time.perf_counter() - finalize_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        log.info(
            "Pipeline timings (ms): total=%.1f stt=%.1f clean=%.1f refine=%.1f "
            "finalize=%.1f input_s=%.2f decode_s=%.2f trimmed=%s refine_needed=%s",
            total_ms,
            stt_ms,
            clean_ms,
            refine_ms,
            finalize_ms,
            input_samples / 16000.0,
            decode_samples / 16000.0,
            trimmed,
            needs_refinement,
        )
        _log_transcript("Final transcription output", cleaned)
        return cleaned

    def _transcribe_adaptive(self, audio: np.ndarray, tech_context: str) -> str:
        """Transcribe short audio directly; chunk long recordings for reliability."""
        duration_s = audio.size / 16000.0
        if duration_s < _LONG_AUDIO_CHUNK_THRESHOLD_S:
            return self._transcribe_with_fallback(audio, tech_context=tech_context)

        chunks = self._split_for_long_transcription(audio)
        log.info(
            "Long recording detected (%.1fs); transcribing in %d chunks",
            duration_s,
            len(chunks),
        )

        parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            part = self._transcribe_with_fallback(chunk, tech_context=tech_context).strip()
            if not part:
                continue
            parts.append(part)
            log.info(
                "Long recording chunk %d/%d decoded (chunk_s=%.1f, words=%d)",
                idx,
                len(chunks),
                chunk.size / 16000.0,
                len(part.split()),
            )

        if not parts:
            return ""

        merged = self._merge_transcript_parts(parts)
        if duration_s >= _LONG_AUDIO_TAIL_PASS_THRESHOLD_S:
            merged = self._append_tail_pass_if_needed(
                merged,
                audio,
                tech_context=tech_context,
            )
        return merged

    @staticmethod
    def _split_for_long_transcription(audio: np.ndarray) -> list[np.ndarray]:
        """Split long audio into overlapping chunks to avoid tail loss."""
        sample_rate = 16000
        chunk_samples = int(_LONG_AUDIO_CHUNK_S * sample_rate)
        overlap_samples = int(_LONG_AUDIO_CHUNK_OVERLAP_S * sample_rate)
        min_final_chunk_samples = int(_LONG_AUDIO_MIN_FINAL_CHUNK_S * sample_rate)
        stride = max(chunk_samples - overlap_samples, sample_rate)
        total = int(audio.size)

        if total <= chunk_samples:
            return [audio]

        chunks: list[np.ndarray] = []
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            remaining = total - end
            if 0 < remaining < min_final_chunk_samples:
                end = total
            chunks.append(audio[start:end])
            if end >= total:
                break
            start += stride
        return chunks

    @classmethod
    def _merge_transcript_parts(cls, parts: list[str]) -> str:
        """Merge chunk transcripts while dropping overlap duplication."""
        cleaned_parts = [part.strip() for part in parts if part and part.strip()]
        if not cleaned_parts:
            return ""
        if len(cleaned_parts) == 1:
            return cleaned_parts[0]

        merged = cleaned_parts[0]
        merged_tokens = cls._word_tokens(merged)
        for part in cleaned_parts[1:]:
            part_tokens = cls._word_tokens(part)
            overlap = cls._find_token_overlap(merged_tokens, part_tokens)
            trimmed_part = cls._drop_leading_tokens(part, overlap)
            if not trimmed_part:
                continue
            merged = f"{merged} {trimmed_part}".strip()
            merged_tokens = cls._word_tokens(merged)
        return merged

    def _append_tail_pass_if_needed(
        self,
        transcript: str,
        audio: np.ndarray,
        tech_context: str,
    ) -> str:
        """Decode the tail of long recordings to prevent dropped final details."""
        if audio.size <= int(_LONG_AUDIO_TAIL_WINDOW_S * 16000):
            return transcript

        tail_samples = int(_LONG_AUDIO_TAIL_WINDOW_S * 16000)
        tail_audio = audio[-tail_samples:]
        tail_text = self._transcribe_with_fallback(
            tail_audio,
            tech_context=tech_context,
        ).strip()
        if not tail_text:
            return transcript
        if self._is_tail_covered(transcript, tail_text):
            return transcript

        merged = self._merge_transcript_parts([transcript, tail_text])
        log.info(
            "Tail-pass appended extra detail (base_words=%d, merged_words=%d)",
            len(transcript.split()),
            len(merged.split()),
        )
        return merged

    @staticmethod
    def _word_tokens(text: str) -> list[str]:
        return [match.group(0).lower() for match in _WORD_TOKEN_RE.finditer(text)]

    @staticmethod
    def _drop_leading_tokens(text: str, token_count: int) -> str:
        if token_count <= 0:
            return text.strip()
        matches = list(_TOKEN_SPLIT_RE.finditer(text))
        if token_count >= len(matches):
            return ""
        return text[matches[token_count - 1].end():].lstrip()

    @classmethod
    def _find_token_overlap(cls, left: list[str], right: list[str]) -> int:
        if not left or not right:
            return 0
        upper = min(_MAX_OVERLAP_WORDS, len(left), len(right))
        for size in range(upper, _MIN_OVERLAP_WORDS - 1, -1):
            if left[-size:] == right[:size]:
                return size
        return 0

    @classmethod
    def _is_tail_covered(cls, full_text: str, tail_text: str) -> bool:
        full_tokens = cls._word_tokens(full_text)
        tail_tokens = cls._word_tokens(tail_text)
        if len(full_tokens) < 6 or len(tail_tokens) < 6:
            return False
        probe_size = min(12, len(tail_tokens))
        probe = tail_tokens[-probe_size:]
        for start in range(len(full_tokens) - probe_size + 1):
            if full_tokens[start:start + probe_size] == probe:
                return True
        return False

    @staticmethod
    def _trim_silence_for_decode(audio: np.ndarray) -> tuple[np.ndarray, bool]:
        """Trim leading/trailing silence before STT to reduce decode latency."""
        if audio.size < _TRIM_FRAME_SAMPLES * 4:
            return audio, False

        usable = (audio.size // _TRIM_FRAME_SAMPLES) * _TRIM_FRAME_SAMPLES
        if usable <= 0:
            return audio, False

        framed = audio[:usable].reshape(-1, _TRIM_FRAME_SAMPLES)
        rms = np.sqrt(np.mean(np.square(framed), axis=1))
        noise_floor = float(np.percentile(rms, 20))
        threshold = max(_TRIM_MIN_RMS, min(_TRIM_MAX_RMS, noise_floor * 2.4))
        active = np.where(rms > threshold)[0]

        if active.size == 0:
            return audio, False

        start = max(int(active[0] * _TRIM_FRAME_SAMPLES) - _TRIM_PADDING_SAMPLES, 0)
        end = min(
            int((active[-1] + 1) * _TRIM_FRAME_SAMPLES) + _TRIM_PADDING_SAMPLES,
            int(audio.size),
        )
        if end - start <= 0:
            return audio, False

        trimmed = audio[start:end]
        if trimmed.size >= audio.size:
            return audio, False

        # Avoid over-trimming low-volume dictation by enforcing a wide minimum
        # window for medium/long recordings.
        if audio.size >= 16000 * 3 and trimmed.size < int(audio.size * 0.4):
            expanded_start = max(start - 16000 // 2, 0)
            expanded_end = min(end + 16000 // 2, int(audio.size))
            trimmed = audio[expanded_start:expanded_end]
            if trimmed.size >= audio.size:
                return audio, False

        return trimmed, True

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
        programmer_mode: bool,
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

        conservative = self.cleaner.clean_conservative(
            raw,
            dictionary_terms,
            programmer_mode=programmer_mode,
        )
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

    def set_transcription_mode(self, mode: str) -> None:
        """Switch between normal and programmer transcription behavior."""
        normalized = "programmer" if str(mode).lower() == "programmer" else "normal"
        self.transcription_mode = normalized
        self.config.transcription_mode = normalized

    def _programmer_mode_enabled(self) -> bool:
        return self.transcription_mode == "programmer"

    def set_language(self, language: str) -> None:
        """Switch transcription language at runtime.

        Supported:
        - "auto": auto-detect language (best for mixed English/German dictation)
        - "en": force English
        - "de": force German
        """
        self.config.language = language
        self.whisper.set_language(language)
