"""Speech-to-text engine wrapper (Parakeet-first, Whisper-compatible fallback)."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# Whisper decode defaults (used only when model_name targets Whisper).
_WHISPER_TEMPERATURE = (0.0, 0.2, 0.4)
_WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
_WHISPER_LOGPROB_THRESHOLD = -1.0
_WHISPER_NO_SPEECH_THRESHOLD = 0.6

# Parakeet guardrails.
_PARAKEET_MIN_AUDIO_S = 1.6
_PARAKEET_LONG_AUDIO_CHUNK_S = 90.0
_PARAKEET_LONG_AUDIO_OVERLAP_S = 12.0
_PARAKEET_LONG_AUDIO_THRESHOLD_S = 95.0


class WhisperEngine:
    """STT engine adapter.

    Name kept for backward compatibility with existing call-sites.
    Uses Parakeet by default, and routes to mlx-whisper when a Whisper model
    identifier is explicitly configured.
    """

    _mlx_whisper = None

    _parakeet_from_pretrained = None
    _parakeet_get_logmel = None
    _parakeet_tokens_to_sentences = None
    _parakeet_sentences_to_result = None
    _parakeet_merge_longest_contiguous = None
    _parakeet_merge_longest_common_subsequence = None
    _parakeet_decoding_config = None
    _mx = None

    def __init__(self, model_name: str = DEFAULT_MODEL, language: str = "auto") -> None:
        self.model_name = str(model_name).strip()
        self.language = language
        self._backend = self._detect_backend(self.model_name)
        self._warmed_up = False
        self._cached_language: str | None = None  # whisper auto-language cache
        self._parakeet_model = None
        self._parakeet_dtype_label = "unknown"

    @staticmethod
    def _detect_backend(model_name: str) -> str:
        normalized = str(model_name).strip().lower()
        return "parakeet" if "parakeet" in normalized else "whisper"

    def _effective_backend(self) -> str:
        detected = self._detect_backend(self.model_name)
        if detected != self._backend:
            log.warning(
                "Correcting STT backend from %s to %s for model %s",
                self._backend,
                detected,
                self.model_name,
            )
            self._backend = detected
        return self._backend

    @classmethod
    def _get_mlx_whisper(cls):
        if cls._mlx_whisper is None:
            import mlx_whisper  # type: ignore[import-untyped]

            cls._mlx_whisper = mlx_whisper
        return cls._mlx_whisper

    @classmethod
    def _load_parakeet_runtime(cls) -> None:
        if cls._parakeet_from_pretrained is not None:
            return

        from parakeet_mlx import (  # type: ignore[import-untyped]
            DecodingConfig,
            SentenceConfig,
            from_pretrained,
        )
        from parakeet_mlx.alignment import (  # type: ignore[import-untyped]
            merge_longest_common_subsequence,
            merge_longest_contiguous,
            sentences_to_result,
            tokens_to_sentences,
        )
        from parakeet_mlx.audio import get_logmel  # type: ignore[import-untyped]
        import mlx.core as mx  # type: ignore[import-untyped]

        # Store callables as static methods to avoid implicit instance binding
        # when accessed as ``self._parakeet_*``.
        cls._parakeet_from_pretrained = staticmethod(from_pretrained)
        cls._parakeet_get_logmel = staticmethod(get_logmel)
        cls._parakeet_tokens_to_sentences = staticmethod(tokens_to_sentences)
        cls._parakeet_sentences_to_result = staticmethod(sentences_to_result)
        cls._parakeet_merge_longest_contiguous = staticmethod(merge_longest_contiguous)
        cls._parakeet_merge_longest_common_subsequence = staticmethod(
            merge_longest_common_subsequence
        )
        cls._parakeet_decoding_config = DecodingConfig(sentence=SentenceConfig())
        cls._mx = mx

    def _ensure_parakeet_model(self):
        self._load_parakeet_runtime()
        if self._parakeet_model is not None:
            return self._parakeet_model

        # Some Apple Silicon variants are more stable with float16 than bfloat16.
        dtype_candidates: list[tuple[str, object | None]] = []
        if self._mx is not None and hasattr(self._mx, "bfloat16"):
            dtype_candidates.append(("bfloat16", self._mx.bfloat16))
        if self._mx is not None and hasattr(self._mx, "float16"):
            dtype_candidates.append(("float16", self._mx.float16))
        dtype_candidates.append(("default", None))

        errors: list[tuple[str, Exception]] = []
        for label, dtype in dtype_candidates:
            try:
                log.info("Loading Parakeet model %s (dtype=%s)", self.model_name, label)
                if dtype is None:
                    model = self._parakeet_from_pretrained(self.model_name)
                else:
                    try:
                        model = self._parakeet_from_pretrained(self.model_name, dtype=dtype)
                    except TypeError:
                        model = self._parakeet_from_pretrained(self.model_name)
                self._parakeet_model = model
                self._parakeet_dtype_label = label
                log.info("Parakeet model loaded (dtype=%s)", label)
                break
            except Exception as exc:
                errors.append((label, exc))
                log.warning(
                    "Parakeet load failed (dtype=%s): %s",
                    label,
                    exc,
                )

        if self._parakeet_model is None:
            summary = ", ".join(
                f"{label}: {type(exc).__name__}" for label, exc in errors
            )
            raise RuntimeError(
                f"Unable to load Parakeet model {self.model_name} ({summary})"
            ) from errors[-1][1]
        return self._parakeet_model

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size <= 0:
            return audio
        return np.ascontiguousarray(audio)

    @staticmethod
    def _pad_for_parakeet_min_duration(audio: np.ndarray) -> np.ndarray:
        min_samples = int(_PARAKEET_MIN_AUDIO_S * 16000)
        if audio.size >= min_samples:
            return audio
        pad_samples = min_samples - int(audio.size)
        return np.pad(audio, (0, pad_samples), mode="constant")

    def transcribe(
        self,
        audio: np.ndarray,
        tech_context: str = "",
        temperature: float | tuple[float, ...] | None = None,
    ) -> str:
        """Transcribe float32 audio array to text."""
        if self._effective_backend() == "parakeet":
            result = self._parakeet_transcribe_result(audio)
            return result.text.strip()
        return self._transcribe_whisper(
            audio,
            tech_context=tech_context,
            temperature=temperature,
        )

    def transcribe_with_segments(
        self,
        audio: np.ndarray,
        tech_context: str = "",
        temperature: float | tuple[float, ...] | None = None,
    ) -> dict:
        """Transcribe and return a dict with sentence-level timestamps."""
        if self._effective_backend() == "parakeet":
            result = self._parakeet_transcribe_result(audio)
            segments: list[dict] = []
            for sentence in result.sentences:
                text = sentence.text.strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": float(sentence.start),
                        "end": float(sentence.end),
                        "text": text,
                        "avg_logprob": float(sentence.confidence),
                    }
                )
            return {"text": result.text.strip(), "segments": segments}
        return self._transcribe_with_segments_whisper(
            audio,
            tech_context=tech_context,
            temperature=temperature,
        )

    def warm_up(self) -> None:
        """Run dummy inference to initialize model/runtime."""
        if self._warmed_up:
            return
        if self._effective_backend() == "parakeet":
            log.info("Warming up Parakeet model %s", self.model_name)
            dummy = np.random.randn(int(_PARAKEET_MIN_AUDIO_S * 16000)).astype(np.float32) * 0.01
            try:
                self.transcribe(dummy, temperature=0.0)
            except Exception as exc:
                raise RuntimeError(
                    f"Parakeet warm-up failed for model '{self.model_name}': {exc}"
                ) from exc
            self._warmed_up = True
            log.info("Parakeet warm-up complete")
            return

        log.info("Warming up Whisper model %s", self.model_name)
        dummy = np.random.randn(32000).astype(np.float32) * 0.01
        try:
            self.transcribe(dummy, temperature=0.0)
        except Exception as exc:
            raise RuntimeError(
                f"Whisper warm-up failed for model '{self.model_name}': {exc}"
            ) from exc
        self._warmed_up = True
        self._cached_language = None
        log.info("Whisper warm-up complete")

    def unload(self) -> None:
        """Release model references and cached compute state."""
        if self._effective_backend() == "parakeet":
            self._parakeet_model = None
            if self._mx is not None:
                try:
                    self._mx.clear_cache()
                except Exception:
                    log.debug("Unable to clear MLX cache", exc_info=True)

    def set_language(self, language: str) -> None:
        self.language = language
        self._cached_language = None

    def _transcribe_whisper(
        self,
        audio: np.ndarray,
        tech_context: str = "",
        temperature: float | tuple[float, ...] | None = None,
    ) -> str:
        mlx_whisper = self._get_mlx_whisper()
        audio = self._prepare_audio(audio)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            verbose=False,
            temperature=temperature if temperature is not None else _WHISPER_TEMPERATURE,
            compression_ratio_threshold=_WHISPER_COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=_WHISPER_LOGPROB_THRESHOLD,
            no_speech_threshold=_WHISPER_NO_SPEECH_THRESHOLD,
            language=self._resolve_whisper_language(),
            initial_prompt=self._build_whisper_prompt(tech_context),
            condition_on_previous_text=False,
        )
        self._cache_detected_language(result)
        return result["text"].strip()

    def _transcribe_with_segments_whisper(
        self,
        audio: np.ndarray,
        tech_context: str = "",
        temperature: float | tuple[float, ...] | None = None,
    ) -> dict:
        mlx_whisper = self._get_mlx_whisper()
        audio = self._prepare_audio(audio)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            verbose=False,
            temperature=temperature if temperature is not None else _WHISPER_TEMPERATURE,
            compression_ratio_threshold=_WHISPER_COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=_WHISPER_LOGPROB_THRESHOLD,
            no_speech_threshold=_WHISPER_NO_SPEECH_THRESHOLD,
            language=self._resolve_whisper_language(),
            initial_prompt=self._build_whisper_prompt(tech_context),
            condition_on_previous_text=False,
        )
        self._cache_detected_language(result)
        return result

    def _parakeet_transcribe_result(self, audio: np.ndarray):
        audio = self._prepare_audio(audio)
        audio = self._pad_for_parakeet_min_duration(audio)
        if audio.size <= 0:
            return self._parakeet_sentences_to_result([])

        duration_s = audio.size / 16000.0
        if duration_s >= _PARAKEET_LONG_AUDIO_THRESHOLD_S:
            return self._parakeet_transcribe_chunked(audio)
        return self._parakeet_transcribe_single(audio)

    def _parakeet_transcribe_single(self, audio: np.ndarray):
        model = self._ensure_parakeet_model()
        mel = self._parakeet_get_logmel(
            self._mx.array(audio),
            model.preprocessor_config,
        )
        return model.generate(mel, decoding_config=self._parakeet_decoding_config)[0]

    def _parakeet_transcribe_chunked(self, audio: np.ndarray):
        sample_rate = 16000
        chunk_samples = int(_PARAKEET_LONG_AUDIO_CHUNK_S * sample_rate)
        overlap_samples = int(_PARAKEET_LONG_AUDIO_OVERLAP_S * sample_rate)
        stride_samples = max(chunk_samples - overlap_samples, sample_rate)

        all_tokens = []
        for start in range(0, audio.size, stride_samples):
            end = min(start + chunk_samples, audio.size)
            if end <= start:
                break

            chunk_audio = self._pad_for_parakeet_min_duration(audio[start:end])
            chunk_result = self._parakeet_transcribe_single(chunk_audio)

            offset_s = start / sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += offset_s
                    token.end = token.start + token.duration

            chunk_tokens = chunk_result.tokens
            if not all_tokens:
                all_tokens = chunk_tokens
            else:
                try:
                    all_tokens = self._parakeet_merge_longest_contiguous(
                        all_tokens,
                        chunk_tokens,
                        overlap_duration=_PARAKEET_LONG_AUDIO_OVERLAP_S,
                    )
                except RuntimeError:
                    all_tokens = self._parakeet_merge_longest_common_subsequence(
                        all_tokens,
                        chunk_tokens,
                        overlap_duration=_PARAKEET_LONG_AUDIO_OVERLAP_S,
                    )

            if end >= audio.size:
                break

        if not all_tokens:
            return self._parakeet_sentences_to_result([])
        sentences = self._parakeet_tokens_to_sentences(
            all_tokens,
            self._parakeet_decoding_config.sentence,
        )
        return self._parakeet_sentences_to_result(sentences)

    def _build_whisper_prompt(self, tech_context: str) -> str:
        if self.language == "de":
            base = (
                "Besprechungsnotizen: Wir haben die API-Änderungen "
                "und den Deployment-Zeitplan besprochen."
            )
        elif self.language == "auto":
            base = (
                "Meeting notes: We discussed the API changes and deployment timeline. "
                "Die Tests laufen erfolgreich."
            )
        else:
            base = (
                "Meeting notes: We discussed the API changes "
                "and deployment timeline."
            )
        if tech_context:
            return f"{base} {tech_context}"
        return base

    def _resolve_whisper_language(self) -> str | None:
        if self.language == "auto":
            return self._cached_language
        return self.language

    def _cache_detected_language(self, result: dict) -> None:
        if self.language != "auto" or self._cached_language is not None:
            return
        detected = result.get("language")
        if detected:
            self._cached_language = detected
            log.info("Cached auto-detected language: %s", detected)
