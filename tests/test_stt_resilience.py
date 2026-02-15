from __future__ import annotations

import unittest
from unittest import mock

from app.config import AppConfig
from app.dictionary import Dictionary
from app.transcription import TranscriptionPipeline


class _FakeEngine:
    def __init__(self, model_name: str, language: str = "auto") -> None:
        self.model_name = model_name
        self.language = language

    def warm_up(self) -> None:
        if "fail" in self.model_name:
            raise RuntimeError(f"simulated warm-up failure: {self.model_name}")

    def transcribe(self, *_args, **_kwargs) -> str:
        if "fail" in self.model_name:
            raise RuntimeError(f"simulated transcription failure: {self.model_name}")
        return "ok"


class SttResilienceTests(unittest.TestCase):
    def test_warmup_prefers_cached_primary_when_max_accuracy_uncached(self) -> None:
        config = AppConfig(
            cleanup_mode="max_accuracy",
            whisper_model="mlx-community/parakeet-tdt-0.6b-v3",
            max_accuracy_whisper_model="mlx-community/parakeet-tdt-0.6b-v2",
        )
        pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())

        with mock.patch.object(
            TranscriptionPipeline,
            "_is_stt_model_cached",
            side_effect=lambda model: model == config.stt_model,
        ):
            models = pipeline._stt_fallback_models(for_warm_up=True)

        self.assertEqual(models[0], config.stt_model)
        self.assertEqual(models[1], config.max_accuracy_stt_model)
        self.assertIn("mlx-community/whisper-large-v3-turbo", models)

    def test_warmup_switches_to_first_working_fallback(self) -> None:
        config = AppConfig(cleanup_mode="standard")
        pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())
        pipeline.stt = _FakeEngine("fail-primary", language=config.language)

        with mock.patch.object(
            pipeline,
            "_stt_fallback_models",
            return_value=["fail-primary", "ok-secondary"],
        ), mock.patch("app.transcription.WhisperEngine", _FakeEngine):
            pipeline._warm_up_stt_with_fallback()

        self.assertEqual(pipeline.stt.model_name, "ok-secondary")


if __name__ == "__main__":
    unittest.main()
