from __future__ import annotations

import unittest

from app.config import DEFAULT_LANGUAGE, DEFAULT_LLM_MODEL, AppConfig


class ConfigLlmModelTests(unittest.TestCase):
    def test_default_llm_model_is_qwen_3b(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.llm_model, DEFAULT_LLM_MODEL)

    def test_alias_4b_model_is_migrated_to_supported_default(self) -> None:
        cfg = AppConfig(llm_model="mlx-community/Mistral-NeMo-Minitron-4B-Instruct")
        self.assertEqual(cfg.llm_model, DEFAULT_LLM_MODEL)

    def test_default_language_is_auto(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.language, DEFAULT_LANGUAGE)

    def test_language_aliases_normalize_to_auto_for_english_german(self) -> None:
        cfg = AppConfig(language="english_german")
        self.assertEqual(cfg.language, "auto")

    def test_default_transcription_mode_is_programmer(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.transcription_mode, "programmer")

    def test_transcription_mode_alias_general_maps_to_normal(self) -> None:
        cfg = AppConfig(transcription_mode="general")
        self.assertEqual(cfg.transcription_mode, "normal")


if __name__ == "__main__":
    unittest.main()
