from __future__ import annotations

import unittest

from app.config import AppConfig
from app.dictionary import Dictionary
from app.transcription import TranscriptionPipeline
from app.transcription.text_cleaner import TextCleaner
from app.transcription.text_refiner import TextRefiner


class TextRefinerGuardTests(unittest.TestCase):
    def test_rejects_answer_for_question_input(self) -> None:
        source = "What is polymorphism in OOP?"
        candidate = "Polymorphism in OOP allows objects to take multiple forms."
        self.assertTrue(TextRefiner._is_answer_like(source=source, candidate=candidate))

    def test_accepts_cleaned_question_output(self) -> None:
        source = "what is polymorphism in oop"
        candidate = "What is polymorphism in OOP?"
        self.assertFalse(TextRefiner._is_answer_like(source=source, candidate=candidate))

    def test_allows_simple_non_question_cleanup(self) -> None:
        source = "I want to change the modularity of the app"
        candidate = "I want to change the modularity of the app."
        self.assertFalse(TextRefiner._is_answer_like(source=source, candidate=candidate))

    def test_rejects_assistant_style_openers(self) -> None:
        source = "Please refactor the parser module"
        candidate = "Sure, you can refactor the parser module by splitting functions."
        self.assertTrue(TextRefiner._is_answer_like(source=source, candidate=candidate))

    def test_sanitize_skips_meta_and_keeps_actual_line(self) -> None:
        output = (
            "This version is concise and directly addresses the question.\n"
            "Please update the parser module."
        )
        self.assertEqual(
            TextRefiner._sanitize_output(output),
            "Please update the parser module.",
        )


class PipelineRefinementGateTests(unittest.TestCase):
    def setUp(self) -> None:
        config = AppConfig(cleanup_mode="standard")
        self.pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())

    def test_question_like_text_skips_refiner(self) -> None:
        self.assertFalse(self.pipeline._should_refine("How do I reset my API key"))
        self.assertFalse(self.pipeline._should_refine("How do I reset my API key?"))
        self.assertFalse(self.pipeline._should_refine("Wie kann ich meinen API-Schluessel zuruecksetzen?"))

    def test_backtrack_text_still_uses_refiner(self) -> None:
        self.assertTrue(self.pipeline._should_refine("Change it to red, sorry blue please"))

    def test_filler_heavy_raw_text_uses_refiner(self) -> None:
        cleaned = "I think we should update parser module."
        raw = "um i think we should basically update parser module"
        self.assertTrue(self.pipeline._should_refine(cleaned, raw_text=raw))

    def test_long_punctuated_text_skips_refiner_for_speed(self) -> None:
        text = (
            "We should ship this after we validate analytics, update the release notes, "
            "and run one final smoke test so nothing regresses in production."
        )
        self.assertFalse(self.pipeline._should_refine(text))


class TextCleanerBehaviorTests(unittest.TestCase):
    def test_no_no_correction_preserves_sentence_context(self) -> None:
        text = (
            "okay we have a problem in the app i want to modify functions file "
            "no no modify text refiner file"
        )
        cleaned = TextCleaner.clean(text)
        self.assertIn("we have a problem in the app", cleaned.lower())
        self.assertIn("i want to modify", cleaned.lower())
        self.assertIn("@ text_refiner", cleaned.lower())
        self.assertNotIn("@ functions", cleaned.lower())

    def test_tags_explicit_and_spoken_file_names(self) -> None:
        explicit = TextCleaner.clean("please update function.py file")
        spoken = TextCleaner.clean("please update function dot py file")
        self.assertIn("@ function.py", explicit)
        self.assertIn("@ function.py", spoken)

    def test_bare_generic_file_reference_is_not_tagged(self) -> None:
        cleaned = TextCleaner.clean("please open the file")
        self.assertEqual(cleaned.lower(), "please open the file")

    def test_collapses_repeated_clauses(self) -> None:
        text = "we should ship today. we should ship today. we should ship today."
        cleaned = TextCleaner.clean(text)
        self.assertEqual(cleaned.lower(), "we should ship today.")

    def test_single_no_statement_does_not_replace_previous_sentence(self) -> None:
        text = "we should enable this for all users. no we don't want that yet."
        cleaned = TextCleaner.clean(text).lower()
        self.assertIn("we should enable this for all users", cleaned)
        self.assertIn("we don't want that yet", cleaned)

    def test_weak_sorry_cue_preserves_context_when_not_edit_command(self) -> None:
        text = "the app is stable in staging. sorry we still need to test payments."
        cleaned = TextCleaner.clean(text).lower()
        self.assertIn("the app is stable in staging", cleaned)
        self.assertIn("we still need to test payments", cleaned)


if __name__ == "__main__":
    unittest.main()
