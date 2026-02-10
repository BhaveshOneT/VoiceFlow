from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

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

    def test_vocab_hints_are_trimmed_to_relevant_entries(self) -> None:
        vocabulary = {
            "plate js": "Plate.js",
            "react dom": "ReactDOM",
            "api key": "API key",
            "unrelated term": "unrelated term",
        }
        hints = TextRefiner._select_vocab_hints(
            "please update plate js and api key handling",
            vocabulary,
            max_hints=3,
        )
        hinted_keys = {wrong for wrong, _ in hints}
        self.assertIn("plate js", hinted_keys)
        self.assertIn("api key", hinted_keys)
        self.assertNotIn("unrelated term", hinted_keys)


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

    def test_long_unpunctuated_text_skips_refiner_for_completeness(self) -> None:
        text = (
            "we should ship this after we validate analytics and update the release notes "
            "and run one final smoke test then follow up with monitoring so nothing "
            "regresses in production and support can track issues quickly"
        )
        self.assertFalse(self.pipeline._should_refine(text))

    def test_truncation_guard_rejects_shortened_refinement(self) -> None:
        source = (
            "okay we are setting up and i think it is good to go but we need to check "
            "if it actually worked or not then we will keep writing more sentences and "
            "more refactoring will follow also i noticed bugs that need to be fixed"
        )
        candidate = "we need to check if it actually worked or not and then also"
        self.assertTrue(
            self.pipeline._is_suspiciously_short_refinement(source, candidate)
        )

    def test_truncation_guard_accepts_similar_length_refinement(self) -> None:
        source = (
            "we need to validate the migration in staging and then write release notes "
            "for the team before we deploy to production"
        )
        candidate = (
            "We need to validate the migration in staging, then write release notes "
            "for the team before deploying to production."
        )
        self.assertFalse(
            self.pipeline._is_suspiciously_short_refinement(source, candidate)
        )

    def test_preserve_completeness_uses_conservative_fallback(self) -> None:
        raw = (
            "we are setting things up and it is good to go but we still need to check "
            "if it actually worked and keep writing more sentences while tracking bugs "
            "that still need fixes also"
        )
        cleaned = "we still need to check if it actually worked also"
        out = self.pipeline._preserve_completeness(
            raw,
            cleaned,
            {},
            programmer_mode=True,
        )
        self.assertGreater(len(out.split()), len(cleaned.split()))
        self.assertIn("setting things up", out.lower())

    def test_trim_silence_reduces_decode_audio_without_cutting_speech(self) -> None:
        speech = (0.02 * np.ones(16000, dtype=np.float32))
        trailing = np.zeros(32000, dtype=np.float32)
        audio = np.concatenate([speech, trailing])
        trimmed, changed = self.pipeline._trim_silence_for_decode(audio)
        self.assertTrue(changed)
        self.assertLess(trimmed.size, audio.size)
        self.assertGreaterEqual(trimmed.size, speech.size)

    def test_trim_silence_keeps_all_silence_audio_unchanged(self) -> None:
        audio = np.zeros(24000, dtype=np.float32)
        trimmed, changed = self.pipeline._trim_silence_for_decode(audio)
        self.assertFalse(changed)
        self.assertEqual(trimmed.size, audio.size)

    def test_long_audio_is_split_into_overlapping_chunks(self) -> None:
        audio = np.zeros(16000 * 190, dtype=np.float32)  # 3m10s
        chunks = self.pipeline._split_for_long_transcription(audio)
        self.assertGreater(len(chunks), 1)
        # First chunk may shift by up to 8000 samples due to silence-aware splitting.
        expected = int(42.0 * 16000)
        self.assertAlmostEqual(chunks[0].size, expected, delta=8000)
        self.assertGreaterEqual(chunks[-1].size, int(12.0 * 16000))

    def test_merge_transcript_parts_removes_overlap(self) -> None:
        merged = self.pipeline._merge_transcript_parts(
            [
                "we should update the parser module and run tests before merge",
                "and run tests before merge then deploy to staging",
            ]
        )
        self.assertIn("deploy to staging", merged.lower())
        self.assertEqual(
            merged.lower().count("and run tests before merge"),
            1,
        )

    def test_tail_coverage_detection(self) -> None:
        full = (
            "we shipped to staging and validated smoke tests then fixed two bugs "
            "before final rollout this morning"
        )
        tail = "fixed two bugs before final rollout this morning"
        self.assertTrue(self.pipeline._is_tail_covered(full, tail))

    def test_transcription_mode_switches_to_normal(self) -> None:
        self.pipeline.set_transcription_mode("normal")
        self.assertFalse(self.pipeline._programmer_mode_enabled())

    def test_process_normal_mode_skips_file_tagging(self) -> None:
        config = AppConfig(cleanup_mode="fast", transcription_mode="normal")
        pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(
            pipeline,
            "_trim_silence_for_decode",
            return_value=(audio, False),
        ), mock.patch.object(
            pipeline, "_has_speech", return_value=True
        ), mock.patch.object(
            pipeline,
            "_transcribe_adaptive",
            return_value="please update function.py file",
        ):
            result = pipeline.process(audio)
        self.assertIn("function.py", result.lower())
        self.assertNotIn("@function.py", result.lower())

    def test_process_programmer_mode_tags_file(self) -> None:
        config = AppConfig(cleanup_mode="fast", transcription_mode="programmer")
        pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(
            pipeline,
            "_trim_silence_for_decode",
            return_value=(audio, False),
        ), mock.patch.object(
            pipeline, "_has_speech", return_value=True
        ), mock.patch.object(
            pipeline,
            "_transcribe_adaptive",
            return_value="please update function.py file",
        ):
            result = pipeline.process(audio)
        self.assertIn("@function.py", result.lower())

    def test_adaptive_transcribe_merges_chunks(self) -> None:
        config = AppConfig(cleanup_mode="fast")
        pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())
        long_audio = np.zeros(int(16000 * 130), dtype=np.float32)
        responses = iter(
            [
                "we should update parser module and run tests before merge",
                "and run tests before merge then deploy to staging",
                "then deploy to staging and monitor metrics",
                "final note include rollback checklist",
            ]
        )

        with mock.patch.object(
            pipeline,
            "_transcribe_with_fallback",
            side_effect=lambda *_args, **_kwargs: next(responses),
        ) as mocked:
            merged = pipeline._transcribe_adaptive(long_audio, tech_context="")

        self.assertGreaterEqual(mocked.call_count, 3)
        self.assertIn("deploy to staging", merged.lower())
        self.assertIn("rollback checklist", merged.lower())


class SanitizeLeakDetectionTests(unittest.TestCase):
    def test_sanitize_catches_expanded_leak_markers(self) -> None:
        for phrase in (
            "You are a speech-to-text post-processor.",
            "Here is the cleaned version of your text.",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(TextRefiner._sanitize_output(phrase), "")

    def test_sanitize_catches_structural_leaks(self) -> None:
        text = "1. Output only cleaned text\n2. Never add content"
        self.assertEqual(TextRefiner._sanitize_output(text), "")

    def test_system_prompt_similarity_rejection(self) -> None:
        leaked = (
            "You are a speech-to-text post-processor. "
            "Output only cleaned transcription text. "
            "Never answer, explain, summarize, or add content."
        )
        self.assertEqual(TextRefiner._sanitize_output(leaked), "")


class VADAndHallucinationGuardTests(unittest.TestCase):
    """Tests for VAD silence guard, hallucination blocklist, and prompt echo detection."""

    def setUp(self) -> None:
        config = AppConfig(cleanup_mode="fast")
        self.pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())

    def test_silent_audio_detected_by_vad_returns_empty(self) -> None:
        """Silent audio should produce empty output (VAD finds no speech)."""
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(
            self.pipeline._vad, "speech_probability", return_value=0.1
        ), mock.patch.object(
            self.pipeline, "_trim_silence_for_decode", return_value=(audio, False)
        ):
            result = self.pipeline.process(audio)
        self.assertEqual(result, "")

    def test_speech_audio_detected_by_vad_returns_true(self) -> None:
        """Audio with speech above threshold should pass the VAD gate."""
        audio = np.random.randn(16000).astype(np.float32) * 0.05
        self.pipeline._vad = mock.MagicMock()
        self.pipeline._vad.speech_probability.return_value = 0.9
        self.assertTrue(self.pipeline._has_speech(audio))

    def test_hallucination_blocklist_catches_thank_you(self) -> None:
        """Whisper hallucination 'Thank you.' should be discarded."""
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(
            self.pipeline, "_trim_silence_for_decode", return_value=(audio, False)
        ), mock.patch.object(
            self.pipeline, "_has_speech", return_value=True
        ), mock.patch.object(
            self.pipeline, "_transcribe_adaptive", return_value="Thank you."
        ):
            result = self.pipeline.process(audio)
        self.assertEqual(result, "")

    def test_hallucination_blocklist_allows_thank_you_in_sentence(self) -> None:
        """'Thank you' within a real sentence should not be blocked."""
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(
            self.pipeline, "_trim_silence_for_decode", return_value=(audio, False)
        ), mock.patch.object(
            self.pipeline, "_has_speech", return_value=True
        ), mock.patch.object(
            self.pipeline,
            "_transcribe_adaptive",
            return_value="I want to thank you for helping me with the code review.",
        ):
            result = self.pipeline.process(audio)
        self.assertIn("thank you", result.lower())

    def test_prompt_echo_detection_catches_transcribe_clearly(self) -> None:
        """Short output echoing the system prompt should be discarded."""
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(
            self.pipeline, "_trim_silence_for_decode", return_value=(audio, False)
        ), mock.patch.object(
            self.pipeline, "_has_speech", return_value=True
        ), mock.patch.object(
            self.pipeline,
            "_transcribe_adaptive",
            return_value="Transcribe clearly with natural punctuation.",
        ):
            result = self.pipeline.process(audio)
        self.assertEqual(result, "")

    def test_prompt_echo_allows_normal_text(self) -> None:
        """Normal transcription text should pass prompt echo detection."""
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(
            self.pipeline, "_trim_silence_for_decode", return_value=(audio, False)
        ), mock.patch.object(
            self.pipeline, "_has_speech", return_value=True
        ), mock.patch.object(
            self.pipeline,
            "_transcribe_adaptive",
            return_value="We need to update the deployment scripts for staging.",
        ):
            result = self.pipeline.process(audio)
        self.assertIn("update the deployment scripts", result.lower())


class TextCleanerBehaviorTests(unittest.TestCase):
    def test_no_no_correction_preserves_sentence_context(self) -> None:
        text = (
            "okay we have a problem in the app i want to modify functions file "
            "no no modify text refiner file"
        )
        cleaned = TextCleaner.clean(text)
        self.assertIn("we have a problem in the app", cleaned.lower())
        self.assertIn("i want to modify", cleaned.lower())
        self.assertIn("@text_refiner", cleaned.lower())
        self.assertNotIn("@functions", cleaned.lower())

    def test_tags_explicit_and_spoken_file_names(self) -> None:
        explicit = TextCleaner.clean("please update function.py file")
        spoken = TextCleaner.clean("please update function dot py file")
        self.assertIn("@function.py", explicit)
        self.assertIn("@function.py", spoken)

    def test_tags_spoken_dmg_filename(self) -> None:
        cleaned = TextCleaner.clean("please update voiceflow dot dmg file")
        self.assertIn("@voiceflow.dmg", cleaned.lower())

    def test_does_not_tag_bare_extension_as_file(self) -> None:
        cleaned = TextCleaner.clean("please update dmg file")
        self.assertNotIn("@dmg", cleaned.lower())
        self.assertIn("dmg file", cleaned.lower())

    def test_sanitizes_lone_extension_tag(self) -> None:
        cleaned = TextCleaner.clean("please update the voiceflow @dmg release")
        self.assertNotIn("@dmg", cleaned.lower())
        self.assertIn("dmg", cleaned.lower())

    def test_spoken_complex_filenames_are_tagged(self) -> None:
        cleaned = TextCleaner.clean(
            "update text underscore refiner dot py and docker dash compose dot yml"
        ).lower()
        self.assertIn("@text_refiner.py", cleaned)
        self.assertIn("@docker-compose.yml", cleaned)

    def test_merges_fragmented_filename_tags(self) -> None:
        cleaned = TextCleaner.clean(
            "update text underscore @refiner.py and @docker-@compose.yml"
        ).lower()
        self.assertIn("@text_refiner.py", cleaned)
        self.assertIn("@docker-compose.yml", cleaned)

    def test_merges_prefixed_tagged_filename_after_rename_verb(self) -> None:
        cleaned = TextCleaner.clean(
            "then rename release @notes.md to @release-notes.md"
        ).lower()
        self.assertNotIn("release @notes.md", cleaned)
        self.assertIn("rename @release-notes.md to @release-notes.md", cleaned)

    def test_merges_prefixed_filename_with_the_file_phrase(self) -> None:
        cleaned = TextCleaner.clean("update the file release notes.md").lower()
        self.assertIn("update the file @release-notes.md", cleaned)
        self.assertNotIn("release @notes.md", cleaned)

    def test_does_not_tag_framework_terms_as_files(self) -> None:
        cleaned = TextCleaner.clean("technical terms like next.js and plate.js")
        self.assertNotIn("@next.js", cleaned.lower())
        self.assertNotIn("@plate.js", cleaned.lower())

    def test_untags_framework_list_with_existing_prefixes(self) -> None:
        cleaned = TextCleaner.clean("technical terms like @next.js, @play.js and @plate.js")
        self.assertIn("technical terms like next.js, play.js and plate.js", cleaned.lower())
        self.assertNotIn("@next.js", cleaned.lower())
        self.assertNotIn("@play.js", cleaned.lower())
        self.assertNotIn("@plate.js", cleaned.lower())

    def test_normal_mode_skips_file_tagging(self) -> None:
        cleaned = TextCleaner.clean(
            "please update function.py file",
            programmer_mode=False,
        )
        self.assertNotIn("@function.py", cleaned.lower())
        self.assertIn("function.py", cleaned.lower())

    def test_programmer_mode_tags_symbol_mentions(self) -> None:
        cleaned = TextCleaner.clean("please refactor function parse_request")
        self.assertIn("@parse_request", cleaned)

    def test_normal_mode_skips_symbol_tagging(self) -> None:
        cleaned = TextCleaner.clean(
            "please refactor function parse_request",
            programmer_mode=False,
        )
        self.assertNotIn("@parse_request", cleaned)

    def test_bare_generic_file_reference_is_not_tagged(self) -> None:
        cleaned = TextCleaner.clean("please open the file")
        self.assertEqual(cleaned.lower(), "please open the file")

    def test_collapses_repeated_clauses(self) -> None:
        text = "we should ship today. we should ship today. we should ship today."
        cleaned = TextCleaner.clean(text)
        self.assertEqual(cleaned.lower(), "we should ship today.")

    def test_dedupes_adjacent_long_sentences(self) -> None:
        text = (
            "The code is a little bit different from the code that we have used in the previous version. "
            "The code is a little bit different from the code that we have used in the previous version."
        )
        cleaned = TextCleaner.clean(text)
        self.assertEqual(
            cleaned.lower().count(
                "the code is a little bit different from the code that we have used in the previous version"
            ),
            1,
        )

    def test_dedupes_repeated_tail_sentence_after_longer_clause(self) -> None:
        text = (
            "The bug appears during long dictation The code is used in the following way. "
            "The code is used in the following way."
        )
        cleaned = TextCleaner.clean(text).lower()
        self.assertEqual(cleaned.count("the code is used in the following way"), 1)
        self.assertIn("long dictation. the code is used", cleaned)

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

    def test_clean_conservative_keeps_context_without_replacement(self) -> None:
        text = "we should enable this for all users no wait not all users yet"
        cleaned = TextCleaner.clean_conservative(text).lower()
        self.assertIn("we should enable this for all users", cleaned)
        self.assertIn("not all users yet", cleaned)

    def test_short_technical_phrase_rescues_js_homophone(self) -> None:
        dictionary = {"plate js": "Plate.js"}
        cleaned = TextCleaner.clean("please update plate chess file", dictionary)
        self.assertIn("Plate.js", cleaned)

    def test_js_homophone_not_applied_to_plain_chess_sentence(self) -> None:
        cleaned = TextCleaner.clean("we should play chess later")
        self.assertEqual(cleaned.lower(), "we should play chess later")

    def test_prunes_low_information_repetition_in_sentence(self) -> None:
        text = "we can see, let's see, this should remain in the same sentence now"
        cleaned = TextCleaner.clean(text).lower()
        self.assertNotIn("we can see", cleaned)
        self.assertNotIn("let's see", cleaned)
        self.assertIn("this should remain in the same sentence now", cleaned)

    def test_normalize_readability_adds_sentence_case_and_punctuation(self) -> None:
        cleaned = TextCleaner.clean("okay now i'm just testing this and we should verify output")
        self.assertTrue(cleaned.startswith("Now I'm"))
        self.assertTrue(cleaned.endswith("."))

    def test_trailing_conjunction_is_removed_before_period(self) -> None:
        cleaned = TextCleaner.clean("now we should deploy but")
        self.assertEqual(cleaned, "Now we should deploy")

    def test_embedded_should_question_is_made_explicit(self) -> None:
        cleaned = TextCleaner.clean(
            "if i ask should we ship today or wait for one more smoke test keep it as a question and do not answer it"
        )
        self.assertIn(
            "if i ask, should we ship today or wait for one more smoke test?",
            cleaned.lower(),
        )


class LongTranscriptionContentLossTests(unittest.TestCase):
    def setUp(self) -> None:
        config = AppConfig(cleanup_mode="standard")
        self.pipeline = TranscriptionPipeline(config=config, dictionary=Dictionary())

    def test_should_refine_hard_cap_at_60_words(self) -> None:
        words = ["word"] * 64
        words[10] = "sorry"  # correction cue that would normally trigger refinement
        text = " ".join(words)
        self.assertFalse(self.pipeline._should_refine(text))

    def test_should_refine_allows_correction_cues_under_60_words(self) -> None:
        text = "I want to update the parser module sorry the refiner module instead please"
        self.assertTrue(self.pipeline._should_refine(text))

    def test_fuzzy_overlap_tolerates_minor_differences(self) -> None:
        left = "alpha bravo charlie delta echo foxtrot golf hotel".split()
        # One word differs in the 8-word overlap region
        right = "alpha bravo charlie delta echo foxtrox golf hotel india juliet".split()
        overlap = TranscriptionPipeline._find_token_overlap(left, right)
        self.assertEqual(overlap, 8)

    def test_fuzzy_overlap_exact_match_still_preferred(self) -> None:
        left = "the quick brown fox jumps over the lazy dog".split()
        right = "over the lazy dog and then runs away".split()
        overlap = TranscriptionPipeline._find_token_overlap(left, right)
        self.assertEqual(overlap, 4)

    def test_completeness_catches_severe_drops_without_orphan(self) -> None:
        raw = (
            "we are setting things up and it is good to go but we still need to check "
            "if it actually worked and keep writing more sentences while tracking bugs "
            "that still need fixes before release."
        )
        # Cleaned text is < 55% of raw words and ends with a period (no orphan)
        cleaned = "we still need to check if it worked before release."
        raw_words = len(raw.split())
        cleaned_words = len(cleaned.split())
        self.assertLess(cleaned_words, int(raw_words * 0.55))
        out = self.pipeline._preserve_completeness(
            raw, cleaned, {}, programmer_mode=True
        )
        self.assertGreater(len(out.split()), cleaned_words)

    def test_split_prefers_silence_boundaries(self) -> None:
        sample_rate = 16000
        duration_s = 130.0
        total_samples = int(duration_s * sample_rate)
        audio = np.full(total_samples, 0.05, dtype=np.float32)
        # Insert a quiet region near where the first split would land (~42s mark)
        quiet_center = int(42.0 * sample_rate)
        quiet_start = max(quiet_center - 3200, 0)
        quiet_end = min(quiet_center + 3200, total_samples)
        audio[quiet_start:quiet_end] = 0.0001

        chunks = TranscriptionPipeline._split_for_long_transcription(audio)
        self.assertGreater(len(chunks), 1)
        first_chunk_end = chunks[0].size
        # The split point should land within the quiet region (+/- window)
        self.assertGreaterEqual(first_chunk_end, quiet_start - 8000)
        self.assertLessEqual(first_chunk_end, quiet_end + 8000)


if __name__ == "__main__":
    unittest.main()
