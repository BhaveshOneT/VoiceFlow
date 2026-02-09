from __future__ import annotations

import types
import unittest

from app.main import VoiceFlowApp


class ModeInferenceTests(unittest.TestCase):
    def test_programmer_mode_for_terminal_name(self) -> None:
        mode = VoiceFlowApp._infer_transcription_mode_for_app(
            app_name="iTerm2",
            bundle_id="com.googlecode.iterm2",
            programmer_hints=["terminal", "iterm", "codex"],
        )
        self.assertEqual(mode, "programmer")

    def test_programmer_mode_for_bundle_hint(self) -> None:
        mode = VoiceFlowApp._infer_transcription_mode_for_app(
            app_name="Some Wrapper",
            bundle_id="com.jetbrains.pycharm",
            programmer_hints=["codex"],
        )
        self.assertEqual(mode, "programmer")

    def test_normal_mode_for_non_coding_app(self) -> None:
        mode = VoiceFlowApp._infer_transcription_mode_for_app(
            app_name="Notes",
            bundle_id="com.apple.Notes",
            programmer_hints=["terminal", "iterm", "codex"],
        )
        self.assertEqual(mode, "normal")

    def test_auto_mode_switch_updates_pipeline_and_config(self) -> None:
        app = object.__new__(VoiceFlowApp)
        app.config = types.SimpleNamespace(
            auto_mode_switch=True,
            transcription_mode="normal",
            programmer_apps=["terminal", "codex", "claude"],
            save=lambda: None,
        )
        seen_modes: list[str] = []
        app.pipeline = types.SimpleNamespace(
            set_transcription_mode=lambda mode: seen_modes.append(mode)
        )
        app._sync_transcription_mode_checkmarks = lambda: None
        app._frontmost_app_info = lambda: ("Terminal", "com.apple.Terminal")

        VoiceFlowApp._apply_auto_transcription_mode(app)

        self.assertEqual(seen_modes, ["programmer"])
        self.assertEqual(app.config.transcription_mode, "programmer")

    def test_auto_mode_switch_noop_when_disabled(self) -> None:
        app = object.__new__(VoiceFlowApp)
        app.config = types.SimpleNamespace(
            auto_mode_switch=False,
            transcription_mode="normal",
            programmer_apps=["terminal"],
            save=lambda: None,
        )
        seen_modes: list[str] = []
        app.pipeline = types.SimpleNamespace(
            set_transcription_mode=lambda mode: seen_modes.append(mode)
        )
        app._sync_transcription_mode_checkmarks = lambda: None
        app._frontmost_app_info = lambda: ("Terminal", "com.apple.Terminal")

        VoiceFlowApp._apply_auto_transcription_mode(app)

        self.assertEqual(seen_modes, [])
        self.assertEqual(app.config.transcription_mode, "normal")


if __name__ == "__main__":
    unittest.main()
