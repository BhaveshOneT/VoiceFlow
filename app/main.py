"""VoiceFlow -- Local AI dictation for macOS."""
from __future__ import annotations

import logging
import threading
import subprocess
from pathlib import Path

import numpy as np
import rumps
from PyObjCTools import AppHelper

from app.config import AppConfig
from app.dictionary import Dictionary
from app.audio.capture import AudioCapture
from app.transcription import TranscriptionPipeline
from app.input.hotkey import HotkeyListener
from app.input.text_inserter import TextInserter
from app.ui.overlay import RecordingOverlay

LOG_DIR = Path.home() / "Library" / "Application Support" / "VoiceFlow" / "logs"
LOG_PATH = LOG_DIR / "voiceflow.log"


def _configure_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
    except Exception:
        # Keep stdout logging even if file logging is unavailable.
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


_configure_logging()
log = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "resources"

# Minimum audio length (in samples at 16 kHz) to bother transcribing.
# 0.3 seconds = 4800 samples -- anything shorter is almost certainly noise.
_MIN_AUDIO_SAMPLES = 4800


def _check_accessibility() -> bool:
    """Return True if the app has Accessibility permission.

    Also triggers the macOS trust prompt when available.
    """
    try:
        from ApplicationServices import (  # type: ignore[import-untyped]
            AXIsProcessTrusted,
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
        trusted = bool(AXIsProcessTrusted())
        if not trusted:
            try:
                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
                trusted = bool(AXIsProcessTrusted())
            except Exception:
                pass
        log.info("Accessibility check: trusted=%s", trusted)
        return trusted
    except ImportError:
        log.warning("ApplicationServices not available; skipping accessibility check")
        return True


class VoiceFlowApp(rumps.App):
    """Menu-bar app that wires audio capture, transcription, and text insertion."""

    def __init__(self) -> None:
        # -- Config & dictionary -----------------------------------------------
        self.config = AppConfig.load()
        self.dictionary = Dictionary.load(Path(self.config.dictionary_path))

        # -- Menu items (created before super().__init__) ----------------------
        self._status_item = rumps.MenuItem("Status: Loading...")
        self._test_item = rumps.MenuItem(
            "Test Recording", callback=self._test_recording
        )
        self._accessibility_item = rumps.MenuItem(
            "Open Accessibility Settings", callback=self._open_accessibility_settings
        )
        self._microphone_item = rumps.MenuItem(
            "Open Microphone Settings", callback=self._open_microphone_settings
        )

        self._fast_item = rumps.MenuItem("Fast Mode", callback=self._set_fast_mode)
        self._standard_item = rumps.MenuItem("Standard Mode", callback=self._set_standard_mode)
        self._max_item = rumps.MenuItem("Max Accuracy Mode", callback=self._set_max_accuracy_mode)

        accuracy_menu = rumps.MenuItem("Accuracy Mode")
        accuracy_menu.update([self._fast_item, self._standard_item, self._max_item])

        super().__init__(
            name="VoiceFlow",
            title="VF ...",
            menu=[
                self._status_item,
                self._test_item,
                None,  # separator
                self._accessibility_item,
                self._microphone_item,
                None,  # separator
                accuracy_menu,
                None,  # separator
            ],
            quit_button="Quit VoiceFlow",
        )

        # Mark the current accuracy mode
        self._sync_mode_checkmarks()

        # -- Audio & pipeline --------------------------------------------------
        self.audio = AudioCapture()
        self.pipeline = TranscriptionPipeline(self.config, self.dictionary)

        # -- Hotkey listener (not started until models are warm) ---------------
        # The new HotkeyListener takes a string key name (e.g. "right_cmd")
        # and uses native macOS NSEvent monitors instead of pynput.
        self.hotkey = HotkeyListener(
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            key=self.config.hotkey,
            mode=self.config.recording_mode,
        )

        # -- Recording overlay ---------------------------------------------------
        self.overlay = RecordingOverlay()

        # -- State flags (guarded by _lock) ------------------------------------
        self._lock = threading.Lock()
        self._processing = False

        # -- Warm up models in background --------------------------------------
        threading.Thread(target=self._warm_up_models, daemon=True).start()

    # ======================================================================
    # Recording lifecycle
    # ======================================================================

    def _on_recording_start(self) -> None:
        """Called when the hotkey is pressed (hold starts)."""
        with self._lock:
            if self._processing:
                log.debug("Still processing previous audio; ignoring new recording")
                return
        log.info("Recording started")
        try:
            self.audio.drain()
            self.audio.start()
        except Exception:
            log.exception("Failed to start microphone capture")
            self._show_error(
                title="Microphone error",
                message="Unable to start audio capture. Check microphone permission.",
            )
            return
        self._set_title("VF \u25cf")  # bullet
        self._set_status("Recording")
        self.overlay.show_recording()

    def _on_recording_stop(self, cancelled: bool = False) -> None:
        """Called when the hotkey is released.

        *cancelled=True* means the hold was shorter than the minimum hold
        duration -- discard the audio.
        """
        if not self.audio.is_active():
            return  # Already stopped (duplicate key event from macOS)

        audio = self.audio.stop()

        if cancelled:
            log.info("Recording cancelled (hold too short)")
            self._set_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            return

        if audio.size < _MIN_AUDIO_SAMPLES:
            log.info("Audio too short (%d samples); discarding", audio.size)
            self._set_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            return

        log.info("Recording stopped; captured %d samples (%.1fs)",
                 audio.size, audio.size / AudioCapture.SAMPLE_RATE)
        self._set_title("VF ...")
        self._set_status("Processing")
        self.overlay.show_processing()
        with self._lock:
            self._processing = True

        # Process in a background thread so the hotkey listener stays responsive.
        threading.Thread(
            target=self._process_audio, args=(audio,), daemon=True
        ).start()

    # ======================================================================
    # Audio processing (runs in background thread)
    # ======================================================================

    def _process_audio(self, audio: np.ndarray) -> None:
        """Run the transcription pipeline and insert the result."""
        try:
            result = self.pipeline.process(audio)
        except Exception as exc:
            log.exception("Transcription failed")
            detail = str(exc).strip() or exc.__class__.__name__
            self._show_error(
                title="Transcription failed",
                message=f"{detail}. Check VoiceFlow logs for details.",
            )
            return
        finally:
            with self._lock:
                self._processing = False

        if not result:
            log.info("Pipeline returned empty result (no speech detected)")
            self._set_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            return

        log.info("Transcription result: %s", result)
        inserted = TextInserter.insert(result, self.config.restore_clipboard)
        if not inserted:
            detail = TextInserter.last_error or "Paste failed"
            if "Accessibility permission required" in detail:
                self.overlay.hide()
                self._set_title("VF")
                self._set_status("Paste permission required")
                self._notify(
                    title="Paste Permission Required",
                    subtitle="VoiceFlow copied text to clipboard",
                    message=(
                        "Enable Accessibility for VoiceFlow to auto-paste. "
                        "You can paste now with Command+V."
                    ),
                )
                self._open_system_settings("Privacy_Accessibility")
                return
            self._show_error(title="Paste failed", message=detail)
            return

        self._set_title("VF")
        self._set_status("Ready")
        self.overlay.hide()

    def _reset_title(self) -> None:
        self._set_title("VF")

    # ======================================================================
    # Test Recording (manual trigger from menu, bypasses hotkey)
    # ======================================================================

    def _test_recording(self, sender: rumps.MenuItem) -> None:
        """Simulate a 3-second recording to test overlay + pipeline."""
        log.info("Test recording triggered from menu")

        with self._lock:
            if self._processing:
                log.info("Already processing; ignoring test")
                return

        # Show overlay immediately
        self.overlay.show_recording()
        self._set_title("VF \u25cf")
        self._set_status("Recording")
        try:
            self.audio.drain()
            self.audio.start()
        except Exception:
            log.exception("Failed to start microphone capture (test)")
            self._show_error(
                title="Microphone error",
                message="Unable to start audio capture. Check microphone permission.",
            )
            return

        def _stop_after_delay():
            import time
            time.sleep(3)
            log.info("Test recording: stopping after 3s")
            AppHelper.callAfter(self._on_recording_stop, False)

        threading.Thread(target=_stop_after_delay, daemon=True).start()

    # ======================================================================
    # Accuracy mode menu callbacks
    # ======================================================================

    def _set_fast_mode(self, sender: rumps.MenuItem) -> None:
        self._switch_mode("fast")

    def _set_standard_mode(self, sender: rumps.MenuItem) -> None:
        self._switch_mode("standard")

    def _set_max_accuracy_mode(self, sender: rumps.MenuItem) -> None:
        self._switch_mode("max_accuracy")

    def _switch_mode(self, mode: str) -> None:
        old_mode = self.config.cleanup_mode
        if old_mode == mode:
            return
        log.info("Switching accuracy mode to %s", mode)
        try:
            self.pipeline.set_cleanup_mode(mode)
        except Exception as exc:
            log.exception("Failed to switch accuracy mode to %s", mode)
            self.config.cleanup_mode = old_mode
            self.config.save()
            try:
                self.pipeline.set_cleanup_mode(old_mode)
            except Exception:
                log.exception("Failed to restore previous accuracy mode %s", old_mode)
            self._sync_mode_checkmarks()
            self._show_error(
                title="Mode switch failed",
                message=f"{exc}",
            )
            return
        self.config.cleanup_mode = mode
        self.config.save()
        self._sync_mode_checkmarks()
        self._set_status(f"{mode.replace('_', ' ').title()} mode")

    def _sync_mode_checkmarks(self) -> None:
        mode = self.config.cleanup_mode
        self._fast_item.state = mode == "fast"
        self._standard_item.state = mode == "standard"
        self._max_item.state = mode == "max_accuracy"

    # ======================================================================
    # Model warm-up (background thread at startup)
    # ======================================================================

    def _warm_up_models(self) -> None:
        """Load and warm up all models, then start the hotkey listener."""
        self._set_status("Loading models...")
        try:
            self.pipeline.warm_up()
        except Exception:
            log.exception("Failed to warm up models")
            self._show_error(
                title="Model load failed",
                message="Failed to warm up models. Check model downloads.",
            )
            return

        active_whisper_model = self.pipeline.whisper.model_name
        if (
            self.config.cleanup_mode == "max_accuracy"
            and active_whisper_model != self.config.max_accuracy_whisper_model
        ):
            self._notify(
                title="Max Accuracy Fallback",
                subtitle="VoiceFlow is using fallback Whisper model",
                message=(
                    f"Configured model unavailable; using {active_whisper_model}."
                ),
            )
            log.warning(
                "Configured max accuracy model unavailable; using fallback model %s",
                active_whisper_model,
            )

        # Check Accessibility permission (needed for hotkeys + text insertion)
        try:
            if not _check_accessibility():
                log.warning("Accessibility permission not granted")
                self._set_status("Accessibility required")
                self._notify(
                    title="Accessibility Permission Required",
                    subtitle="VoiceFlow needs Accessibility access",
                    message=(
                        "Open System Settings > Privacy & Security > "
                        "Accessibility and enable VoiceFlow, then restart."
                    ),
                )
        except Exception:
            log.exception("Accessibility check failed (non-fatal)")

        # Start the hotkey listener (dispatches to main thread internally)
        self.hotkey.start()
        log.info("Hotkey listener start requested")

        self._set_status("Ready")
        self._set_title("VF")
        log.info("VoiceFlow ready")

    # ======================================================================
    # Cleanup
    # ======================================================================

    def terminate(self) -> None:  # called by rumps on quit
        log.info("Shutting down VoiceFlow")
        self.hotkey.stop()
        if self.audio.is_active():
            self.audio.stop()
        if self.pipeline.refiner and self.pipeline.refiner.loaded:
            self.pipeline.refiner.unload()

    # ======================================================================
    # Helpers
    # ======================================================================

    def _set_status(self, status: str) -> None:
        AppHelper.callAfter(self._set_status_on_main_thread, status)

    def _set_status_on_main_thread(self, status: str) -> None:
        self._status_item.title = f"Status: {status}"

    def _set_title(self, title: str) -> None:
        AppHelper.callAfter(self._set_title_on_main_thread, title)

    def _set_title_on_main_thread(self, title: str) -> None:
        self.title = title

    def _notify(self, title: str, subtitle: str, message: str) -> None:
        AppHelper.callAfter(self._notify_on_main_thread, title, subtitle, message)

    def _notify_on_main_thread(self, title: str, subtitle: str, message: str) -> None:
        rumps.notification(title=title, subtitle=subtitle, message=message)

    def _show_error(self, title: str, message: str) -> None:
        log.error("%s: %s", title, message)
        self._set_title("VF !")
        self._set_status(title)
        self.overlay.hide()
        try:
            self._notify(
                title=title,
                subtitle="VoiceFlow",
                message=message,
            )
        except Exception:
            log.debug("Notification failed", exc_info=True)
        threading.Timer(2.0, self._reset_title).start()

    def _open_accessibility_settings(self, sender: rumps.MenuItem) -> None:
        self._open_system_settings("Privacy_Accessibility")

    def _open_microphone_settings(self, sender: rumps.MenuItem) -> None:
        self._open_system_settings("Privacy_Microphone")

    def _open_system_settings(self, pane: str) -> None:
        url = f"x-apple.systempreferences:com.apple.preference.security?{pane}"
        try:
            subprocess.run(["open", url], check=False)
        except Exception:
            log.exception("Failed to open System Settings: %s", pane)


def main() -> None:
    # Required for PyInstaller: prevents child processes (e.g. multiprocessing
    # resource_tracker) from re-executing the full app on macOS.
    import multiprocessing
    multiprocessing.freeze_support()

    app = VoiceFlowApp()
    app.run()


if __name__ == "__main__":
    main()
