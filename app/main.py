"""VoiceFlow -- Local AI dictation for macOS."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import rumps

from app.config import AppConfig
from app.dictionary import Dictionary
from app.audio.capture import AudioCapture
from app.transcription import TranscriptionPipeline
from app.input.hotkey import HotkeyListener
from app.input.text_inserter import TextInserter
from app.ui.overlay import RecordingOverlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "resources"

# Minimum audio length (in samples at 16 kHz) to bother transcribing.
# 0.3 seconds = 4800 samples -- anything shorter is almost certainly noise.
_MIN_AUDIO_SAMPLES = 4800


def _check_accessibility() -> bool:
    """Return True if the app has Accessibility permission."""
    try:
        from ApplicationServices import AXIsProcessTrusted  # type: ignore[import-untyped]
        trusted = AXIsProcessTrusted()
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
        self.audio.drain()
        self.audio.start()
        self.title = "VF \u25cf"  # bullet
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
            self.title = "VF"
            self.overlay.hide()
            return

        if audio.size < _MIN_AUDIO_SAMPLES:
            log.info("Audio too short (%d samples); discarding", audio.size)
            self.title = "VF"
            self.overlay.hide()
            return

        log.info("Recording stopped; captured %d samples (%.1fs)",
                 audio.size, audio.size / AudioCapture.SAMPLE_RATE)
        self.title = "VF ..."
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
            if result:
                log.info("Transcription result: %s", result)
                TextInserter.insert(result, self.config.restore_clipboard)
            else:
                log.info("Pipeline returned empty result (no speech detected)")
        except Exception:
            log.exception("Error during audio processing")
            self.title = "VF !"
            self.overlay.hide()
            # Brief error indicator, then reset
            threading.Timer(2.0, self._reset_title).start()
            return
        finally:
            with self._lock:
                self._processing = False

        self.title = "VF"
        self.overlay.hide()

    def _reset_title(self) -> None:
        self.title = "VF"

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
        self.title = "VF \u25cf"
        self.audio.drain()
        self.audio.start()

        def _stop_after_delay():
            import time
            time.sleep(3)
            log.info("Test recording: stopping after 3s")
            # Dispatch stop to main thread context
            self._on_recording_stop(cancelled=False)

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
        if self.config.cleanup_mode == mode:
            return
        log.info("Switching accuracy mode to %s", mode)
        self.config.cleanup_mode = mode
        self.config.save()
        self.pipeline.set_cleanup_mode(mode)
        self._sync_mode_checkmarks()
        self._status_item.title = f"Status: {mode.replace('_', ' ').title()} mode"

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
        self._status_item.title = "Status: Loading models..."
        try:
            self.pipeline.warm_up()
        except Exception:
            log.exception("Failed to warm up models")
            self._status_item.title = "Status: Model load failed"
            self.title = "VF !"
            return

        # Check Accessibility permission (needed for hotkeys + text insertion)
        try:
            if not _check_accessibility():
                log.warning("Accessibility permission not granted")
                rumps.notification(
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

        self._status_item.title = "Status: Ready"
        self.title = "VF"
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


def main() -> None:
    # Required for PyInstaller: prevents child processes (e.g. multiprocessing
    # resource_tracker) from re-executing the full app on macOS.
    import multiprocessing
    multiprocessing.freeze_support()

    app = VoiceFlowApp()
    app.run()


if __name__ == "__main__":
    main()
