"""VoiceFlow -- Local AI dictation & meeting intelligence for macOS.

PySide6-based entry point.  Replaces the legacy rumps menu-bar-only app
with a full windowed application while preserving all dictation features.
"""
from __future__ import annotations

import logging
import os
import platform
import signal as _signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from app.audio.capture import AudioCapture
from app.config import AppConfig
from app.core.signals import AppSignals
from app.dictionary import Dictionary
from app.input.hotkey import HotkeyListener
from app.input.text_inserter import TextInserter
from app.transcription import TranscriptionPipeline
from app.ui.overlay import RecordingOverlay
from app.ui_qt.main_window import MainWindow
from app.ui_qt.styles.stylesheets import app_stylesheet

LOG_DIR = Path.home() / "Library" / "Application Support" / "VoiceFlow" / "logs"
LOG_PATH = LOG_DIR / "voiceflow.log"


def _configure_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
    except Exception as exc:
        logging.getLogger(__name__).debug("File logging unavailable: %s", exc)
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
_MIN_AUDIO_SAMPLES = 4800

_PROGRAMMER_BUNDLE_HINTS = (
    "com.apple.terminal",
    "com.googlecode.iterm2",
    "dev.warp.warp-stable",
    "com.microsoft.vscode",
    "com.todesktop.230313mzl4w4u92",
    "com.todesktop.230313mzl4w4u92.helper",
    "com.anthropic.claudefordesktop",
    "com.openai.codex",
    "com.github.atom",
    "com.jetbrains.",
)


def _check_accessibility() -> bool:
    """Return True if the app has Accessibility permission."""
    try:
        from ApplicationServices import (
            AXIsProcessTrusted,
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
        trusted = bool(AXIsProcessTrusted())
        if not trusted:
            try:
                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
                trusted = bool(AXIsProcessTrusted())
            except Exception as exc:
                log.debug("Accessibility prompt check failed: %s", exc)
        log.info("Accessibility check: trusted=%s", trusted)
        return trusted
    except ImportError:
        log.warning("ApplicationServices not available; skipping accessibility check")
        return True


class _MainThreadInvoker(QObject):
    """Dispatch callables to the main Qt thread via a queued signal.

    When ``invoke(fn)`` is called from a background thread, Qt automatically
    uses a queued connection because the QObject lives on the main thread.
    The connected slot ``_execute`` therefore runs on the main thread.
    """

    _call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._call.connect(self._execute)

    @staticmethod
    def _execute(fn: object) -> None:
        try:
            fn()  # type: ignore[operator]
        except Exception:
            logging.getLogger(__name__).debug(
                "Main-thread callback failed", exc_info=True
            )

    def invoke(self, fn: object) -> None:
        self._call.emit(fn)


def _log_system_info() -> None:
    """Log system diagnostics at startup."""
    log.info(
        "System: macOS %s, Python %s, PID %d",
        platform.mac_ver()[0] or "unknown",
        platform.python_version(),
        os.getpid(),
    )
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        mem_gb = (page_size * page_count) / (1024 ** 3)
        log.info("Physical memory: %.1f GB", mem_gb)
    except (ValueError, OSError):
        pass


class VoiceFlowApplication:
    """Main application orchestrator -- owns all components.

    Replaces the legacy ``VoiceFlowApp(rumps.App)`` with PySide6.
    """

    def __init__(self) -> None:
        # -- macOS: register as a proper GUI application -------------------
        # Must happen BEFORE QApplication is created so macOS treats the
        # process as a regular app (Dock icon, Cmd+Tab, windows stay front).
        try:
            from AppKit import NSApp, NSApplication
            NSApplication.sharedApplication()
            NSApp.setActivationPolicy_(0)  # 0 = NSApplicationActivationPolicyRegular
        except Exception:
            log.debug("Failed to set activation policy (non-fatal)")

        # -- Qt app --------------------------------------------------------
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        self.qt_app.setApplicationName("VoiceFlow")
        self.qt_app.setOrganizationName("VoiceFlow")
        self.qt_app.setQuitOnLastWindowClosed(False)
        self.qt_app.setStyleSheet(app_stylesheet())

        # -- Signals (cross-thread safe) -----------------------------------
        self.signals = AppSignals()
        self._invoker = _MainThreadInvoker()

        # -- Config & dictionary -------------------------------------------
        self.config = AppConfig.load()
        self.dictionary = Dictionary.load(Path(self.config.dictionary_path))

        # -- Audio & pipeline ----------------------------------------------
        self.audio = AudioCapture()
        self.pipeline = TranscriptionPipeline(self.config, self.dictionary)

        # -- Hotkey listener -----------------------------------------------
        self.hotkey = HotkeyListener(
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            key=self.config.hotkey,
            mode=self.config.recording_mode,
        )

        # -- Recording overlay (AppKit, still native) ----------------------
        self.overlay = RecordingOverlay()

        # -- State ---------------------------------------------------------
        self._lock = threading.Lock()
        self._processing = False
        self._target_app_pid: int | None = None  # PID of app to paste into

        # -- System tray ---------------------------------------------------
        self._tray = QSystemTrayIcon()
        self._build_tray_menu()
        self._set_tray_title("VF ...")
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

        # -- Main window ---------------------------------------------------
        self.window = MainWindow(self.config, self.signals)

        # -- Auto-transcription on meeting stop ----------------------------
        self.signals.meeting_recording_stopped.connect(self._on_meeting_stopped)

        # -- Live settings wiring ------------------------------------------
        self.signals.hotkey_changed.connect(self._on_hotkey_setting_changed)
        self.signals.language_changed.connect(self._on_language_setting_changed)
        self.signals.accuracy_changed.connect(self._on_accuracy_setting_changed)
        self.signals.transcription_mode_changed.connect(
            self._on_transcription_mode_setting_changed
        )

        # -- Start model warm-up -------------------------------------------
        threading.Thread(target=self._warm_up_models, daemon=True).start()

    # ==================================================================
    # System tray
    # ==================================================================

    def _build_tray_menu(self) -> None:
        menu = QMenu()

        self._status_action = QAction("Status: Loading...")
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)

        menu.addSeparator()

        show_action = QAction("Show Window")
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)

        menu.addSeparator()

        quit_action = QAction("Quit VoiceFlow")
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)

    def _set_tray_title(self, title: str) -> None:
        def _do() -> None:
            self._tray.setToolTip(title)
        self._run_on_main_thread(_do)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_window()

    def _show_window(self) -> None:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        # macOS: force Python process to the foreground (required when
        # running from terminal or when the process lacks a .app bundle).
        try:
            from AppKit import NSApp
            NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass

    def _quit(self) -> None:
        shutdown_start = time.perf_counter()
        log.info("Shutting down VoiceFlow...")

        # Stop hotkey listener
        try:
            self.hotkey.stop()
        except Exception:
            log.debug("Error stopping hotkey listener", exc_info=True)

        # Stop any active audio capture
        try:
            if self.audio.is_active():
                self.audio.stop()
        except Exception:
            log.debug("Error stopping audio capture", exc_info=True)

        # Clear processing state
        with self._lock:
            self._processing = False

        # Unload refiner model
        try:
            if self.pipeline.refiner and self.pipeline.refiner.loaded:
                self.pipeline.refiner.unload()
        except Exception:
            log.debug("Error unloading refiner", exc_info=True)

        # Unload STT model if method exists
        try:
            stt_engine = getattr(self.pipeline, "stt", getattr(self.pipeline, "whisper", None))
            if stt_engine and hasattr(stt_engine, "unload"):
                stt_engine.unload()
        except Exception:
            log.debug("Error unloading STT model", exc_info=True)

        # Close database
        try:
            db = getattr(getattr(self, "window", None), "_db", None)
            if db and hasattr(db, "close"):
                db.close()
        except Exception:
            log.debug("Error closing database", exc_info=True)

        # Release PID lock
        _release_single_instance_lock()

        # Hide tray
        try:
            self._tray.hide()
        except Exception:
            log.debug("Error hiding tray", exc_info=True)

        shutdown_ms = (time.perf_counter() - shutdown_start) * 1000.0
        log.info("VoiceFlow shutdown complete (%.1fms)", shutdown_ms)
        self.qt_app.quit()

    # ==================================================================
    # Recording lifecycle (identical logic to legacy main.py)
    # ==================================================================

    def _on_recording_start(self) -> None:
        with self._lock:
            if self._processing:
                log.debug("Still processing previous audio; ignoring new recording")
                return
        log.info("Recording started")
        # Remember which app was frontmost so we paste into it later.
        # With PySide6 as a visible app, VoiceFlow could steal focus.
        # If VoiceFlow itself is frontmost, don't capture it -- leave
        # _target_app_pid as None so we fall through to the retry logic.
        front_pid = self._frontmost_app_pid()
        if front_pid != os.getpid():
            self._target_app_pid = front_pid
        else:
            self._target_app_pid = None
            log.debug("VoiceFlow is frontmost at recording start; skipping PID capture")
        self._apply_auto_transcription_mode()
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
        self._set_tray_title("VF \u25cf")
        self._set_status("Recording")
        self.overlay.show_recording()

    def _on_recording_stop(self, cancelled: bool = False) -> None:
        if not self.audio.is_active():
            return

        capture_stop_started = time.perf_counter()
        audio = self.audio.stop()
        capture_stop_ms = (time.perf_counter() - capture_stop_started) * 1000.0

        if cancelled:
            log.info("Recording cancelled (hold too short)")
            self._set_tray_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            return

        if audio.size < _MIN_AUDIO_SAMPLES:
            log.info("Audio too short (%d samples); discarding", audio.size)
            self._set_tray_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            return

        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < 0.003:
            log.info("Audio is silence (rms=%.6f); discarding", rms)
            self._set_tray_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            return

        log.info(
            "Recording stopped; captured %d samples (%.1fs), capture_stop_ms=%.1f",
            audio.size,
            audio.size / AudioCapture.SAMPLE_RATE,
            capture_stop_ms,
        )
        self._set_tray_title("VF ...")
        self._set_status("Processing")
        self.overlay.show_processing()
        with self._lock:
            self._processing = True

        threading.Thread(
            target=self._process_audio, args=(audio,), daemon=True
        ).start()

    # ==================================================================
    # Audio processing (background thread)
    # ==================================================================

    def _process_audio(self, audio: np.ndarray) -> None:
        process_started = time.perf_counter()
        pipeline_ms = 0.0
        try:
            pipeline_started = time.perf_counter()
            result = self.pipeline.process(audio)
            pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
        except Exception as exc:
            log.exception("Transcription failed")
            detail = str(exc).strip() or exc.__class__.__name__
            self._show_error(
                title="Transcription failed",
                message=f"{detail}. Check VoiceFlow logs for details.",
            )
            with self._lock:
                self._processing = False
            return

        if not result:
            log.info("Pipeline returned empty result (no speech detected)")
            self._set_tray_title("VF")
            self._set_status("Ready")
            self.overlay.hide()
            with self._lock:
                self._processing = False
            return

        log.info(
            "Transcription result ready (chars=%d, words=%d)",
            len(result),
            len(result.split()),
        )
        # Ensure the user's target app is frontmost before pasting.
        # With PySide6 as a visible app, VoiceFlow could have stolen focus.
        self._reactivate_target_app()
        # Verify we're not about to paste into ourselves.
        own_pid = os.getpid()
        current_front = self._frontmost_app_pid()
        if current_front == own_pid:
            log.warning(
                "VoiceFlow is still frontmost after reactivation; "
                "retrying activation"
            )
            self._reactivate_target_app()
            time.sleep(0.25)
        else:
            time.sleep(0.15)  # let macOS complete the app switch

        paste_started = time.perf_counter()
        inserted = TextInserter.insert(result, self.config.restore_clipboard)
        paste_ms = (time.perf_counter() - paste_started) * 1000.0
        total_ms = (time.perf_counter() - process_started) * 1000.0
        log.info(
            "End-to-end post-record timings (ms): pipeline=%.1f paste=%.1f total=%.1f",
            pipeline_ms,
            paste_ms,
            total_ms,
        )

        # Release the processing lock AFTER paste completes to prevent
        # overlapping dictations from racing on the clipboard / Cmd+V.
        with self._lock:
            self._processing = False

        if not inserted:
            detail = TextInserter.last_error or "Paste failed"
            if "Accessibility permission required" in detail:
                self.overlay.hide()
                self._set_tray_title("VF")
                self._set_status("Paste permission required")
                self._run_on_main_thread(lambda: self._tray.showMessage(
                    "Paste Permission Required",
                    "Enable Accessibility for VoiceFlow to auto-paste. "
                    "You can paste now with Command+V.",
                    QSystemTrayIcon.MessageIcon.Warning,
                    5000,
                ))
                self._open_system_settings("Privacy_Accessibility")
                return
            self._show_error(title="Paste failed", message=detail)
            return

        self.signals.transcription_complete.emit(result)
        self._set_tray_title("VF")
        self._set_status("Ready")
        self.overlay.hide()

    # ==================================================================
    # Auto transcription mode switching
    # ==================================================================

    def _apply_auto_transcription_mode(self) -> None:
        if not self.config.auto_mode_switch:
            return
        app_name, bundle_id = self._frontmost_app_info()
        desired = self._infer_transcription_mode_for_app(
            app_name=app_name,
            bundle_id=bundle_id,
            programmer_hints=self.config.programmer_apps,
        )
        if desired == self.config.transcription_mode:
            return
        log.info(
            "Auto-switching transcription mode to %s (frontmost app=%s, bundle=%s)",
            desired,
            app_name or "<unknown>",
            bundle_id or "<unknown>",
        )
        try:
            self.pipeline.set_transcription_mode(desired)
            self.config.transcription_mode = desired
            self.config.save()
        except Exception:
            log.exception("Failed auto-switching transcription mode to %s", desired)

    @staticmethod
    def _frontmost_app_info() -> tuple[str, str]:
        try:
            import AppKit as AK
            app = AK.NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is None:
                return "", ""
            name = str(app.localizedName() or "")
            bundle_id = str(app.bundleIdentifier() or "")
            return name, bundle_id
        except Exception:
            log.debug("Could not resolve frontmost app", exc_info=True)
            return "", ""

    @staticmethod
    def _infer_transcription_mode_for_app(
        app_name: str,
        bundle_id: str,
        programmer_hints: list[str],
    ) -> str:
        app_name_l = app_name.strip().lower()
        bundle_l = bundle_id.strip().lower()
        hints = [hint.strip().lower() for hint in programmer_hints if hint.strip()]
        if any(hint in app_name_l for hint in hints):
            return "programmer"
        if any(hint in bundle_l for hint in hints):
            return "programmer"
        if any(bundle_hint in bundle_l for bundle_hint in _PROGRAMMER_BUNDLE_HINTS):
            return "programmer"
        return "normal"

    # ==================================================================
    # Model warm-up (background thread)
    # ==================================================================

    def _warm_up_models(self) -> None:
        self._set_status("Loading speech model...")
        self.signals.model_loading.emit("stt")
        try:
            self.pipeline.warm_up_for_realtime()
        except Exception:
            log.exception("Failed to warm up speech model")
            self._show_error(
                title="Model load failed",
                message="Failed to warm up speech model. Check model downloads.",
            )
            return

        self.signals.model_loaded.emit("stt")

        active_stt_model = self.pipeline.stt.model_name
        if (
            self.config.cleanup_mode == "max_accuracy"
            and active_stt_model != self.config.max_accuracy_stt_model
        ):
            _fallback_msg = f"Configured model unavailable; using {active_stt_model}."
            self._run_on_main_thread(lambda: self._tray.showMessage(
                "Max Accuracy Fallback",
                _fallback_msg,
                QSystemTrayIcon.MessageIcon.Warning,
                5000,
            ))

        # Check Accessibility permission
        try:
            if not _check_accessibility():
                log.warning("Accessibility permission not granted")
                self._set_status("Accessibility required")
                self._run_on_main_thread(lambda: self._tray.showMessage(
                    "Accessibility Permission Required",
                    "Open System Settings > Privacy & Security > "
                    "Accessibility and enable VoiceFlow, then restart.",
                    QSystemTrayIcon.MessageIcon.Warning,
                    8000,
                ))
        except Exception:
            log.exception("Accessibility check failed (non-fatal)")

        # Start the hotkey listener
        self.hotkey.start()
        log.info("Hotkey listener start requested")

        self._set_status("Ready")
        self._set_tray_title("VF")
        log.info("VoiceFlow ready")

        # Load optional LLM refiner in background
        if self.pipeline.refiner and not self.pipeline.refiner.loaded:
            threading.Thread(target=self._warm_up_refiner_background, daemon=True).start()

    def _warm_up_refiner_background(self) -> None:
        self.signals.model_loading.emit("refiner")
        try:
            self.pipeline.warm_up_refiner()
            self.signals.model_loaded.emit("refiner")
        except Exception:
            log.exception("Background LLM warm-up failed (continuing without refiner)")

    # ==================================================================
    # Meeting auto-transcription + summarization
    # ==================================================================

    def _on_meeting_stopped(self, meeting_id: str) -> None:
        """Triggered when a meeting recording stops."""
        if not self.config.auto_transcribe_on_stop:
            log.info("Auto-transcribe disabled; skipping for meeting %s", meeting_id)
            return
        threading.Thread(
            target=self._process_meeting, args=(meeting_id,), daemon=True
        ).start()

    def _process_meeting(self, meeting_id: str) -> None:
        """Background: transcribe and optionally summarize a meeting."""
        import uuid as _uuid

        from app.core.model_manager import ModelManager
        from app.storage.models import MeetingStatus, MeetingSummary
        from app.transcription.diarization import SpeakerDiarizer
        from app.transcription.meeting_summarizer import MeetingSummarizer
        from app.transcription.meeting_transcriber import MeetingTranscriber

        db = self.window._db
        meeting = db.get_meeting(meeting_id)
        if not meeting or not meeting.audio_path:
            log.warning("Cannot process meeting %s: missing audio path", meeting_id)
            return

        audio_path = Path(meeting.audio_path)
        if not audio_path.exists():
            log.warning("Audio file missing for meeting %s: %s", meeting_id, audio_path)
            return

        model_manager = ModelManager(self.signals)
        model_manager.register_stt(
            load_fn=lambda: self.pipeline.warm_up_for_realtime(),
            unload_fn=lambda: None,
        )

        diarizer = None
        if SpeakerDiarizer.is_available():
            diarizer = SpeakerDiarizer(auth_token=self.config.pyannote_auth_token or None)
            model_manager.register_diarizer(
                load_fn=diarizer.load,
                unload_fn=diarizer.unload,
            )
        else:
            log.info("pyannote.audio not installed; skipping diarization for meeting %s", meeting_id)

        try:
            transcriber = MeetingTranscriber(
                stt=self.pipeline.stt,
                diarizer=diarizer,
                model_manager=model_manager,
                db=db,
                signals=self.signals,
                transcription_provider=self.config.meeting_transcription_provider,
                openai_api_key=self.config.openai_api_key,
            )
            log.info("Starting auto-transcription for meeting %s", meeting_id)
            self._set_status("Transcribing meeting...")
            transcriber.process_meeting(meeting_id, audio_path)
        except Exception:
            log.exception("Auto-transcription failed for meeting %s", meeting_id)
            self._set_status("Meeting transcription failed")
            return

        # Optional auto-summarization
        if self.config.auto_summarize:
            try:
                self._set_status("Summarizing meeting...")
                db.update_meeting(meeting_id, status=MeetingStatus.SUMMARIZING)
                segments = db.get_segments(meeting_id)
                transcript_text = "\n".join(seg.text for seg in segments if seg.text)
                if transcript_text.strip():
                    summarizer = MeetingSummarizer(
                        provider=self.config.summary_provider,
                        api_key=self.config.openai_api_key,
                    )
                    result = summarizer.summarize(transcript_text)
                    summary = MeetingSummary(
                        id=str(_uuid.uuid4()),
                        meeting_id=meeting_id,
                        summary_text=result.summary_text,
                        key_points=result.key_points,
                        action_items=result.action_items,
                    )
                    db.save_summary(summary)
                    db.update_meeting(meeting_id, status=MeetingStatus.COMPLETE)
                    log.info("Auto-summarization complete for meeting %s", meeting_id)
            except Exception:
                log.exception("Auto-summarization failed for meeting %s", meeting_id)

        self._set_status("Ready")

    # ==================================================================
    # Live settings handlers (connected to AppSignals)
    # ==================================================================

    def _on_hotkey_setting_changed(self, key: str) -> None:
        log.info("Applying hotkey change: %s", key)
        self.hotkey.update_key(key)

    def _on_language_setting_changed(self, language: str) -> None:
        log.info("Applying language change: %s", language)
        try:
            self.pipeline.set_language(language)
        except Exception:
            log.exception("Failed to apply language change")

    def _on_accuracy_setting_changed(self, mode: str) -> None:
        log.info("Applying accuracy mode change: %s", mode)
        try:
            self.pipeline.set_cleanup_mode(mode)
        except Exception:
            log.exception("Failed to apply accuracy mode change")

    def _on_transcription_mode_setting_changed(self, mode: str) -> None:
        log.info("Applying transcription mode change: %s", mode)
        try:
            self.pipeline.set_transcription_mode(mode)
        except Exception:
            log.exception("Failed to apply transcription mode change")

    # ==================================================================
    # Helpers
    # ==================================================================

    def _run_on_main_thread(self, callback: object) -> None:
        """Schedule *callback* to execute on the main Qt thread.

        If already on the main thread the callback runs immediately;
        otherwise it is dispatched via ``_MainThreadInvoker``.
        """
        if threading.current_thread() is threading.main_thread():
            try:
                callback()  # type: ignore[operator]
            except Exception:
                log.debug("Main-thread callback failed", exc_info=True)
        else:
            self._invoker.invoke(callback)

    def _set_status(self, status: str) -> None:
        def _do() -> None:
            self._status_action.setText(f"Status: {status}")
        self._run_on_main_thread(_do)
        self.signals.status_changed.emit(status)

    def _show_error(self, title: str, message: str) -> None:
        log.error("%s: %s", title, message)
        self.signals.error_occurred.emit(title, message)
        self.signals.status_changed.emit(title)

        def _do() -> None:
            self._tray.setToolTip("VF !")
            self._status_action.setText(f"Status: {title}")
            try:
                self._tray.showMessage(
                    title, message, QSystemTrayIcon.MessageIcon.Critical, 5000,
                )
            except Exception:
                log.debug("Tray notification failed", exc_info=True)
            QTimer.singleShot(2000, lambda: self._tray.setToolTip("VF"))

        self._run_on_main_thread(_do)
        self.overlay.hide()  # already thread-safe via AppHelper.callAfter

    @staticmethod
    def _open_system_settings(pane: str) -> None:
        try:
            import AppKit as AK
            url = f"x-apple.systempreferences:com.apple.preference.security?{pane}"
            ns_url = AK.NSURL.URLWithString_(url)
            if ns_url:
                AK.NSWorkspace.sharedWorkspace().openURL_(ns_url)
        except Exception:
            log.exception("Failed to open System Settings: %s", pane)

    @staticmethod
    def _frontmost_app_pid() -> int | None:
        """Return the PID of the currently frontmost macOS application."""
        try:
            from AppKit import NSWorkspace
            front = NSWorkspace.sharedWorkspace().frontmostApplication()
            if front:
                return front.processIdentifier()
        except Exception:
            log.debug("Could not get frontmost app PID", exc_info=True)
        return None

    def _reactivate_target_app(self) -> None:
        """Reactivate the app that was frontmost when recording started."""
        pid = self._target_app_pid
        if pid is None:
            return
        try:
            from AppKit import NSRunningApplication
            target = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if target and not target.isTerminated():
                # NSApplicationActivateIgnoringOtherApps = 1 << 1 = 2
                target.activateWithOptions_(2)
                log.debug("Reactivated target app PID=%d (%s)", pid, target.localizedName())
            else:
                log.debug("Target app PID=%d no longer running", pid)
        except Exception:
            log.debug("Failed to reactivate target app PID=%d", pid, exc_info=True)

    # ==================================================================
    # Run
    # ==================================================================

    def run(self) -> int:
        """Start the application event loop."""
        self._show_window()
        return self.qt_app.exec()


# Backward-compatible class alias for legacy imports/tests.
VoiceFlowApp = VoiceFlowApplication

_LOCK_PATH = Path.home() / "Library" / "Application Support" / "VoiceFlow" / ".lock"


def _acquire_single_instance_lock() -> bool:
    """Ensure only one VoiceFlow process runs at a time.

    Uses a PID-based lock file.  Returns True if we acquired the lock.
    If another instance is alive, prints a message and returns False.
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LOCK_PATH.exists():
        try:
            old_pid = int(_LOCK_PATH.read_text().strip())
            # Check if that PID is still alive
            os.kill(old_pid, 0)
            # Process exists -- refuse to start
            print(f"VoiceFlow is already running (PID {old_pid}). Exiting.")
            return False
        except (ValueError, OSError):
            pass  # stale lock or process gone
    _LOCK_PATH.write_text(str(os.getpid()))
    return True


def _release_single_instance_lock() -> None:
    try:
        if _LOCK_PATH.exists():
            pid_in_file = int(_LOCK_PATH.read_text().strip())
            if pid_in_file == os.getpid():
                _LOCK_PATH.unlink()
    except Exception:
        pass


def main() -> None:
    import atexit
    import multiprocessing
    multiprocessing.freeze_support()

    _log_system_info()

    if not _acquire_single_instance_lock():
        sys.exit(1)
    atexit.register(_release_single_instance_lock)

    app = VoiceFlowApplication()

    # Allow Ctrl+C / SIGTERM to trigger clean shutdown
    _signal.signal(_signal.SIGINT, lambda *_: app._quit())
    _signal.signal(_signal.SIGTERM, lambda *_: app._quit())

    # Periodic timer lets Python process signal handlers during the Qt
    # event loop (which otherwise blocks signal delivery).
    # Must be stored on `app` to prevent garbage collection.
    app._signal_timer = QTimer()
    app._signal_timer.start(500)
    app._signal_timer.timeout.connect(lambda: None)

    sys.exit(app.run())


if __name__ == "__main__":
    main()
