"""Global hotkey listener using native macOS NSEvent monitors.

Uses NSEvent.addGlobalMonitorForEventsMatchingMask_handler_ which
integrates directly with the AppKit event loop (provided by rumps)
rather than creating a separate CFRunLoop thread like pynput does.
This is more reliable in PyInstaller .app bundles.
"""

import logging
import threading
import time
from typing import Callable

import AppKit
from PyObjCTools import AppHelper

log = logging.getLogger(__name__)

# Modifier key codes (from IOKit/hidsystem/ev_keymap.h)
KEYCODE_MAP: dict[str, int] = {
    "right_cmd": 0x36,
    "left_cmd": 0x37,
    "right_ctrl": 0x3E,
    "left_ctrl": 0x3B,
    "right_alt": 0x3D,
    "left_alt": 0x3A,
    "right_shift": 0x3C,
    "left_shift": 0x38,
}

# Map key codes to their NSEventModifierFlag
_MODIFIER_FLAG_FOR_KEYCODE: dict[int, int] = {
    0x36: AppKit.NSEventModifierFlagCommand,
    0x37: AppKit.NSEventModifierFlagCommand,
    0x3E: AppKit.NSEventModifierFlagControl,
    0x3B: AppKit.NSEventModifierFlagControl,
    0x3D: AppKit.NSEventModifierFlagOption,
    0x3A: AppKit.NSEventModifierFlagOption,
    0x3C: AppKit.NSEventModifierFlagShift,
    0x38: AppKit.NSEventModifierFlagShift,
}


class HotkeyListener:
    """Listens for global hotkey events for dictation control.

    Primary mode (push_to_talk): Press and HOLD key to record, release to stop.
    Secondary mode (toggle): Double-press to start, double-press to stop.
    """

    def __init__(
        self,
        on_recording_start: Callable[[], None],
        on_recording_stop: Callable[[bool], None],  # cancelled: bool
        key: str = "right_cmd",
        mode: str = "push_to_talk",
        min_hold_ms: int = 200,
        double_press_window_ms: int = 300,
    ):
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop
        self._key_code = KEYCODE_MAP.get(key, 0x36)
        self._modifier_flag = _MODIFIER_FLAG_FOR_KEYCODE.get(self._key_code, 0)
        self._mode = mode
        self._min_hold_ms = min_hold_ms
        self._double_press_window_ms = double_press_window_ms

        self._global_monitor = None
        self._local_monitor = None
        self._lock = threading.Lock()

        # Push-to-talk state
        self._key_held = False
        self._press_time: float = 0.0
        self._recording = False

        # Toggle mode state
        self._last_press_time: float = 0.0
        self._toggle_armed = False

    def start(self) -> None:
        """Start listening for hotkey events.

        Safe to call from any thread -- dispatches to the main thread
        where the AppKit event loop is running.
        """
        AppHelper.callAfter(self._start_on_main_thread)

    def _start_on_main_thread(self) -> None:
        if self._global_monitor is not None:
            return

        mask = AppKit.NSEventMaskFlagsChanged

        self._global_monitor = (
            AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask, self._handle_event
            )
        )
        self._local_monitor = (
            AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                mask, self._handle_local_event
            )
        )

        if self._global_monitor is None:
            log.error(
                "Failed to create global event monitor -- "
                "Accessibility permission may be missing"
            )
        else:
            log.info(
                "Hotkey listener started (NSEvent monitor, keycode=0x%02X)",
                self._key_code,
            )

    def stop(self) -> None:
        """Stop listening and clean up."""
        if self._global_monitor is not None:
            AppKit.NSEvent.removeMonitor_(self._global_monitor)
            self._global_monitor = None
        if self._local_monitor is not None:
            AppKit.NSEvent.removeMonitor_(self._local_monitor)
            self._local_monitor = None
        with self._lock:
            self._key_held = False
            self._recording = False
            self._toggle_armed = False

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_local_event(self, event):
        """Handle modifier events when our app is focused."""
        self._handle_event(event)
        return event  # Local monitors MUST return the event

    def _handle_event(self, event) -> None:
        """Handle an NSFlagsChanged event from any app."""
        try:
            if event.keyCode() != self._key_code:
                return

            is_pressed = bool(event.modifierFlags() & self._modifier_flag)

            if self._mode == "push_to_talk":
                if is_pressed:
                    self._handle_push_to_talk_press()
                else:
                    self._handle_push_to_talk_release()
            else:
                if is_pressed:
                    self._handle_toggle_press()
        except Exception:
            log.exception("Error in hotkey event handler")

    # ------------------------------------------------------------------
    # Push-to-talk mode
    # ------------------------------------------------------------------

    def _handle_push_to_talk_press(self) -> None:
        with self._lock:
            if self._key_held:
                return  # Key repeat from macOS -- ignore
            self._key_held = True
            self._press_time = time.monotonic()

        log.debug("Hotkey pressed -- starting recording")
        self._on_recording_start()
        with self._lock:
            self._recording = True

    def _handle_push_to_talk_release(self) -> None:
        with self._lock:
            if not self._key_held:
                return
            self._key_held = False
            press_time = self._press_time
            was_recording = self._recording
            self._recording = False

        if not was_recording:
            return

        hold_duration_ms = (time.monotonic() - press_time) * 1000

        if hold_duration_ms < self._min_hold_ms:
            log.debug("Hold too short (%.0fms) -- cancelling", hold_duration_ms)
            self._on_recording_stop(cancelled=True)
            return

        log.debug("Hotkey released after %.0fms -- stopping recording", hold_duration_ms)
        self._on_recording_stop(cancelled=False)

    # ------------------------------------------------------------------
    # Toggle mode
    # ------------------------------------------------------------------

    def _handle_toggle_press(self) -> None:
        now = time.monotonic()
        with self._lock:
            elapsed_ms = (now - self._last_press_time) * 1000
            self._last_press_time = now

            if elapsed_ms > self._double_press_window_ms:
                self._toggle_armed = True
                return

            if not self._toggle_armed:
                return

            # Double-press detected
            self._toggle_armed = False

            if not self._recording:
                self._recording = True
                start = True
            else:
                self._recording = False
                start = False

        if start:
            self._on_recording_start()
        else:
            self._on_recording_stop(cancelled=False)
