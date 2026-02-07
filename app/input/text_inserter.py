"""Text insertion via clipboard + simulated Cmd+V."""

from __future__ import annotations

import logging
import time
from typing import Optional

from AppKit import NSPasteboard, NSPasteboardTypeString
import Quartz

log = logging.getLogger(__name__)

try:  # Accessibility permission check (optional)
    from ApplicationServices import AXIsProcessTrusted  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - depends on macOS frameworks
    AXIsProcessTrusted = None


class TextInserter:
    """Inserts text at cursor position via clipboard + Cmd+V.

    This is the only universally reliable method on macOS.
    Works in every app that supports paste.
    """

    last_error: Optional[str] = None

    SETTLE_DELAY = 0.05   # 50ms for pasteboard to settle
    PASTE_DELAY = 0.12    # base delay for paste to complete
    RESTORE_DELAY = 0.08  # base delay before restoring clipboard
    _LONG_TEXT_CHARS = 180
    _VERY_LONG_TEXT_CHARS = 420

    @staticmethod
    def is_accessibility_trusted() -> bool:
        """Return whether macOS Accessibility permission is granted."""
        if AXIsProcessTrusted is None:
            return True
        try:
            return bool(AXIsProcessTrusted())
        except Exception as exc:
            log.warning("Accessibility check failed: %s", exc)
            return True

    @staticmethod
    def insert(text: str, restore_clipboard: bool = True) -> bool:
        """Insert text at current cursor position.

        Saves clipboard, sets our text, simulates Cmd+V, restores clipboard.
        """
        TextInserter.last_error = None
        if not text:
            return True

        original: Optional[str] = None
        if restore_clipboard:
            original = TextInserter._get_clipboard()

        try:
            TextInserter._set_clipboard(text)
            time.sleep(TextInserter.SETTLE_DELAY)

            if not TextInserter.is_accessibility_trusted():
                TextInserter.last_error = (
                    "Accessibility permission required for paste. "
                    "Text has been copied to your clipboard."
                )
                return False

            TextInserter._simulate_paste()
            time.sleep(TextInserter._paste_delay_for_text(text))

            if restore_clipboard and original is not None:
                time.sleep(TextInserter._restore_delay_for_text(text))
                TextInserter._set_clipboard(original)
            return True
        except Exception as exc:
            log.exception("Text insertion failed")
            TextInserter.last_error = str(exc)
            try:
                TextInserter._set_clipboard(text)
            except Exception as clipboard_exc:
                log.debug(
                    "Failed to preserve text in clipboard after paste error: %s",
                    clipboard_exc,
                )
            return False

    @classmethod
    def _paste_delay_for_text(cls, text: str) -> float:
        text_len = len(text)
        if text_len <= cls._LONG_TEXT_CHARS:
            return cls.PASTE_DELAY
        # Longer pastes need more settle time in some editors.
        scaled = cls.PASTE_DELAY + min((text_len - cls._LONG_TEXT_CHARS) / 700.0, 1.10)
        if text_len >= cls._VERY_LONG_TEXT_CHARS:
            scaled = max(scaled, 0.55)
        return min(scaled, 1.22)

    @classmethod
    def _restore_delay_for_text(cls, text: str) -> float:
        text_len = len(text)
        if text_len <= cls._LONG_TEXT_CHARS:
            return cls.RESTORE_DELAY
        # Delay clipboard restore longer for big dictations to avoid clipping.
        scaled = cls.RESTORE_DELAY + min((text_len - cls._LONG_TEXT_CHARS) / 520.0, 1.80)
        if text_len >= cls._VERY_LONG_TEXT_CHARS:
            scaled = max(scaled, 0.95)
        return min(scaled, 2.00)

    @staticmethod
    def _get_clipboard() -> Optional[str]:
        """Read current clipboard string content."""
        pb = NSPasteboard.generalPasteboard()
        return pb.stringForType_(NSPasteboardTypeString)

    @staticmethod
    def _set_clipboard(text: str) -> None:
        """Set clipboard to the given string."""
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)

    @staticmethod
    def _simulate_paste() -> None:
        """Simulate Cmd+V using Quartz CGEvent.

        Key code 9 = 'v' on US keyboard layout.
        """
        source = Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState
        )
        if source is None:
            raise RuntimeError("Failed to create CGEvent source")

        key_down = Quartz.CGEventCreateKeyboardEvent(source, 9, True)
        key_up = Quartz.CGEventCreateKeyboardEvent(source, 9, False)
        if key_down is None or key_up is None:
            raise RuntimeError("Failed to create CGEvent for paste")

        Quartz.CGEventSetFlags(key_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetFlags(key_up, Quartz.kCGEventFlagMaskCommand)

        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
