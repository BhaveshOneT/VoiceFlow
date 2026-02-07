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
    PASTE_DELAY = 0.10    # 100ms for paste to complete
    RESTORE_DELAY = 0.05  # 50ms before restoring clipboard

    @staticmethod
    def insert(text: str, restore_clipboard: bool = True) -> bool:
        """Insert text at current cursor position.

        Saves clipboard, sets our text, simulates Cmd+V, restores clipboard.
        """
        TextInserter.last_error = None
        if not text:
            return True

        if AXIsProcessTrusted is not None:
            try:
                if not AXIsProcessTrusted():
                    TextInserter._set_clipboard(text)
                    TextInserter.last_error = (
                        "Accessibility permission required for paste"
                    )
                    return False
            except Exception as exc:
                log.warning("Accessibility check failed: %s", exc)

        original: Optional[str] = None
        if restore_clipboard:
            original = TextInserter._get_clipboard()

        try:
            TextInserter._set_clipboard(text)
            time.sleep(TextInserter.SETTLE_DELAY)

            TextInserter._simulate_paste()
            time.sleep(TextInserter.PASTE_DELAY)

            if restore_clipboard and original is not None:
                time.sleep(TextInserter.RESTORE_DELAY)
                TextInserter._set_clipboard(original)
            return True
        except Exception as exc:
            log.exception("Text insertion failed")
            TextInserter.last_error = str(exc)
            try:
                TextInserter._set_clipboard(text)
            except Exception:
                pass
            return False

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
