"""Text insertion via clipboard + simulated Cmd+V."""

import time
from typing import Optional

from AppKit import NSPasteboard, NSPasteboardTypeString
import Quartz


class TextInserter:
    """Inserts text at cursor position via clipboard + Cmd+V.

    This is the only universally reliable method on macOS.
    Works in every app that supports paste.
    """

    SETTLE_DELAY = 0.05   # 50ms for pasteboard to settle
    PASTE_DELAY = 0.10    # 100ms for paste to complete
    RESTORE_DELAY = 0.05  # 50ms before restoring clipboard

    @staticmethod
    def insert(text: str, restore_clipboard: bool = True) -> None:
        """Insert text at current cursor position.

        Saves clipboard, sets our text, simulates Cmd+V, restores clipboard.
        """
        if not text:
            return

        original: Optional[str] = None
        if restore_clipboard:
            original = TextInserter._get_clipboard()

        TextInserter._set_clipboard(text)
        time.sleep(TextInserter.SETTLE_DELAY)

        TextInserter._simulate_paste()
        time.sleep(TextInserter.PASTE_DELAY)

        if restore_clipboard and original is not None:
            time.sleep(TextInserter.RESTORE_DELAY)
            TextInserter._set_clipboard(original)

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

        key_down = Quartz.CGEventCreateKeyboardEvent(source, 9, True)
        key_up = Quartz.CGEventCreateKeyboardEvent(source, 9, False)

        Quartz.CGEventSetFlags(key_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetFlags(key_up, Quartz.kCGEventFlagMaskCommand)

        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
