from __future__ import annotations

import unittest

from app.input.text_inserter import TextInserter


class TextInserterDelayTests(unittest.TestCase):
    def test_paste_delay_scales_with_text_length(self) -> None:
        short = TextInserter._paste_delay_for_text("x" * 120)
        long_text = TextInserter._paste_delay_for_text("x" * 1200)
        ultra = TextInserter._paste_delay_for_text("x" * 3200)
        self.assertGreater(long_text, short)
        self.assertGreaterEqual(ultra, long_text)

    def test_restore_delay_scales_for_long_dictation(self) -> None:
        short = TextInserter._restore_delay_for_text("x" * 120)
        long_text = TextInserter._restore_delay_for_text("x" * 1200)
        ultra = TextInserter._restore_delay_for_text("x" * 3200)
        self.assertGreater(long_text, short)
        self.assertGreaterEqual(ultra, long_text)

    def test_ultra_long_dictation_uses_safe_restore_window(self) -> None:
        delay = TextInserter._restore_delay_for_text("x" * 3200)
        self.assertGreaterEqual(delay, 3.4)


if __name__ == "__main__":
    unittest.main()
