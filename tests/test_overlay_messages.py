from __future__ import annotations

import unittest

from app.ui.recording_messages import message_for_elapsed


class OverlayMessageTests(unittest.TestCase):
    def test_message_tiers(self) -> None:
        self.assertEqual(message_for_elapsed(0), "Listening...")
        self.assertEqual(message_for_elapsed(19.9), "Listening...")
        self.assertEqual(message_for_elapsed(20.0), "Locked in. Keep going...")
        self.assertEqual(message_for_elapsed(29.9), "Locked in. Keep going...")
        self.assertEqual(message_for_elapsed(30.0), "Great flow. Keep going...")
        self.assertEqual(message_for_elapsed(59.9), "Great flow. Keep going...")
        self.assertEqual(message_for_elapsed(60.0), "Nice detail. Keep going...")
        self.assertEqual(message_for_elapsed(89.9), "Nice detail. Keep going...")
        self.assertEqual(message_for_elapsed(90.0), "A bit faster if you can.")


if __name__ == "__main__":
    unittest.main()
