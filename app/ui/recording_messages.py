"""Timed microcopy for recording overlay."""
from __future__ import annotations


def message_for_elapsed(seconds: float) -> str:
    """Return overlay copy based on elapsed recording time."""
    if seconds < 20.0:
        return "Listening..."
    if seconds < 30.0:
        return "Locked in. Keep going..."
    if seconds < 60.0:
        return "Great flow. Keep going..."
    if seconds < 90.0:
        return "Nice detail. Keep going..."
    return "A bit faster if you can."
