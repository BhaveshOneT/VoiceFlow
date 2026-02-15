"""Enumerate audio input devices and detect iPhone Continuity Microphone."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import sounddevice as sd

log = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    id: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False
    is_iphone: bool = False


def list_input_devices() -> list[AudioDevice]:
    """Return all available audio input devices."""
    devices: list[AudioDevice] = []
    try:
        default_input = sd.default.device[0]  # (input, output) tuple
    except Exception:
        default_input = -1

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"]
        devices.append(AudioDevice(
            id=idx,
            name=name,
            channels=dev["max_input_channels"],
            sample_rate=dev["default_samplerate"],
            is_default=(idx == default_input),
            is_iphone=_is_iphone_device(name),
        ))
    return devices


def get_default_input_device() -> AudioDevice | None:
    """Return the system default input device."""
    for dev in list_input_devices():
        if dev.is_default:
            return dev
    devs = list_input_devices()
    return devs[0] if devs else None


def get_continuity_microphone() -> AudioDevice | None:
    """Return the iPhone Continuity Microphone device, if available."""
    for dev in list_input_devices():
        if dev.is_iphone:
            return dev
    return None


def _is_iphone_device(name: str) -> bool:
    lower = name.lower()
    return "iphone" in lower
