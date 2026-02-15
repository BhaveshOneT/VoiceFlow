"""Audio device selector dropdown with status indicators."""
from __future__ import annotations

import logging

from PySide6.QtWidgets import QComboBox, QWidget

from app.core.audio_device_manager import AudioDevice, list_input_devices

log = logging.getLogger(__name__)


class DeviceSelector(QComboBox):
    """Dropdown for selecting audio input device.

    Refreshes the device list each time the dropdown is opened,
    supporting hot-plug of devices (e.g. iPhone Continuity Mic).
    Marks the default device and shows channel count info.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._devices: list[AudioDevice] = []
        self._refresh_devices()

    def showPopup(self) -> None:  # noqa: N802
        """Refresh device list when the user opens the dropdown."""
        self._refresh_devices()
        super().showPopup()

    def selected_device_id(self) -> int | None:
        """Return the sounddevice device index, or None for default."""
        idx = self.currentIndex()
        if idx < 0 or idx >= len(self._devices):
            return None
        return self._devices[idx].id

    def _refresh_devices(self) -> None:
        current_id = self.selected_device_id()
        self.blockSignals(True)
        self.clear()
        self._devices = list_input_devices()

        select_idx = 0
        for i, dev in enumerate(self._devices):
            label = dev.name
            if dev.is_default:
                label += " \u2713 Default"
            if dev.is_iphone:
                label = f"\U0001F4F1 {label}"
            # Show channel count for multi-channel devices
            if dev.channels > 1:
                label += f" ({dev.channels}ch)"
            self.addItem(label)
            if current_id is not None and dev.id == current_id:
                select_idx = i

        if self._devices:
            self.setCurrentIndex(select_idx)
        self.blockSignals(False)
