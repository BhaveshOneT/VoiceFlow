"""Settings page -- user preferences for dictation, audio, and transcription."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui_qt.styles.theme import Colors, Spacing

if TYPE_CHECKING:
    from app.config import AppConfig
    from app.core.signals import AppSignals

log = logging.getLogger(__name__)


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    return line


def _section_header(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("subheading")
    return label


def _setting_row(label_text: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(Spacing.LG)
    label = QLabel(label_text)
    label.setMinimumWidth(180)
    row.addWidget(label)
    row.addWidget(widget, 1)
    return row


class SettingsPage(QWidget):
    """Preferences panel for dictation, audio device, transcription, etc."""

    def __init__(
        self,
        config: "AppConfig",
        signals: "AppSignals",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._signals = signals

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        title = QLabel("Settings")
        title.setObjectName("heading")
        layout.addWidget(title)

        # -- Hotkey -------------------------------------------------------
        layout.addWidget(_section_header("Hotkey"))
        self._hotkey_combo = QComboBox()
        self._hotkey_combo.addItems([
            "Right Cmd", "Left Cmd", "Right Ctrl", "Left Ctrl",
            "Right Alt", "Left Alt", "Right Shift", "Left Shift",
        ])
        _key_map = {
            "right_cmd": 0, "left_cmd": 1, "right_ctrl": 2, "left_ctrl": 3,
            "right_alt": 4, "left_alt": 5, "right_shift": 6, "left_shift": 7,
        }
        self._hotkey_combo.setCurrentIndex(_key_map.get(config.hotkey, 0))
        self._hotkey_combo.currentIndexChanged.connect(self._on_hotkey_changed)
        layout.addLayout(_setting_row("Push-to-talk key", self._hotkey_combo))

        layout.addWidget(_separator())

        # -- Transcription Mode -------------------------------------------
        layout.addWidget(_section_header("Transcription"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Normal", "Programmer"])
        self._mode_combo.setCurrentIndex(0 if config.transcription_mode == "normal" else 1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addLayout(_setting_row("Mode", self._mode_combo))

        # -- Accuracy -----------------------------------------------------
        self._accuracy_combo = QComboBox()
        self._accuracy_combo.addItems(["Fast", "Standard", "Max Accuracy"])
        _acc_map = {"fast": 0, "standard": 1, "max_accuracy": 2}
        self._accuracy_combo.setCurrentIndex(_acc_map.get(config.cleanup_mode, 1))
        self._accuracy_combo.currentIndexChanged.connect(self._on_accuracy_changed)
        layout.addLayout(_setting_row("Accuracy", self._accuracy_combo))

        # -- Language -----------------------------------------------------
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(["Auto (English + German)", "English", "German"])
        _lang_map = {"auto": 0, "en": 1, "de": 2}
        self._lang_combo.setCurrentIndex(_lang_map.get(config.language, 0))
        self._lang_combo.currentIndexChanged.connect(self._on_language_changed)
        layout.addLayout(_setting_row("Language", self._lang_combo))

        layout.addWidget(_separator())

        # -- Meetings -----------------------------------------------------
        layout.addWidget(_section_header("Meetings"))

        self._auto_transcribe_cb = QCheckBox("Auto-transcribe when recording stops")
        self._auto_transcribe_cb.setChecked(config.auto_transcribe_on_stop)
        self._auto_transcribe_cb.toggled.connect(self._on_auto_transcribe_changed)
        layout.addWidget(self._auto_transcribe_cb)

        self._auto_summarize_cb = QCheckBox("Auto-summarize after transcription")
        self._auto_summarize_cb.setChecked(config.auto_summarize)
        self._auto_summarize_cb.toggled.connect(self._on_auto_summarize_changed)
        layout.addWidget(self._auto_summarize_cb)

        self._meeting_tx_provider_combo = QComboBox()
        self._meeting_tx_provider_combo.addItems(["Local (Parakeet)", "OpenAI API"])
        self._meeting_tx_provider_combo.setCurrentIndex(
            0 if config.meeting_transcription_provider == "local" else 1
        )
        self._meeting_tx_provider_combo.currentIndexChanged.connect(
            self._on_meeting_tx_provider_changed
        )
        layout.addLayout(_setting_row("Transcription Provider", self._meeting_tx_provider_combo))

        self._provider_combo = QComboBox()
        self._provider_combo.addItems(["Local (Qwen)", "OpenAI API"])
        self._provider_combo.setCurrentIndex(0 if config.summary_provider == "local" else 1)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        layout.addLayout(_setting_row("Summary Provider", self._provider_combo))

        api_key_row = QHBoxLayout()
        api_key_row.setSpacing(Spacing.SM)
        self._api_key_input = QLineEdit()
        self._api_key_input.setPlaceholderText("sk-...")
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setText(config.openai_api_key)
        self._api_key_input.editingFinished.connect(self._on_api_key_changed)
        self._api_key_input.textChanged.connect(self._validate_api_key)
        api_key_row.addWidget(self._api_key_input, 1)

        self._api_key_status = QLabel()
        self._api_key_status.setFixedWidth(20)
        api_key_row.addWidget(self._api_key_status)

        api_key_wrapper = QWidget()
        api_key_wrapper.setLayout(api_key_row)
        layout.addLayout(_setting_row("OpenAI API Key", api_key_wrapper))
        self._validate_api_key(config.openai_api_key)

        layout.addWidget(_separator())

        # -- System -------------------------------------------------------
        layout.addWidget(_section_header("System"))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(Spacing.MD)
        accessibility_btn = QPushButton("Open Accessibility Settings")
        accessibility_btn.clicked.connect(self._open_accessibility)
        btn_row.addWidget(accessibility_btn)

        microphone_btn = QPushButton("Open Microphone Settings")
        microphone_btn.clicked.connect(self._open_microphone)
        btn_row.addWidget(microphone_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(content)

    # -- Callbacks --------------------------------------------------------

    def _on_hotkey_changed(self, index: int) -> None:
        keys = [
            "right_cmd", "left_cmd", "right_ctrl", "left_ctrl",
            "right_alt", "left_alt", "right_shift", "left_shift",
        ]
        if 0 <= index < len(keys):
            self._config.hotkey = keys[index]
            self._config.save()
            self._signals.hotkey_changed.emit(keys[index])
            log.info("Hotkey changed to %s", self._config.hotkey)

    def _on_mode_changed(self, index: int) -> None:
        mode = "normal" if index == 0 else "programmer"
        self._config.transcription_mode = mode
        self._config.save()
        self._signals.transcription_mode_changed.emit(mode)
        log.info("Transcription mode changed to %s", mode)

    def _on_accuracy_changed(self, index: int) -> None:
        modes = ["fast", "standard", "max_accuracy"]
        if 0 <= index < len(modes):
            self._config.cleanup_mode = modes[index]
            self._config.save()
            self._signals.accuracy_changed.emit(modes[index])
            log.info("Accuracy mode changed to %s", self._config.cleanup_mode)

    def _on_language_changed(self, index: int) -> None:
        langs = ["auto", "en", "de"]
        if 0 <= index < len(langs):
            self._config.language = langs[index]
            self._config.save()
            self._signals.language_changed.emit(langs[index])
            log.info("Language changed to %s", self._config.language)

    def _on_auto_transcribe_changed(self, checked: bool) -> None:
        self._config.auto_transcribe_on_stop = checked
        self._config.save()
        log.info("Auto-transcribe on stop: %s", checked)

    def _on_auto_summarize_changed(self, checked: bool) -> None:
        self._config.auto_summarize = checked
        self._config.save()
        log.info("Auto-summarize: %s", checked)

    def _on_meeting_tx_provider_changed(self, index: int) -> None:
        self._config.meeting_transcription_provider = "local" if index == 0 else "openai"
        self._config.save()
        log.info(
            "Meeting transcription provider changed to %s",
            self._config.meeting_transcription_provider,
        )

    def _on_provider_changed(self, index: int) -> None:
        self._config.summary_provider = "local" if index == 0 else "openai"
        self._config.save()
        log.info("Summary provider changed to %s", self._config.summary_provider)

    def _on_api_key_changed(self) -> None:
        self._config.openai_api_key = self._api_key_input.text().strip()
        self._config.save()
        self._validate_api_key(self._api_key_input.text())
        log.info("OpenAI API key updated")

    def _validate_api_key(self, text: str) -> None:
        """Show green check or red X based on API key format."""
        text = text.strip()
        if not text:
            self._api_key_status.setText("")
            return
        valid = text.startswith("sk-") and 20 <= len(text) <= 200
        if valid:
            self._api_key_status.setText("\u2705")
            self._api_key_status.setToolTip("Valid API key format")
        else:
            self._api_key_status.setText("\u274C")
            self._api_key_status.setToolTip(
                "Invalid format: should start with 'sk-' and be 20-200 chars"
            )

    def _open_accessibility(self) -> None:
        self._open_system_settings("Privacy_Accessibility")

    def _open_microphone(self) -> None:
        self._open_system_settings("Privacy_Microphone")

    @staticmethod
    def _open_system_settings(pane: str) -> None:
        try:
            import AppKit
            url = f"x-apple.systempreferences:com.apple.preference.security?{pane}"
            ns_url = AppKit.NSURL.URLWithString_(url)
            if ns_url:
                AppKit.NSWorkspace.sharedWorkspace().openURL_(ns_url)
        except Exception:
            log.exception("Failed to open System Settings: %s", pane)
