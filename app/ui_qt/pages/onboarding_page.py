"""First-run onboarding wizard.

Steps:
  1. Welcome + feature tour
  2. Microphone permission request
  3. Accessibility permission request
  4. Hotkey configuration
  5. Ready!
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui_qt.styles.theme import Spacing

if TYPE_CHECKING:
    from app.config import AppConfig
    from app.core.signals import AppSignals

log = logging.getLogger(__name__)

# Step definitions: (icon, title, description)
_STEPS = [
    (
        "\U0001F3A4",
        "Welcome to VoiceFlow",
        "Local AI dictation and meeting intelligence for macOS.\n\n"
        "Dictate text anywhere with push-to-talk.\n"
        "Record meetings with speaker diarization.\n"
        "Get AI summaries and action items.",
    ),
    (
        "\U0001F3A4",
        "Microphone Access",
        "VoiceFlow needs microphone permission to capture your speech.\n"
        "Click 'Grant Access' to open System Settings.",
    ),
    (
        "\u2328",
        "Accessibility Access",
        "VoiceFlow needs Accessibility permission for hotkeys\n"
        "and auto-pasting transcribed text.",
    ),
    (
        "\u2699",
        "Configure Hotkey",
        "Choose which key to hold for push-to-talk dictation.",
    ),
    (
        "\u2705",
        "You're All Set!",
        "VoiceFlow is ready. Hold your hotkey to start dictating.\n"
        "Click 'Start Meeting' on the Dashboard to record meetings.",
    ),
]


class _WizardStep(QWidget):
    """Base class for a wizard step."""

    def __init__(self, icon: str, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XXL, Spacing.XXL, Spacing.XXL, Spacing.XXL)
        layout.setSpacing(Spacing.LG)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Large icon
        icon_label = QLabel(icon)
        icon_label.setObjectName("heading")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setObjectName("heading")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setObjectName("caption")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setMaximumWidth(500)
        layout.addWidget(desc_label)


class OnboardingPage(QWidget):
    """Multi-step first-run wizard."""

    completed = Signal()

    def __init__(
        self,
        config: "AppConfig",
        signals: "AppSignals",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._signals = signals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress indicator (step dots)
        self._progress_row = QHBoxLayout()
        self._progress_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_row.setSpacing(Spacing.SM)
        self._dots: list[QLabel] = []
        for i in range(len(_STEPS)):
            dot = QLabel()
            dot.setFixedSize(8, 8)
            dot.setObjectName("status_dot_ready" if i == 0 else "status_dot_processing")
            self._dots.append(dot)
            self._progress_row.addWidget(dot)
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, Spacing.LG, 0, 0)
        progress_layout.addLayout(self._progress_row)
        layout.addWidget(progress_widget)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        # Build steps
        for i, (icon, title, desc) in enumerate(_STEPS):
            step = _WizardStep(icon, title, desc)

            # Step-specific widgets
            if i == 1:  # Microphone
                mic_btn = QPushButton("Grant Microphone Access")
                mic_btn.setObjectName("primary_button")
                mic_btn.clicked.connect(self._request_microphone)
                step.layout().addWidget(mic_btn, 0, Qt.AlignmentFlag.AlignCenter)
            elif i == 2:  # Accessibility
                access_btn = QPushButton("Grant Accessibility Access")
                access_btn.setObjectName("primary_button")
                access_btn.clicked.connect(self._request_accessibility)
                step.layout().addWidget(access_btn, 0, Qt.AlignmentFlag.AlignCenter)
            elif i == 3:  # Hotkey
                self._hotkey_combo = QComboBox()
                self._hotkey_combo.addItems([
                    "Right Cmd (Recommended)", "Left Cmd", "Right Ctrl", "Left Ctrl",
                    "Right Alt", "Left Alt", "Right Shift", "Left Shift",
                ])
                self._hotkey_combo.setMinimumWidth(300)
                step.layout().addWidget(self._hotkey_combo, 0, Qt.AlignmentFlag.AlignCenter)

            self._stack.addWidget(step)

        # Navigation buttons
        nav = QHBoxLayout()
        nav.setContentsMargins(Spacing.XXL, 0, Spacing.XXL, Spacing.XL)

        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        self._back_btn.hide()
        nav.addWidget(self._back_btn)

        nav.addStretch()

        # Step counter label
        self._step_label = QLabel()
        self._step_label.setObjectName("caption")
        nav.addWidget(self._step_label)

        nav.addStretch()

        self._next_btn = QPushButton("Get Started")
        self._next_btn.setObjectName("primary_button")
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        layout.addLayout(nav)

        self._update_nav()

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx >= self._stack.count() - 1:
            self._finish()
            return
        self._stack.setCurrentIndex(idx + 1)
        self._update_nav()

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        total = self._stack.count()
        self._back_btn.setVisible(idx > 0)

        if idx >= total - 1:
            self._next_btn.setText("Finish")
        elif idx == 0:
            self._next_btn.setText("Get Started")
        else:
            self._next_btn.setText("Next")

        self._step_label.setText(f"Step {idx + 1} of {total}")

        # Update progress dots
        for i, dot in enumerate(self._dots):
            if i < idx:
                dot.setObjectName("status_dot_ready")  # completed = green
            elif i == idx:
                dot.setObjectName("status_dot_recording")  # current = accent
            else:
                dot.setObjectName("status_dot_processing")  # upcoming = muted
            dot.style().unpolish(dot)
            dot.style().polish(dot)

    def _finish(self) -> None:
        # Save hotkey selection
        keys = [
            "right_cmd", "left_cmd", "right_ctrl", "left_ctrl",
            "right_alt", "left_alt", "right_shift", "left_shift",
        ]
        idx = self._hotkey_combo.currentIndex()
        if 0 <= idx < len(keys):
            self._config.hotkey = keys[idx]
            self._config.save()
        self.completed.emit()

    @staticmethod
    def _request_microphone() -> None:
        try:
            import AppKit
            url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
            ns_url = AppKit.NSURL.URLWithString_(url)
            if ns_url:
                AppKit.NSWorkspace.sharedWorkspace().openURL_(ns_url)
        except Exception:
            log.exception("Failed to open Microphone settings")

    @staticmethod
    def _request_accessibility() -> None:
        try:
            import AppKit
            url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            ns_url = AppKit.NSURL.URLWithString_(url)
            if ns_url:
                AppKit.NSWorkspace.sharedWorkspace().openURL_(ns_url)
        except Exception:
            log.exception("Failed to open Accessibility settings")
