"""Main application window with sidebar navigation."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.signals import AppSignals
from app.storage.audio_store import AudioStore
from app.storage.database import MeetingDatabase
from app.ui_qt.pages.dashboard_page import DashboardPage
from app.ui_qt.pages.meeting_detail_page import MeetingDetailPage
from app.ui_qt.pages.meetings_page import MeetingsPage
from app.ui_qt.pages.onboarding_page import OnboardingPage
from app.ui_qt.pages.recording_page import RecordingPage
from app.ui_qt.pages.settings_page import SettingsPage
from app.ui_qt.styles.theme import Spacing

if TYPE_CHECKING:
    from app.config import AppConfig

log = logging.getLogger(__name__)

# Sidebar nav items: (icon, label, page_index)
_NAV_ITEMS = [
    ("\u2302", "Dashboard"),    # ⌂
    ("\U0001F399", "Meetings"),  # 🎙
    ("\u23FA", "Recording"),    # ⏺
    ("\u2699", "Settings"),     # ⚙
]


class MainWindow(QWidget):
    """Top-level window: sidebar on the left, stacked content on the right.

    Uses QWidget (not QMainWindow) so we get a clean frameless-ish look
    while still having native title bar chrome on macOS.
    """

    PAGE_DASHBOARD = 0
    PAGE_MEETINGS = 1
    PAGE_RECORDING = 2
    PAGE_MEETING_DETAIL = 3
    PAGE_SETTINGS = 4

    def __init__(
        self,
        config: "AppConfig",
        signals: AppSignals,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._signals = signals

        self.setWindowTitle("VoiceFlow")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        # -- Root stack: onboarding | main content -------------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._root_stack = QStackedWidget()
        outer.addWidget(self._root_stack)

        # Storage layer (shared across pages)
        self._db = MeetingDatabase()
        self._audio_store = AudioStore()

        # Onboarding page (index 0)
        self._onboarding = OnboardingPage(config, signals)
        self._onboarding.completed.connect(self._finish_onboarding)
        self._root_stack.addWidget(self._onboarding)

        # Main content container (index 1): sidebar | content
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(Spacing.SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, Spacing.LG, 0, Spacing.LG)
        sidebar_layout.setSpacing(2)

        # App title in sidebar (styled via QSS #sidebar_title)
        app_title = QLabel("VoiceFlow")
        app_title.setObjectName("sidebar_title")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(app_title)

        # Navigation buttons with icons
        self._nav_buttons: list[QPushButton] = []
        for i, (icon, label) in enumerate(_NAV_ITEMS):
            btn = QPushButton(f"  {icon}  {label}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self._navigate(idx))
            sidebar_layout.addWidget(btn)
            self._nav_buttons.append(btn)

        sidebar_layout.addStretch()

        # Version at bottom of sidebar (styled via QSS #sidebar_version)
        version_label = QLabel("v1.0.0")
        version_label.setObjectName("sidebar_version")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(version_label)

        main_layout.addWidget(sidebar)

        # Content stack
        self._stack = QStackedWidget()

        self._dashboard = DashboardPage(signals, self._db)
        self._meetings = MeetingsPage(signals, self._db)
        self._recording = RecordingPage(signals, self._db, self._audio_store)
        self._meeting_detail = MeetingDetailPage(signals, self._db)
        self._settings = SettingsPage(config, signals)

        self._stack.addWidget(self._dashboard)      # index 0
        self._stack.addWidget(self._meetings)        # index 1
        self._stack.addWidget(self._recording)       # index 2
        self._stack.addWidget(self._meeting_detail)  # index 3
        self._stack.addWidget(self._settings)        # index 4

        # Wire dashboard "Start Meeting" button to navigate to recording page
        self._dashboard.start_meeting_button.clicked.connect(
            lambda: self._navigate(self.PAGE_RECORDING)
        )

        # Wire meetings page card click to detail page
        self._meetings.meeting_selected.connect(self._open_meeting_detail)

        # Wire detail page back button
        self._meeting_detail.back_requested.connect(
            lambda: self._navigate(self.PAGE_MEETINGS)
        )

        # Refresh meetings list when returning from recording
        signals.meeting_recording_stopped.connect(
            lambda _: self._meetings.refresh()
        )
        # Also refresh dashboard recent meetings
        signals.meeting_recording_stopped.connect(
            lambda _: self._dashboard.refresh_recent()
        )

        main_layout.addWidget(self._stack, 1)
        self._root_stack.addWidget(main_container)

        # Show onboarding or main content based on config
        if config.onboarding_complete:
            self._root_stack.setCurrentIndex(1)
            self._navigate(self.PAGE_DASHBOARD)
        else:
            self._root_stack.setCurrentIndex(0)

        # -- Keyboard shortcuts -------------------------------------------
        quit_action = QAction(self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self._quit_app)
        self.addAction(quit_action)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _finish_onboarding(self) -> None:
        """Called when the onboarding wizard completes."""
        self._config.onboarding_complete = True
        self._config.save()
        self._root_stack.setCurrentIndex(1)
        self._navigate(self.PAGE_DASHBOARD)
        log.info("Onboarding complete")

    def _navigate(self, page_index: int) -> None:
        self._stack.setCurrentIndex(page_index)
        # Meeting detail page doesn't have a sidebar button
        nav_mapping = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3}  # page -> nav button
        active_btn = nav_mapping.get(page_index, -1)
        for i, btn in enumerate(self._nav_buttons):
            btn.setProperty("active", "true" if i == active_btn else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        # Refresh meetings page when navigating to it
        if page_index == self.PAGE_MEETINGS:
            self._meetings.refresh()

    def _open_meeting_detail(self, meeting_id: str) -> None:
        self._meeting_detail.load_meeting(meeting_id)
        self._navigate(self.PAGE_MEETING_DETAIL)

    # ------------------------------------------------------------------
    # Window behaviour
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Hide window on close instead of quitting (tray keeps running)."""
        event.ignore()
        self.hide()
        log.info("Main window hidden (close to tray)")

    def _quit_app(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def dashboard(self) -> DashboardPage:
        return self._dashboard

    @property
    def meetings_page(self) -> MeetingsPage:
        return self._meetings

    @property
    def recording_page(self) -> RecordingPage:
        return self._recording

    @property
    def settings_page(self) -> SettingsPage:
        return self._settings
