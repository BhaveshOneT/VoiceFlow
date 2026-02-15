"""Debounced search bar widget with search icon and focus styling."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from app.ui_qt.styles.theme import Colors, Spacing


class SearchBar(QWidget):
    """Search input with debounced signal emission (300ms).

    Features:
    - Built-in clear button (via Qt)
    - Search icon prefix
    - Focus/blur styling through QSS
    """

    search_changed = Signal(str)

    def __init__(self, placeholder: str = "Search...", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        # Search icon
        icon_label = QLabel("\U0001F50D")
        icon_label.setFixedWidth(20)
        icon_label.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; font-size: 14px;")
        layout.addWidget(icon_label)
        self._icon = icon_label

        self._input = QLineEdit()
        self._input.setPlaceholderText(placeholder)
        self._input.setClearButtonEnabled(True)
        self._input.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._input)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._emit_search)

    @property
    def text(self) -> str:
        return self._input.text()

    def clear(self) -> None:
        """Programmatically clear the search input."""
        self._input.clear()

    def _on_text_changed(self, text: str) -> None:
        self._debounce.start()
        # Tint icon when text is present
        if text:
            self._icon.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 14px;")
        else:
            self._icon.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; font-size: 14px;")

    def _emit_search(self) -> None:
        self.search_changed.emit(self._input.text().strip())
