"""QSS stylesheets for VoiceFlow -- dark macOS-native aesthetic.

Every interactive element has hover, pressed, focus, and disabled states.
Selectors use objectName (#id) or dynamic property ([class="card"]) to
keep styling declarative and out of widget code.
"""
from __future__ import annotations

from app.ui_qt.styles.theme import Colors, Spacing, Typography


def app_stylesheet() -> str:
    """Return the global QSS applied to QApplication."""
    return f"""
/* ── Global ────────────────────────────────────────── */
* {{
    font-family: {Typography.FONT_FAMILY};
}}

QWidget {{
    background-color: {Colors.BG_PRIMARY};
    color: {Colors.TEXT_PRIMARY};
    font-size: {Typography.SIZE_BODY}px;
    outline: none;
}}

/* ── Main Window ───────────────────────────────────── */
QMainWindow {{
    background-color: {Colors.BG_PRIMARY};
}}

/* ══════════════════════════════════════════════════════
   SIDEBAR  (macOS Finder-style navigation)
   ══════════════════════════════════════════════════════ */
#sidebar {{
    background-color: {Colors.BG_SECONDARY};
    border-right: 1px solid {Colors.SEPARATOR};
    padding: {Spacing.SM}px 0px;
}}

/* Sidebar app title */
#sidebar QLabel#sidebar_title {{
    font-size: {Typography.SIZE_H3}px;
    font-weight: {Typography.WEIGHT_SEMIBOLD};
    color: {Colors.TEXT_PRIMARY};
    letter-spacing: {Typography.SPACING_TIGHT};
    padding: {Spacing.SM}px {Spacing.MD}px {Spacing.LG}px {Spacing.MD}px;
    background-color: transparent;
}}

/* Sidebar nav buttons – Finder-like */
#sidebar QPushButton {{
    background-color: transparent;
    border: none;
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    color: {Colors.TEXT_SECONDARY};
    font-size: {Typography.SIZE_BODY}px;
    font-weight: {Typography.WEIGHT_MEDIUM};
    padding: {Spacing.SM}px {Spacing.MD}px;
    text-align: left;
    min-height: {Spacing.SIDEBAR_ITEM_HEIGHT}px;
    margin: 1px {Spacing.SM}px;
}}

#sidebar QPushButton:hover {{
    background-color: {Colors.BG_HOVER};
    color: {Colors.TEXT_PRIMARY};
}}

#sidebar QPushButton:pressed {{
    background-color: {Colors.BG_ACTIVE};
    color: {Colors.TEXT_PRIMARY};
}}

#sidebar QPushButton:focus {{
    border: 2px solid {Colors.FOCUS_RING};
    padding: {Spacing.SM - 2}px {Spacing.MD - 2}px;
}}

/* Active sidebar item – highlighted like macOS sidebar selection */
#sidebar QPushButton[active="true"] {{
    background-color: {Colors.ACCENT_SUBTLE};
    color: {Colors.ACCENT};
    font-weight: {Typography.WEIGHT_SEMIBOLD};
}}

#sidebar QPushButton[active="true"]:hover {{
    background-color: {Colors.ACCENT};
    color: white;
}}

/* Sidebar version label */
#sidebar QLabel#sidebar_version {{
    color: {Colors.TEXT_TERTIARY};
    font-size: {Typography.SIZE_SMALL}px;
    background-color: transparent;
}}

/* ══════════════════════════════════════════════════════
   CARDS  (elevated surface panels)
   ══════════════════════════════════════════════════════ */
.card {{
    background-color: {Colors.BG_TERTIARY};
    border: 1px solid {Colors.BORDER};
    border-radius: {Spacing.BORDER_RADIUS}px;
    padding: {Spacing.CARD_PADDING}px;
}}

.card:hover {{
    border-color: {Colors.BORDER_LIGHT};
    background-color: #3E3E40;
}}

/* ══════════════════════════════════════════════════════
   BUTTONS  (full hierarchy: default, primary, danger)
   ══════════════════════════════════════════════════════ */

/* -- Default / secondary buttons ---------------------- */
QPushButton {{
    background-color: {Colors.BG_TERTIARY};
    border: 1px solid {Colors.BORDER};
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    color: {Colors.TEXT_PRIMARY};
    padding: {Spacing.SM}px {Spacing.LG}px;
    font-size: {Typography.SIZE_BODY}px;
    font-weight: {Typography.WEIGHT_MEDIUM};
    min-height: 28px;
}}

QPushButton:hover {{
    background-color: {Colors.BG_HOVER};
    border-color: {Colors.BORDER_LIGHT};
}}

QPushButton:pressed {{
    background-color: {Colors.BG_ACTIVE};
    border-color: {Colors.BORDER_LIGHT};
}}

QPushButton:focus {{
    border: 2px solid {Colors.FOCUS_RING};
    padding: {Spacing.SM - 1}px {Spacing.LG - 1}px;
}}

QPushButton:disabled {{
    background-color: {Colors.DISABLED_BG};
    border-color: {Colors.BORDER};
    color: {Colors.DISABLED_TEXT};
}}

/* -- Primary button (accent blue, filled) ------------- */
QPushButton#primary_button {{
    background-color: {Colors.ACCENT};
    border: 1px solid {Colors.ACCENT};
    color: white;
    font-weight: {Typography.WEIGHT_SEMIBOLD};
}}

QPushButton#primary_button:hover {{
    background-color: {Colors.ACCENT_HOVER};
    border-color: {Colors.ACCENT_HOVER};
}}

QPushButton#primary_button:pressed {{
    background-color: {Colors.ACCENT_PRESSED};
    border-color: {Colors.ACCENT_PRESSED};
}}

QPushButton#primary_button:focus {{
    border: 2px solid white;
}}

QPushButton#primary_button:disabled {{
    background-color: {Colors.ACCENT};
    border-color: {Colors.ACCENT};
    color: rgba(255, 255, 255, 0.4);
}}

/* -- Danger button (destructive actions) -------------- */
QPushButton#danger_button {{
    background-color: {Colors.DANGER};
    border: 1px solid {Colors.DANGER};
    color: white;
    font-weight: {Typography.WEIGHT_SEMIBOLD};
}}

QPushButton#danger_button:hover {{
    background-color: #FF6961;
    border-color: #FF6961;
}}

QPushButton#danger_button:pressed {{
    background-color: #CC3A32;
    border-color: #CC3A32;
}}

QPushButton#danger_button:focus {{
    border: 2px solid white;
}}

QPushButton#danger_button:disabled {{
    background-color: {Colors.DANGER};
    border-color: {Colors.DANGER};
    color: rgba(255, 255, 255, 0.4);
}}

/* -- Ghost / link button (transparent bg) ------------- */
QPushButton#ghost_button {{
    background-color: transparent;
    border: none;
    color: {Colors.ACCENT};
    font-weight: {Typography.WEIGHT_MEDIUM};
    padding: {Spacing.XS}px {Spacing.SM}px;
}}

QPushButton#ghost_button:hover {{
    background-color: {Colors.ACCENT_SUBTLE};
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
}}

QPushButton#ghost_button:pressed {{
    color: {Colors.ACCENT_PRESSED};
}}

/* ══════════════════════════════════════════════════════
   LABELS  (text hierarchy)
   ══════════════════════════════════════════════════════ */
QLabel {{
    background-color: transparent;
    border: none;
}}

QLabel#heading {{
    font-size: {Typography.SIZE_H2}px;
    font-weight: {Typography.WEIGHT_BOLD};
    color: {Colors.TEXT_PRIMARY};
    letter-spacing: {Typography.SPACING_TIGHT};
}}

QLabel#subheading {{
    font-size: {Typography.SIZE_H3}px;
    font-weight: {Typography.WEIGHT_SEMIBOLD};
    color: {Colors.TEXT_PRIMARY};
}}

QLabel#caption {{
    font-size: {Typography.SIZE_CAPTION}px;
    color: {Colors.TEXT_SECONDARY};
}}

/* ── Meeting title (editable inline) ─────────────── */
QLineEdit#meeting_title_edit {{
    font-size: {Typography.SIZE_H2}px;
    font-weight: {Typography.WEIGHT_SEMIBOLD};
    border: 1px solid transparent;
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    padding: {Spacing.XS}px;
    background-color: transparent;
}}

QLineEdit#meeting_title_edit:hover {{
    border-color: {Colors.BORDER};
    background-color: {Colors.BG_TERTIARY};
}}

QLineEdit#meeting_title_edit:focus {{
    border: 2px solid {Colors.FOCUS_RING};
    padding: {Spacing.XS - 1}px;
    background-color: {Colors.BG_INPUT};
}}

/* ══════════════════════════════════════════════════════
   TEXT INPUTS  (QLineEdit, QTextEdit)
   ══════════════════════════════════════════════════════ */
QLineEdit, QTextEdit {{
    background-color: {Colors.BG_INPUT};
    border: 1px solid {Colors.BORDER};
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    color: {Colors.TEXT_PRIMARY};
    padding: {Spacing.SM}px {Spacing.MD}px;
    selection-background-color: {Colors.ACCENT};
    selection-color: white;
    min-height: 28px;
}}

QLineEdit:hover, QTextEdit:hover {{
    border-color: {Colors.BORDER_LIGHT};
}}

QLineEdit:focus, QTextEdit:focus {{
    border: 2px solid {Colors.BORDER_FOCUS};
    padding: {Spacing.SM - 1}px {Spacing.MD - 1}px;
}}

QLineEdit:disabled, QTextEdit:disabled {{
    background-color: {Colors.DISABLED_BG};
    color: {Colors.DISABLED_TEXT};
    border-color: {Colors.BORDER};
}}

QLineEdit::placeholder {{
    color: {Colors.TEXT_TERTIARY};
}}

/* ══════════════════════════════════════════════════════
   COMBOBOX  (native-feeling dropdown)
   ══════════════════════════════════════════════════════ */
QComboBox {{
    background-color: {Colors.BG_TERTIARY};
    border: 1px solid {Colors.BORDER};
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    color: {Colors.TEXT_PRIMARY};
    padding: {Spacing.SM}px {Spacing.MD}px;
    padding-right: 28px;
    min-height: 28px;
}}

QComboBox:hover {{
    border-color: {Colors.BORDER_LIGHT};
    background-color: {Colors.BG_HOVER};
}}

QComboBox:focus {{
    border: 2px solid {Colors.FOCUS_RING};
    padding: {Spacing.SM - 1}px {Spacing.MD - 1}px;
    padding-right: 27px;
}}

QComboBox:disabled {{
    background-color: {Colors.DISABLED_BG};
    color: {Colors.DISABLED_TEXT};
    border-color: {Colors.BORDER};
}}

/* Drop-down arrow area */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border: none;
    border-left: 1px solid {Colors.BORDER};
    border-top-right-radius: {Spacing.BORDER_RADIUS_SM}px;
    border-bottom-right-radius: {Spacing.BORDER_RADIUS_SM}px;
    background-color: transparent;
}}

QComboBox::down-arrow {{
    image: none;
    width: 10px;
    height: 10px;
    border-left: 2px solid {Colors.TEXT_SECONDARY};
    border-bottom: 2px solid {Colors.TEXT_SECONDARY};
    margin-right: 4px;
    margin-bottom: 3px;
}}

/* Dropdown popup list */
QComboBox QAbstractItemView {{
    background-color: {Colors.BG_SECONDARY};
    border: 1px solid {Colors.BORDER_LIGHT};
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    color: {Colors.TEXT_PRIMARY};
    padding: {Spacing.XS}px;
    selection-background-color: {Colors.ACCENT};
    selection-color: white;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    padding: {Spacing.SM}px {Spacing.MD}px;
    border-radius: {Spacing.XS}px;
    min-height: 24px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {Colors.BG_HOVER};
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {Colors.ACCENT};
    color: white;
}}

/* ══════════════════════════════════════════════════════
   CHECKBOX  (custom check indicator)
   ══════════════════════════════════════════════════════ */
QCheckBox {{
    spacing: {Spacing.SM}px;
    color: {Colors.TEXT_PRIMARY};
    font-size: {Typography.SIZE_BODY}px;
    background-color: transparent;
}}

QCheckBox:disabled {{
    color: {Colors.DISABLED_TEXT};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {Colors.BORDER_LIGHT};
    border-radius: 4px;
    background-color: {Colors.BG_INPUT};
}}

QCheckBox::indicator:hover {{
    border-color: {Colors.ACCENT};
    background-color: {Colors.BG_HOVER};
}}

QCheckBox::indicator:checked {{
    background-color: {Colors.ACCENT};
    border-color: {Colors.ACCENT};
    image: none;
}}

QCheckBox::indicator:checked:hover {{
    background-color: {Colors.ACCENT_HOVER};
    border-color: {Colors.ACCENT_HOVER};
}}

QCheckBox::indicator:disabled {{
    background-color: {Colors.DISABLED_BG};
    border-color: {Colors.BORDER};
}}

QCheckBox:focus {{
    outline: none;
}}

/* ══════════════════════════════════════════════════════
   PROGRESS BAR  (thin accent-colored)
   ══════════════════════════════════════════════════════ */
QProgressBar {{
    background-color: {Colors.BG_TERTIARY};
    border: none;
    border-radius: 4px;
    max-height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {Colors.ACCENT};
    border-radius: 4px;
}}

/* ══════════════════════════════════════════════════════
   SCROLL AREA + SCROLLBAR  (thin macOS-style)
   ══════════════════════════════════════════════════════ */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

/* Vertical */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px 1px 2px 1px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {Colors.BORDER};
    border-radius: 3px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {Colors.BORDER_LIGHT};
}}

QScrollBar::handle:vertical:pressed {{
    background-color: {Colors.TEXT_TERTIARY};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

/* Horizontal */
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 1px 2px 1px 2px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: {Colors.BORDER};
    border-radius: 3px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {Colors.BORDER_LIGHT};
}}

QScrollBar::handle:horizontal:pressed {{
    background-color: {Colors.TEXT_TERTIARY};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ══════════════════════════════════════════════════════
   SEPARATOR  (horizontal rules)
   ══════════════════════════════════════════════════════ */
QFrame[frameShape="4"] {{
    color: {Colors.SEPARATOR};
    max-height: 1px;
    background-color: {Colors.SEPARATOR};
}}

/* ══════════════════════════════════════════════════════
   TOOLTIP
   ══════════════════════════════════════════════════════ */
QToolTip {{
    background-color: {Colors.BG_SECONDARY};
    border: 1px solid {Colors.BORDER_LIGHT};
    color: {Colors.TEXT_PRIMARY};
    padding: {Spacing.XS}px {Spacing.SM}px;
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    font-size: {Typography.SIZE_CAPTION}px;
}}

/* ══════════════════════════════════════════════════════
   STATUS INDICATORS  (shape + color for accessibility)
   ══════════════════════════════════════════════════════ */

/* Ready: filled circle + green */
#status_dot_ready {{
    background-color: {Colors.STATUS_READY};
    border: 2px solid {Colors.STATUS_READY};
    border-radius: 7px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}

/* Recording: pulsing red ring (outer ring effect) */
#status_dot_recording {{
    background-color: {Colors.STATUS_RECORDING};
    border: 2px solid #FF6B6B;
    border-radius: 7px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}

/* Processing: amber diamond-ish (square with small radius) */
#status_dot_processing {{
    background-color: {Colors.STATUS_PROCESSING};
    border: 2px solid {Colors.STATUS_PROCESSING};
    border-radius: 3px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}

/* Error: red square (no radius) */
#status_dot_error {{
    background-color: {Colors.STATUS_ERROR};
    border: 2px solid {Colors.STATUS_ERROR};
    border-radius: 0px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}

/* ══════════════════════════════════════════════════════
   STACKED WIDGET (no extra borders)
   ══════════════════════════════════════════════════════ */
QStackedWidget {{
    background-color: {Colors.BG_PRIMARY};
    border: none;
}}

/* ══════════════════════════════════════════════════════
   TAB WIDGET (if used in future)
   ══════════════════════════════════════════════════════ */
QTabWidget::pane {{
    border: 1px solid {Colors.BORDER};
    border-radius: {Spacing.BORDER_RADIUS_SM}px;
    background-color: {Colors.BG_PRIMARY};
}}

QTabBar::tab {{
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: {Colors.TEXT_SECONDARY};
    padding: {Spacing.SM}px {Spacing.LG}px;
    font-weight: {Typography.WEIGHT_MEDIUM};
}}

QTabBar::tab:hover {{
    color: {Colors.TEXT_PRIMARY};
    background-color: {Colors.BG_HOVER};
}}

QTabBar::tab:selected {{
    color: {Colors.ACCENT};
    border-bottom-color: {Colors.ACCENT};
    font-weight: {Typography.WEIGHT_SEMIBOLD};
}}
"""
