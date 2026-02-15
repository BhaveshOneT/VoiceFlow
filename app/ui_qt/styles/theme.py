"""Color palette, typography, and spacing constants for VoiceFlow UI.

Designed for a dark macOS-native aesthetic.  All colours are hex strings
suitable for QSS stylesheets.  Grays follow Apple's systemGray scale for
an authentic dark-mode feel.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Colour palette (dark mode, macOS-native feel)
# ---------------------------------------------------------------------------

class Colors:
    # Backgrounds – layered from deepest to most elevated
    BG_PRIMARY = "#1C1C1E"        # Main window (systemGray6)
    BG_SECONDARY = "#2C2C2E"      # Sidebar, panels (systemGray5)
    BG_TERTIARY = "#3A3A3C"       # Cards, elevated surfaces (systemGray4)
    BG_HOVER = "#48484A"          # Hover states (systemGray3)
    BG_ACTIVE = "#545456"         # Active/selected states
    BG_INPUT = "#1C1C1E"          # Text input fields (recessed)

    # Text – Apple HIG dark-mode hierarchy
    TEXT_PRIMARY = "#F5F5F7"      # Primary text (near-white)
    TEXT_SECONDARY = "#98989D"    # Secondary / muted (systemGray)
    TEXT_TERTIARY = "#636366"     # Placeholder, disabled (systemGray2)
    TEXT_INVERSE = "#1C1C1E"      # Text on light / accent backgrounds

    # Accent – macOS system blue
    ACCENT = "#0A84FF"            # System blue
    ACCENT_HOVER = "#409CFF"      # Blue hover (lighter)
    ACCENT_PRESSED = "#0060D0"    # Blue pressed (darker)
    ACCENT_SUBTLE = "#0A84FF26"   # Blue at 15% opacity for subtle highlights

    # Semantic status
    SUCCESS = "#30D158"           # Green (systemGreen)
    WARNING = "#FF9F0A"           # Orange-amber (systemOrange)
    DANGER = "#FF453A"            # Red (systemRed)
    INFO = "#64D2FF"              # Cyan (systemCyan)

    # Recording status (aliases for clarity)
    STATUS_RECORDING = "#FF3B30"  # Red – recording active
    STATUS_PROCESSING = "#FF9F0A" # Amber – processing
    STATUS_READY = "#30D158"      # Green – ready
    STATUS_ERROR = "#FF453A"      # Red – error

    # Borders
    BORDER = "#38383A"            # Default border (Apple dark separator)
    BORDER_LIGHT = "#48484A"      # Hover/lighter border
    BORDER_FOCUS = "#0A84FF"      # Focus ring (accent blue)

    # Disabled
    DISABLED_TEXT = "#48484A"     # Disabled text (systemGray3)
    DISABLED_BG = "#2C2C2E"      # Disabled background

    # Separator
    SEPARATOR = "#38383A"

    # Focus ring
    FOCUS_RING = "#0A84FF"        # Keyboard focus outline


# ---------------------------------------------------------------------------
# Typography (SF-compatible system font stack for Qt on macOS)
# ---------------------------------------------------------------------------

class Typography:
    # .AppleSystemUIFont is what Qt resolves to SF Pro on macOS
    FONT_FAMILY = ".AppleSystemUIFont, -apple-system, 'SF Pro Text', 'Helvetica Neue', sans-serif"
    FONT_FAMILY_MONO = "'SF Mono', 'Menlo', 'Monaco', monospace"

    # Type scale (px)
    SIZE_DISPLAY = 28
    SIZE_H1 = 24
    SIZE_H2 = 20
    SIZE_H3 = 16
    SIZE_BODY = 14
    SIZE_CAPTION = 12
    SIZE_SMALL = 11

    # Weights (QSS font-weight)
    WEIGHT_REGULAR = "400"
    WEIGHT_MEDIUM = "500"
    WEIGHT_SEMIBOLD = "600"
    WEIGHT_BOLD = "700"

    # Letter spacing (px) – subtle tightening for large text
    SPACING_TIGHT = "-0.5px"
    SPACING_NORMAL = "0px"
    SPACING_WIDE = "0.5px"

    # Line height multipliers (for reference; QSS line-height is limited)
    LINE_HEIGHT_TIGHT = 1.2
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.75


# ---------------------------------------------------------------------------
# Spacing & geometry
# ---------------------------------------------------------------------------

class Spacing:
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32

    SIDEBAR_WIDTH = 220
    SIDEBAR_ITEM_HEIGHT = 36
    BORDER_RADIUS = 8
    BORDER_RADIUS_SM = 6
    BORDER_RADIUS_LG = 12
    CARD_PADDING = 16
