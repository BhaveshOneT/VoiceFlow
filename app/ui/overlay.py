"""Floating recording-indicator overlay for VoiceFlow.

Displays a WisprFlow-style pill at the bottom-center of the screen that shows
the current recording/processing state.  Built with AppKit so it stays on top
of all windows, ignores mouse events, and appears on every Space.
"""
from __future__ import annotations

import logging

import AppKit
import Quartz
from PyObjCTools import AppHelper

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PILL_WIDTH = 160
_PILL_HEIGHT = 36
_BOTTOM_MARGIN = 72  # distance from bottom of screen
_CORNER_RADIUS = _PILL_HEIGHT / 2

_DOT_DIAMETER = 10
_DOT_LEFT_PADDING = 16

_LABEL_FONT_SIZE = 13.0

_FADE_DURATION = 0.25
_PULSE_DURATION = 1.0


def _main_screen_frame() -> AppKit.NSRect:
    """Return the frame of the main screen (includes menu bar area)."""
    return AppKit.NSScreen.mainScreen().frame()


# ---------------------------------------------------------------------------
# RecordingOverlay
# ---------------------------------------------------------------------------

class RecordingOverlay:
    """Non-interactive floating pill overlay.

    All public methods are safe to call from any thread -- they dispatch
    to the main thread via ``AppHelper.callAfter``.
    """

    def __init__(self) -> None:
        self._panel: AppKit.NSPanel | None = None
        self._dot_layer: Quartz.CALayer | None = None
        self._label: AppKit.NSTextField | None = None
        self._built = False

    # ------------------------------------------------------------------
    # Lazy construction (must happen on the main thread)
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        """Create the panel + views if they haven't been created yet."""
        if self._built:
            return
        try:
            self._build()
            self._built = True
            log.info("Overlay panel built successfully")
        except Exception:
            log.exception("Failed to build overlay panel")

    def _build(self) -> None:
        screen = _main_screen_frame()
        x = (screen.size.width - _PILL_WIDTH) / 2
        y = _BOTTOM_MARGIN

        frame = AppKit.NSMakeRect(x, y, _PILL_WIDTH, _PILL_HEIGHT)

        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(AppKit.NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setIgnoresMouseEvents_(True)
        panel.setCanBecomeKeyWindow_(False)
        panel.setCanBecomeMainWindow_(False)
        panel.setHasShadow_(True)
        panel.setAnimationBehavior_(AppKit.NSWindowAnimationBehaviorNone)

        # Visible on all Spaces / full-screen apps
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # --- vibrancy background ---
        content_frame = AppKit.NSMakeRect(0, 0, _PILL_WIDTH, _PILL_HEIGHT)
        vibrancy = AppKit.NSVisualEffectView.alloc().initWithFrame_(content_frame)
        vibrancy.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
        vibrancy.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
        vibrancy.setState_(AppKit.NSVisualEffectStateActive)
        vibrancy.setWantsLayer_(True)
        vibrancy.layer().setCornerRadius_(_CORNER_RADIUS)
        vibrancy.layer().setMasksToBounds_(True)
        panel.contentView().addSubview_(vibrancy)

        # --- red dot (recording indicator) ---
        dot_y = (_PILL_HEIGHT - _DOT_DIAMETER) / 2
        dot_frame = AppKit.NSMakeRect(_DOT_LEFT_PADDING, dot_y, _DOT_DIAMETER, _DOT_DIAMETER)
        dot_view = AppKit.NSView.alloc().initWithFrame_(dot_frame)
        dot_view.setWantsLayer_(True)
        dot_layer = dot_view.layer()
        dot_layer.setBackgroundColor_(
            AppKit.NSColor.redColor().CGColor()
        )
        dot_layer.setCornerRadius_(_DOT_DIAMETER / 2)
        vibrancy.addSubview_(dot_view)
        self._dot_layer = dot_layer
        self._dot_view = dot_view

        # --- label ---
        label_x = _DOT_LEFT_PADDING + _DOT_DIAMETER + 8
        label_width = _PILL_WIDTH - label_x - 12
        label_frame = AppKit.NSMakeRect(label_x, 0, label_width, _PILL_HEIGHT)
        label = AppKit.NSTextField.labelWithString_("Recording")
        label.setFrame_(label_frame)
        label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(_LABEL_FONT_SIZE, AppKit.NSFontWeightMedium))
        label.setTextColor_(AppKit.NSColor.whiteColor())
        label.setAlignment_(AppKit.NSTextAlignmentLeft)
        label.setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
        vibrancy.addSubview_(label)
        self._label = label

        panel.setAlphaValue_(0.0)
        panel.orderFront_(None)

        self._panel = panel

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def show_recording(self) -> None:
        """Show the pill with a pulsing red dot and 'Recording' label."""
        AppHelper.callAfter(self._show_recording)

    def show_processing(self) -> None:
        """Switch the pill to 'Processing...' (no pulsing dot)."""
        AppHelper.callAfter(self._show_processing)

    def hide(self) -> None:
        """Fade out and hide the pill."""
        AppHelper.callAfter(self._hide)

    # ------------------------------------------------------------------
    # Main-thread implementations
    # ------------------------------------------------------------------

    def _show_recording(self) -> None:
        try:
            self._ensure_built()
            if not self._built:
                return
            self._label.setStringValue_("Recording")
            self._dot_view.setHidden_(False)
            self._start_pulse()
            self._fade_in()
        except Exception:
            log.exception("Error showing recording overlay")

    def _show_processing(self) -> None:
        try:
            self._ensure_built()
            if not self._built:
                return
            self._label.setStringValue_("Processing...")
            self._stop_pulse()
            self._dot_view.setHidden_(True)
            self._fade_in()
        except Exception:
            log.exception("Error showing processing overlay")

    def _hide(self) -> None:
        try:
            if not self._built or self._panel is None:
                return
            self._stop_pulse()
            self._fade_out()
        except Exception:
            log.exception("Error hiding overlay")

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def _fade_in(self) -> None:
        try:
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
            self._panel.animator().setAlphaValue_(1.0)
            AppKit.NSAnimationContext.endGrouping()
        except Exception:
            # Fallback: set alpha directly without animation
            log.debug("Animation fallback: setting alpha directly")
            self._panel.setAlphaValue_(1.0)

    def _fade_out(self) -> None:
        try:
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
            self._panel.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()
        except Exception:
            log.debug("Animation fallback: setting alpha directly")
            self._panel.setAlphaValue_(0.0)

    def _start_pulse(self) -> None:
        if self._dot_layer is None:
            return
        try:
            pulse = Quartz.CABasicAnimation.animationWithKeyPath_("opacity")
            pulse.setFromValue_(1.0)
            pulse.setToValue_(0.3)
            pulse.setDuration_(_PULSE_DURATION)
            pulse.setAutoreverses_(True)
            pulse.setRepeatCount_(float("inf"))
            pulse.setTimingFunction_(
                Quartz.CAMediaTimingFunction.functionWithName_(
                    Quartz.kCAMediaTimingFunctionEaseInEaseOut
                )
            )
            self._dot_layer.addAnimation_forKey_(pulse, "pulse")
        except Exception:
            log.debug("Pulse animation failed (non-fatal)")

    def _stop_pulse(self) -> None:
        if self._dot_layer is None:
            return
        try:
            self._dot_layer.removeAnimationForKey_("pulse")
            self._dot_layer.setOpacity_(1.0)
        except Exception:
            pass
