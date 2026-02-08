"""Floating recording-indicator overlay for VoiceFlow.

Displays a WisprFlow-style pill at the bottom-center of the screen that shows
the current recording/processing state.  Built with AppKit so it stays on top
of all windows, ignores mouse events, and appears on every Space.
"""
from __future__ import annotations

import logging
import math
import threading
import time

import AppKit
import Quartz
from PyObjCTools import AppHelper

from app.ui.recording_messages import message_for_elapsed

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PILL_WIDTH = 220
_PILL_HEIGHT = 44
_BOTTOM_MARGIN = 72  # distance from bottom of screen
_CORNER_RADIUS = _PILL_HEIGHT / 2
_PILL_MIN_WIDTH = 188
_PILL_MAX_WIDTH = 420
_TEXT_WIDTH_BUFFER = 30

_ICON_DIAMETER = 22
_ICON_LEFT_PADDING = 14
_ICON_TEXT_GAP = 10
_LABEL_RIGHT_PADDING = 16

_LABEL_FONT_SIZE = 13.0

_FADE_DURATION = 0.2
_PULSE_DURATION = 1.0
_LIFT_PIXELS = 6.0
_LABEL_TRANSITION_DURATION = 0.16


def _main_screen_frame() -> AppKit.NSRect:
    """Return the frame of the main screen (includes menu bar area)."""
    screen = AppKit.NSScreen.mainScreen()
    if screen is None:
        screens = AppKit.NSScreen.screens()
        if screens:
            screen = screens[0]
    if screen is None:
        raise RuntimeError("No screen available to place overlay")
    return screen.frame()


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
        self._dot_view: AppKit.NSView | None = None
        self._container_view: AppKit.NSView | None = None
        self._ring_layer: Quartz.CALayer | None = None
        self._eq_layers: list[Quartz.CALayer] = []
        self._spinner: AppKit.NSProgressIndicator | None = None
        self._container_layer: Quartz.CALayer | None = None
        self._built = False
        self._recording_started_at: float | None = None
        self._recording_token = 0
        self._last_label_text = ""

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
        panel.setHidesOnDeactivate_(False)
        panel.setFloatingPanel_(True)
        panel.setHasShadow_(True)
        panel.setAnimationBehavior_(AppKit.NSWindowAnimationBehaviorNone)
        try:
            panel.setAppearance_(
                AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark)
            )
        except Exception:
            log.debug("Failed to force dark overlay appearance")

        # Visible on all Spaces / full-screen apps
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # --- vibrancy background (with fallback to solid view) ---
        content_frame = AppKit.NSMakeRect(0, 0, _PILL_WIDTH, _PILL_HEIGHT)
        container: AppKit.NSView
        try:
            vibrancy = AppKit.NSVisualEffectView.alloc().initWithFrame_(content_frame)
            vibrancy.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
            vibrancy.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            vibrancy.setState_(AppKit.NSVisualEffectStateActive)
            vibrancy.setWantsLayer_(True)
            vibrancy.layer().setCornerRadius_(_CORNER_RADIUS)
            vibrancy.layer().setMasksToBounds_(True)
            vibrancy.layer().setBorderWidth_(0.8)
            vibrancy.layer().setBorderColor_(
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.18).CGColor()
            )
            vibrancy.layer().setShadowColor_(AppKit.NSColor.blackColor().CGColor())
            vibrancy.layer().setShadowOpacity_(0.22)
            vibrancy.layer().setShadowRadius_(12.0)
            vibrancy.layer().setShadowOffset_(AppKit.NSMakeSize(0, -3))
            panel.contentView().addSubview_(vibrancy)
            container = vibrancy
        except Exception:
            log.exception("Failed to create vibrancy view; using solid fallback")
            solid = AppKit.NSView.alloc().initWithFrame_(content_frame)
            solid.setWantsLayer_(True)
            solid.layer().setBackgroundColor_(
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.09, 0.9).CGColor()
            )
            solid.layer().setCornerRadius_(_CORNER_RADIUS)
            solid.layer().setMasksToBounds_(True)
            solid.layer().setBorderWidth_(0.8)
            solid.layer().setBorderColor_(
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.16).CGColor()
            )
            panel.contentView().addSubview_(solid)
            container = solid
        self._container_view = container
        self._container_layer = container.layer()

        # --- red dot (recording indicator) ---
        dot_y = (_PILL_HEIGHT - _ICON_DIAMETER) / 2
        dot_frame = AppKit.NSMakeRect(
            _ICON_LEFT_PADDING, dot_y, _ICON_DIAMETER, _ICON_DIAMETER
        )
        dot_view = AppKit.NSView.alloc().initWithFrame_(dot_frame)
        dot_view.setWantsLayer_(True)
        dot_layer = dot_view.layer()
        dot_layer.setBackgroundColor_(
            AppKit.NSColor.systemRedColor().CGColor()
        )
        dot_layer.setCornerRadius_(_ICON_DIAMETER / 2)
        dot_layer.setMasksToBounds_(False)
        dot_layer.setShadowColor_(AppKit.NSColor.systemRedColor().CGColor())
        dot_layer.setShadowOpacity_(0.4)
        dot_layer.setShadowRadius_(6.0)
        dot_layer.setShadowOffset_(AppKit.NSMakeSize(0, 0))
        ring_layer = Quartz.CALayer.layer()
        ring_layer.setFrame_(dot_view.bounds())
        ring_layer.setBorderWidth_(2.0)
        ring_layer.setBorderColor_(
            AppKit.NSColor.systemRedColor().colorWithAlphaComponent_(0.7).CGColor()
        )
        ring_layer.setCornerRadius_(_ICON_DIAMETER / 2)
        ring_layer.setOpacity_(0.0)
        dot_layer.addSublayer_(ring_layer)
        self._ring_layer = ring_layer

        # Equalizer bars look more dynamic than a static glyph.
        self._eq_layers = []
        bar_w = 3.0
        bar_gap = 2.6
        bar_h = 10.0
        total_w = (bar_w * 3) + (bar_gap * 2)
        start_x = (_ICON_DIAMETER - total_w) / 2
        base_y = (_ICON_DIAMETER - bar_h) / 2
        for idx in range(3):
            bar = Quartz.CALayer.layer()
            bar.setFrame_(
                AppKit.NSMakeRect(
                    start_x + idx * (bar_w + bar_gap),
                    base_y,
                    bar_w,
                    bar_h,
                )
            )
            bar.setCornerRadius_(bar_w / 2)
            bar.setBackgroundColor_(
                AppKit.NSColor.whiteColor().colorWithAlphaComponent_(0.95).CGColor()
            )
            dot_layer.addSublayer_(bar)
            self._eq_layers.append(bar)

        container.addSubview_(dot_view)
        self._dot_layer = dot_layer
        self._dot_view = dot_view

        spinner = AppKit.NSProgressIndicator.alloc().initWithFrame_(
            AppKit.NSMakeRect(_ICON_LEFT_PADDING, dot_y, _ICON_DIAMETER, _ICON_DIAMETER)
        )
        spinner.setStyle_(AppKit.NSProgressIndicatorSpinningStyle)
        spinner.setDisplayedWhenStopped_(False)
        spinner.setControlSize_(AppKit.NSControlSizeSmall)
        spinner.stopAnimation_(None)
        spinner.setHidden_(True)
        container.addSubview_(spinner)
        self._spinner = spinner

        # --- label ---
        label_x = _ICON_LEFT_PADDING + _ICON_DIAMETER + _ICON_TEXT_GAP
        label_width = _PILL_WIDTH - label_x - _LABEL_RIGHT_PADDING
        label_height = 18
        label_y = (_PILL_HEIGHT - label_height) / 2
        label_frame = AppKit.NSMakeRect(label_x, label_y, label_width, label_height)
        label = AppKit.NSTextField.labelWithString_("Listening...")
        label.setFrame_(label_frame)
        label.setFont_(
            AppKit.NSFont.systemFontOfSize_weight_(
                _LABEL_FONT_SIZE, AppKit.NSFontWeightMedium
            )
        )
        label.setWantsLayer_(True)
        label.setTextColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.98, 1.0))
        label.setAlignment_(AppKit.NSTextAlignmentLeft)
        label.setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
        label.setUsesSingleLineMode_(True)
        try:
            label.layer().setShadowColor_(AppKit.NSColor.blackColor().CGColor())
            label.layer().setShadowOpacity_(0.78)
            label.layer().setShadowRadius_(2.2)
            label.layer().setShadowOffset_(AppKit.NSMakeSize(0, -1))
        except Exception:
            log.debug("Failed to apply label shadow")
        container.addSubview_(label)
        self._label = label
        self._last_label_text = "Listening..."

        panel.setAlphaValue_(0.0)
        panel.orderFront_(None)

        self._panel = panel
        self._update_layout_for_text("Listening...", animated=False)

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
            self._recording_started_at = time.monotonic()
            self._recording_token += 1
            token = self._recording_token
            self._set_recording_message(0.0, animated=False)
            self._dot_view.setHidden_(False)
            if self._spinner is not None:
                self._spinner.stopAnimation_(None)
                self._spinner.setHidden_(True)
            self._start_pulse()
            self._fade_in()
            self._schedule_recording_tick(token)
        except Exception:
            log.exception("Error showing recording overlay")

    def _show_processing(self) -> None:
        try:
            self._ensure_built()
            if not self._built:
                return
            self._recording_started_at = None
            self._recording_token += 1
            self._set_label_text("Transcribing...", animated=True)
            self._stop_pulse()
            if self._dot_view is not None:
                self._dot_view.setHidden_(True)
            if self._spinner is not None:
                self._spinner.setHidden_(False)
                self._spinner.startAnimation_(None)
            self._fade_in()
        except Exception:
            log.exception("Error showing processing overlay")

    def _hide(self) -> None:
        try:
            if not self._built or self._panel is None:
                return
            self._recording_started_at = None
            self._recording_token += 1
            self._stop_pulse()
            if self._spinner is not None:
                self._spinner.stopAnimation_(None)
                self._spinner.setHidden_(True)
            self._fade_out()
        except Exception:
            log.exception("Error hiding overlay")

    def _schedule_recording_tick(self, token: int) -> None:
        def _tick() -> None:
            AppHelper.callAfter(self._recording_tick, token)

        timer = threading.Timer(1.0, _tick)
        timer.daemon = True
        timer.start()

    def _recording_tick(self, token: int) -> None:
        if token != self._recording_token:
            return
        if self._recording_started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self._recording_started_at)
        self._set_recording_message(elapsed, animated=True)
        self._schedule_recording_tick(token)

    def _set_recording_message(self, elapsed_seconds: float, animated: bool) -> None:
        self._set_label_text(message_for_elapsed(elapsed_seconds), animated=animated)

    def _set_label_text(self, text: str, animated: bool) -> None:
        if self._label is None:
            return
        if text == self._last_label_text:
            return
        self._last_label_text = text
        if animated:
            try:
                self._label.setAlphaValue_(0.0)
            except Exception:
                pass
        self._label.setStringValue_(text)
        self._update_layout_for_text(text, animated=animated)
        if animated:
            try:
                AppKit.NSAnimationContext.beginGrouping()
                AppKit.NSAnimationContext.currentContext().setDuration_(
                    _LABEL_TRANSITION_DURATION
                )
                self._label.animator().setAlphaValue_(1.0)
                AppKit.NSAnimationContext.endGrouping()
            except Exception:
                self._label.setAlphaValue_(1.0)

    def _update_layout_for_text(self, text: str, animated: bool) -> None:
        if self._panel is None or self._label is None or self._container_view is None:
            return
        font = self._label.font() or AppKit.NSFont.systemFontOfSize_(_LABEL_FONT_SIZE)
        attrs = {AppKit.NSFontAttributeName: font}
        text_size = AppKit.NSString.stringWithString_(text).sizeWithAttributes_(attrs)
        desired_width = (
            _ICON_LEFT_PADDING
            + _ICON_DIAMETER
            + _ICON_TEXT_GAP
            + int(math.ceil(text_size.width))
            + _LABEL_RIGHT_PADDING
            + _TEXT_WIDTH_BUFFER
        )
        width = max(_PILL_MIN_WIDTH, min(_PILL_MAX_WIDTH, desired_width))

        panel_frame = self._panel.frame()
        if (
            int(panel_frame.size.width) != int(width)
            or int(panel_frame.size.height) != int(_PILL_HEIGHT)
        ):
            screen = _main_screen_frame()
            new_frame = AppKit.NSMakeRect(
                (screen.size.width - width) / 2,
                _BOTTOM_MARGIN,
                width,
                _PILL_HEIGHT,
            )
            self._panel.setFrame_display_(new_frame, True)

        container_frame = AppKit.NSMakeRect(0, 0, width, _PILL_HEIGHT)
        self._container_view.setFrame_(container_frame)
        if self._container_layer is not None:
            self._container_layer.setCornerRadius_(_PILL_HEIGHT / 2)

        icon_y = (_PILL_HEIGHT - _ICON_DIAMETER) / 2
        icon_frame = AppKit.NSMakeRect(
            _ICON_LEFT_PADDING,
            icon_y,
            _ICON_DIAMETER,
            _ICON_DIAMETER,
        )
        if self._dot_view is not None:
            self._dot_view.setFrame_(icon_frame)
        if self._spinner is not None:
            self._spinner.setFrame_(icon_frame)

        label_x = _ICON_LEFT_PADDING + _ICON_DIAMETER + _ICON_TEXT_GAP
        label_h = 18
        label_w = max(56, width - label_x - _LABEL_RIGHT_PADDING)
        label_y = (_PILL_HEIGHT - label_h) / 2
        self._label.setFrame_(AppKit.NSMakeRect(label_x, label_y, label_w, label_h))

        if animated and self._container_layer is not None:
            try:
                pop = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
                pop.setFromValue_(1.0)
                pop.setToValue_(1.015)
                pop.setDuration_(0.12)
                pop.setAutoreverses_(True)
                pop.setRepeatCount_(1)
                self._container_layer.addAnimation_forKey_(pop, "labelPop")
            except Exception:
                log.debug("Label pop animation skipped")

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def _fade_in(self) -> None:
        try:
            if self._panel is None:
                return
            try:
                self._panel.orderFrontRegardless()
            except Exception:
                self._panel.orderFront_(None)
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
            self._panel.animator().setAlphaValue_(1.0)
            AppKit.NSAnimationContext.endGrouping()
            self._animate_entrance()
        except Exception:
            # Fallback: set alpha directly without animation
            log.debug("Animation fallback: setting alpha directly")
            if self._panel is not None:
                self._panel.setAlphaValue_(1.0)

    def _fade_out(self) -> None:
        try:
            if self._panel is None:
                return
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(_FADE_DURATION)
            self._panel.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()
            if self._container_layer is not None:
                self._container_layer.removeAnimationForKey_("overlayEntrance")
        except Exception:
            log.debug("Animation fallback: setting alpha directly")
            if self._panel is not None:
                self._panel.setAlphaValue_(0.0)

    def _animate_entrance(self) -> None:
        if self._container_layer is None:
            return
        try:
            scale = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
            scale.setFromValue_(0.965)
            scale.setToValue_(1.0)
            scale.setDuration_(0.22)
            lift = Quartz.CABasicAnimation.animationWithKeyPath_("transform.translation.y")
            lift.setFromValue_(-_LIFT_PIXELS)
            lift.setToValue_(0.0)
            lift.setDuration_(0.22)
            group = Quartz.CAAnimationGroup.animation()
            group.setAnimations_([scale, lift])
            group.setDuration_(0.22)
            group.setTimingFunction_(
                Quartz.CAMediaTimingFunction.functionWithName_(
                    Quartz.kCAMediaTimingFunctionEaseOut
                )
            )
            self._container_layer.addAnimation_forKey_(group, "overlayEntrance")
        except Exception:
            log.debug("Overlay entrance animation failed (non-fatal)")

    def _start_pulse(self) -> None:
        if self._dot_layer is None:
            return
        try:
            icon_pulse = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
            icon_pulse.setFromValue_(1.0)
            icon_pulse.setToValue_(1.08)
            icon_pulse.setDuration_(0.65)
            icon_pulse.setAutoreverses_(True)
            icon_pulse.setRepeatCount_(float("inf"))
            self._dot_layer.addAnimation_forKey_(icon_pulse, "iconPulse")
            if self._container_layer is not None:
                breathe = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
                breathe.setFromValue_(1.0)
                breathe.setToValue_(1.018)
                breathe.setDuration_(1.4)
                breathe.setAutoreverses_(True)
                breathe.setRepeatCount_(float("inf"))
                self._container_layer.addAnimation_forKey_(breathe, "containerBreathe")

            if self._ring_layer is not None:
                ring_scale = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
                ring_scale.setFromValue_(1.0)
                ring_scale.setToValue_(1.8)
                ring_scale.setDuration_(_PULSE_DURATION)
                ring_scale.setRepeatCount_(float("inf"))
                ring_fade = Quartz.CABasicAnimation.animationWithKeyPath_("opacity")
                ring_fade.setFromValue_(0.9)
                ring_fade.setToValue_(0.0)
                ring_fade.setDuration_(_PULSE_DURATION)
                ring_fade.setRepeatCount_(float("inf"))
                group = Quartz.CAAnimationGroup.animation()
                group.setAnimations_([ring_scale, ring_fade])
                group.setDuration_(_PULSE_DURATION)
                group.setRepeatCount_(float("inf"))
                group.setTimingFunction_(
                    Quartz.CAMediaTimingFunction.functionWithName_(
                        Quartz.kCAMediaTimingFunctionEaseOut
                    )
                )
                self._ring_layer.addAnimation_forKey_(group, "ringPulse")
                self._ring_layer.setOpacity_(1.0)
            for idx, bar in enumerate(self._eq_layers):
                wave = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale.y")
                wave.setFromValue_(0.32 + (idx * 0.08))
                wave.setToValue_(1.0)
                wave.setDuration_(0.34 + (idx * 0.07))
                wave.setAutoreverses_(True)
                wave.setRepeatCount_(float("inf"))
                wave.setTimingFunction_(
                    Quartz.CAMediaTimingFunction.functionWithName_(
                        Quartz.kCAMediaTimingFunctionEaseInEaseOut
                    )
                )
                wave.setBeginTime_(Quartz.CACurrentMediaTime() + (idx * 0.07))
                bar.addAnimation_forKey_(wave, f"eqWave{idx}")
        except Exception:
            log.debug("Pulse animation failed (non-fatal)")

    def _stop_pulse(self) -> None:
        if self._dot_layer is None:
            return
        try:
            self._dot_layer.removeAnimationForKey_("iconPulse")
            self._dot_layer.setOpacity_(1.0)
            self._dot_layer.setTransform_(Quartz.CATransform3DIdentity)
            if self._container_layer is not None:
                self._container_layer.removeAnimationForKey_("containerBreathe")
                self._container_layer.setTransform_(Quartz.CATransform3DIdentity)
            if self._ring_layer is not None:
                self._ring_layer.removeAnimationForKey_("ringPulse")
                self._ring_layer.setOpacity_(0.0)
                self._ring_layer.setTransform_(Quartz.CATransform3DIdentity)
            for idx, bar in enumerate(self._eq_layers):
                bar.removeAnimationForKey_(f"eqWave{idx}")
                bar.setTransform_(Quartz.CATransform3DIdentity)
        except Exception as exc:
            log.debug("Failed to stop pulse animation cleanly: %s", exc)
