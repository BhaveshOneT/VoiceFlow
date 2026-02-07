#!/usr/bin/env python3
"""Generate VoiceFlow app icon assets.

Creates:
- app/resources/icons/voiceflow-logo-1024.png
- app/resources/icons/VoiceFlow.iconset/*
- app/resources/icons/voiceflow.icns
"""
from __future__ import annotations

from pathlib import Path

import AppKit

ROOT = Path(__file__).resolve().parents[1]
ICONS_DIR = ROOT / "app" / "resources" / "icons"
MASTER_PNG = ICONS_DIR / "voiceflow-logo-1024.png"
ICONSET_DIR = ICONS_DIR / "VoiceFlow.iconset"
ICNS_PATH = ICONS_DIR / "voiceflow.icns"


def _nscolor(r: int, g: int, b: int, a: float = 1.0) -> AppKit.NSColor:
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
        r / 255.0, g / 255.0, b / 255.0, a
    )


def _draw_gradient_polygon(
    points: list[tuple[float, float]],
    start: AppKit.NSColor,
    end: AppKit.NSColor,
    angle: float,
) -> None:
    path = AppKit.NSBezierPath.bezierPath()
    path.moveToPoint_(points[0])
    for point in points[1:]:
        path.lineToPoint_(point)
    path.closePath()

    gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(start, end)
    AppKit.NSGraphicsContext.saveGraphicsState()
    path.addClip()
    gradient.drawInBezierPath_angle_(path, angle)
    AppKit.NSGraphicsContext.restoreGraphicsState()


def generate_master_png(output: Path) -> None:
    size = 1024
    image = AppKit.NSImage.alloc().initWithSize_(AppKit.NSMakeSize(size, size))
    image.lockFocus()

    AppKit.NSColor.blackColor().setFill()
    AppKit.NSBezierPath.fillRect_(AppKit.NSMakeRect(0, 0, size, size))

    g1 = _nscolor(120, 225, 92)
    g2 = _nscolor(49, 172, 78)
    g3 = _nscolor(82, 206, 84)

    # V left wing (top-left to bottom-center)
    _draw_gradient_polygon(
        [
            (285, 600),
            (355, 600),
            (510, 345),
            (572, 450),
            (436, 680),
            (372, 680),
        ],
        g1,
        g2,
        42.0,
    )

    # F diagonal stem
    _draw_gradient_polygon(
        [
            (560, 600),
            (662, 600),
            (542, 394),
            (452, 210),
            (365, 210),
            (510, 470),
        ],
        g1,
        g2,
        42.0,
    )

    # F top bar
    _draw_gradient_polygon(
        [
            (548, 600),
            (748, 600),
            (716, 546),
            (518, 546),
        ],
        g3,
        g2,
        0.0,
    )

    # F mid bar
    _draw_gradient_polygon(
        [
            (536, 498),
            (676, 498),
            (648, 444),
            (505, 444),
        ],
        g3,
        g2,
        0.0,
    )

    # Top hook line
    hook_path = AppKit.NSBezierPath.bezierPath()
    hook_path.setLineWidth_(30.0)
    hook_path.setLineCapStyle_(AppKit.NSRoundLineCapStyle)
    hook_path.setLineJoinStyle_(AppKit.NSRoundLineJoinStyle)
    hook_path.moveToPoint_((468, 562))
    hook_path.lineToPoint_((468, 760))
    hook_path.lineToPoint_((700, 760))
    hook_path.lineToPoint_((700, 640))
    _nscolor(95, 214, 87).setStroke()
    hook_path.stroke()

    image.unlockFocus()

    tiff_data = image.TIFFRepresentation()
    bitmap = AppKit.NSBitmapImageRep.imageRepWithData_(tiff_data)
    png_data = bitmap.representationUsingType_properties_(
        AppKit.NSBitmapImageFileTypePNG, {}
    )
    png_data.writeToFile_atomically_(str(output), True)


def _run(cmd: list[str]) -> None:
    import subprocess

    subprocess.run(cmd, check=True)


def generate_iconset(master: Path, iconset_dir: Path, icns_path: Path) -> None:
    iconset_dir.mkdir(parents=True, exist_ok=True)
    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for size, name in sizes:
        out = iconset_dir / name
        _run(
            [
                "/usr/bin/sips",
                "-z",
                str(size),
                str(size),
                str(master),
                "--out",
                str(out),
            ]
        )
    _run(
        [
            "/usr/bin/iconutil",
            "-c",
            "icns",
            str(iconset_dir),
            "-o",
            str(icns_path),
        ]
    )


def main() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    generate_master_png(MASTER_PNG)
    generate_iconset(MASTER_PNG, ICONSET_DIR, ICNS_PATH)
    print(f"Generated: {MASTER_PNG}")
    print(f"Generated: {ICNS_PATH}")


if __name__ == "__main__":
    main()
