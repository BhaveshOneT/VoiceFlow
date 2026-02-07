#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# build.sh — Build VoiceFlow.app and optionally create a DMG
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APP_NAME="VoiceFlow"
DIST="$ROOT/dist"
BUILD="$ROOT/build"
APP="$DIST/$APP_NAME.app"

echo "==> Cleaning previous builds..."
rm -rf "$DIST" "$BUILD"

echo "==> Building $APP_NAME.app with PyInstaller..."
pyinstaller VoiceFlow.spec --noconfirm 2>&1

# ── Verify bundle structure ──────────────────────────────────
echo ""
echo "==> Verifying app bundle..."

BINARY="$APP/Contents/MacOS/$APP_NAME"
PLIST="$APP/Contents/Info.plist"

errors=0

if [ ! -f "$BINARY" ]; then
    echo "  ERROR: Missing binary at $BINARY"
    errors=$((errors + 1))
else
    echo "  OK: Binary exists"
fi

if [ ! -f "$PLIST" ]; then
    echo "  ERROR: Missing Info.plist"
    errors=$((errors + 1))
else
    echo "  OK: Info.plist exists"
    # Verify microphone usage description is present
    if /usr/libexec/PlistBuddy -c "Print :NSMicrophoneUsageDescription" "$PLIST" &>/dev/null; then
        echo "  OK: NSMicrophoneUsageDescription present"
    else
        echo "  WARN: NSMicrophoneUsageDescription missing from plist"
    fi
fi

# Check for mlx_whisper assets (tokenizer files)
ASSETS_DIR=$(find "$APP" -type d -name "assets" -path "*/mlx_whisper/*" 2>/dev/null | head -1)
if [ -n "$ASSETS_DIR" ]; then
    echo "  OK: mlx_whisper assets found at $ASSETS_DIR"
else
    echo "  WARN: mlx_whisper assets directory not found (tokenizer may fail)"
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "BUILD FAILED: $errors error(s) found."
    exit 1
fi

echo ""
echo "==> $APP_NAME.app built successfully at: $APP"

# ── Optional DMG creation ────────────────────────────────────
if command -v create-dmg &>/dev/null; then
    DMG="$DIST/$APP_NAME.dmg"
    echo ""
    echo "==> Creating DMG installer..."
    create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 150 190 \
        --app-drop-link 450 190 \
        --no-internet-enable \
        "$DMG" \
        "$APP"
    echo "==> DMG created at: $DMG"
else
    echo ""
    echo "NOTE: Install 'create-dmg' (brew install create-dmg) to auto-generate a DMG."
    echo "      For now, you can run the app directly: open \"$APP\""
fi

echo ""
echo "Done."
