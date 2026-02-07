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
DMG="$DIST/$APP_NAME.dmg"
CODESIGN_IDENTITY="${VOICEFLOW_CODESIGN_IDENTITY:-}"
NOTARY_PROFILE="${VOICEFLOW_NOTARY_PROFILE:-}"

echo "==> Cleaning previous builds..."
rm -rf "$DIST" "$BUILD"

echo "==> Building $APP_NAME.app with PyInstaller..."
PYI_CMD=""
if command -v pyinstaller >/dev/null 2>&1; then
    PYI_CMD="pyinstaller"
elif [ -x "$ROOT/.venv/bin/pyinstaller" ]; then
    PYI_CMD="$ROOT/.venv/bin/pyinstaller"
else
    echo "ERROR: PyInstaller not found. Install it in PATH or at $ROOT/.venv/bin/pyinstaller"
    exit 1
fi

"$PYI_CMD" VoiceFlow.spec --noconfirm 2>&1

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

# ── Optional signing ────────────────────────────────────────────
if [ -n "$CODESIGN_IDENTITY" ]; then
    echo ""
    echo "==> Code signing app bundle..."
    /usr/bin/codesign \
        --force \
        --deep \
        --timestamp \
        --options runtime \
        --sign "$CODESIGN_IDENTITY" \
        "$APP"
    /usr/bin/codesign --verify --deep --strict "$APP"
    echo "==> App signed successfully with identity: $CODESIGN_IDENTITY"
else
    echo ""
    echo "==> Skipping code signing (set VOICEFLOW_CODESIGN_IDENTITY to enable)"
fi

# ── DMG creation ───────────────────────────────────────────────
echo ""
echo "==> Creating DMG installer..."
rm -f "$DMG"

if command -v create-dmg &>/dev/null; then
    if create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 150 190 \
        --app-drop-link 450 190 \
        --no-internet-enable \
        "$DMG" \
        "$APP"; then
        echo "==> DMG created at: $DMG"
    else
        echo "WARN: create-dmg failed; falling back to hdiutil DMG generation."
        rm -f "$DMG"
    fi
fi

if [ ! -f "$DMG" ]; then
    STAGING="$(mktemp -d "$BUILD/dmg-staging.XXXXXX")"
    cp -R "$APP" "$STAGING/"
    ln -s /Applications "$STAGING/Applications"
    hdiutil create \
        -volname "$APP_NAME" \
        -srcfolder "$STAGING" \
        -ov \
        -format UDZO \
        "$DMG" >/dev/null
    rm -rf "$STAGING"
    echo "==> DMG created at: $DMG (hdiutil fallback)"
fi

# ── Optional notarization + stapling ───────────────────────────
if [ -n "$NOTARY_PROFILE" ]; then
    if [ -z "$CODESIGN_IDENTITY" ]; then
        echo "ERROR: VOICEFLOW_NOTARY_PROFILE is set but VOICEFLOW_CODESIGN_IDENTITY is missing."
        exit 1
    fi
    echo ""
    echo "==> Submitting DMG for notarization (profile: $NOTARY_PROFILE)..."
    xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
    echo "==> Stapling notarization tickets..."
    xcrun stapler staple "$APP"
    xcrun stapler staple "$DMG"
    echo "==> Notarization and stapling complete."
else
    echo ""
    echo "==> Skipping notarization (set VOICEFLOW_NOTARY_PROFILE to enable)"
fi

echo ""
echo "Done."
