# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for VoiceFlow.app.

Usage (via build script):
    ./scripts/build.sh

Or directly:
    pyinstaller VoiceFlow.spec
"""
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

ROOT = os.path.abspath(".")
SITE_PKGS = os.path.join(ROOT, ".venv", "lib", "python3.12", "site-packages")

# ---------------------------------------------------------------------------
# Collect problematic packages comprehensively.
#
# These packages use dynamic imports, namespace packages, native extensions,
# or runtime data files that PyInstaller's import tracer can't discover:
#
#   mlx        — namespace package (no __init__.py), native .so, libmlx.dylib,
#                Metal shaders (mlx.metallib)
#   mlx_lm     — dynamically imports model modules via importlib.import_module
#                based on model config (e.g. mlx_lm.models.qwen2)
#   mlx_whisper— has assets/ data dir with tokenizer files
#   transformers— massive dynamic model loading via Auto* classes
# ---------------------------------------------------------------------------

all_datas = []
all_binaries = []
all_hiddenimports = []

for pkg in ["mlx", "mlx_lm", "mlx_whisper", "transformers"]:
    datas, binaries, hiddenimports = collect_all(pkg)
    all_datas += datas
    all_binaries += binaries
    all_hiddenimports += hiddenimports

# App resources
app_resources = [("app/resources/default_dictionary.json", "resources")]

# sounddevice needs its PortAudio binary
sounddevice_data = os.path.join(SITE_PKGS, "_sounddevice_data")
sounddevice_datas = [(sounddevice_data, "_sounddevice_data")] if os.path.isdir(sounddevice_data) else []

# certifi CA bundle
certifi_datas = collect_data_files("certifi")

a = Analysis(
    ["app/main.py"],
    pathex=[ROOT],
    binaries=all_binaries,
    datas=app_resources + sounddevice_datas + certifi_datas + all_datas,
    hiddenimports=[
        "PyObjCTools.AppHelper",
        "AppKit",
        "Quartz",
        "HIServices",
        "ApplicationServices",
        "CoreFoundation",
        "sounddevice",
        "rumps",
        "certifi",
    ] + all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoiceFlow",
)

app = BUNDLE(
    coll,
    name="VoiceFlow.app",
    bundle_identifier="com.voiceflow.dictation",
    info_plist={
        "CFBundleName": "VoiceFlow",
        "CFBundleDisplayName": "VoiceFlow",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "13.0",
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": (
            "VoiceFlow needs microphone access to capture your speech for "
            "local AI transcription."
        ),
        "NSAppleEventsUsageDescription": (
            "VoiceFlow uses Apple Events to insert transcribed text into "
            "the active application."
        ),
    },
)
