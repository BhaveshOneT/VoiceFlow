"""py2app build configuration for VoiceFlow (legacy, prefer scripts/build.sh).

Usage:
    python setup.py py2app

Creates VoiceFlow.app in the dist/ directory.
Note: The primary build method uses PyInstaller via scripts/build.sh.
"""
import sys

sys.setrecursionlimit(5000)

from setuptools import setup

APP = ["app/main.py"]
DATA_FILES = [
    ("resources", ["app/resources/default_dictionary.json"]),
]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": None,  # TODO: Add app icon (VoiceFlow.icns)
    "plist": {
        "CFBundleName": "VoiceFlow",
        "CFBundleDisplayName": "VoiceFlow",
        "CFBundleIdentifier": "com.voiceflow.dictation",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "13.0",
        "LSUIElement": False,  # Windowed app with Dock icon
        "NSMicrophoneUsageDescription": (
            "VoiceFlow needs microphone access to capture your speech for "
            "local AI transcription."
        ),
        "NSAppleEventsUsageDescription": (
            "VoiceFlow uses Apple Events to insert transcribed text into "
            "the active application."
        ),
    },
    "includes": [
        "PyObjCTools.AppHelper",
        "sounddevice",
    ],
    "packages": [
        "app",
        "rumps",
        "numpy",
        "pynput",
        "onnxruntime",
        "parakeet_mlx",
        "mlx_whisper",
        "mlx_lm",
        "huggingface_hub",
        "transformers",
        "safetensors",
        "certifi",
    ],
    "excludes": [
        "tkinter",
        "test",
        "unittest",
        "email",
        "xmlrpc",
    ],
}

setup(
    name="VoiceFlow",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
