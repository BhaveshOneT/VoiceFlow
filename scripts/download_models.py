#!/usr/bin/env python3
"""Pre-download and cache all AI models for VoiceFlow.

Run this before first launch so the app starts instantly:
    python scripts/download_models.py

Models are cached in the HuggingFace Hub cache (~/.cache/huggingface/).
"""
from __future__ import annotations

import argparse
import logging
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Model definitions ────────────────────────────────────────────────────────

WHISPER_MODELS = [
    "mlx-community/whisper-large-v3-turbo",      # default (standard + fast)
    "mlx-community/whisper-large-v3",             # max_accuracy mode
]

LLM_MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",  # text refiner
]

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_PATH = Path.home() / ".cache" / "voiceflow" / "silero_vad.onnx"


# ── Download functions ───────────────────────────────────────────────────────

def download_hf_model(repo_id: str) -> None:
    """Download a HuggingFace model using snapshot_download (supports resume)."""
    from huggingface_hub import snapshot_download

    log.info("Downloading %s ...", repo_id)
    path = snapshot_download(repo_id)
    log.info("  Cached at: %s", path)


def download_silero_vad() -> None:
    """Download the Silero VAD ONNX model."""
    if SILERO_VAD_PATH.exists():
        log.info("Silero VAD already cached at %s", SILERO_VAD_PATH)
        return

    log.info("Downloading Silero VAD ONNX model ...")
    SILERO_VAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(SILERO_VAD_URL)
    with urllib.request.urlopen(req, context=ctx) as resp:
        SILERO_VAD_PATH.write_bytes(resp.read())
    log.info("  Cached at: %s", SILERO_VAD_PATH)


def download_all(include_max_accuracy: bool = True) -> None:
    """Download all required models."""
    models = list(WHISPER_MODELS) if include_max_accuracy else WHISPER_MODELS[:1]
    models.extend(LLM_MODELS)

    log.info("=" * 60)
    log.info("VoiceFlow Model Downloader")
    log.info("=" * 60)

    # Silero VAD (tiny, ~2MB)
    download_silero_vad()

    # HuggingFace models
    for repo_id in models:
        try:
            download_hf_model(repo_id)
        except Exception as e:
            log.error("Failed to download %s: %s", repo_id, e)
            sys.exit(1)

    log.info("=" * 60)
    log.info("All models downloaded successfully!")
    log.info("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoiceFlow AI models")
    parser.add_argument(
        "--skip-max-accuracy",
        action="store_true",
        help="Skip the large whisper-large-v3 model (saves ~3GB)",
    )
    args = parser.parse_args()
    download_all(include_max_accuracy=not args.skip_max_accuracy)


if __name__ == "__main__":
    main()
