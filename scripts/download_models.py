#!/usr/bin/env python3
"""Pre-download and cache all AI models for VoiceFlow.

Run this before first launch so the app starts instantly:
    python scripts/download_models.py

Models are cached in the HuggingFace Hub cache (~/.cache/huggingface/).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import ssl
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import certifi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Model definitions ────────────────────────────────────────────────────────

MODEL_SPECS = [
    {
        "repo_id": "mlx-community/parakeet-tdt-0.6b-v3",
        "revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    },
    {
        "repo_id": "mlx-community/parakeet-tdt-0.6b-v2",
        "revision": "8ae155301e23d820d82aa60d24817c900e69e487",
    },
    {
        "repo_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "revision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    },
]

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
ALLOWED_DOWNLOAD_HOSTS = {"github.com", "raw.githubusercontent.com"}
SILERO_VAD_PATH = Path.home() / ".cache" / "voiceflow" / "silero_vad.onnx"


# ── Download functions ───────────────────────────────────────────────────────

def download_hf_model(repo_id: str, revision: str) -> None:
    """Download a HuggingFace model using pinned revision."""
    from huggingface_hub import snapshot_download

    log.info("Downloading %s @ %s ...", repo_id, revision[:12])
    path = snapshot_download(repo_id, revision=revision)  # nosec B615
    log.info("  Cached at: %s", path)


def download_silero_vad() -> None:
    """Download the Silero VAD ONNX model."""
    if SILERO_VAD_PATH.exists() and _sha256_file(SILERO_VAD_PATH) == SILERO_VAD_SHA256:
        log.info("Silero VAD already cached and verified at %s", SILERO_VAD_PATH)
        return

    log.info("Downloading Silero VAD ONNX model ...")
    parsed = urlparse(SILERO_VAD_URL)
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_DOWNLOAD_HOSTS:
        raise RuntimeError(f"Blocked model URL: {SILERO_VAD_URL}")

    SILERO_VAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(SILERO_VAD_URL)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:  # nosec B310
        content = resp.read()

    digest = hashlib.sha256(content).hexdigest()
    if digest != SILERO_VAD_SHA256:
        raise RuntimeError("Silero VAD checksum mismatch; refusing unverified model.")

    tmp_path = SILERO_VAD_PATH.with_suffix(".onnx.tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(SILERO_VAD_PATH)
    log.info("  Cached at: %s", SILERO_VAD_PATH)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(include_max_accuracy: bool = True) -> None:
    """Download all required models."""
    models = list(MODEL_SPECS) if include_max_accuracy else [MODEL_SPECS[0], MODEL_SPECS[2]]

    log.info("=" * 60)
    log.info("VoiceFlow Model Downloader")
    log.info("=" * 60)

    # Silero VAD (tiny, ~2MB)
    download_silero_vad()

    # HuggingFace models
    for model in models:
        try:
            download_hf_model(model["repo_id"], model["revision"])
        except Exception as e:
            log.error("Failed to download %s: %s", model["repo_id"], e)
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
        help="Skip the secondary Parakeet model (saves ~650MB)",
    )
    args = parser.parse_args()
    download_all(include_max_accuracy=not args.skip_max_accuracy)


if __name__ == "__main__":
    main()
