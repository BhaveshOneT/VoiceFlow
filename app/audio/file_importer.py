"""Import audio files (.wav, .mp3, .m4a, .aac, .flac, .ogg) as 16kHz mono WAV.

Uses ffmpeg subprocess for non-WAV formats and soundfile for WAV resampling.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
TARGET_SAMPLE_RATE = 16000


class AudioFileImporter:
    """Converts various audio formats to 16kHz mono WAV."""

    @staticmethod
    def import_file(source: Path, target_dir: Path) -> Path:
        """Convert *source* to 16kHz mono WAV in *target_dir*.

        Returns the path to the output WAV file.

        Raises:
            ValueError: If the file extension is unsupported.
            FileNotFoundError: If the source file doesn't exist.
            RuntimeError: If ffmpeg is needed but not installed.
        """
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        ext = source.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {ext}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / "recording.wav"

        if ext == ".wav":
            return AudioFileImporter._convert_wav(source, output)
        return AudioFileImporter._convert_with_ffmpeg(source, output)

    @staticmethod
    def _convert_wav(source: Path, output: Path) -> Path:
        """Resample a WAV file to 16kHz mono using soundfile."""
        try:
            import soundfile as sf
            import numpy as np

            data, sr = sf.read(source, dtype="float32")
            # Convert to mono if stereo
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Simple resampling if needed
            if sr != TARGET_SAMPLE_RATE:
                # Linear interpolation resampling (good enough for speech)
                duration = len(data) / sr
                target_len = int(duration * TARGET_SAMPLE_RATE)
                indices = np.linspace(0, len(data) - 1, target_len)
                data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
            sf.write(str(output), data, TARGET_SAMPLE_RATE, subtype="PCM_16")
            log.info("WAV imported: %s -> %s", source, output)
            return output
        except ImportError:
            # Fall back to ffmpeg
            return AudioFileImporter._convert_with_ffmpeg(source, output)

    @staticmethod
    def _convert_with_ffmpeg(source: Path, output: Path) -> Path:
        """Convert any audio file to 16kHz mono WAV via ffmpeg."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required to import non-WAV audio files. "
                "Install it with: brew install ffmpeg"
            )

        cmd = [
            ffmpeg, "-y", "-i", str(source),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            str(output),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")

        log.info("Audio imported via ffmpeg: %s -> %s", source, output)
        return output

    @staticmethod
    def is_supported(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS
