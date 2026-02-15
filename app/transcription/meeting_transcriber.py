"""Orchestrates meeting transcription: local STT + pyannote diarization + alignment.

Pipeline:
  1. Local STT transcribes audio into timestamped segments.
  2. pyannote diarizes audio into speaker-labeled time regions.
  3. Alignment merges the two: each transcript segment gets the speaker with
     the greatest time overlap from diarization.
  4. Results are persisted to the SQLite database.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import soundfile as sf

from app.storage.models import (
    MeetingStatus,
    Speaker,
    TranscriptSegment,
)
from app.transcription.diarization import DiarizationSegment, SpeakerDiarizer

if TYPE_CHECKING:
    from app.core.model_manager import ModelManager
    from app.core.signals import AppSignals
    from app.storage.database import MeetingDatabase
    from app.transcription.whisper_engine import WhisperEngine

log = logging.getLogger(__name__)

# Speaker color palette
_SPEAKER_COLORS = [
    "#0A84FF",  # Blue
    "#FF9500",  # Orange
    "#30D158",  # Green
    "#FF375F",  # Pink
    "#BF5AF2",  # Purple
    "#FFD60A",  # Yellow
    "#64D2FF",  # Teal
    "#FF6482",  # Coral
]


class MeetingTranscriber:
    """Orchestrates full meeting processing pipeline.

    Progress is reported as 0.0-1.0:
      0.0  - 0.45  transcription (local STT)
      0.45 - 0.80  diarization (pyannote)
      0.80 - 0.95  alignment + DB write
      0.95 - 1.00  cleanup
    """

    def __init__(
        self,
        stt: "WhisperEngine",
        diarizer: Optional[SpeakerDiarizer],
        model_manager: "ModelManager",
        db: "MeetingDatabase",
        signals: "AppSignals | None" = None,
        transcription_provider: str = "local",
        openai_api_key: str = "",
    ) -> None:
        self._stt = stt
        self._diarizer = diarizer
        self._model_manager = model_manager
        self._db = db
        self._signals = signals
        self._transcription_provider = transcription_provider
        self._openai_api_key = openai_api_key

    def process_meeting(
        self,
        meeting_id: str,
        audio_path: Path,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Run the full transcription + diarization pipeline for a meeting.

        Updates meeting status in the database throughout.
        """
        def _emit(progress: float) -> None:
            if on_progress:
                on_progress(progress)
            if self._signals:
                self._signals.meeting_transcription_progress.emit(meeting_id, progress)

        try:
            # -- Phase 1: Transcription ----------------------------------------
            self._db.update_meeting(meeting_id, status=MeetingStatus.TRANSCRIBING)
            _emit(0.05)

            if self._transcription_provider == "openai" and self._openai_api_key:
                log.info("Transcribing via OpenAI Whisper API: %s", audio_path)
                _emit(0.10)
                stt_result = self._transcribe_openai(audio_path)
            else:
                log.info("Loading audio for local transcription: %s", audio_path)
                audio_data, sr = sf.read(str(audio_path), dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                # Resample to 16kHz if needed
                if sr != 16000:
                    duration = len(audio_data) / sr
                    target_len = int(duration * 16000)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data) - 1, target_len),
                        np.arange(len(audio_data)),
                        audio_data,
                    ).astype(np.float32)

                self._model_manager.prepare_for_meeting_transcription()
                _emit(0.10)

                log.info("Transcribing meeting audio (local)...")
                stt_result = self._stt.transcribe_with_segments(audio_data)

            stt_segments = stt_result.get("segments", [])
            log.info("Transcription produced %d segments", len(stt_segments))
            _emit(0.45)

            # -- Phase 2: Speaker diarization (optional) ----------------------
            diar_segments: list[DiarizationSegment] = []
            diarization_ok = False

            if self._diarizer is not None:
                self._db.update_meeting(meeting_id, status=MeetingStatus.DIARIZING)
                self._model_manager.prepare_for_diarization()
                _emit(0.50)

                if self._diarizer.load():
                    log.info("Running speaker diarization...")
                    diar_segments = self._diarizer.diarize(audio_path)
                    diarization_ok = True
                else:
                    log.warning("Diarization unavailable — skipping, single-speaker fallback")
            else:
                log.info("No diarizer provided — skipping diarization")

            _emit(0.80)

            # -- Phase 3: Alignment + DB write --------------------------------
            if diarization_ok and diar_segments:
                log.info("Aligning transcription with diarization...")
                speakers_set: set[str] = {s.speaker for s in diar_segments}
            else:
                log.info("Using single-speaker fallback (no diarization)")
                speakers_set = {"Speaker 1"}

            speaker_map: dict[str, Speaker] = {}
            for idx, label in enumerate(sorted(speakers_set)):
                speaker = Speaker(
                    id=str(uuid.uuid4()),
                    meeting_id=meeting_id,
                    label=label,
                    color=_SPEAKER_COLORS[idx % len(_SPEAKER_COLORS)],
                )
                speaker_map[label] = speaker
                self._db.add_speaker(speaker)

            # Align each STT segment to a speaker.
            transcript_segments: list[TranscriptSegment] = []
            default_label = "Speaker 1"
            for ws in stt_segments:
                start = ws.get("start", 0.0)
                end = ws.get("end", 0.0)
                text = ws.get("text", "").strip()
                if not text:
                    continue

                if diarization_ok and diar_segments:
                    speaker_label = self._find_best_speaker(start, end, diar_segments)
                else:
                    speaker_label = default_label
                speaker = speaker_map.get(speaker_label)

                transcript_segments.append(TranscriptSegment(
                    id=str(uuid.uuid4()),
                    meeting_id=meeting_id,
                    speaker_id=speaker.id if speaker else "",
                    speaker_label=speaker_label,
                    start_time=start,
                    end_time=end,
                    text=text,
                    confidence=ws.get("avg_logprob", 0.0),
                ))

            self._db.add_segments(transcript_segments)
            _emit(0.95)

            # -- Finalize ---------------------------------------------------
            self._db.update_meeting(meeting_id, status=MeetingStatus.COMPLETE)
            _emit(1.0)

            if self._signals:
                self._signals.meeting_transcription_complete.emit(meeting_id)

            log.info(
                "Meeting transcription complete: %d segments, %d speakers",
                len(transcript_segments), len(speakers_set),
            )

        except Exception as exc:
            log.exception("Meeting transcription failed: %s", meeting_id)
            self._db.update_meeting(
                meeting_id,
                status=MeetingStatus.ERROR,
                error_message=str(exc),
            )
            raise

    # -- OpenAI Whisper API transcription ---------------------------------

    _OPENAI_MAX_FILE_BYTES = 24 * 1024 * 1024  # 24 MB safety margin (limit is 25 MB)

    def _transcribe_openai(self, audio_path: Path) -> dict:
        """Transcribe audio via OpenAI Whisper API.

        For files >24MB, splits into chunks and concatenates results.
        Returns dict matching local STT format:
        ``{"text": "...", "segments": [{"start", "end", "text"}, ...]}``
        """
        file_size = audio_path.stat().st_size
        if file_size <= self._OPENAI_MAX_FILE_BYTES:
            return self._openai_transcribe_file(audio_path, time_offset=0.0)

        # Split large files into chunks
        log.info("Audio file is %d bytes (>24MB); splitting into chunks", file_size)
        return self._openai_transcribe_chunked(audio_path)

    def _openai_transcribe_file(self, file_path: Path, time_offset: float = 0.0) -> dict:
        """Send a single audio file to OpenAI Whisper API."""
        import urllib.request

        boundary = uuid.uuid4().hex
        content_type = f"multipart/form-data; boundary={boundary}"

        # Build multipart body
        parts: list[bytes] = []
        # model field
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"whisper-1\r\n".encode()
        )
        # response_format field
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
            f"verbose_json\r\n".encode()
        )
        # timestamp_granularities field
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="timestamp_granularities[]"\r\n\r\n'
            f"segment\r\n".encode()
        )
        # file field
        file_data = file_path.read_bytes()
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n".encode()
            + file_data
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())

        body = b"".join(parts)

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._openai_api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )

        log.info("Sending audio to OpenAI Whisper API (%d bytes)...", len(file_data))
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        # Normalize to local STT format with optional time offset.
        segments = []
        for seg in data.get("segments", []):
            segments.append({
                "start": seg.get("start", 0.0) + time_offset,
                "end": seg.get("end", 0.0) + time_offset,
                "text": seg.get("text", ""),
            })

        return {"text": data.get("text", ""), "segments": segments}

    def _openai_transcribe_chunked(self, audio_path: Path) -> dict:
        """Split a large audio file into <24MB WAV chunks and transcribe each."""
        import tempfile

        audio_data, sr = sf.read(str(audio_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Calculate chunk size in samples: 24MB / (4 bytes per float32 + WAV header overhead)
        # WAV at 16kHz mono float32 = 64KB/s -> ~375s per 24MB
        bytes_per_sample = 4  # float32
        wav_overhead = 44  # standard WAV header
        samples_per_chunk = (self._OPENAI_MAX_FILE_BYTES - wav_overhead) // bytes_per_sample

        all_segments: list[dict] = []
        all_texts: list[str] = []
        chunk_idx = 0
        pos = 0

        while pos < len(audio_data):
            chunk = audio_data[pos : pos + samples_per_chunk]
            time_offset = pos / sr

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                sf.write(str(tmp_path), chunk, sr)

            try:
                log.info(
                    "Transcribing chunk %d (offset=%.1fs, samples=%d)",
                    chunk_idx, time_offset, len(chunk),
                )
                result = self._openai_transcribe_file(tmp_path, time_offset=time_offset)
                all_segments.extend(result.get("segments", []))
                all_texts.append(result.get("text", ""))
            finally:
                tmp_path.unlink(missing_ok=True)

            pos += samples_per_chunk
            chunk_idx += 1

        return {"text": " ".join(all_texts), "segments": all_segments}

    @staticmethod
    def _find_best_speaker(
        start: float,
        end: float,
        diar_segments: list[DiarizationSegment],
    ) -> str:
        """Find the diarization speaker with the most overlap to [start, end]."""
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for ds in diar_segments:
            overlap_start = max(start, ds.start)
            overlap_end = min(end, ds.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds.speaker

        return best_speaker
