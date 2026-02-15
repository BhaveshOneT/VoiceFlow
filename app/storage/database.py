"""SQLite database for meeting persistence.

Uses WAL mode for concurrent reads during live transcription writes.
Thread-safe via threading.Lock on all write operations.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from app.storage.models import (
    Meeting,
    MeetingStatus,
    MeetingSummary,
    Speaker,
    TranscriptSegment,
)

log = logging.getLogger(__name__)

_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "VoiceFlow"
_DB_PATH = _APP_SUPPORT / "meetings.db"

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    duration_seconds REAL DEFAULT 0.0,
    audio_path TEXT DEFAULT '',
    status TEXT DEFAULT 'recording',
    device_name TEXT DEFAULT '',
    error_message TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_meetings_created ON meetings(created_at);

CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    display_name TEXT DEFAULT '',
    color TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_speakers_meeting ON speakers(meeting_id);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_id TEXT DEFAULT '',
    speaker_label TEXT DEFAULT '',
    start_time REAL DEFAULT 0.0,
    end_time REAL DEFAULT 0.0,
    text TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_segments_meeting_time
    ON transcript_segments(meeting_id, start_time);

CREATE TABLE IF NOT EXISTS meeting_summaries (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    summary_text TEXT DEFAULT '',
    key_points TEXT DEFAULT '[]',
    action_items TEXT DEFAULT '[]',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting ON meeting_summaries(meeting_id);
"""


class MeetingDatabase:
    """Thread-safe SQLite database for meetings."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._init_db()

    def _connect(self, *, _retries: int = 3) -> sqlite3.Connection:
        last_exc: Exception | None = None
        for attempt in range(_retries):
            try:
                conn = sqlite3.connect(str(self._db_path), timeout=10.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.row_factory = sqlite3.Row
                return conn
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if attempt < _retries - 1:
                    wait = 0.1 * (2 ** attempt)
                    log.warning(
                        "DB connect attempt %d/%d failed: %s (retry in %.1fs)",
                        attempt + 1, _retries, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    @contextmanager
    def _write_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Acquire write lock and yield a connection with auto-commit/rollback."""
        with self._write_lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._write_connection() as conn:
            conn.executescript(_SCHEMA_SQL)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (_SCHEMA_VERSION,),
                )
            log.info("Database initialized at %s (v%d)", self._db_path, _SCHEMA_VERSION)

    # ------------------------------------------------------------------
    # Meeting CRUD
    # ------------------------------------------------------------------

    def create_meeting(
        self,
        title: str,
        device_name: str = "",
        audio_path: str = "",
    ) -> Meeting:
        meeting = Meeting(
            id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.now(),
            device_name=device_name,
            audio_path=audio_path,
        )
        with self._write_connection() as conn:
            conn.execute(
                "INSERT INTO meetings (id, title, created_at, status, device_name, audio_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    meeting.id,
                    meeting.title,
                    meeting.created_at.isoformat(),
                    meeting.status.value,
                    meeting.device_name,
                    meeting.audio_path,
                ),
            )
        return meeting

    def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()
            return self._row_to_meeting(row) if row else None
        finally:
            conn.close()

    def list_meetings(self, limit: int = 50, offset: int = 0) -> list[Meeting]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM meetings ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_meeting(r) for r in rows]
        finally:
            conn.close()

    def update_meeting(self, meeting_id: str, **kwargs) -> None:
        allowed = {
            "title", "duration_seconds", "audio_path", "status",
            "device_name", "error_message",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "status" in updates and isinstance(updates["status"], MeetingStatus):
            updates["status"] = updates["status"].value

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [meeting_id]
        with self._write_connection() as conn:
            conn.execute(
                f"UPDATE meetings SET {set_clause} WHERE id = ?", values
            )

    def delete_meeting(self, meeting_id: str) -> None:
        with self._write_connection() as conn:
            conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))

    def search_meetings(self, query: str) -> list[Meeting]:
        conn = self._connect()
        try:
            like = f"%{query}%"
            rows = conn.execute(
                "SELECT DISTINCT m.* FROM meetings m "
                "LEFT JOIN transcript_segments ts ON ts.meeting_id = m.id "
                "WHERE m.title LIKE ? OR ts.text LIKE ? "
                "ORDER BY m.created_at DESC LIMIT 50",
                (like, like),
            ).fetchall()
            return [self._row_to_meeting(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Transcript segments
    # ------------------------------------------------------------------

    def add_segments(self, segments: list[TranscriptSegment]) -> None:
        if not segments:
            return
        with self._write_connection() as conn:
            conn.executemany(
                "INSERT INTO transcript_segments "
                "(id, meeting_id, speaker_id, speaker_label, start_time, end_time, text, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.id or str(uuid.uuid4()),
                        s.meeting_id,
                        s.speaker_id,
                        s.speaker_label,
                        s.start_time,
                        s.end_time,
                        s.text,
                        s.confidence,
                    )
                    for s in segments
                ],
            )

    def get_segments(self, meeting_id: str) -> list[TranscriptSegment]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM transcript_segments WHERE meeting_id = ? ORDER BY start_time",
                (meeting_id,),
            ).fetchall()
            return [self._row_to_segment(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Speakers
    # ------------------------------------------------------------------

    def add_speaker(self, speaker: Speaker) -> None:
        with self._write_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO speakers (id, meeting_id, label, display_name, color) "
                "VALUES (?, ?, ?, ?, ?)",
                (speaker.id, speaker.meeting_id, speaker.label, speaker.display_name, speaker.color),
            )

    def get_speakers(self, meeting_id: str) -> list[Speaker]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM speakers WHERE meeting_id = ?", (meeting_id,)
            ).fetchall()
            return [
                Speaker(
                    id=r["id"],
                    meeting_id=r["meeting_id"],
                    label=r["label"],
                    display_name=r["display_name"],
                    color=r["color"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def update_speaker_name(self, speaker_id: str, display_name: str) -> None:
        with self._write_connection() as conn:
            conn.execute(
                "UPDATE speakers SET display_name = ? WHERE id = ?",
                (display_name, speaker_id),
            )
            conn.execute(
                "UPDATE transcript_segments SET speaker_label = ? WHERE speaker_id = ?",
                (display_name, speaker_id),
            )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def save_summary(self, summary: MeetingSummary) -> None:
        with self._write_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meeting_summaries "
                "(id, meeting_id, summary_text, key_points, action_items, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    summary.id or str(uuid.uuid4()),
                    summary.meeting_id,
                    summary.summary_text,
                    json.dumps(summary.key_points),
                    json.dumps(summary.action_items),
                    (summary.created_at or datetime.now()).isoformat(),
                ),
            )

    def get_summary(self, meeting_id: str) -> Optional[MeetingSummary]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM meeting_summaries WHERE meeting_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (meeting_id,),
            ).fetchone()
            if not row:
                return None
            return MeetingSummary(
                id=row["id"],
                meeting_id=row["meeting_id"],
                summary_text=row["summary_text"],
                key_points=json.loads(row["key_points"]) if row["key_points"] else [],
                action_items=json.loads(row["action_items"]) if row["action_items"] else [],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Checkpoint WAL and release resources for clean shutdown."""
        try:
            with self._write_lock:
                conn = sqlite3.connect(str(self._db_path), timeout=5.0)
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    conn.close()
            log.info("Database closed: %s", self._db_path)
        except Exception:
            log.debug("Database close error", exc_info=True)

    # ------------------------------------------------------------------
    # Row mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_meeting(row: sqlite3.Row) -> Meeting:
        return Meeting(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            duration_seconds=row["duration_seconds"],
            audio_path=row["audio_path"],
            status=MeetingStatus(row["status"]),
            device_name=row["device_name"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _row_to_segment(row: sqlite3.Row) -> TranscriptSegment:
        return TranscriptSegment(
            id=row["id"],
            meeting_id=row["meeting_id"],
            speaker_id=row["speaker_id"],
            speaker_label=row["speaker_label"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            text=row["text"],
            confidence=row["confidence"],
        )
