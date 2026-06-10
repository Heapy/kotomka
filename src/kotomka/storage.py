from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .models import JobCreate, JobRecord, JobStatus


class JobStore:
    def __init__(self, db_path: Path, jobs_dir: Path) -> None:
        self.db_path = db_path
        self.jobs_dir = jobs_dir
        self._lock = RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )
            conn.commit()

    def create_job(self, payload: JobCreate) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        artifact_dir = self.jobs_dir / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, status, progress, message, error, created_at, updated_at,
                    input_json, artifact_dir, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "queued",
                    0,
                    "Queued",
                    None,
                    now.isoformat(),
                    now.isoformat(),
                    payload.model_dump_json(),
                    str(artifact_dir),
                    None,
                ),
            )
            conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_job(row)

    def list_requeueable_jobs(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def list_jobs(self, *, limit: int = 100) -> list[JobRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: int | None = None,
        message: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> JobRecord:
        current = self.get_job(job_id)
        next_status = status or current.status
        next_progress = current.progress if progress is None else max(0, min(100, int(progress)))
        next_message = current.message if message is None else message
        next_error = current.error if error is None else error
        next_result = current.result if result is None else result
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, message = ?, error = ?, updated_at = ?, result_json = ?
                WHERE id = ?
                """,
                (
                    next_status,
                    next_progress,
                    next_message,
                    next_error,
                    now,
                    json.dumps(next_result, ensure_ascii=False, default=str) if next_result is not None else None,
                    job_id,
                ),
            )
            conn.commit()
        return self.get_job(job_id)

    def retry_job(self, job_id: str, *, payload: JobCreate | None = None) -> JobRecord:
        current = self.get_job(job_id)
        next_input = payload or current.input
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, message = ?, error = ?, updated_at = ?, input_json = ?, result_json = ?
                WHERE id = ?
                """,
                ("queued", 0, "Queued", None, now, next_input.model_dump_json(), None, current.id),
            )
            conn.commit()
        return self.get_job(job_id)

    def delete_job(self, job_id: str) -> JobRecord:
        current = self.get_job(job_id)
        artifact_dir = current.artifact_dir.resolve()
        jobs_dir = self.jobs_dir.resolve()
        if current.status not in {"completed", "failed"}:
            raise ValueError("Only completed or failed jobs can be deleted")
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (current.id,))
            conn.commit()
        if artifact_dir.exists() and artifact_dir != jobs_dir and jobs_dir in artifact_dir.parents:
            shutil.rmtree(artifact_dir)
        return current

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        result_json = row["result_json"]
        return JobRecord(
            id=str(row["id"]),
            status=row["status"],
            progress=int(row["progress"]),
            message=str(row["message"]),
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            input=JobCreate.model_validate_json(row["input_json"]),
            artifact_dir=Path(row["artifact_dir"]),
            result=json.loads(result_json) if result_json else None,
        )
