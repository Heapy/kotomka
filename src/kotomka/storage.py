from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .artifacts import FrameCleanup, prune_frame_files
from .models import JobCreate, JobRecord, JobStatus
from .reporting import load_report


class _Unset:
    pass


_UNSET = _Unset()


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
                    is_read INTEGER NOT NULL DEFAULT 0,
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
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "is_read" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN is_read INTEGER NOT NULL DEFAULT 0")
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
                    id, status, is_read, progress, message, error, created_at, updated_at,
                    input_json, artifact_dir, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "queued",
                    0,
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

    def list_jobs(self, *, limit: int = 100, include_read: bool = False) -> list[JobRecord]:
        query = "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?"
        if not include_read:
            query = "SELECT * FROM jobs WHERE is_read = 0 ORDER BY created_at DESC LIMIT ?"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                query,
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
        error: str | None | _Unset = _UNSET,
        result: dict[str, Any] | None | _Unset = _UNSET,
    ) -> JobRecord:
        changes: dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if status is not None:
            changes["status"] = status
        if progress is not None:
            changes["progress"] = max(0, min(100, int(progress)))
        if message is not None:
            changes["message"] = message
        if error is not _UNSET:
            changes["error"] = error
        if result is not _UNSET:
            changes["result_json"] = json.dumps(result, ensure_ascii=False, default=str) if result is not None else None
        assignments = ", ".join(f"{column} = ?" for column in changes)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ? RETURNING *",
                [*changes.values(), job_id],
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_job(row)

    def retry_job(self, job_id: str, *, payload: JobCreate | None = None) -> JobRecord:
        current = self.get_job(job_id)
        next_input = payload or current.input
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, is_read = ?, progress = ?, message = ?, error = ?, updated_at = ?, input_json = ?, result_json = ?
                WHERE id = ? AND status IN ('failed', 'completed')
                """,
                ("queued", 0, 0, "Queued", None, now, next_input.model_dump_json(), None, current.id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                # Either the job doesn't exist (already excluded by get_job above) or a
                # concurrent retry already claimed it; the WHERE clause makes the
                # check-and-flip atomic under self._lock, so only one caller wins.
                raise ValueError(f"job {job_id} is not retryable from status {current.status!r}")
        return self.get_job(job_id)

    def set_job_read(self, job_id: str, is_read: bool) -> JobRecord:
        self.get_job(job_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET is_read = ?, updated_at = ? WHERE id = ?",
                (1 if is_read else 0, now, job_id),
            )
            conn.commit()
        return self.get_job(job_id)

    def completed_job_ids(self) -> list[str]:
        with self._lock, self._connect() as conn:
            return [row["id"] for row in conn.execute("SELECT id FROM jobs WHERE status = 'completed'")]

    def cleanup_frames(self, job_id: str, *, dry_run: bool = False) -> FrameCleanup | None:
        # A database write lock also excludes retries in another server process.
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status, artifact_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] != "completed":
                return None
            artifact_dir = Path(row["artifact_dir"])
            if artifact_dir.is_symlink() or artifact_dir.resolve().parent != self.jobs_dir.resolve():
                raise ValueError("Job artifacts must be inside the jobs directory")
            report = load_report(artifact_dir / "report.json")
            return prune_frame_files(artifact_dir, report, dry_run=dry_run)

    def delete_job(self, job_id: str) -> JobRecord:
        current = self.get_job(job_id)
        artifact_dir = current.artifact_dir.resolve()
        jobs_dir = self.jobs_dir.resolve()
        if current.status not in {"completed", "failed"}:
            raise ValueError("Only completed or failed jobs can be deleted")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE id = ? AND status IN ('completed', 'failed')", (current.id,)
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise ValueError("Job is no longer terminal or has already been deleted")
        if artifact_dir.exists() and artifact_dir != jobs_dir and jobs_dir in artifact_dir.parents:
            shutil.rmtree(artifact_dir)
        return current

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        result_json = row["result_json"]
        return JobRecord(
            id=str(row["id"]),
            status=row["status"],
            is_read=bool(row["is_read"]),
            progress=int(row["progress"]),
            message=str(row["message"]),
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            input=JobCreate.model_validate_json(row["input_json"]),
            artifact_dir=Path(row["artifact_dir"]),
            result=json.loads(result_json) if result_json else None,
        )
