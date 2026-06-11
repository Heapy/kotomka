import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kotomka.models import JobCreate
from kotomka.storage import JobStore


def test_delete_job_removes_terminal_record_and_artifacts(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    marker = job.artifact_dir / "marker.txt"
    marker.write_text("artifact", encoding="utf-8")
    store.update_job(job.id, status="completed", progress=100, message="Completed")

    deleted = store.delete_job(job.id)

    assert deleted.id == job.id
    assert not job.artifact_dir.exists()
    with pytest.raises(KeyError):
        store.get_job(job.id)


def test_delete_job_rejects_active_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))

    with pytest.raises(ValueError):
        store.delete_job(job.id)

    assert store.get_job(job.id).status == "queued"


def test_read_jobs_are_hidden_from_default_list(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    unread = store.create_job(JobCreate(source_url="https://example.com/unread"))
    read = store.create_job(JobCreate(source_url="https://example.com/read"))

    updated = store.set_job_read(read.id, True)

    assert updated.is_read is True
    assert [job.id for job in store.list_jobs()] == [unread.id]
    assert {job.id for job in store.list_jobs(include_read=True)} == {unread.id, read.id}


def test_retry_job_resets_read_state(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    store.set_job_read(job.id, True)

    retried = store.retry_job(job.id)

    assert retried.is_read is False
    assert retried.status == "queued"


def test_init_db_migrates_existing_jobs_to_unread(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    jobs_dir = tmp_path / "jobs"
    artifact_dir = jobs_dir / "legacy-job"
    artifact_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    payload = JobCreate(source_url="https://example.com/video")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
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
        conn.execute(
            """
            INSERT INTO jobs (
                id, status, progress, message, error, created_at, updated_at,
                input_json, artifact_dir, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-job",
                "completed",
                100,
                "Completed",
                None,
                now,
                now,
                payload.model_dump_json(),
                str(artifact_dir),
                None,
            ),
        )

    store = JobStore(db_path, jobs_dir)
    job = store.get_job("legacy-job")

    assert job.is_read is False
    assert [listed.id for listed in store.list_jobs()] == ["legacy-job"]
