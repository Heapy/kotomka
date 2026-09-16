import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kotomka.models import JobCreate
from kotomka.storage import JobStore


def test_update_can_clear_error_and_result_without_resetting_other_fields(tmp_path):
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    store.update_job(job.id, status="failed", progress=42, error="old failure", result={"old": True})
    store.update_job(job.id, message="new message")
    assert store.get_job(job.id).error == "old failure"
    cleared = store.update_job(job.id, error=None, result=None)
    assert cleared.error is None and cleared.result is None
    assert (cleared.status, cleared.progress, cleared.message) == ("failed", 42, "new message")
    assert store.get_job(job.id) == cleared


def test_update_uses_one_statement_and_returns_the_written_record(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    statements = []
    connect = store._connect
    def traced_connect():
        conn = connect()
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(store, "_connect", traced_connect)
    result = store.update_job(job.id, progress=200)
    assert result.progress == 100
    data_statements = [statement for statement in statements if statement.lstrip().startswith(("UPDATE", "SELECT"))]
    assert len(data_statements) == 1
    with pytest.raises(KeyError):
        store.update_job("does-not-exist", error=None)


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


def test_delete_does_not_remove_a_job_retried_after_its_status_read(tmp_path: Path, monkeypatch) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    concurrent_store = JobStore(store.db_path, store.jobs_dir)
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    store.update_job(job.id, status="completed")
    marker = job.artifact_dir / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    get_job = store.get_job

    def read_then_retry(job_id):
        snapshot = get_job(job_id)
        concurrent_store.retry_job(job_id)
        return snapshot

    monkeypatch.setattr(store, "get_job", read_then_retry)
    with pytest.raises(ValueError):
        store.delete_job(job.id)

    assert concurrent_store.get_job(job.id).status == "queued"
    assert marker.read_text(encoding="utf-8") == "keep"


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
    store.update_job(job.id, status="completed", progress=100, message="Completed")

    retried = store.retry_job(job.id)

    assert retried.is_read is False
    assert retried.status == "queued"


def test_retry_job_rejects_queued_or_running_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))

    with pytest.raises(ValueError):
        store.retry_job(job.id)

    store.update_job(job.id, status="running", progress=50, message="Working")
    with pytest.raises(ValueError):
        store.retry_job(job.id)

    assert store.get_job(job.id).status == "running"


def test_retry_job_is_atomic_against_concurrent_retries(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    store.update_job(job.id, status="failed", progress=10, message="boom", error="boom")

    first = store.retry_job(job.id)
    assert first.status == "queued"

    with pytest.raises(ValueError):
        store.retry_job(job.id)


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
