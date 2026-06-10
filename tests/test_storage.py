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
