from pathlib import Path
import sqlite3

import pytest

from kotomka.models import FrameSelection, JobCreate, Report, Transcript, VideoMetadata
from kotomka.reporting import save_report
from kotomka.storage import JobStore


def completed_job(tmp_path):
    store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = store.create_job(JobCreate(source_url="https://example.com/video"))
    frames = job.artifact_dir / "frames"
    frames.mkdir()
    (frames / "keep.png").write_bytes(b"selected")
    (frames / "unused.png").write_bytes(b"unused")
    (frames / "diagnostic.txt").write_text("keep non-image artifacts")
    report = Report(
        video=VideoMetadata(source_url="https://example.com/video"), summary="test", sections=[],
        frames=[FrameSelection(frame_id="keep", timestamp_s=0, image_path="keep.png", score=1)],
        transcript=Transcript(segments=[]),
    )
    save_report(report, job.artifact_dir / "report.json")
    store.update_job(job.id, status="completed")
    return store, job


def test_cleanup_keeps_all_report_frames_and_supports_dry_run(tmp_path):
    store, job = completed_job(tmp_path)
    preview = store.cleanup_frames(job.id, dry_run=True)
    assert (preview.files, preview.bytes) == (1, 6)
    assert (job.artifact_dir / "frames/unused.png").exists()
    result = store.cleanup_frames(job.id)
    assert result == preview
    assert {p.name for p in (job.artifact_dir / "frames").iterdir()} == {"keep.png", "diagnostic.txt"}
    assert store.cleanup_frames(job.id).files == 0


def test_cleanup_skips_retried_jobs(tmp_path):
    store, job = completed_job(tmp_path)
    store.retry_job(job.id)
    assert store.cleanup_frames(job.id) is None
    assert (job.artifact_dir / "frames/unused.png").exists()


@pytest.mark.parametrize("broken", ["report", "frame", "symlink"])
def test_cleanup_refuses_incomplete_or_external_artifacts(tmp_path, broken):
    store, job = completed_job(tmp_path)
    if broken == "report":
        (job.artifact_dir / "report.json").write_text("{}")
    else:
        kept = job.artifact_dir / "frames/keep.png"
        kept.unlink()
        if broken == "symlink":
            external = tmp_path / "external.png"
            external.write_bytes(b"external")
            kept.symlink_to(external)
    with pytest.raises(ValueError):
        store.cleanup_frames(job.id)
    assert (job.artifact_dir / "frames/unused.png").exists()


def test_cleanup_holds_database_write_lock_during_deletion(tmp_path, monkeypatch):
    import kotomka.storage as storage
    store, job = completed_job(tmp_path)
    original = storage.prune_frame_files

    def concurrent_retry(*args, **kwargs):
        with sqlite3.connect(store.db_path, timeout=0) as db:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                db.execute("UPDATE jobs SET status = 'queued' WHERE id = ?", (job.id,))
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "prune_frame_files", concurrent_retry)
    store.cleanup_frames(job.id)
    assert store.get_job(job.id).status == "completed"
