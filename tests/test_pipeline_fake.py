from __future__ import annotations

import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

import kotomka.worker as worker_module
import kotomka.source as source_module
from kotomka.config import Settings
from kotomka.models import JobCreate, SourceArtifact
from kotomka.providers.llm.fake import FakeLlmProvider
from kotomka.reporting import load_report
from kotomka.source import LocalFileSourceProvider
from kotomka.storage import JobStore
from kotomka.worker import JobWorker

needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required"
)


def make_fixture_video(tmp_path: Path) -> Path:
    video = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=1:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    return video


def make_worker(tmp_path: Path) -> tuple[JobStore, JobWorker]:
    settings = Settings(
        data_dir=tmp_path / "data",
        stt_provider="fake",
        llm_provider="fake",
        frame_interval_seconds=1,
        max_frames_for_llm=4,
    )
    store = JobStore(settings.db_path, settings.jobs_dir)
    worker = JobWorker(store=store, settings=settings, source_provider=LocalFileSourceProvider())
    return store, worker


@needs_ffmpeg
def test_pipeline_fake_end_to_end(tmp_path: Path) -> None:
    video = make_fixture_video(tmp_path)
    store, worker = make_worker(tmp_path)
    job = store.create_job(
        JobCreate(source_url=video.as_uri(), output_language="ru", stt_provider="fake", llm_provider="fake")
    )
    worker.process(job.id)
    completed = store.get_job(job.id)
    assert completed.status == "completed"
    report = load_report(completed.artifact_dir / "report.json")
    assert report.summary
    assert report.transcript.segments
    assert report.assessment is not None
    assert report.assessment.verdict
    assert (completed.artifact_dir / "frames.json").exists()
    assert (completed.artifact_dir / "media" / "audio.flac").exists()
    assert (completed.artifact_dir / "transcript_raw.json").exists()
    assert {p.name for p in (completed.artifact_dir / "frames").glob("*.png")} == {
        frame.image_path for frame in report.frames
    }


@needs_ffmpeg
def test_pipeline_completes_when_assessment_fails(tmp_path: Path, monkeypatch) -> None:
    class BrokenAssessmentLlm(FakeLlmProvider):
        def assess_report(self, **kwargs):
            raise RuntimeError("assessment exploded")

    monkeypatch.setattr(worker_module, "get_llm_provider", lambda name: BrokenAssessmentLlm())
    video = make_fixture_video(tmp_path)
    store, worker = make_worker(tmp_path)
    job = store.create_job(
        JobCreate(source_url=video.as_uri(), output_language="ru", stt_provider="fake", llm_provider="fake")
    )
    worker.process(job.id)
    completed = store.get_job(job.id)
    assert completed.status == "completed"
    report = load_report(completed.artifact_dir / "report.json")
    assert report.assessment is None
    assert report.summary


@needs_ffmpeg
def test_cleanup_database_failure_cannot_fail_an_already_retried_job(tmp_path, monkeypatch):
    video = make_fixture_video(tmp_path)
    store, worker = make_worker(tmp_path)
    job = store.create_job(JobCreate(source_url=video.as_uri(), stt_provider="fake", llm_provider="fake"))
    def cleanup(job_id):
        assert store.get_job(job_id).status == "completed"
        store.retry_job(job_id)
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(store, "cleanup_frames", cleanup)
    worker.process(job.id)
    assert store.get_job(job.id).status == "queued"
    assert store.get_job(job.id).error is None


class DelayedSourceProvider(LocalFileSourceProvider):
    def __init__(self, *, delay_seconds: float, events: list[tuple[str, float]]) -> None:
        self.delay_seconds = delay_seconds
        self.events = events

    def fetch(self, payload: JobCreate, artifact_dir: Path) -> SourceArtifact:
        self.events.append(("start", time.monotonic()))
        time.sleep(self.delay_seconds)
        result = super().fetch(payload, artifact_dir)
        self.events.append(("end", time.monotonic()))
        return result


@needs_ffmpeg
def test_worker_pool_serializes_downloads(tmp_path: Path) -> None:
    video = make_fixture_video(tmp_path)
    events: list[tuple[str, float]] = []
    settings = Settings(
        data_dir=tmp_path / "data",
        stt_provider="fake",
        llm_provider="fake",
        frame_interval_seconds=1,
        max_frames_for_llm=4,
        worker_pool_size=2,
    )
    store = JobStore(settings.db_path, settings.jobs_dir)
    worker = JobWorker(
        store=store,
        settings=settings,
        source_provider=DelayedSourceProvider(delay_seconds=0.3, events=events),
    )
    jobs = [
        store.create_job(JobCreate(source_url=video.as_uri(), stt_provider="fake", llm_provider="fake"))
        for _ in range(2)
    ]

    # Jobs are already "queued" in the store, so worker.start() picks them up
    # via its startup requeue; enqueuing them again here would double-process.
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            statuses = [store.get_job(job.id).status for job in jobs]
            if all(status == "completed" for status in statuses):
                break
            time.sleep(0.05)
    finally:
        worker.stop()

    assert len(worker._threads) == 2
    assert [store.get_job(job.id).status for job in jobs] == ["completed", "completed"]

    starts = sorted(timestamp for kind, timestamp in events if kind == "start")
    ends = sorted(timestamp for kind, timestamp in events if kind == "end")
    assert len(starts) == 2 and len(ends) == 2
    # The second download must not begin until the first one has finished.
    assert starts[1] >= ends[0]


def test_duration_limit_is_checked_before_audio_extraction(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "too-long.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(source_module, "ffprobe_duration", lambda path: 20)
    extract = Mock(return_value=tmp_path / "audio.flac")
    monkeypatch.setattr(worker_module, "extract_audio", extract)
    store, worker = make_worker(tmp_path)
    worker.settings.max_video_duration_seconds = 10
    job = store.create_job(JobCreate(source_url=video.as_uri(), stt_provider="fake", llm_provider="fake"))

    worker.process(job.id)

    assert store.get_job(job.id).status == "failed"
    assert "longer" in store.get_job(job.id).error
    extract.assert_not_called()


def test_audio_extraction_does_not_block_the_next_download(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(source_module, "ffprobe_duration", lambda path: 3)
    first_audio_started = Event()
    release_audio = Event()
    second_download_finished = Event()
    store, worker = make_worker(tmp_path)
    worker.settings.worker_pool_size = 2
    jobs = [store.create_job(JobCreate(source_url=video.as_uri(), stt_provider="fake", llm_provider="fake")) for _ in range(2)]
    fetch = worker.source_provider.fetch
    downloads = []

    def tracked_fetch(payload, artifact_dir):
        result = fetch(payload, artifact_dir)
        downloads.append(artifact_dir)
        if len(downloads) == 2:
            second_download_finished.set()
        return result

    def extract(video_path, audio_path):
        first_audio_started.set()
        assert release_audio.wait(timeout=5)
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(worker.source_provider, "fetch", tracked_fetch)
    monkeypatch.setattr(worker_module, "extract_audio", extract)
    monkeypatch.setattr(worker_module, "extract_candidate_frames", lambda *args, **kwargs: [])
    worker.start()
    try:
        assert first_audio_started.wait(timeout=2)
        assert second_download_finished.wait(timeout=1), "The download permit is still held during audio extraction"
    finally:
        release_audio.set()
        worker._queue.join()
        worker.stop()
    assert all(store.get_job(job.id).status == "completed" for job in jobs)


def test_worker_restarts_after_stop(tmp_path: Path) -> None:
    _, worker = make_worker(tmp_path)

    worker.start()
    try:
        assert worker._threads and all(thread.is_alive() for thread in worker._threads)
    finally:
        worker.stop()
    assert all(not thread.is_alive() for thread in worker._threads)

    worker.start()
    try:
        assert all(thread.is_alive() for thread in worker._threads)
    finally:
        worker.stop()


def test_worker_skips_a_deleted_queue_entry(tmp_path: Path, monkeypatch) -> None:
    _, worker = make_worker(tmp_path)
    monkeypatch.setattr(worker.source_provider, "fetch", lambda *args: pytest.fail("Deleted job was processed"))
    worker.process("already-deleted")


def test_worker_continues_after_an_unexpected_job_exception(tmp_path: Path, monkeypatch) -> None:
    _, worker = make_worker(tmp_path)
    worker.settings.worker_pool_size = 1
    processed = Event()

    def process(job_id):
        if job_id == "broken":
            raise RuntimeError("unexpected storage failure")
        processed.set()

    monkeypatch.setattr(worker, "process", process)
    worker.enqueue("broken")
    worker.enqueue("next")
    worker.start()
    try:
        assert processed.wait(timeout=2), "The worker stopped before the next queued job"
        assert worker._threads[0].is_alive()
    finally:
        worker.stop()
