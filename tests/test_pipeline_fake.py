from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

import kotomka.worker as worker_module
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
