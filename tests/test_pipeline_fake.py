from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from kotomka.config import Settings
from kotomka.models import JobCreate
from kotomka.reporting import load_report
from kotomka.source import LocalFileSourceProvider
from kotomka.storage import JobStore
from kotomka.worker import JobWorker


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required")
def test_pipeline_fake_end_to_end(tmp_path: Path) -> None:
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
    settings = Settings(
        data_dir=tmp_path / "data",
        stt_provider="fake",
        llm_provider="fake",
        frame_interval_seconds=1,
        max_frames_for_llm=4,
    )
    store = JobStore(settings.db_path, settings.jobs_dir)
    worker = JobWorker(store=store, settings=settings, source_provider=LocalFileSourceProvider())
    job = store.create_job(
        JobCreate(source_url=video.as_uri(), output_language="ru", stt_provider="fake", llm_provider="fake")
    )
    worker.process(job.id)
    completed = store.get_job(job.id)
    assert completed.status == "completed"
    report = load_report(completed.artifact_dir / "report.json")
    assert report.summary
    assert report.transcript.segments
    assert (completed.artifact_dir / "frames.json").exists()

