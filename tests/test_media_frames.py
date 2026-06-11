from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from kotomka.media import (
    blur_score,
    compute_gap_fill_timestamps,
    dedupe_frames,
    detect_plateaus,
    extract_candidate_frames,
)
from kotomka.models import CandidateFrame

needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required"
)


def test_detect_plateaus_finds_stable_runs() -> None:
    hashes = [(float(ts), value) for ts, value in enumerate([0, 0, 0, 0, 50, 50, 50, 50, 99])]
    plateaus = detect_plateaus(hashes, max_distance=3, min_dwell_s=3.0)
    assert plateaus == [(2.0, 3.0), (6.0, 3.0)]  # second-to-last sample of each run


def test_detect_plateaus_ignores_short_runs_and_noise() -> None:
    hashes = [(float(ts), value) for ts, value in enumerate([0, 50, 0, 50, 0, 50])]
    assert detect_plateaus(hashes, max_distance=3, min_dwell_s=3.0) == []
    assert detect_plateaus([(0.0, 1)], max_distance=3, min_dwell_s=1.0) == []


def test_compute_gap_fill_timestamps_fills_only_large_gaps() -> None:
    fill = compute_gap_fill_timestamps([0.0, 30.0, 200.0], duration_s=260.0, max_gap_s=60.0, stride_s=15.0)
    assert fill == [45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0, 195.0]
    assert compute_gap_fill_timestamps([10.0, 50.0], duration_s=60.0, max_gap_s=60.0, stride_s=15.0) == []
    assert compute_gap_fill_timestamps([], duration_s=0.0, max_gap_s=60.0, stride_s=15.0) == []


def test_blur_score_separates_sharp_from_blurred(tmp_path: Path) -> None:
    sharp = Image.new("L", (320, 240))
    draw = ImageDraw.Draw(sharp)
    for x in range(0, 320, 16):
        for y in range(0, 240, 16):
            if (x + y) % 32 == 0:
                draw.rectangle([x, y, x + 15, y + 15], fill=255)
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=6))
    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"
    sharp.save(sharp_path)
    blurred.save(blurred_path)
    assert blur_score(sharp_path) > blur_score(blurred_path)


def test_dedupe_prefers_plateau_over_scene(tmp_path: Path) -> None:
    image = Image.new("RGB", (160, 120), color=(200, 30, 30))
    plateau_path = tmp_path / "plateau.png"
    scene_path = tmp_path / "scene.png"
    image.save(plateau_path)
    image.save(scene_path)
    frames = [
        CandidateFrame(frame_id="plateau-0001", timestamp_s=10.0, path=plateau_path, source="plateau", dwell_s=8.0),
        CandidateFrame(frame_id="scene-0001", timestamp_s=10.2, path=scene_path, source="scene"),
    ]
    kept = dedupe_frames(frames)
    assert [frame.frame_id for frame in kept] == ["plateau-0001"]


@needs_ffmpeg
def test_extract_candidate_frames_detects_slides(tmp_path: Path) -> None:
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    for index in range(2):
        slide = Image.new("RGB", (320, 240), color=(245, 245, 245))
        draw = ImageDraw.Draw(slide)
        offset = 30 + index * 120
        draw.rectangle([offset, 40, offset + 80, 200], fill=(20, 60, 160) if index == 0 else (160, 40, 20))
        draw.rectangle([10, 10, 310, 30], fill=(0, 0, 0))
        slide.save(slides_dir / f"slide{index}.png")
    video = tmp_path / "slides.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-t",
            "6",
            "-framerate",
            "5",
            "-i",
            str(slides_dir / "slide0.png"),
            "-loop",
            "1",
            "-t",
            "6",
            "-framerate",
            "5",
            "-i",
            str(slides_dir / "slide1.png"),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            str(video),
        ],
        check=True,
    )
    frames_dir = tmp_path / "frames"

    frames = extract_candidate_frames(
        video,
        frames_dir,
        duration_s=12.0,
        interval_seconds=2,
        max_gap_seconds=60,
        plateau_min_dwell_s=3.0,
        plateau_hash_distance=3,
    )

    plateau_frames = [frame for frame in frames if frame.source == "plateau"]
    assert len(plateau_frames) == 2
    first, second = plateau_frames
    assert 2.0 <= first.timestamp_s <= 5.0
    assert 8.0 <= second.timestamp_s <= 11.0
    assert all(frame.dwell_s and frame.dwell_s >= 3.0 for frame in plateau_frames)
    assert frames == sorted(frames, key=lambda frame: frame.timestamp_s)
    assert not (frames_dir / "thumbs_work").exists()
