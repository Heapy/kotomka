from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from kotomka.media import (
    _extract_frames_at,
    extract_frame_at,
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


def test_batch_extraction_does_not_return_old_files_without_timestamps(tmp_path, monkeypatch):
    (tmp_path / "candidate_00001.png").write_bytes(b"old")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""))
    pending = CandidateFrame(frame_id="f1", timestamp_s=1, path=tmp_path / "pending.png")
    assert _extract_frames_at(Path("video.mp4"), tmp_path, [pending]) == []


def test_empty_seek_does_not_return_a_previous_frame(tmp_path, monkeypatch):
    (tmp_path / "frame.png").write_bytes(b"old")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""))
    assert extract_frame_at(Path("video.mp4"), tmp_path, 1000, "frame.png") is None


@needs_ffmpeg
def test_extraction_uses_two_passes_without_thumbnail_files(tmp_path, monkeypatch):
    video = tmp_path / "motion.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=4:duration=8", "-pix_fmt", "yuv420p", str(video)], check=True)
    popen = subprocess.Popen
    commands = []
    def tracked(command, *args, **kwargs):
        commands.append(command)
        return popen(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", tracked)
    frames_dir = tmp_path / "frames"
    result = extract_candidate_frames(video, frames_dir, duration_s=8, max_gap_seconds=2, interval_seconds=1, max_candidates=3)
    assert result
    assert len(commands) == 2
    assert any("split" in argument for argument in commands[0])
    assert not list(frames_dir.rglob("thumb*"))
    assert len(list(frames_dir.glob("*.png"))) <= 3


@needs_ffmpeg
@pytest.mark.parametrize("count", [150, 151, 512])
def test_batch_extraction_handles_large_candidate_budgets(tmp_path, count):
    video = tmp_path / "motion.mkv"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=64x48:rate=10:duration={count / 5 + 1}",
                    "-c:v", "ffv1", str(video)], check=True)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    candidates = [
        CandidateFrame(frame_id=str(index), timestamp_s=index / 5 + 0.01,
                       path=frames_dir / "pending.png", source="periodic")
        for index in range(count)
    ]

    frames = _extract_frames_at(video, frames_dir, candidates)

    assert [frame.frame_id for frame in frames] == [frame.frame_id for frame in candidates]
    assert [frame.timestamp_s for frame in frames] == pytest.approx(
        [index / 5 + 0.1 for index in range(count)]
    )
    assert len(list(frames_dir.glob("*.png"))) == count
    assert all(frame.path.is_file() for frame in frames)


@needs_ffmpeg
def test_batch_extraction_maps_close_targets_to_actual_vfr_timestamps(tmp_path):
    from kotomka.media import _analyze_video
    for index, color in enumerate(["white", "black", "white"]):
        Image.new("RGB", (160, 120), color).save(tmp_path / f"vfr-{index}.png")
    manifest = tmp_path / "vfr.txt"
    manifest.write_text("file 'vfr-0.png'\nduration 0.2\nfile 'vfr-1.png'\nduration 0.7\nfile 'vfr-2.png'\nduration 0.1\nfile 'vfr-2.png'\n")
    video = tmp_path / "vfr.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
                    "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(video)], check=True)
    _, scenes = _analyze_video(video, min_dwell_s=3, max_distance=3)
    assert scenes[0] == pytest.approx(0.2, abs=0.03)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    candidates = [CandidateFrame(frame_id=str(index), timestamp_s=ts, path=frames_dir / "pending.png",
                                 source="plateau" if index == 1 else "periodic")
                  for index, ts in enumerate([0.1, 0.15, 0.25])]
    frames = _extract_frames_at(video, frames_dir, candidates)
    assert [frame.frame_id for frame in frames] == ["1", "2"]
    assert frames[0].timestamp_s == pytest.approx(0.2, abs=0.03)
    assert frames[1].timestamp_s == pytest.approx(0.9, abs=0.03)
    with Image.open(frames[0].path) as first, Image.open(frames[1].path) as second:
        assert max(first.getpixel((0, 0))) < 10
        assert second.getpixel((0, 0))[2] > 200


@needs_ffmpeg
def test_blur_fallback_can_share_a_timestamp_with_a_rejected_plateau(tmp_path):
    video = tmp_path / "static.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=white:size=160x120:rate=2:duration=4", str(video)], check=True)
    frames = extract_candidate_frames(video, tmp_path / "frames", duration_s=4, blur_threshold=10000)
    assert len(frames) == 1
    assert frames[0].source == "periodic"


@needs_ffmpeg
def test_analysis_and_extraction_use_the_same_video_stream(tmp_path):
    video = tmp_path / "multiple-streams.mkv"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=white:size=160x120:rate=2:duration=4", "-f", "lavfi", "-i",
                    "color=red:size=320x240:rate=2:duration=4", "-map", "0:v", "-map", "1:v",
                    "-c:v", "libx264", "-disposition:v:0", "0", "-disposition:v:1", "default", str(video)], check=True)
    frames = extract_candidate_frames(video, tmp_path / "frames", duration_s=4)
    assert frames
    with Image.open(frames[0].path) as image:
        assert image.size == (160, 120)
        assert min(image.getpixel((0, 0))) > 240


@needs_ffmpeg
def test_plateau_detection_preserves_a_changed_digit_before_deduplication(tmp_path):
    font = ImageFont.load_default(size=28)
    for index, number in enumerate([1, 9]):
        slide = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(slide)
        draw.rectangle((0, 0, 1280, 100), fill="#17385c")
        draw.text((50, 30), "Database benchmark", font=font, fill="white")
        draw.text((60, 220), f"Latency: {number} ms", font=font, fill="black")
        slide.save(tmp_path / f"number-{index}.png")
    manifest = tmp_path / "numbers.txt"
    manifest.write_text("file 'number-0.png'\nduration 6\nfile 'number-1.png'\nduration 6\nfile 'number-1.png'\n")
    video = tmp_path / "numbers.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
                    "-vf", "fps=25", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-t", "12", str(video)], check=True)
    frames = extract_candidate_frames(video, tmp_path / "frames", duration_s=12)
    assert len(frames) == 2
    assert frames[0].timestamp_s < 6 <= frames[1].timestamp_s
    assert all(frame.source == "plateau" for frame in frames)


def test_detect_plateaus_finds_stable_runs() -> None:
    hashes = [(float(ts), value) for ts, value in enumerate([0, 0, 0, 0, 50, 50, 50, 50, 99])]
    plateaus = detect_plateaus(hashes, max_distance=3, min_dwell_s=3.0)
    assert plateaus == [(2.0, 3.0), (6.0, 3.0)]  # second-to-last sample of each run


def test_detect_plateaus_ignores_short_runs_and_noise() -> None:
    hashes = [(float(ts), value) for ts, value in enumerate([0, 50, 0, 50, 0, 50])]
    assert detect_plateaus(hashes, max_distance=3, min_dwell_s=3.0) == []
    assert detect_plateaus([(0.0, 1)], max_distance=3, min_dwell_s=1.0) == []


def test_plateau_hashes_compare_to_run_anchor_not_only_previous_sample():
    hashes = [(float(index), index) for index in range(9)]
    assert detect_plateaus(hashes, max_distance=3, min_dwell_s=3) == [(2, 3), (6, 3)]


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
def test_dedupe_collapses_a_repeated_slide_after_h264_encoding(tmp_path: Path) -> None:
    font = ImageFont.load_default(size=28)
    for index in range(2):
        slide = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(slide)
        draw.rectangle((0, 0, 1280, 110), fill="#17385c")
        draw.text((45, 30), "Production checklist" if not index else "Next topic", font=font, fill="white")
        for row, text in enumerate(["1. Check service health", "2. Confirm replica status", "3. Never log access tokens"]):
            draw.text((60, 190 + 80 * row), text if not index else f"Another topic {row}", font=font, fill="black")
        slide.save(tmp_path / f"slide-{index}.png")
    concat = tmp_path / "slides.txt"
    concat.write_text(
        "file 'slide-0.png'\nduration 2\nfile 'slide-1.png'\nduration 2\nfile 'slide-0.png'\nduration 2\nfile 'slide-0.png'\n",
        encoding="utf-8",
    )
    video = tmp_path / "slides.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-vf", "fps=25", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    frames = []
    for index, timestamp in enumerate([0.5, 4.5]):
        path = tmp_path / f"frame-{index}.png"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", str(path),
        ], check=True)
        frames.append(CandidateFrame(frame_id=str(index), timestamp_s=timestamp, path=path, source="plateau"))
    assert [frame.frame_id for frame in dedupe_frames(frames)] == ["0"]


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
    old_images = {frame.path: frame.path.read_bytes() for frame in frames}
    retried = extract_candidate_frames(video, frames_dir, duration_s=12)
    assert {frame.path for frame in retried}.isdisjoint(old_images)
    assert all(path.read_bytes() == content for path, content in old_images.items())
