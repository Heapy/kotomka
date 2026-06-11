from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

from .models import CandidateFrame
from .utils import parse_showinfo_timestamps


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary not found on PATH: {name}")
    return path


def run_command(command: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, timeout=timeout, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")
    return result


def ffprobe_duration(path: Path) -> float:
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(video_path: Path, output_path: Path) -> Path:
    require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "flac",
            str(output_path),
        ]
    )
    return output_path


def extract_candidate_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    duration_s: float,
    interval_seconds: int = 15,
) -> list[CandidateFrame]:
    require_binary("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[CandidateFrame] = []
    candidates.extend(_extract_scene_frames(video_path, frames_dir))
    if len(candidates) < 3:
        candidates.extend(
            _extract_periodic_frames(
                video_path,
                frames_dir,
                duration_s=duration_s,
                interval_seconds=max(1, interval_seconds),
            )
        )
    return dedupe_frames(sorted(candidates, key=lambda item: item.timestamp_s))


def dedupe_frames(frames: list[CandidateFrame], *, max_distance: int = 6) -> list[CandidateFrame]:
    try:
        import imagehash
    except ImportError:
        return frames
    kept: list[tuple[CandidateFrame, object]] = []
    for frame in frames:
        try:
            with Image.open(frame.path) as image:
                fingerprint = imagehash.phash(image)
        except Exception:
            continue
        duplicate = any(abs(fingerprint - prior_hash) <= max_distance for _, prior_hash in kept)
        if not duplicate:
            kept.append((frame, fingerprint))
    return [frame for frame, _ in kept]


def _extract_scene_frames(video_path: Path, frames_dir: Path) -> list[CandidateFrame]:
    pattern = frames_dir / "scene_%05d.png"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(video_path),
            "-vf",
            "select=gt(scene\\,0.35),showinfo",
            "-vsync",
            "vfr",
            str(pattern),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    timestamps = parse_showinfo_timestamps(result.stderr)
    paths = sorted(frames_dir.glob("scene_*.png"))
    candidates: list[CandidateFrame] = []
    for index, path in enumerate(paths):
        timestamp = timestamps[index] if index < len(timestamps) else float(index)
        candidates.append(CandidateFrame(frame_id=f"scene-{index + 1:04d}", timestamp_s=timestamp, path=path, source="scene"))
    return candidates


def _extract_periodic_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    duration_s: float,
    interval_seconds: int,
) -> list[CandidateFrame]:
    pattern = frames_dir / "periodic_%05d.png"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval_seconds}",
            str(pattern),
        ]
    )
    paths = sorted(frames_dir.glob("periodic_*.png"))
    candidates: list[CandidateFrame] = []
    for index, path in enumerate(paths):
        timestamp = min(float(index * interval_seconds), max(0.0, duration_s))
        candidates.append(
            CandidateFrame(frame_id=f"periodic-{index + 1:04d}", timestamp_s=timestamp, path=path, source="periodic")
        )
    return candidates
