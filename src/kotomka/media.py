from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from .models import CandidateFrame
from .utils import parse_showinfo_timestamps

_SOURCE_PRIORITY = {"plateau": 0, "scene": 1, "periodic": 2}


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
    max_gap_seconds: int = 60,
    plateau_min_dwell_s: float = 3.0,
    plateau_hash_distance: int = 3,
    blur_threshold: float = 0.0,
) -> list[CandidateFrame]:
    """Collect candidate frames from three sources.

    Plateau detection finds slide-like content (stable stretches of the video,
    sampled near their end so builds/animations have finished), scene detection
    catches camera cuts, and gap filling guarantees no stretch longer than
    `max_gap_seconds` is left without a candidate. Dedupe prefers plateau frames
    over scene frames over periodic fillers.
    """
    require_binary("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[CandidateFrame] = []
    candidates.extend(
        _extract_plateau_frames(
            video_path,
            frames_dir,
            min_dwell_s=plateau_min_dwell_s,
            max_distance=plateau_hash_distance,
        )
    )
    candidates.extend(_extract_scene_frames(video_path, frames_dir))
    if blur_threshold > 0:
        candidates = [frame for frame in candidates if blur_score(frame.path) >= blur_threshold]
    fill_timestamps = compute_gap_fill_timestamps(
        [frame.timestamp_s for frame in candidates],
        duration_s=duration_s,
        max_gap_s=float(max_gap_seconds),
        stride_s=float(max(1, interval_seconds)),
    )
    if not candidates and not fill_timestamps and duration_s > 0:
        fill_timestamps = [round(duration_s / 2, 3)]
    candidates.extend(_extract_periodic_frames(video_path, frames_dir, fill_timestamps))
    ordered = sorted(candidates, key=lambda item: (_SOURCE_PRIORITY.get(item.source, 9), item.timestamp_s))
    return sorted(dedupe_frames(ordered), key=lambda item: item.timestamp_s)


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


def _extract_plateau_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    min_dwell_s: float,
    max_distance: int,
) -> list[CandidateFrame]:
    try:
        import imagehash
    except ImportError:
        return []
    work_dir = frames_dir / "thumbs_work"
    try:
        hashes: list[tuple[float, object]] = []
        for timestamp, path in _sample_thumbnails(video_path, work_dir):
            try:
                with Image.open(path) as image:
                    hashes.append((timestamp, imagehash.phash(image)))
            except Exception:
                continue
        plateaus = detect_plateaus(hashes, max_distance=max_distance, min_dwell_s=min_dwell_s)
    except Exception:
        return []
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    candidates: list[CandidateFrame] = []
    for index, (timestamp, dwell) in enumerate(plateaus):
        path = extract_frame_at(video_path, frames_dir, timestamp, name=f"plateau_{index + 1:05d}.png")
        if path is None:
            continue
        candidates.append(
            CandidateFrame(
                frame_id=f"plateau-{index + 1:04d}",
                timestamp_s=timestamp,
                path=path,
                source="plateau",
                dwell_s=dwell,
            )
        )
    return candidates


def _sample_thumbnails(video_path: Path, work_dir: Path, *, fps: float = 1.0) -> list[tuple[float, Path]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    pattern = work_dir / "thumb_%06d.png"
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
            f"fps={fps},scale=160:-2,format=gray",
            str(pattern),
        ]
    )
    paths = sorted(work_dir.glob("thumb_*.png"))
    return [(index / fps, path) for index, path in enumerate(paths)]


def detect_plateaus(
    hashes: list[tuple[float, object]],
    *,
    max_distance: int,
    min_dwell_s: float,
) -> list[tuple[float, float]]:
    """Find stable runs in a sampled hash sequence.

    Returns (timestamp, dwell_s) per plateau. The timestamp is the second-to-last
    sample of the run, so slide builds and the first crossfade frame are avoided.
    """
    if len(hashes) < 2:
        return []
    plateaus: list[tuple[float, float]] = []
    run_start = 0
    for index in range(1, len(hashes) + 1):
        if index < len(hashes) and abs(hashes[index][1] - hashes[index - 1][1]) <= max_distance:  # type: ignore[operator]
            continue
        last = index - 1
        dwell = hashes[last][0] - hashes[run_start][0]
        if dwell >= min_dwell_s:
            pick = max(run_start, last - 1)
            plateaus.append((hashes[pick][0], dwell))
        run_start = index
    return plateaus


def compute_gap_fill_timestamps(
    timestamps: list[float],
    *,
    duration_s: float,
    max_gap_s: float,
    stride_s: float,
) -> list[float]:
    """Timestamps to extract so no candidate gap exceeds `max_gap_s`."""
    if duration_s <= 0 or stride_s <= 0:
        return []
    fill: list[float] = []
    boundaries = [0.0, *sorted(timestamps), duration_s]
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left <= max_gap_s:
            continue
        position = left + stride_s
        while position < right:
            fill.append(round(position, 3))
            position += stride_s
    return fill


def extract_frame_at(video_path: Path, frames_dir: Path, timestamp_s: float, name: str) -> Path | None:
    output = frames_dir / name
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp_s):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not output.exists():
        return None
    return output


def blur_score(path: Path) -> float:
    """Edge-energy sharpness proxy; transition and motion blur score low."""
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            if gray.width > 640:
                gray = gray.resize((640, max(1, round(gray.height * 640 / gray.width))))
            edges = gray.filter(ImageFilter.FIND_EDGES)
            return float(ImageStat.Stat(edges).rms[0])
    except Exception:
        return float("inf")


def _extract_periodic_frames(video_path: Path, frames_dir: Path, timestamps: list[float]) -> list[CandidateFrame]:
    candidates: list[CandidateFrame] = []
    for index, timestamp in enumerate(timestamps):
        path = extract_frame_at(video_path, frames_dir, timestamp, name=f"periodic_{index + 1:05d}.png")
        if path is None:
            continue
        candidates.append(
            CandidateFrame(frame_id=f"periodic-{index + 1:04d}", timestamp_s=timestamp, path=path, source="periodic")
        )
    return candidates
