from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

from PIL import Image, ImageChops, ImageFilter, ImageStat

from .models import CandidateFrame, Chapter
from .utils import parse_showinfo_timestamps

_SOURCE_PRIORITY = {"plateau": 0, "scene": 1, "periodic": 2}


def limit_candidates(
    frames: list[CandidateFrame], *, limit: int, chapters: list[Chapter] | None = None,
) -> list[CandidateFrame]:
    """Reserve chapter coverage, then divide the remaining budget across time."""
    if limit <= 0:
        return []
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id))
    if len(ordered) <= limit:
        return ordered

    def rank(frame: CandidateFrame, target: float = 0):
        return (_SOURCE_PRIORITY.get(frame.source, 9), -(frame.dwell_s or 0),
                abs(frame.timestamp_s - target), frame.timestamp_s, frame.frame_id)

    picks: dict[str, CandidateFrame] = {}
    for chapter in chapters or []:
        candidates = [frame for frame in ordered if chapter.start_s <= frame.timestamp_s < chapter.end_s]
        if candidates:
            pick = min(candidates, key=lambda frame: rank(frame, (chapter.start_s + chapter.end_s) / 2))
            picks[pick.frame_id] = pick
    if len(picks) > limit:
        return limit_candidates(list(picks.values()), limit=limit)
    slots = limit - len(picks)
    if slots:
        start, end = ordered[0].timestamp_s, ordered[-1].timestamp_s
        buckets: dict[int, list[CandidateFrame]] = {}
        for frame in ordered:
            if frame.frame_id in picks:
                continue
            bucket = min(slots - 1, int((frame.timestamp_s - start) * slots / (end - start))) if end > start else 0
            buckets.setdefault(bucket, []).append(frame)
        for bucket, candidates in buckets.items():
            target = start + (bucket + 0.5) * (end - start) / slots
            pick = min(candidates, key=lambda frame: rank(frame, target))
            picks[pick.frame_id] = pick
    for frame in sorted(ordered, key=rank):
        if len(picks) >= limit:
            break
        picks.setdefault(frame.frame_id, frame)
    return sorted(picks.values(), key=lambda frame: (frame.timestamp_s, frame.frame_id))


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
    max_candidates: int = 150,
    chapters: list[Chapter] | None = None,
) -> list[CandidateFrame]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".extract-", dir=frames_dir) as temporary:
        work_dir = Path(temporary)
        candidates = _collect_candidate_frames(
            video_path, work_dir, duration_s=duration_s, interval_seconds=interval_seconds,
            max_gap_seconds=max_gap_seconds, plateau_min_dwell_s=plateau_min_dwell_s,
            plateau_hash_distance=plateau_hash_distance, blur_threshold=blur_threshold,
            max_candidates=max_candidates, chapters=chapters,
        )
        result = []
        for frame in candidates:
            path = frames_dir / f"{work_dir.name.removeprefix('.extract-')}-{frame.path.name}"
            frame.path.replace(path)
            result.append(frame.model_copy(update={"path": path}))
        return result


def _collect_candidate_frames(
    video_path: Path, frames_dir: Path, *, duration_s: float, interval_seconds: int,
    max_gap_seconds: int, plateau_min_dwell_s: float, plateau_hash_distance: int,
    blur_threshold: float, max_candidates: int, chapters: list[Chapter] | None,
) -> list[CandidateFrame]:
    require_binary("ffmpeg")
    plateaus, scenes = _analyze_video(
        video_path, min_dwell_s=plateau_min_dwell_s, max_distance=plateau_hash_distance,
    )
    candidates = [
        CandidateFrame(frame_id=f"plateau-{index:04d}", timestamp_s=timestamp,
                       path=frames_dir / "pending.png", source="plateau", dwell_s=dwell)
        for index, (timestamp, dwell) in enumerate(plateaus, 1)
    ]
    candidates.extend(
        CandidateFrame(frame_id=f"scene-{index:04d}", timestamp_s=timestamp,
                       path=frames_dir / "pending.png", source="scene")
        for index, timestamp in enumerate(scenes, 1)
    )
    fill = compute_gap_fill_timestamps(
        [frame.timestamp_s for frame in candidates], duration_s=duration_s,
        max_gap_s=float(max_gap_seconds), stride_s=float(max(1, interval_seconds)),
    )
    if not candidates and not fill and duration_s > 0:
        fill = [duration_s / 2]
    candidates.extend(
        CandidateFrame(frame_id=f"periodic-{index:04d}", timestamp_s=timestamp,
                       path=frames_dir / "pending.png", source="periodic")
        for index, timestamp in enumerate(fill, 1)
    )
    candidates = limit_candidates(candidates, limit=max_candidates, chapters=chapters)
    # Reserve one emergency sample in the same extraction pass for a blur gate
    # that rejects every scene/plateau. It is discarded when normal picks survive.
    if blur_threshold > 0 and candidates and not any(frame.source == "periodic" for frame in candidates):
        candidates.append(CandidateFrame(frame_id="blur-fallback", timestamp_s=duration_s / 2,
                                         path=frames_dir / "pending.png", source="periodic"))
    extracted = _extract_frames_at(video_path, frames_dir, candidates)
    usable = [frame for frame in extracted if frame.frame_id != "blur-fallback" and (
        frame.source == "periodic" or blur_threshold <= 0 or blur_score(frame.path) >= blur_threshold
    )]
    if not usable:
        usable = [frame for frame in extracted if frame.frame_id == "blur-fallback"]
    ordered = sorted(usable, key=lambda frame: (_SOURCE_PRIORITY.get(frame.source, 9), frame.timestamp_s))
    return sorted(dedupe_frames(ordered), key=lambda frame: frame.timestamp_s)


def _analyze_video(video_path: Path, *, min_dwell_s: float, max_distance: int) -> tuple[list[tuple[float, float]], list[float]]:
    """One decoder feeds scene detection and a streaming 1 fps grayscale sampler."""
    import imagehash

    side = 160
    graph = (
        "[0:v:0]setpts=PTS-STARTPTS,split=2[thumb][scene];"
        f"[thumb]fps=1,scale={side}:{side}:force_original_aspect_ratio=decrease,"
        f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2,format=gray[thumbs];"
        "[scene]select='gt(scene,0.35)',showinfo,nullsink"
    )
    hashes: list[tuple[float, object]] = []
    with TemporaryFile() as log:
        command = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(video_path),
            "-filter_complex", graph, "-map", "[thumbs]", "-c:v", "rawvideo", "-pix_fmt", "gray",
            "-f", "rawvideo", "pipe:1",
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log) as process:
            try:
                assert process.stdout is not None
                while pixels := process.stdout.read(side * side):
                    if len(pixels) != side * side:
                        raise RuntimeError("ffmpeg returned an incomplete thumbnail")
                    image = Image.frombytes("L", (side, side), pixels)
                    hashes.append((float(len(hashes)), imagehash.phash(image)))
                code = process.wait()
            except BaseException:
                process.kill()
                raise
        log.seek(0)
        output = log.read().decode("utf-8", errors="replace")
        if code:
            raise RuntimeError(f"Video analysis failed: {output[-4000:]}")
    return detect_plateaus(hashes, max_distance=max_distance, min_dwell_s=min_dwell_s), parse_showinfo_timestamps(output)


def _extract_frames_at(video_path: Path, frames_dir: Path, candidates: list[CandidateFrame]) -> list[CandidateFrame]:
    """Extract the first decoded frame at/after each target in one sequential pass.

    Several targets can map to the same actual PTS in VFR media; keep the strongest
    source for that image and record its actual timestamp, never its list index.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda frame: frame.timestamp_s)
    times = sorted({max(0, frame.timestamp_s) for frame in ordered})
    crossed = "+".join(f"gte(t,{value:.6f})*lt(prev_selected_t,{value:.6f})" for value in times)
    expression = f"if(isnan(prev_selected_t),gte(t,{times[0]:.6f}),gt({crossed},0))"
    result = run_command([
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(video_path),
        "-vf", f"setpts=PTS-STARTPTS,select='{expression}',showinfo", "-an", "-fps_mode", "vfr",
        str(frames_dir / "candidate_%05d.png"),
    ])
    frames: list[CandidateFrame] = []
    position = 0
    for index, timestamp in enumerate(parse_showinfo_timestamps(result.stderr), 1):
        eligible = []
        while position < len(ordered) and ordered[position].timestamp_s <= timestamp + 1e-5:
            eligible.append(ordered[position])
            position += 1
        path = frames_dir / f"candidate_{index:05d}.png"
        if not eligible or not path.is_file():
            continue
        pick = min(eligible, key=lambda frame: (_SOURCE_PRIORITY.get(frame.source, 9), -(frame.dwell_s or 0)))
        frames.append(pick.model_copy(update={"path": path, "timestamp_s": timestamp}))
        fallback = next((frame for frame in eligible if frame.frame_id == "blur-fallback"), None)
        if fallback is not None and pick is not fallback:
            frames.append(fallback.model_copy(update={"path": path, "timestamp_s": timestamp}))
    return frames


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
        duplicate = any(
            abs(fingerprint - prior_hash) <= max_distance and _same_visual_content(frame.path, prior.path)
            for prior, prior_hash in kept
        )
        if not duplicate:
            kept.append((frame, fingerprint))
    return [frame for frame, _ in kept]


def _same_visual_content(first: Path, second: Path, *, max_pixel_delta: int = 8) -> bool:
    # pHash can collide even when a number or bullet changes. Only codec-level
    # pixel differences are safe to discard before OCR sees the frames.
    try:
        with Image.open(first) as left, Image.open(second) as right:
            if left.size != right.size:
                return False
            left_pixels, right_pixels = left.convert("RGBA"), right.convert("RGBA")
            difference = ImageChops.difference(left_pixels, right_pixels)
            peak_delta = max(maximum for _minimum, maximum in difference.getextrema())
            if peak_delta <= max_pixel_delta:
                return True
            if peak_delta > 128:
                return False
            # Suppress codec ringing locally, without averaging away a changed
            # glyph merely because it occupies a small fraction of the slide.
            difference = ImageChops.difference(
                left_pixels.filter(ImageFilter.GaussianBlur(1.5)),
                right_pixels.filter(ImageFilter.GaussianBlur(1.5)),
            )
            return all(maximum <= max_pixel_delta for _minimum, maximum in difference.getextrema())
    except (OSError, ValueError):
        return False


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
    output.unlink(missing_ok=True)
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
