from __future__ import annotations

import math
from typing import NamedTuple

from .models import Chapter, Transcript, TranscriptSegment

LOW_CONFIDENCE_MARKER = "[low-confidence]"


class TranscriptChunk(NamedTuple):
    title: str | None
    start_s: float
    end_s: float
    segments: list[TranscriptSegment]


def format_segment_line(segment: TranscriptSegment, *, low_confidence_below: float = 0.0) -> str:
    flag = ""
    if low_confidence_below > 0 and segment.confidence is not None and segment.confidence < low_confidence_below:
        flag = f" {LOW_CONFIDENCE_MARKER}"
    return f"[{segment.start_s:.1f}-{segment.end_s:.1f}] {segment.speaker}{flag}: {segment.text}"


def format_transcript(
    transcript: Transcript,
    *,
    max_chars: int | None = None,
    low_confidence_below: float = 0.0,
) -> str:
    lines = [format_segment_line(segment, low_confidence_below=low_confidence_below) for segment in transcript.segments]
    if max_chars is None:
        return "\n".join(lines)
    return _join_whole_lines(lines, max_chars)


def window_excerpt(
    transcript: Transcript,
    *,
    start_s: float,
    end_s: float,
    margin_s: float = 30.0,
    max_chars: int = 6000,
    low_confidence_below: float = 0.0,
) -> str:
    low = min(start_s, end_s) - margin_s
    high = max(start_s, end_s) + margin_s
    lines = [
        format_segment_line(segment, low_confidence_below=low_confidence_below)
        for segment in transcript.segments
        if segment.end_s >= low and segment.start_s <= high
    ]
    return _join_whole_lines(lines, max_chars)


def chunk_transcript(
    transcript: Transcript,
    chapters: list[Chapter],
    *,
    target_seconds: int = 600,
    min_seconds: int = 120,
    duration_s: float | None = None,
) -> list[TranscriptChunk]:
    """Split a transcript into chapter-aligned chunks for map-reduce summarization.

    Adjacent chapters shorter than `min_seconds` are merged, chapters longer than
    twice `target_seconds` are split evenly, and without chapters fixed
    `target_seconds` windows are used. Segments are never split: each goes to the
    window containing its start time, so window edges snap to segment boundaries.
    """
    duration = duration_s or transcript.duration_s
    if duration <= 0:
        duration = max((segment.end_s for segment in transcript.segments), default=0.0)
    windows = _chapter_windows(chapters, duration, min_seconds) if chapters else []
    if not windows:
        windows = _fixed_windows(duration, target_seconds)
    windows = _split_long_windows(windows, target_seconds)
    chunks = [TranscriptChunk(title=title, start_s=start, end_s=end, segments=[]) for title, start, end in windows]
    if not chunks:
        return []
    for segment in transcript.segments:
        index = _window_index(windows, segment.start_s)
        chunks[index].segments.append(segment)
    return [chunk for chunk in chunks if chunk.segments]


def _chapter_windows(chapters: list[Chapter], duration: float, min_seconds: int) -> list[tuple[str | None, float, float]]:
    bounded = []
    for chapter in sorted(chapters, key=lambda item: item.start_s):
        start = max(0.0, chapter.start_s)
        end = min(duration, chapter.end_s) if duration > 0 else chapter.end_s
        if end > start:
            bounded.append((chapter.title or None, start, end))
    if not bounded:
        return []
    merged: list[list] = []
    for title, start, end in bounded:
        if merged and (merged[-1][2] - merged[-1][1]) < min_seconds:
            merged[-1][2] = end
        else:
            merged.append([title, start, end])
    if len(merged) > 1 and (merged[-1][2] - merged[-1][1]) < min_seconds:
        merged[-2][2] = merged[-1][2]
        merged.pop()
    merged[0][1] = 0.0
    if duration > 0:
        merged[-1][2] = max(merged[-1][2], duration)
    return [(title, start, end) for title, start, end in merged]


def _fixed_windows(duration: float, target_seconds: int) -> list[tuple[str | None, float, float]]:
    if duration <= 0:
        return [(None, 0.0, float("inf"))]
    step = max(1, int(target_seconds))
    return [(None, float(start), float(min(start + step, duration))) for start in range(0, int(math.ceil(duration)), step)]


def _split_long_windows(
    windows: list[tuple[str | None, float, float]], target_seconds: int
) -> list[tuple[str | None, float, float]]:
    result: list[tuple[str | None, float, float]] = []
    for title, start, end in windows:
        length = end - start
        if not math.isfinite(length) or length <= 2 * target_seconds:
            result.append((title, start, end))
            continue
        parts = max(2, math.ceil(length / target_seconds))
        step = length / parts
        for index in range(parts):
            part_title = title if index == 0 else (f"{title} (cont.)" if title else None)
            result.append((part_title, start + index * step, start + (index + 1) * step))
    return result


def _window_index(windows: list[tuple[str | None, float, float]], timestamp: float) -> int:
    for index, (_, start, end) in enumerate(windows):
        if start <= timestamp < end:
            return index
    return 0 if timestamp < windows[0][1] else len(windows) - 1


def _join_whole_lines(lines: list[str], max_chars: int) -> str:
    kept: list[str] = []
    total = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if total + cost > max_chars:
            break
        kept.append(line)
        total += cost
    return "\n".join(kept)
