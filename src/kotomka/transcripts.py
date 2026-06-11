from __future__ import annotations

from .models import Transcript, TranscriptSegment

LOW_CONFIDENCE_MARKER = "[low-confidence]"


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
