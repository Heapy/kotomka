from __future__ import annotations

import re
from pathlib import Path

from .models import Report, ReportSection
from .utils import read_json, write_json

CITATION_PATTERN = re.compile(r"\[((?:\d+(?:\.\d+)?\s*,\s*)*\d+(?:\.\d+)?)\]")

CODE_FENCE = "```"


def save_report(report: Report, path: Path) -> None:
    write_json(path, report.model_dump())


def load_report(path: Path) -> Report:
    return Report.model_validate(read_json(path))


def normalize_report(report: Report, *, tolerance_s: float = 5.0) -> Report:
    """Deterministically clean LLM-produced timestamps so rendered citations stay truthful.

    Citations are snapped to nearby transcript segment starts, values beyond the video
    duration are clamped, section bounds are ordered and clamped, and frame references
    that do not exist in the report are dropped. Inline ``[12.3]`` groups in prose are
    rewritten only when every value in the group is clearly a timestamp, so bracketed
    numbers that are part of regular text or code survive untouched.
    """
    duration = report.transcript.duration_s or report.video.duration_s
    starts = sorted({segment.start_s for segment in report.transcript.segments})
    known_frame_ids = {frame.frame_id for frame in report.frames}
    can_snap = duration > 0 and bool(starts)

    def snap_value(value: float) -> float:
        nearest = min(starts, key=lambda start: abs(start - value))
        if abs(nearest - value) <= tolerance_s:
            return nearest
        return min(max(value, 0.0), duration)

    def normalize_text(text: str) -> str:
        if not can_snap:
            return text
        return _normalize_inline_citations(text, starts, tolerance_s, duration)

    sections: list[ReportSection] = []
    for section in report.sections:
        start_s, end_s = sorted((section.start_s, section.end_s))
        if can_snap:
            start_s = min(max(start_s, 0.0), duration)
            end_s = min(max(end_s, 0.0), duration)
            citations = sorted({snap_value(value) for value in section.citations})
        else:
            citations = sorted(set(section.citations))
        sections.append(
            section.model_copy(
                update={
                    "start_s": start_s,
                    "end_s": end_s,
                    "citations": citations,
                    "frame_ids": [frame_id for frame_id in section.frame_ids if frame_id in known_frame_ids],
                    "body": normalize_text(section.body),
                }
            )
        )
    return report.model_copy(update={"summary": normalize_text(report.summary), "sections": sections})


def _normalize_inline_citations(text: str, starts: list[float], tolerance_s: float, duration: float) -> str:
    if not text:
        return text
    chunks = text.split(CODE_FENCE)
    for index in range(0, len(chunks), 2):
        chunks[index] = CITATION_PATTERN.sub(
            lambda match: _rewrite_citation_group(match, starts, tolerance_s, duration),
            chunks[index],
        )
    return CODE_FENCE.join(chunks)


def _rewrite_citation_group(match: re.Match[str], starts: list[float], tolerance_s: float, duration: float) -> str:
    rewritten: list[float] = []
    for raw in match.group(1).split(","):
        value = float(raw.strip())
        nearest = min(starts, key=lambda start: abs(start - value))
        if value > duration:
            rewritten.append(nearest if abs(nearest - value) <= tolerance_s else duration)
        elif abs(nearest - value) <= 1e-6:
            rewritten.append(nearest)
        elif abs(nearest - value) <= tolerance_s and value > tolerance_s:
            # Snapping tiny values would rewrite prose like "array [1, 2]" into [0];
            # values at or below the tolerance are left for the human eye instead.
            rewritten.append(nearest)
        else:
            return match.group(0)
    deduped = list(dict.fromkeys(rewritten))
    return "[" + ", ".join(_format_seconds(value) for value in deduped) + "]"


def _format_seconds(value: float) -> str:
    return f"{value:g}"
