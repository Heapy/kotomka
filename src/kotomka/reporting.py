from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .models import Report, ReportSection
from .utils import read_json, write_json

CITATION_PATTERN = re.compile(r"\[t=((?:\d+(?:\.\d+)?\s*,\s*)*\d+(?:\.\d+)?)\]")
CODE_PATTERN = re.compile(r"(`{3,}|~{3,})[\s\S]*?(?:\1|$)|(`+)[\s\S]*?\2")


def substitute_citations(text: str, replace: Callable[[re.Match[str]], str]) -> str:
    """Transform explicit timestamp markup only outside fenced and inline code."""
    result: list[str] = []
    end = 0
    for code in CODE_PATTERN.finditer(text):
        result.append(CITATION_PATTERN.sub(replace, text[end:code.start()]))
        result.append(code.group())
        end = code.end()
    result.append(CITATION_PATTERN.sub(replace, text[end:]))
    return "".join(result)


def save_report(report: Report, path: Path) -> None:
    write_json(path, report.model_dump())


def load_report(path: Path) -> Report:
    return Report.model_validate(read_json(path))


def normalize_report(report: Report, *, tolerance_s: float = 5.0) -> Report:
    """Deterministically clean LLM-produced timestamps so rendered citations stay truthful.

    Citations are snapped to nearby transcript segment starts, values beyond the video
    duration are clamped, section bounds are ordered and clamped, and frame references
    that do not exist in the report are dropped. Only explicit ``[t=12.3]`` groups
    in prose are rewritten; ordinary bracketed numbers and code stay untouched.
    """
    duration = report.video.duration_s or report.transcript.duration_s
    starts = sorted({
        segment.start_s for segment in report.transcript.segments
        if 0 <= segment.start_s and (duration <= 0 or segment.start_s <= duration)
    })
    known_frame_ids = {frame.frame_id for frame in report.frames}

    def snap_value(value: float) -> float:
        nearest = min(starts, key=lambda start: abs(start - value), default=None)
        if nearest is not None and abs(nearest - value) <= tolerance_s:
            value = nearest
        return min(max(value, 0.0), duration)

    def normalize_text(text: str) -> str:
        if duration <= 0:
            return text
        return _normalize_inline_citations(text, starts, tolerance_s, duration)

    sections: list[ReportSection] = []
    for section in report.sections:
        start_s, end_s = sorted((section.start_s, section.end_s))
        if duration > 0:
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
    return substitute_citations(text, lambda match: _rewrite_citation_group(match, starts, tolerance_s, duration))


def _rewrite_citation_group(match: re.Match[str], starts: list[float], tolerance_s: float, duration: float) -> str:
    rewritten: list[float] = []
    for raw in match.group(1).split(","):
        value = float(raw.strip())
        nearest = min(starts, key=lambda start: abs(start - value), default=None)
        if nearest is not None and abs(nearest - value) <= tolerance_s:
            value = nearest
        rewritten.append(min(max(value, 0.0), duration))
    deduped = list(dict.fromkeys(rewritten))
    return "[t=" + ", ".join(_format_seconds(value) for value in deduped) + "]"


def _format_seconds(value: float) -> str:
    return f"{value:g}"
