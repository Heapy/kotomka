from __future__ import annotations

import re
from importlib import util
from pathlib import Path

from .models import CandidateFrame

_TOKEN = re.compile(r"[\w'-]+")


def ocr_available() -> bool:
    return util.find_spec("ocrmac") is not None


def ocr_image(path: Path) -> str | None:
    """Recognize text via Apple Vision (ocrmac). Never raises; None when unavailable or failed."""
    if not ocr_available():
        return None
    try:
        from ocrmac import ocrmac

        annotations = ocrmac.OCR(str(path)).recognize()
        lines = [str(text).strip() for text, _confidence, _bbox in annotations if str(text).strip()]
        return "\n".join(lines) or None
    except Exception:
        return None


def annotate_frames_with_ocr(frames: list[CandidateFrame]) -> list[CandidateFrame]:
    return [frame.model_copy(update={"ocr_text": ocr_image(frame.path)}) for frame in frames]


def dedupe_ocr_supersets(frames: list[CandidateFrame], *, window_s: float = 90.0) -> list[CandidateFrame]:
    """Drop bullet-build predecessors among chronologically ordered frames.

    A frame whose OCR tokens are (almost) contained in a later frame within
    `window_s` is an incomplete version of the same slide mid-build; the later,
    complete slide is kept. The comparison stays time-windowed because builds are
    temporally adjacent, while far-apart slides sharing template text (footers,
    logos) must not match.
    """
    tokens = [_token_set(frame.ocr_text) for frame in frames]
    kept: list[CandidateFrame] = []
    for index, frame in enumerate(frames):
        current = tokens[index]
        superseded = False
        if len(current) >= 3:
            for later in range(index + 1, len(frames)):
                if frames[later].timestamp_s - frame.timestamp_s > window_s:
                    break
                later_tokens = tokens[later]
                if len(later_tokens) <= len(current):
                    continue
                if len(current - later_tokens) <= len(current) // 10:
                    superseded = True
                    break
        if not superseded:
            kept.append(frame)
    return kept


def _token_set(text: str | None) -> frozenset[str]:
    if not text:
        return frozenset()
    return frozenset(token for token in _TOKEN.findall(text.lower()) if len(token) > 1)
