from __future__ import annotations

import re
from collections import Counter
from importlib import util
from pathlib import Path

from PIL import Image, ImageChops

from .models import CandidateFrame

_TOKEN = re.compile(r"[+-]?\d+(?:[.,]\d+)*%?|[\w'-]+|[<>=]+")


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

    Require an exact text superset and unchanged existing foreground on a flat
    slide background. Uncertain matches, OCR omissions, charts and changed metrics
    survive for scoring rather than being mistaken for incremental bullet builds.
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
                if current <= later_tokens and _only_adds_content(frame.path, frames[later].path):
                    superseded = True
                    break
        if not superseded:
            kept.append(frame)
    return kept


def _token_set(text: str | None) -> Counter[str]:
    return Counter(_TOKEN.findall((text or "").casefold()))


def _only_adds_content(first: Path, second: Path) -> bool:
    try:
        with Image.open(first) as image:
            before = image.convert("RGB")
        with Image.open(second) as image:
            after = image.convert("RGB")
        if before.size != after.size:
            return False
        w, h = before.size
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        background = before.getpixel(corners[0])
        if any(max(abs(a - b) for a, b in zip(image.getpixel(point), background)) > 8
               for image in (before, after) for point in corners):
            return False
        delta = ImageChops.difference(before, Image.new("RGB", before.size, background))
        red, green, blue = delta.split()
        foreground = ImageChops.lighter(ImageChops.lighter(red, green), blue).point(lambda value: 255 if value > 16 else 0)
        changes = ImageChops.difference(before, after)
        changes = Image.composite(changes, Image.new("RGB", before.size), foreground)
        return max(high for _, high in changes.getextrema()) <= 8
    except (OSError, ValueError):
        return False
