from __future__ import annotations

from typing import TypeVar

from .models import CandidateFrame, FrameSelection

Frame = TypeVar("Frame", CandidateFrame, FrameSelection)


def sample_timeline(frames: list[Frame], limit: int, *, reserved: list[Frame] | None = None) -> list[Frame]:
    """Greedily fill the largest uncovered time gaps, preserving reserved picks."""
    if limit <= 0:
        return []
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id))
    if len(ordered) <= limit:
        return ordered
    reserved_ids = {frame.frame_id for frame in reserved or []}
    selected = [frame for frame in ordered if frame.frame_id in reserved_ids]
    if len(selected) > limit:
        return sample_timeline(selected, limit)
    if not selected:
        if limit == 1:
            midpoint = (ordered[0].timestamp_s + ordered[-1].timestamp_s) / 2
            return [min(ordered, key=lambda frame: abs(frame.timestamp_s - midpoint))]
        selected = [ordered[0], ordered[-1]]
    selected_ids = {frame.frame_id for frame in selected}
    remaining = [frame for frame in ordered if frame.frame_id not in selected_ids]
    distances = [min(abs(frame.timestamp_s - pick.timestamp_s) for pick in selected) for frame in remaining]
    while len(selected) < limit and remaining:
        index = max(range(len(remaining)), key=lambda i: (
            distances[i], getattr(remaining[i], "score", 0), -remaining[i].timestamp_s,
        ))
        pick = remaining.pop(index)
        distances.pop(index)
        selected.append(pick)
        distances = [min(distance, abs(frame.timestamp_s - pick.timestamp_s))
                     for frame, distance in zip(remaining, distances)]
    return sorted(selected, key=lambda frame: (frame.timestamp_s, frame.frame_id))
