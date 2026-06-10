from pathlib import Path

from kotomka.models import CandidateFrame, FrameSelection, Transcript
from kotomka.providers.llm.base import LlmProvider
from kotomka.worker import _fallback_frame_selection, _score_frames_across_timeline, _select_diverse_frames


class RecordingLlm(LlmProvider):
    name = "recording"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def score_frames(self, frames: list[CandidateFrame], transcript: Transcript) -> list[FrameSelection]:
        del transcript
        self.batches.append([frame.frame_id for frame in frames])
        return [
            FrameSelection(
                frame_id=frame.frame_id,
                timestamp_s=frame.timestamp_s,
                image_path=frame.path.name,
                score=1.0 if frame.frame_id in {"f-00", "f-49"} else 0.5,
                caption=frame.frame_id,
            )
            for frame in frames
        ]

    def build_report(self, **kwargs):
        raise NotImplementedError


def test_score_frames_batches_entire_timeline(tmp_path: Path) -> None:
    frames = _frames(tmp_path, count=50)
    llm = RecordingLlm()

    selected = _score_frames_across_timeline(
        llm,
        frames,
        Transcript(),
        batch_size=10,
        max_selected=4,
        min_gap_seconds=20,
    )

    assert len(llm.batches) == 5
    assert llm.batches[0][0] == "f-00"
    assert llm.batches[-1][-1] == "f-49"
    assert "f-49" in {frame.frame_id for frame in selected}


def test_select_diverse_frames_prefers_spread_then_fills() -> None:
    selections = [
        FrameSelection(frame_id="a", timestamp_s=0, image_path="a.png", score=0.99),
        FrameSelection(frame_id="b", timestamp_s=2, image_path="b.png", score=0.98),
        FrameSelection(frame_id="c", timestamp_s=100, image_path="c.png", score=0.7),
    ]

    selected = _select_diverse_frames(selections, max_selected=2, min_gap_seconds=20)

    assert [frame.frame_id for frame in selected] == ["a", "c"]


def test_fallback_selection_spans_timeline(tmp_path: Path) -> None:
    selected = _fallback_frame_selection(_frames(tmp_path, count=10), max_selected=3)

    assert [frame.frame_id for frame in selected] == ["f-00", "f-04", "f-09"]


def _frames(tmp_path: Path, *, count: int) -> list[CandidateFrame]:
    frames: list[CandidateFrame] = []
    for index in range(count):
        path = tmp_path / f"f-{index:02d}.png"
        path.write_bytes(b"png")
        frames.append(CandidateFrame(frame_id=f"f-{index:02d}", timestamp_s=float(index * 10), path=path))
    return frames
