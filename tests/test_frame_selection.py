from pathlib import Path

from kotomka.models import CandidateFrame, Chapter, FrameSelection, Transcript
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


def test_select_diverse_frames_guarantees_chapter_coverage() -> None:
    selections = [
        FrameSelection(frame_id="a", timestamp_s=10, image_path="a.png", score=0.99),
        FrameSelection(frame_id="b", timestamp_s=15, image_path="b.png", score=0.98),
        FrameSelection(frame_id="c", timestamp_s=300, image_path="c.png", score=0.5),
    ]
    chapters = [
        Chapter(title="Intro", start_s=0, end_s=100),
        Chapter(title="Demo", start_s=200, end_s=400),
        Chapter(title="Empty", start_s=400, end_s=500),
    ]

    selected = _select_diverse_frames(selections, max_selected=2, min_gap_seconds=20, chapters=chapters)

    # without chapters the two top-scored frames (a, b) would win; the Demo
    # chapter's only candidate displaces the lower of them
    assert [frame.frame_id for frame in selected] == ["a", "c"]


def test_select_diverse_frames_caps_chapter_picks_by_score() -> None:
    selections = [
        FrameSelection(frame_id="low", timestamp_s=50, image_path="l.png", score=0.3),
        FrameSelection(frame_id="high", timestamp_s=150, image_path="h.png", score=0.9),
        FrameSelection(frame_id="mid", timestamp_s=250, image_path="m.png", score=0.6),
    ]
    chapters = [
        Chapter(title="One", start_s=0, end_s=100),
        Chapter(title="Two", start_s=100, end_s=200),
        Chapter(title="Three", start_s=200, end_s=300),
    ]

    selected = _select_diverse_frames(selections, max_selected=2, min_gap_seconds=0, chapters=chapters)

    assert [frame.frame_id for frame in selected] == ["high", "mid"]


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
