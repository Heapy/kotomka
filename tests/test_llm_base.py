import json
from pathlib import Path
from typing import Any

from kotomka.config import get_settings
from kotomka.models import (
    CandidateFrame,
    FrameSelection,
    SourceArtifact,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    VideoMetadata,
)
from kotomka.providers.llm.json_base import JsonLlmProviderBase, frame_selections_from_payload


class StubJsonLlm(JsonLlmProviderBase):
    name = "stub"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.payloads = list(payloads)

    def _request_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.payloads.pop(0)

    def _scoring_model(self) -> str | None:
        return "stub-scoring-model"


def make_transcript() -> Transcript:
    words = [TranscriptWord(start_s=0, end_s=1, text="intro")]
    return Transcript(
        language="en",
        duration_s=300,
        speakers=["Speaker A"],
        segments=[
            TranscriptSegment(start_s=0, end_s=10, speaker="Speaker A", text="intro words", words=words),
            TranscriptSegment(start_s=100, end_s=110, speaker="Speaker A", text="middle words"),
            TranscriptSegment(start_s=200, end_s=210, speaker="Speaker A", text="late words"),
        ],
        words=words,
    )


def make_frames(tmp_path: Path, *, count: int, start_ts: float = 100.0) -> list[CandidateFrame]:
    frames = []
    for index in range(count):
        path = tmp_path / f"frame-{index:02d}.png"
        path.write_bytes(b"\x89PNG fake")
        frames.append(CandidateFrame(frame_id=f"frame-{index:02d}", timestamp_s=start_ts + index, path=path))
    return frames


def make_source() -> SourceArtifact:
    return SourceArtifact(
        metadata=VideoMetadata(source_url="https://example.com/v", title="Talk", duration_s=300),
        video_path=Path("/nonexistent/video.mp4"),
        audio_path=Path("/nonexistent/audio.flac"),
    )


def test_score_frames_uses_windowed_excerpt_and_scoring_model(tmp_path: Path) -> None:
    frames = make_frames(tmp_path, count=3, start_ts=100.0)
    stub = StubJsonLlm([{"frames": []}])

    stub.score_frames(frames, make_transcript())

    call = stub.calls[0]
    assert "middle words" in call["text"]
    assert "intro words" not in call["text"]
    assert "late words" not in call["text"]
    assert call["model"] == "stub-scoring-model"
    assert call["image_detail"] == "low"
    assert len(call["images"]) == 3
    assert "frame_id=frame-00" in call["images"][0].label


def test_build_report_sends_compact_transcript_without_words(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame-00.png").write_bytes(b"\x89PNG fake")
    selections = [
        FrameSelection(frame_id="frame-00", timestamp_s=100.0, image_path="frame-00.png", score=0.9, caption="cap")
    ]
    stub = StubJsonLlm([{"summary": "s", "sections": []}])

    report = stub.build_report(
        source=make_source(),
        transcript=make_transcript(),
        frames=selections,
        output_language="ru",
        work_dir=tmp_path,
    )

    call = stub.calls[0]
    assert '"words"' not in call["text"]
    assert "[0.0-10.0] Speaker A: intro words" in call["text"]
    assert call["image_detail"] == "high"
    assert call.get("model") is None
    assert len(call["images"]) == 1
    assert report.summary == "s"


def test_build_report_caps_images_and_skips_missing_files(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    selections = []
    for index in range(20):
        name = f"frame-{index:02d}.png"
        if index != 3:
            (frames_dir / name).write_bytes(b"\x89PNG fake")
        selections.append(
            FrameSelection(frame_id=f"frame-{index:02d}", timestamp_s=float(index), image_path=name, score=0.9)
        )
    stub = StubJsonLlm([{"summary": "s", "sections": []}])

    stub.build_report(
        source=make_source(),
        transcript=make_transcript(),
        frames=selections,
        output_language="ru",
        work_dir=tmp_path,
    )

    images = stub.calls[0]["images"]
    assert len(images) == 15  # 16 cap minus one missing file
    assert all(image.path.exists() for image in images)


def test_build_report_without_work_dir_sends_no_images() -> None:
    stub = StubJsonLlm([{"summary": "s", "sections": []}])
    stub.build_report(
        source=make_source(),
        transcript=make_transcript(),
        frames=[FrameSelection(frame_id="x", timestamp_s=0, image_path="x.png", score=0.9)],
        output_language="ru",
    )
    assert stub.calls[0]["images"] == []


def test_build_report_skips_malformed_sections(tmp_path: Path) -> None:
    stub = StubJsonLlm(
        [
            {
                "summary": "s",
                "sections": [
                    {"title": "ok", "start_s": 0, "end_s": 10, "body": "b", "frame_ids": [], "citations": []},
                    {"title": "missing fields"},
                    "not-a-dict",
                ],
            }
        ]
    )
    report = stub.build_report(
        source=make_source(),
        transcript=make_transcript(),
        frames=[],
        output_language="ru",
        work_dir=tmp_path,
    )
    assert [section.title for section in report.sections] == ["ok"]


def test_build_report_applies_speaker_names_and_strips_words(tmp_path: Path) -> None:
    stub = StubJsonLlm(
        [
            {
                "summary": "s",
                "sections": [],
                "speaker_names": [
                    {"label": "Speaker A", "name": "Alice"},
                    {"label": "", "name": "ignored"},
                ],
            }
        ]
    )
    transcript = make_transcript()
    report = stub.build_report(
        source=make_source(),
        transcript=transcript,
        frames=[],
        output_language="ru",
        work_dir=tmp_path,
    )
    assert report.transcript.speakers == ["Alice"]
    assert {segment.speaker for segment in report.transcript.segments} == {"Alice"}
    assert report.transcript.words is None
    assert all(segment.words is None for segment in report.transcript.segments)
    # the source transcript object is untouched
    assert transcript.speakers == ["Speaker A"]
    assert transcript.words is not None


def test_build_report_map_reduces_long_transcripts(tmp_path: Path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "report_single_pass_max_chars", 10)
    monkeypatch.setattr(settings, "report_chunk_target_seconds", 100)
    notes_payload = {
        "chunk_summary": "covers things",
        "notes": [{"kind": "number", "text": "42 rps", "timestamp_s": 5.0, "importance": 0.9}],
    }
    report_payload = {"summary": "s", "sections": [], "speaker_names": []}
    stub = StubJsonLlm([notes_payload, notes_payload, notes_payload, report_payload])

    report = stub.build_report(
        source=make_source(),
        transcript=make_transcript(),
        frames=[],
        output_language="ru",
        work_dir=tmp_path,
    )

    assert report.summary == "s"
    assert len(stub.calls) == 4  # three 100s chunks with segments + one reduce
    notes_calls = [call for call in stub.calls if call["schema_name"] == "chunk_notes"]
    assert len(notes_calls) == 3
    reduce_call = stub.calls[-1]
    assert reduce_call["schema_name"] == "video_report"
    assert "42 rps" in reduce_call["text"]
    assert "intro words" not in reduce_call["text"]
    saved_notes = json.loads((tmp_path / "notes.json").read_text(encoding="utf-8"))
    assert len(saved_notes) == 3
    assert saved_notes[0]["notes"][0]["text"] == "42 rps"


def test_build_report_tolerates_failed_notes_chunk(tmp_path: Path, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "report_single_pass_max_chars", 10)
    monkeypatch.setattr(settings, "report_chunk_target_seconds", 100)

    class FlakyStub(StubJsonLlm):
        def _request_json(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["schema_name"] == "chunk_notes" and not self.calls:
                self.calls.append(kwargs)
                raise RuntimeError("boom")
            return super()._request_json(**kwargs)

    notes_payload = {"chunk_summary": "ok", "notes": []}
    stub = FlakyStub([notes_payload, notes_payload, {"summary": "s", "sections": [], "speaker_names": []}])

    report = stub.build_report(
        source=make_source(),
        transcript=make_transcript(),
        frames=[],
        output_language="ru",
        work_dir=tmp_path,
    )

    assert report.summary == "s"
    saved_notes = json.loads((tmp_path / "notes.json").read_text(encoding="utf-8"))
    assert len(saved_notes) == 3
    assert saved_notes[0]["chunk_summary"] == ""


def test_frame_selections_from_payload_filters_low_scores(tmp_path: Path) -> None:
    frames = make_frames(tmp_path, count=2)
    payload = {
        "frames": [
            {"frame_id": "frame-00", "score": 0.9, "content_type": "slide", "caption": "c", "reason": "r", "ocr_text": None},
            {"frame_id": "frame-01", "score": 0.2, "content_type": "slide", "caption": "c", "reason": "r", "ocr_text": None},
            {"frame_id": "ghost", "score": 0.9, "content_type": "slide", "caption": "c", "reason": "r", "ocr_text": None},
        ]
    }
    selections = frame_selections_from_payload(payload, frames)
    assert [selection.frame_id for selection in selections] == ["frame-00"]
