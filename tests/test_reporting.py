from kotomka.models import (
    FrameSelection,
    Report,
    ReportSection,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)
from kotomka.reporting import normalize_report


def make_report(*, summary: str = "", sections: list[ReportSection] | None = None) -> Report:
    transcript = Transcript(
        language="en",
        duration_s=180,
        speakers=["Speaker A"],
        segments=[
            TranscriptSegment(start_s=0, end_s=45, speaker="Speaker A", text="intro"),
            TranscriptSegment(start_s=45, end_s=120, speaker="Speaker A", text="middle"),
            TranscriptSegment(start_s=120, end_s=180, speaker="Speaker A", text="end"),
        ],
    )
    frames = [
        FrameSelection(frame_id="f1", timestamp_s=10, image_path="f1.png", score=0.9),
        FrameSelection(frame_id="f2", timestamp_s=130, image_path="f2.png", score=0.8),
    ]
    return Report(
        video=VideoMetadata(source_url="https://example.com/v", duration_s=180),
        summary=summary,
        sections=sections or [],
        frames=frames,
        transcript=transcript,
    )


def make_section(**overrides) -> ReportSection:
    defaults = dict(title="Section", start_s=0.0, end_s=60.0, body="", frame_ids=[], citations=[])
    defaults.update(overrides)
    return ReportSection(**defaults)


def test_citations_snap_clamp_dedupe_sort() -> None:
    report = make_report(sections=[make_section(citations=[47.0, 44.0, 300.0, 47.0])])
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.sections[0].citations == [45.0, 180.0]


def test_citations_outside_tolerance_stay_put() -> None:
    report = make_report(sections=[make_section(citations=[30.0])])
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.sections[0].citations == [30.0]


def test_section_bounds_are_ordered_and_clamped() -> None:
    report = make_report(sections=[make_section(start_s=250.0, end_s=100.0)])
    normalized = normalize_report(report, tolerance_s=5.0)
    section = normalized.sections[0]
    assert section.start_s == 100.0
    assert section.end_s == 180.0


def test_unknown_frame_ids_are_dropped() -> None:
    report = make_report(sections=[make_section(frame_ids=["f1", "ghost", "f2"])])
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.sections[0].frame_ids == ["f1", "f2"]


def test_inline_citations_snap_in_summary_and_body() -> None:
    report = make_report(
        summary="Discussed at [44.8].",
        sections=[make_section(body="See [44.8, 121] for details.")],
    )
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.summary == "Discussed at [45]."
    assert normalized.sections[0].body == "See [45, 120] for details."


def test_inline_over_duration_value_is_clamped() -> None:
    report = make_report(sections=[make_section(body="Wrap-up at [999].")])
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.sections[0].body == "Wrap-up at [180]."


def test_inline_prose_numbers_survive() -> None:
    body = "Store them in array [1, 2] before use, see [7, 9]."
    report = make_report(sections=[make_section(body=body)])
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.sections[0].body == body


def test_inline_code_blocks_are_untouched() -> None:
    body = "Snap [44.8] here.\n```python\nitems = data[44]\n```\nAnd [44.8] here."
    report = make_report(sections=[make_section(body=body)])
    normalized = normalize_report(report, tolerance_s=5.0)
    assert "items = data[44]" in normalized.sections[0].body
    assert normalized.sections[0].body.startswith("Snap [45] here.")
    assert normalized.sections[0].body.endswith("And [45] here.")


def test_normalize_is_idempotent() -> None:
    report = make_report(
        summary="Discussed at [44.8].",
        sections=[make_section(body="See [44.8, 121].", citations=[44.0], frame_ids=["f1"])],
    )
    once = normalize_report(report, tolerance_s=5.0)
    twice = normalize_report(once, tolerance_s=5.0)
    assert twice == once


def test_empty_transcript_passes_through_numbers() -> None:
    report = make_report(sections=[make_section(citations=[999.0], body="See [999].")])
    report = report.model_copy(
        update={
            "transcript": Transcript(language="en", duration_s=0, segments=[]),
            "video": VideoMetadata(source_url="https://example.com/v", duration_s=0),
        }
    )
    normalized = normalize_report(report, tolerance_s=5.0)
    assert normalized.sections[0].citations == [999.0]
    assert normalized.sections[0].body == "See [999]."
