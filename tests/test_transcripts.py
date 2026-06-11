from kotomka.models import Transcript, TranscriptSegment
from kotomka.transcripts import format_segment_line, format_transcript, window_excerpt


def make_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=300,
        speakers=["Speaker A"],
        segments=[
            TranscriptSegment(start_s=0, end_s=10, speaker="Speaker A", text="intro words", confidence=0.9),
            TranscriptSegment(start_s=100, end_s=110, speaker="Speaker A", text="middle words", confidence=0.3),
            TranscriptSegment(start_s=200, end_s=210, speaker="Speaker A", text="late words"),
        ],
    )


def test_format_segment_line_flags_low_confidence() -> None:
    segment = TranscriptSegment(start_s=1.25, end_s=4.5, speaker="Speaker B", text="hello", confidence=0.2)
    assert format_segment_line(segment) == "[1.2-4.5] Speaker B: hello"
    assert format_segment_line(segment, low_confidence_below=0.5) == "[1.2-4.5] Speaker B [low-confidence]: hello"
    no_confidence = segment.model_copy(update={"confidence": None})
    assert "[low-confidence]" not in format_segment_line(no_confidence, low_confidence_below=0.5)


def test_format_transcript_joins_and_truncates_whole_lines() -> None:
    transcript = make_transcript()
    full = format_transcript(transcript)
    assert full.splitlines() == [
        "[0.0-10.0] Speaker A: intro words",
        "[100.0-110.0] Speaker A: middle words",
        "[200.0-210.0] Speaker A: late words",
    ]
    first_line = full.splitlines()[0]
    truncated = format_transcript(transcript, max_chars=len(first_line) + 5)
    assert truncated == first_line


def test_window_excerpt_selects_overlapping_segments_with_margin() -> None:
    transcript = make_transcript()
    excerpt = window_excerpt(transcript, start_s=100, end_s=110, margin_s=30)
    assert "middle words" in excerpt
    assert "intro words" not in excerpt
    assert "late words" not in excerpt
    wide = window_excerpt(transcript, start_s=100, end_s=110, margin_s=95)
    assert "intro words" in wide
    assert "late words" in wide
