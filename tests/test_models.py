import pytest

from kotomka.models import JobCreate, Report, Transcript, TranscriptSegment


def test_job_create_validates_url() -> None:
    with pytest.raises(ValueError):
        JobCreate(source_url="not-a-url")
    assert JobCreate(source_url="https://youtube.com/watch?v=x").source_url.startswith("https://")


def test_job_create_validates_speakers_expected() -> None:
    assert JobCreate(source_url="https://example.com/v").speakers_expected is None
    assert JobCreate(source_url="https://example.com/v", speakers_expected=3).speakers_expected == 3
    with pytest.raises(ValueError):
        JobCreate(source_url="https://example.com/v", speakers_expected=0)
    with pytest.raises(ValueError):
        JobCreate(source_url="https://example.com/v", speakers_expected=11)


def test_report_without_assessment_still_loads() -> None:
    legacy = {
        "video": {"source_url": "https://example.com/v", "title": "Old"},
        "summary": "Old summary",
        "sections": [],
        "frames": [],
        "transcript": {"language": "en", "duration_s": 10, "speakers": [], "segments": []},
        "generated_at": "2026-01-01T00:00:00+00:00",
        "output_language": "ru",
    }
    report = Report.model_validate(legacy)
    assert report.assessment is None


def test_transcript_dedupes_speakers() -> None:
    transcript = Transcript(
        speakers=["Speaker A", "Speaker A", "Speaker B"],
        segments=[TranscriptSegment(start_s=0, end_s=1, speaker="Speaker A", text="hello")],
    )
    assert transcript.speakers == ["Speaker A", "Speaker B"]

