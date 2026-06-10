import pytest

from kotomka.models import JobCreate, Transcript, TranscriptSegment


def test_job_create_validates_url() -> None:
    with pytest.raises(ValueError):
        JobCreate(source_url="not-a-url")
    assert JobCreate(source_url="https://youtube.com/watch?v=x").source_url.startswith("https://")


def test_transcript_dedupes_speakers() -> None:
    transcript = Transcript(
        speakers=["Speaker A", "Speaker A", "Speaker B"],
        segments=[TranscriptSegment(start_s=0, end_s=1, speaker="Speaker A", text="hello")],
    )
    assert transcript.speakers == ["Speaker A", "Speaker B"]

