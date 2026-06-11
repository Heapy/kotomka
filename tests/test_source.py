import json
from pathlib import Path

from kotomka.source import MAX_DESCRIPTION_CHARS, _metadata_from_info


def write_info(tmp_path: Path, data: dict) -> Path:
    info_path = tmp_path / "source.info.json"
    info_path.write_text(json.dumps(data), encoding="utf-8")
    return info_path


def test_metadata_from_info_maps_extended_fields(tmp_path: Path) -> None:
    info_path = write_info(
        tmp_path,
        {
            "title": "Talk",
            "webpage_url": "https://example.com/watch?v=1",
            "duration": 120,
            "uploader": "Uploader",
            "channel": "Channel",
            "thumbnail": "https://example.com/thumb.jpg",
            "description": "  Slides and notes  ",
            "tags": ["python", "  ", 42, "FastAPI"],
            "upload_date": "20260115",
            "language": "ru",
            "chapters": [
                {"title": "Intro", "start_time": 0, "end_time": 60},
                {"title": "Bad", "start_time": 90, "end_time": 30},
                {"title": "Demo", "start_time": "60", "end_time": 120.5},
                "not-a-dict",
            ],
        },
    )
    metadata = _metadata_from_info(info_path, "https://example.com/watch?v=1")
    assert metadata.description == "Slides and notes"
    assert metadata.tags == ["python", "FastAPI"]
    assert metadata.upload_date == "2026-01-15"
    assert metadata.language == "ru"
    assert metadata.channel == "Channel"
    assert [chapter.title for chapter in metadata.chapters] == ["Intro", "Demo"]
    assert metadata.chapters[1].start_s == 60.0
    assert metadata.chapters[1].end_s == 120.5


def test_metadata_from_info_defaults_when_fields_missing(tmp_path: Path) -> None:
    info_path = write_info(tmp_path, {"title": "Bare", "uploader": "Someone"})
    metadata = _metadata_from_info(info_path, "https://example.com/v")
    assert metadata.description is None
    assert metadata.tags == []
    assert metadata.upload_date is None
    assert metadata.language is None
    assert metadata.channel == "Someone"
    assert metadata.chapters == []


def test_metadata_from_info_missing_file(tmp_path: Path) -> None:
    metadata = _metadata_from_info(tmp_path / "absent.json", "https://example.com/v")
    assert metadata.title == "Untitled video"
    assert metadata.chapters == []


def test_metadata_from_info_caps_description(tmp_path: Path) -> None:
    info_path = write_info(tmp_path, {"title": "Long", "description": "x" * (MAX_DESCRIPTION_CHARS + 500)})
    metadata = _metadata_from_info(info_path, "https://example.com/v")
    assert metadata.description is not None
    assert len(metadata.description) == MAX_DESCRIPTION_CHARS
