import json
from pathlib import Path

import pytest

from kotomka.models import JobCreate
from kotomka.source import MAX_DESCRIPTION_CHARS, _find_downloaded_video, _metadata_from_info
from kotomka.source import _yt_dlp_command, _yt_dlp_cookie_args, _yt_dlp_error_message


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


def test_find_downloaded_video_ignores_audio_artifacts(tmp_path: Path) -> None:
    (tmp_path / "source.mp4").write_bytes(b"v" * 10)
    (tmp_path / "audio.flac").write_bytes(b"a" * 100)
    (tmp_path / "audio.mp3").write_bytes(b"a" * 100)
    (tmp_path / "source.info.json").write_text("{}", encoding="utf-8")
    assert _find_downloaded_video(tmp_path).name == "source.mp4"


def test_metadata_from_info_caps_description(tmp_path: Path) -> None:
    info_path = write_info(tmp_path, {"title": "Long", "description": "x" * (MAX_DESCRIPTION_CHARS + 500)})
    metadata = _metadata_from_info(info_path, "https://example.com/v")
    assert metadata.description is not None
    assert len(metadata.description) == MAX_DESCRIPTION_CHARS


def test_yt_dlp_command_uses_browser_cookies() -> None:
    payload = JobCreate(source_url="https://example.com/v", cookies_from_browser="firefox:default-release")
    command = _yt_dlp_command(payload, "source.%(ext)s")
    assert command[-3:] == ["--cookies-from-browser", "firefox:default-release", "https://example.com/v"]


def test_yt_dlp_command_uses_cookies_file(tmp_path: Path) -> None:
    cookies_path = tmp_path / "youtube-cookies.txt"
    cookies_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    payload = JobCreate(source_url="https://example.com/v", cookies_file=str(cookies_path))
    command = _yt_dlp_command(payload, "source.%(ext)s")
    assert command[-3:] == ["--cookies", str(cookies_path), "https://example.com/v"]


def test_yt_dlp_cookie_args_rejects_two_cookie_sources(tmp_path: Path) -> None:
    cookies_path = tmp_path / "youtube-cookies.txt"
    cookies_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    payload = JobCreate(source_url="https://example.com/v", cookies_from_browser="firefox", cookies_file=str(cookies_path))
    with pytest.raises(ValueError, match="Use either Cookies browser or Cookies file"):
        _yt_dlp_cookie_args(payload)


def test_yt_dlp_cookie_args_rejects_unknown_browser() -> None:
    payload = JobCreate(source_url="https://example.com/v", cookies_from_browser="netscape")
    with pytest.raises(ValueError, match="Unsupported cookies browser 'netscape'"):
        _yt_dlp_cookie_args(payload)


def test_yt_dlp_cookie_args_requires_existing_cookie_file(tmp_path: Path) -> None:
    payload = JobCreate(source_url="https://example.com/v", cookies_file=str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError, match="Cookies file not found"):
        _yt_dlp_cookie_args(payload)


def test_yt_dlp_error_message_explains_rotated_youtube_cookies() -> None:
    error = (
        "Command failed: yt-dlp --cookies-from-browser firefox https://www.youtube.com/watch?v=pT-nyd6o_WY\n"
        "WARNING: [youtube] The provided YouTube account cookies are no longer valid.\n"
        "ERROR: [youtube] pT-nyd6o_WY: Sign in to confirm you're not a bot."
    )
    payload = JobCreate(source_url="https://www.youtube.com/watch?v=pT-nyd6o_WY", cookies_from_browser="firefox")
    command = _yt_dlp_command(payload, "data/jobs/abc/media/source.%(ext)s")
    message = _yt_dlp_error_message(error, payload, command=command)
    assert "yt-dlp could not download this YouTube video." in message
    assert "yt-dlp command:" in message
    assert "--cookies-from-browser firefox" in message
    assert "https://www.youtube.com/watch?v=pT-nyd6o_WY" in message
    assert "export a fresh Netscape cookies.txt file" in message
    assert "Browser cookie source used: firefox." in message
