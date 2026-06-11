from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .media import extract_audio, ffprobe_duration, require_binary, run_command
from .models import Chapter, JobCreate, SourceArtifact, VideoMetadata

MAX_DESCRIPTION_CHARS = 4000

# audio.mp3 is the pre-FLAC artifact name; reprocessed jobs may still have one on disk.
AUDIO_ARTIFACT_NAMES = {"audio.flac", "audio.mp3"}


class SourceProvider(ABC):
    @abstractmethod
    def fetch(self, payload: JobCreate, artifact_dir: Path) -> SourceArtifact:
        raise NotImplementedError


class YtDlpSourceProvider(SourceProvider):
    def fetch(self, payload: JobCreate, artifact_dir: Path) -> SourceArtifact:
        if payload.source_url.startswith("file://"):
            return LocalFileSourceProvider().fetch(payload, artifact_dir)

        require_binary("yt-dlp")
        media_dir = artifact_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(media_dir / "source.%(ext)s")
        command = [
            "yt-dlp",
            "--no-playlist",
            "--write-info-json",
            "--merge-output-format",
            "mp4",
            "-f",
            "bv*+ba/best",
            "-o",
            output_template,
        ]
        if payload.cookies_from_browser:
            command.extend(["--cookies-from-browser", payload.cookies_from_browser])
        command.append(payload.source_url)
        run_command(command, timeout=60 * 60)

        info_path = media_dir / "source.info.json"
        video_path = _find_downloaded_video(media_dir)
        metadata = _metadata_from_info(info_path, payload.source_url)
        duration = metadata.duration_s or ffprobe_duration(video_path)
        metadata.duration_s = duration
        audio_path = extract_audio(video_path, media_dir / "audio.flac")
        return SourceArtifact(metadata=metadata, video_path=video_path, audio_path=audio_path, info_path=info_path)


class LocalFileSourceProvider(SourceProvider):
    def fetch(self, payload: JobCreate, artifact_dir: Path) -> SourceArtifact:
        parsed = urlparse(payload.source_url)
        if parsed.scheme != "file":
            raise ValueError("LocalFileSourceProvider only accepts file:// URLs")
        source_path = Path(unquote(parsed.path)).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        media_dir = artifact_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        video_path = media_dir / f"source{source_path.suffix or '.mp4'}"
        if source_path.resolve() != video_path.resolve():
            shutil.copy2(source_path, video_path)
        duration = ffprobe_duration(video_path)
        audio_path = extract_audio(video_path, media_dir / "audio.flac")
        metadata = VideoMetadata(source_url=payload.source_url, title=source_path.stem, duration_s=duration)
        return SourceArtifact(metadata=metadata, video_path=video_path, audio_path=audio_path)


def _find_downloaded_video(media_dir: Path) -> Path:
    ignored_suffixes = {".json", ".part", ".ytdl"}
    candidates = [path for path in media_dir.iterdir() if path.is_file() and path.suffix not in ignored_suffixes]
    candidates = [path for path in candidates if not path.name.endswith(".info.json") and path.name not in AUDIO_ARTIFACT_NAMES]
    if not candidates:
        raise RuntimeError("yt-dlp did not produce a video file")
    return max(candidates, key=lambda path: path.stat().st_size)


def _metadata_from_info(info_path: Path, source_url: str) -> VideoMetadata:
    if not info_path.exists():
        return VideoMetadata(source_url=source_url, title="Untitled video")
    data = json.loads(info_path.read_text(encoding="utf-8"))
    description = str(data.get("description") or "").strip()[:MAX_DESCRIPTION_CHARS] or None
    return VideoMetadata(
        source_url=source_url,
        title=str(data.get("title") or "Untitled video"),
        webpage_url=data.get("webpage_url"),
        duration_s=float(data.get("duration") or 0),
        uploader=data.get("uploader"),
        thumbnail_url=data.get("thumbnail"),
        description=description,
        tags=[tag.strip() for tag in data.get("tags") or [] if isinstance(tag, str) and tag.strip()],
        upload_date=_normalize_upload_date(data.get("upload_date")),
        language=str(data.get("language") or "").strip() or None,
        channel=str(data.get("channel") or data.get("uploader") or "").strip() or None,
        chapters=_chapters_from_info(data.get("chapters")),
    )


def _normalize_upload_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return text or None


def _chapters_from_info(value: Any) -> list[Chapter]:
    chapters: list[Chapter] = []
    if not isinstance(value, list):
        return chapters
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start_s = float(item.get("start_time") or 0)
            end_s = float(item.get("end_time") or 0)
        except (TypeError, ValueError):
            continue
        if end_s < start_s:
            continue
        chapters.append(Chapter(title=str(item.get("title") or "").strip(), start_s=max(0.0, start_s), end_s=end_s))
    return chapters

