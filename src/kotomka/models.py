from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


JobStatus = Literal["queued", "running", "completed", "failed"]


class JobCreate(BaseModel):
    source_url: str
    output_language: str = "ru"
    stt_provider: str | None = None
    llm_provider: str | None = None
    cookies_from_browser: str | None = None

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("source_url is required")
        if candidate.startswith(("http://", "https://", "file://")):
            return candidate
        raise ValueError("source_url must be http(s) or file URL")

    @field_validator("output_language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        candidate = value.strip() or "ru"
        if len(candidate) > 32:
            raise ValueError("output_language is too long")
        return candidate


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    is_read: bool = False
    progress: int = 0
    message: str = ""
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    input: JobCreate
    artifact_dir: Path
    result: dict | None = None


class Chapter(BaseModel):
    title: str = ""
    start_s: float = 0
    end_s: float = 0


class VideoMetadata(BaseModel):
    source_url: str
    title: str = "Untitled video"
    webpage_url: str | None = None
    duration_s: float = 0
    uploader: str | None = None
    thumbnail_url: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    upload_date: str | None = None
    language: str | None = None
    channel: str | None = None
    chapters: list[Chapter] = Field(default_factory=list)


class SourceArtifact(BaseModel):
    metadata: VideoMetadata
    video_path: Path
    audio_path: Path
    info_path: Path | None = None


class TranscriptWord(BaseModel):
    start_s: float
    end_s: float
    text: str
    confidence: float | None = None
    speaker: str | None = None


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    speaker: str = "Speaker 1"
    text: str
    confidence: float | None = None
    words: list[TranscriptWord] | None = None


class Transcript(BaseModel):
    language: str = "unknown"
    duration_s: float = 0
    speakers: list[str] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    words: list[TranscriptWord] | None = None

    @field_validator("speakers")
    @classmethod
    def dedupe_speakers(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for speaker in value:
            normalized = speaker.strip() or "Speaker"
            if normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
        return result


class CandidateFrame(BaseModel):
    frame_id: str
    timestamp_s: float
    path: Path
    source: Literal["scene", "periodic"] = "periodic"


class FrameSelection(BaseModel):
    frame_id: str
    timestamp_s: float
    image_path: str
    content_type: str = "unknown"
    score: float = 0
    caption: str = ""
    reason: str = ""
    ocr_text: str | None = None


class ReportSection(BaseModel):
    title: str
    start_s: float
    end_s: float
    body: str
    frame_ids: list[str] = Field(default_factory=list)
    citations: list[float] = Field(default_factory=list)


class Report(BaseModel):
    video: VideoMetadata
    summary: str
    sections: list[ReportSection]
    frames: list[FrameSelection]
    transcript: Transcript
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    output_language: str = "ru"
