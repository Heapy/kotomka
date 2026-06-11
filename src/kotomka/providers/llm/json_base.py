from __future__ import annotations

import base64
import json
import traceback
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from ...config import Settings, get_settings
from ...models import CandidateFrame, FrameSelection, Report, ReportSection, SourceArtifact, Transcript, VideoMetadata
from ...transcripts import chunk_transcript, format_segment_line, format_transcript, window_excerpt
from ...utils import write_json
from .base import LlmProvider
from .json_helpers import FRAME_SCORE_SCHEMA, NOTES_SCHEMA, REPORT_SCHEMA
from .prompts import FRAME_SCORE_INSTRUCTIONS, NOTES_INSTRUCTIONS, REPORT_INSTRUCTIONS

MIN_FRAME_SCORE = 0.45


class ImageInput(NamedTuple):
    path: Path
    label: str


class JsonLlmProviderBase(LlmProvider):
    """Shared scoring/report orchestration; subclasses implement only the JSON transport.

    Transports receive flat JSON schemas. The OpenAI transport enforces them in
    strict mode; the Codex transport can only hint at them in the prompt, so all
    payload parsing here stays lenient.
    """

    def _request_json(
        self,
        *,
        instructions: str,
        text: str,
        images: list[ImageInput],
        image_detail: str,
        schema_name: str,
        schema: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _scoring_model(self) -> str | None:
        return None

    def score_frames(self, frames: list[CandidateFrame], transcript: Transcript) -> list[FrameSelection]:
        if not frames:
            return []
        settings = get_settings()
        excerpt = window_excerpt(
            transcript,
            start_s=min(frame.timestamp_s for frame in frames),
            end_s=max(frame.timestamp_s for frame in frames),
            margin_s=float(settings.transcript_excerpt_margin_seconds),
            low_confidence_below=settings.transcript_low_confidence_threshold,
        )
        payload = self._request_json(
            instructions=FRAME_SCORE_INSTRUCTIONS,
            text=(
                f"Transcript excerpt:\n{excerpt}\n\n"
                f"Score these frame IDs: {[frame.frame_id for frame in frames]}"
            ),
            images=[ImageInput(path=frame.path, label=candidate_frame_label(frame)) for frame in frames],
            image_detail=settings.scoring_image_detail,
            schema_name="frame_scores",
            schema=FRAME_SCORE_SCHEMA,
            model=self._scoring_model(),
        )
        return frame_selections_from_payload(payload, frames)

    def build_report(
        self,
        *,
        source: SourceArtifact,
        transcript: Transcript,
        frames: list[FrameSelection],
        output_language: str,
        work_dir: Path | None = None,
    ) -> Report:
        settings = get_settings()
        context = {
            "output_language": output_language,
            "video": metadata_summary(source.metadata),
            "selected_frames": [
                {
                    "frame_id": frame.frame_id,
                    "timestamp_s": frame.timestamp_s,
                    "content_type": frame.content_type,
                    "caption": frame.caption,
                    "ocr_text": frame.ocr_text,
                }
                for frame in frames
            ],
        }
        transcript_text = format_transcript(
            transcript,
            low_confidence_below=settings.transcript_low_confidence_threshold,
        )
        if len(transcript_text) <= settings.report_single_pass_max_chars:
            knowledge = "Transcript:\n" + transcript_text
        else:
            knowledge = "Transcript notes by chunk:\n" + self._chunk_notes_text(source, transcript, work_dir, settings)
        payload = self._request_json(
            instructions=REPORT_INSTRUCTIONS,
            text=json.dumps(context, ensure_ascii=False, default=str) + "\n\n" + knowledge,
            images=report_images(frames, work_dir, max_images=settings.report_max_images),
            image_detail=settings.report_image_detail,
            schema_name="video_report",
            schema=REPORT_SCHEMA,
        )
        speaker_names = speaker_name_mapping(payload.get("speaker_names"))
        return Report(
            video=source.metadata,
            summary=str(payload.get("summary") or ""),
            sections=coerce_sections(payload.get("sections")),
            frames=frames,
            transcript=display_transcript(transcript, speaker_names),
            output_language=output_language,
        )

    def _chunk_notes_text(
        self,
        source: SourceArtifact,
        transcript: Transcript,
        work_dir: Path | None,
        settings: Settings,
    ) -> str:
        chunks = chunk_transcript(
            transcript,
            source.metadata.chapters,
            target_seconds=settings.report_chunk_target_seconds,
            duration_s=source.metadata.duration_s or transcript.duration_s,
        )
        collected: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            chunk_text = "\n".join(
                format_segment_line(segment, low_confidence_below=settings.transcript_low_confidence_threshold)
                for segment in chunk.segments
            )
            header = {
                "video_title": source.metadata.title,
                "chunk_index": index + 1,
                "chunk_count": len(chunks),
                "chunk_title": chunk.title,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
            }
            try:
                payload = self._request_json(
                    instructions=NOTES_INSTRUCTIONS,
                    text=json.dumps(header, ensure_ascii=False) + "\n\nTranscript chunk:\n" + chunk_text,
                    images=[],
                    image_detail=settings.scoring_image_detail,
                    schema_name="chunk_notes",
                    schema=NOTES_SCHEMA,
                )
            except Exception:
                # One unparseable chunk must not sink the whole report.
                traceback.print_exc()
                payload = {}
            collected.append(
                {
                    "title": chunk.title,
                    "start_s": chunk.start_s,
                    "end_s": chunk.end_s,
                    **coerce_notes(payload),
                }
            )
        if work_dir is not None:
            write_json(work_dir / "notes.json", collected)
        return notes_to_text(collected)


def candidate_frame_label(frame: CandidateFrame) -> str:
    return f"frame_id={frame.frame_id} timestamp_s={frame.timestamp_s}"


def selection_label(selection: FrameSelection) -> str:
    return f"frame_id={selection.frame_id} timestamp_s={selection.timestamp_s}"


def report_images(frames: list[FrameSelection], work_dir: Path | None, *, max_images: int) -> list[ImageInput]:
    if work_dir is None or max_images <= 0:
        return []
    images: list[ImageInput] = []
    for selection in frames[:max_images]:
        path = work_dir / "frames" / selection.image_path
        if path.exists():
            images.append(ImageInput(path=path, label=selection_label(selection)))
    return images


def metadata_summary(metadata: VideoMetadata) -> dict[str, Any]:
    return {
        "title": metadata.title,
        "uploader": metadata.uploader,
        "channel": metadata.channel,
        "upload_date": metadata.upload_date,
        "duration_s": metadata.duration_s,
        "language": metadata.language,
        "description": metadata.description,
        "tags": metadata.tags,
        "chapters": [
            {"title": chapter.title, "start_s": chapter.start_s, "end_s": chapter.end_s}
            for chapter in metadata.chapters
        ],
        "webpage_url": metadata.webpage_url,
    }


def coerce_notes(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    notes: list[dict[str, Any]] = []
    for item in payload.get("notes") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            timestamp = float(item.get("timestamp_s") or 0)
            importance = float(item.get("importance") or 0)
        except (TypeError, ValueError):
            continue
        notes.append(
            {
                "kind": str(item.get("kind") or "claim"),
                "text": text,
                "timestamp_s": timestamp,
                "importance": importance,
            }
        )
    return {"chunk_summary": str(payload.get("chunk_summary") or ""), "notes": notes}


def notes_to_text(collected: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(collected):
        title = chunk.get("title") or "untitled"
        lines.append(f"Chunk {index + 1}: {title} [{chunk.get('start_s', 0):.0f}-{chunk.get('end_s', 0):.0f}s]")
        summary = str(chunk.get("chunk_summary") or "").strip()
        if summary:
            lines.append(f"Summary: {summary}")
        for note in chunk.get("notes") or []:
            lines.append(f"- [{note['timestamp_s']:.1f}] {note['kind']}: {note['text']}")
        lines.append("")
    return "\n".join(lines).strip()


def speaker_name_mapping(items: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not isinstance(items, list):
        return mapping
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        name = str(item.get("name") or "").strip()
        if label and name and label != name:
            mapping[label] = name
    return mapping


def display_transcript(transcript: Transcript, speaker_names: dict[str, str]) -> Transcript:
    """Transcript copy embedded in the report: real speaker names, no word arrays.

    The word-level data stays in the transcript.json artifact; nothing in the
    report rendering uses it and it inflates report.json by megabytes.
    """
    segments = [
        segment.model_copy(update={"speaker": speaker_names.get(segment.speaker, segment.speaker), "words": None})
        for segment in transcript.segments
    ]
    speakers = list(dict.fromkeys(speaker_names.get(speaker, speaker) for speaker in transcript.speakers))
    return transcript.model_copy(update={"segments": segments, "speakers": speakers, "words": None})


def coerce_sections(items: Any) -> list[ReportSection]:
    sections: list[ReportSection] = []
    if not isinstance(items, list):
        return sections
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            sections.append(ReportSection.model_validate(item))
        except ValidationError:
            continue
    return sections


def frame_selections_from_payload(payload: dict[str, Any], frames: list[CandidateFrame]) -> list[FrameSelection]:
    by_id = {frame.frame_id: frame for frame in frames}
    selections: list[FrameSelection] = []
    for item in payload.get("frames", []):
        if not isinstance(item, dict):
            continue
        frame = by_id.get(str(item.get("frame_id") or ""))
        if frame is None:
            continue
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            continue
        if score < MIN_FRAME_SCORE:
            continue
        selections.append(
            FrameSelection(
                frame_id=frame.frame_id,
                timestamp_s=frame.timestamp_s,
                image_path=frame.path.name,
                content_type=str(item.get("content_type") or "unknown"),
                score=score,
                caption=str(item.get("caption") or ""),
                reason=str(item.get("reason") or ""),
                ocr_text=item.get("ocr_text") if isinstance(item.get("ocr_text"), str) else None,
            )
        )
    return sorted(selections, key=lambda item: item.timestamp_s)


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
