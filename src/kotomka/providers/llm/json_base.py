from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from ...config import get_settings
from ...models import CandidateFrame, FrameSelection, Report, ReportSection, SourceArtifact, Transcript, VideoMetadata
from ...transcripts import format_transcript, window_excerpt
from .base import LlmProvider
from .json_helpers import FRAME_SCORE_SCHEMA, REPORT_SCHEMA
from .prompts import FRAME_SCORE_INSTRUCTIONS, REPORT_INSTRUCTIONS

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
        payload = self._request_json(
            instructions=REPORT_INSTRUCTIONS,
            text=json.dumps(context, ensure_ascii=False, default=str) + "\n\nTranscript:\n" + transcript_text,
            images=report_images(frames, work_dir, max_images=settings.report_max_images),
            image_detail=settings.report_image_detail,
            schema_name="video_report",
            schema=REPORT_SCHEMA,
        )
        return Report(
            video=source.metadata,
            summary=str(payload.get("summary") or ""),
            sections=coerce_sections(payload.get("sections")),
            frames=frames,
            transcript=transcript,
            output_language=output_language,
        )


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
