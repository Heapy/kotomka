from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from ...config import get_settings
from ...models import CandidateFrame, FrameSelection, Report, ReportSection, SourceArtifact, Transcript
from .base import LlmProvider
from .json_helpers import FRAME_SCORE_SCHEMA, REPORT_SCHEMA, parse_json_object


class OpenAiResponsesProvider(LlmProvider):
    name = "openai"

    def __init__(self, *, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for llm_provider=openai")
        self.client = OpenAI(api_key=api_key)
        self.model = model or get_settings().openai_model

    def score_frames(self, frames: list[CandidateFrame], transcript: Transcript) -> list[FrameSelection]:
        if not frames:
            return []
        payload = self._request_json(
            instructions=FRAME_SCORE_INSTRUCTIONS,
            text=f"Transcript excerpt:\n{_transcript_excerpt(transcript)}\n\nScore these frame IDs: {[frame.frame_id for frame in frames]}",
            images=frames,
            schema_name="frame_scores",
            schema=FRAME_SCORE_SCHEMA,
        )
        return _frame_selections_from_payload(payload, frames)

    def build_report(
        self,
        *,
        source: SourceArtifact,
        transcript: Transcript,
        frames: list[FrameSelection],
        output_language: str,
    ) -> Report:
        payload = self._request_json(
            instructions=REPORT_INSTRUCTIONS,
            text=json.dumps(
                {
                    "output_language": output_language,
                    "video": source.metadata.model_dump(),
                    "transcript": transcript.model_dump(),
                    "selected_frames": [frame.model_dump() for frame in frames],
                },
                ensure_ascii=False,
                default=str,
            ),
            images=[],
            schema_name="video_report",
            schema=REPORT_SCHEMA,
        )
        sections = [ReportSection.model_validate(item) for item in payload.get("sections", [])]
        return Report(
            video=source.metadata,
            summary=str(payload.get("summary") or ""),
            sections=sections,
            frames=frames,
            transcript=transcript,
            output_language=output_language,
        )

    def _request_json(
        self,
        *,
        instructions: str,
        text: str,
        images: list[CandidateFrame],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for frame in images:
            content.append({"type": "input_text", "text": f"frame_id={frame.frame_id} timestamp_s={frame.timestamp_s}"})
            content.append({"type": "input_image", "image_url": _image_data_url(frame.path), "detail": "low"})
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return parse_json_object(getattr(response, "output_text", "") or "")


FRAME_SCORE_INSTRUCTIONS = """You select knowledge-bearing video frames.
Keep slides, diagrams, charts, code, whiteboards, documents, product screens, and visual examples.
Reject mood shots, audience shots, repeated talking heads, and frames with no durable information.
Return only the requested JSON."""

REPORT_INSTRUCTIONS = """Create a knowledge-preserving report from a transcript and selected frames.
Write in the requested output language. Keep the transcript unchanged.
Every section must include timestamp citations in seconds. Return only JSON."""


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _transcript_excerpt(transcript: Transcript, max_chars: int = 6000) -> str:
    parts = [f"[{segment.start_s:.1f}-{segment.end_s:.1f}] {segment.speaker}: {segment.text}" for segment in transcript.segments]
    return "\n".join(parts)[:max_chars]


def _frame_selections_from_payload(payload: dict[str, Any], frames: list[CandidateFrame]) -> list[FrameSelection]:
    by_id = {frame.frame_id: frame for frame in frames}
    selections: list[FrameSelection] = []
    for item in payload.get("frames", []):
        if not isinstance(item, dict):
            continue
        frame = by_id.get(str(item.get("frame_id") or ""))
        if frame is None:
            continue
        score = float(item.get("score") or 0)
        if score < 0.45:
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
