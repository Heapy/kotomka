from __future__ import annotations

from pathlib import Path

from ...models import CandidateFrame, FrameSelection, Report, ReportSection, SourceArtifact, Transcript
from .base import LlmProvider


class FakeLlmProvider(LlmProvider):
    name = "fake"

    def score_frames(self, frames: list[CandidateFrame], transcript: Transcript) -> list[FrameSelection]:
        del transcript
        selected: list[FrameSelection] = []
        for index, frame in enumerate(frames[:12]):
            selected.append(
                FrameSelection(
                    frame_id=frame.frame_id,
                    timestamp_s=frame.timestamp_s,
                    image_path=frame.path.name,
                    content_type="slide_or_screen",
                    score=max(0.1, 0.95 - index * 0.03),
                    caption=f"Representative frame at {int(frame.timestamp_s)}s",
                    reason="Fake provider keeps early deduplicated frames for offline testing.",
                )
            )
        return selected

    def build_report(
        self,
        *,
        source: SourceArtifact,
        transcript: Transcript,
        frames: list[FrameSelection],
        output_language: str,
        work_dir: Path | None = None,
    ) -> Report:
        del work_dir
        sections: list[ReportSection] = []
        if transcript.segments:
            for index, segment in enumerate(transcript.segments):
                matched_frames = [
                    frame.frame_id
                    for frame in frames
                    if segment.start_s <= frame.timestamp_s <= max(segment.end_s, segment.start_s + 1)
                ]
                sections.append(
                    ReportSection(
                        title=f"Part {index + 1}",
                        start_s=segment.start_s,
                        end_s=segment.end_s,
                        body=segment.text,
                        frame_ids=matched_frames[:2],
                        citations=[segment.start_s],
                    )
                )
        else:
            sections.append(
                ReportSection(
                    title="Overview",
                    start_s=0,
                    end_s=source.metadata.duration_s,
                    body="No transcript segments were produced.",
                    frame_ids=[frame.frame_id for frame in frames[:2]],
                    citations=[0],
                )
            )
        return Report(
            video=source.metadata,
            summary=f"Offline fake summary for '{source.metadata.title}'. Replace fake providers with AssemblyAI and a live LLM provider for real analysis.",
            sections=sections,
            frames=frames,
            transcript=transcript,
            output_language=output_language,
        )
