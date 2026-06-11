from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...models import (
    CandidateFrame,
    FrameSelection,
    Report,
    ReportAssessment,
    SourceArtifact,
    Transcript,
    VideoMetadata,
)


class LlmProvider(ABC):
    name: str

    @abstractmethod
    def score_frames(self, frames: list[CandidateFrame], transcript: Transcript) -> list[FrameSelection]:
        raise NotImplementedError

    @abstractmethod
    def build_report(
        self,
        *,
        source: SourceArtifact,
        transcript: Transcript,
        frames: list[FrameSelection],
        output_language: str,
        work_dir: Path | None = None,
    ) -> Report:
        """Build the report. `work_dir` is the job artifact dir, used to resolve
        selected frame images so vision-capable providers can ground the report."""
        raise NotImplementedError

    def recaption_frames(
        self,
        selections: list[FrameSelection],
        *,
        work_dir: Path,
        transcript: Transcript | None = None,
    ) -> list[FrameSelection]:
        """Optionally re-caption the selected frames with a higher-fidelity pass.
        Defaults to a passthrough so simple providers and test doubles keep working."""
        del work_dir, transcript
        return selections

    def assess_report(
        self,
        *,
        report: Report,
        metadata: VideoMetadata,
        output_language: str,
    ) -> ReportAssessment | None:
        """Critically assess originality, freshness, and usefulness of a finished
        report. Defaults to no assessment so simple providers and test doubles
        keep working."""
        del report, metadata, output_language
        return None

