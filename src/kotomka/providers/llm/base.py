from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...models import CandidateFrame, FrameSelection, Report, SourceArtifact, Transcript


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

