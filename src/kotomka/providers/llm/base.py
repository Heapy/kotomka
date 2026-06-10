from __future__ import annotations

from abc import ABC, abstractmethod

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
    ) -> Report:
        raise NotImplementedError

