from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...models import Transcript, VideoMetadata


class SttProvider(ABC):
    name: str

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        metadata: VideoMetadata,
        *,
        speakers_expected: int | None = None,
        raw_path: Path | None = None,
    ) -> Transcript:
        """Transcribe audio into a normalized transcript.

        `speakers_expected` is an optional diarization hint. When `raw_path` is
        given, the provider saves its raw response payload there for debugging
        and reprocessing without paying for transcription again.
        """
        raise NotImplementedError

