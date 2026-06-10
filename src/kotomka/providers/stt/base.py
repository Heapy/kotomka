from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...models import Transcript, VideoMetadata


class SttProvider(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, metadata: VideoMetadata) -> Transcript:
        raise NotImplementedError

