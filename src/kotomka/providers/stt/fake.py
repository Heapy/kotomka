from __future__ import annotations

from pathlib import Path

from ...models import Transcript, TranscriptSegment, VideoMetadata
from .base import SttProvider


class FakeSttProvider(SttProvider):
    name = "fake"

    def transcribe(self, audio_path: Path, metadata: VideoMetadata) -> Transcript:
        del audio_path
        duration = max(60.0, float(metadata.duration_s or 180.0))
        speakers = ["Speaker A", "Speaker B"]
        segments = [
            TranscriptSegment(
                start_s=0,
                end_s=min(duration, 45),
                speaker="Speaker A",
                text=f"This is a fake transcript for {metadata.title}. It introduces the main topic and explains why it matters.",
                confidence=1.0,
            ),
            TranscriptSegment(
                start_s=min(duration, 45),
                end_s=min(duration, 120),
                speaker="Speaker B",
                text="The video then walks through the important ideas, examples, constraints, and practical takeaways.",
                confidence=1.0,
            ),
            TranscriptSegment(
                start_s=min(duration, 120),
                end_s=duration,
                speaker="Speaker A",
                text="The conclusion summarizes the decisions, tradeoffs, and next steps that should be remembered.",
                confidence=1.0,
            ),
        ]
        return Transcript(language="en", duration_s=duration, speakers=speakers, segments=segments)

