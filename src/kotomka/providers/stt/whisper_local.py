from __future__ import annotations

import math
from importlib import util
from pathlib import Path

from ...config import get_settings
from ...models import Transcript, TranscriptSegment, TranscriptWord, VideoMetadata
from ...utils import write_json
from .base import SttProvider


def whisper_available() -> bool:
    return util.find_spec("faster_whisper") is not None


class WhisperLocalSttProvider(SttProvider):
    """Offline transcription via faster-whisper.

    No diarization: every segment is labeled "Speaker 1". The model is
    constructed lazily inside transcribe() because the first call downloads
    multi-gigabyte weights.
    """

    name = "whisper"

    def __init__(self, *, model_name: str | None = None, compute_type: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.whisper_model
        self.compute_type = compute_type or settings.whisper_compute_type

    def transcribe(
        self,
        audio_path: Path,
        metadata: VideoMetadata,
        *,
        speakers_expected: int | None = None,
        raw_path: Path | None = None,
    ) -> Transcript:
        del speakers_expected  # no diarization support
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_name, compute_type=self.compute_type)
        language = (metadata.language or "").strip().split("-")[0].lower() or None
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        transcript = whisper_segments_to_transcript(
            segments_iter,
            language=getattr(info, "language", None) or language or "unknown",
            fallback_duration=metadata.duration_s,
        )
        if raw_path is not None:
            write_json(
                raw_path,
                {
                    "provider": "whisper",
                    "model": self.model_name,
                    "language": transcript.language,
                    "segments": [segment.model_dump() for segment in transcript.segments],
                },
            )
        return transcript


def whisper_segments_to_transcript(segments_iter, *, language: str, fallback_duration: float = 0) -> Transcript:
    segments: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []
    for segment in segments_iter:
        segment_words = [
            TranscriptWord(
                start_s=float(word.start),
                end_s=float(word.end),
                text=str(word.word).strip(),
                confidence=_probability(getattr(word, "probability", None)),
                speaker="Speaker 1",
            )
            for word in (getattr(segment, "words", None) or [])
        ]
        words.extend(segment_words)
        segments.append(
            TranscriptSegment(
                start_s=float(segment.start),
                end_s=float(segment.end),
                speaker="Speaker 1",
                text=str(segment.text).strip(),
                confidence=_probability_from_logprob(getattr(segment, "avg_logprob", None)),
                words=segment_words or None,
            )
        )
    duration = max([segment.end_s for segment in segments], default=fallback_duration or 0)
    return Transcript(
        language=str(language),
        duration_s=duration,
        speakers=["Speaker 1"] if segments else [],
        segments=segments,
        words=words or None,
    )


def _probability(value) -> float | None:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    return None


def _probability_from_logprob(value) -> float | None:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, math.exp(float(value))))
    return None
