from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from ...models import Transcript, TranscriptSegment, TranscriptWord, VideoMetadata
from .base import SttProvider


class AssemblyAiSttProvider(SttProvider):
    name = "assemblyai"

    def __init__(self, *, poll_seconds: float = 3.0) -> None:
        self.poll_seconds = poll_seconds

    def transcribe(self, audio_path: Path, metadata: VideoMetadata) -> Transcript:
        api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is required for stt_provider=assemblyai")
        headers = {"authorization": api_key}
        with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as client:
            with audio_path.open("rb") as handle:
                upload_response = client.post("https://api.assemblyai.com/v2/upload", headers=headers, content=handle)
            upload_response.raise_for_status()
            audio_url = upload_response.json()["upload_url"]
            start_response = client.post(
                "https://api.assemblyai.com/v2/transcript",
                headers={**headers, "content-type": "application/json"},
                json={
                    "audio_url": audio_url,
                    "speaker_labels": True,
                    "language_detection": True,
                },
            )
            start_response.raise_for_status()
            transcript_id = start_response.json()["id"]
            while True:
                poll_response = client.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
                poll_response.raise_for_status()
                payload = poll_response.json()
                status = payload.get("status")
                if status == "completed":
                    return assemblyai_payload_to_transcript(payload, fallback_duration=metadata.duration_s)
                if status == "error":
                    raise RuntimeError(f"AssemblyAI transcription failed: {payload.get('error')}")
                time.sleep(self.poll_seconds)


def assemblyai_payload_to_transcript(payload: dict[str, Any], *, fallback_duration: float = 0) -> Transcript:
    utterances = payload.get("utterances")
    segments: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []
    speakers: list[str] = []
    if isinstance(utterances, list) and utterances:
        for utterance in utterances:
            if not isinstance(utterance, dict):
                continue
            speaker = f"Speaker {utterance.get('speaker') or '?'}"
            if speaker not in speakers:
                speakers.append(speaker)
            segment_words = [_assembly_word_to_model(word) for word in utterance.get("words") or [] if isinstance(word, dict)]
            words.extend(segment_words)
            segments.append(
                TranscriptSegment(
                    start_s=_ms_to_seconds(utterance.get("start")),
                    end_s=_ms_to_seconds(utterance.get("end")),
                    speaker=speaker,
                    text=str(utterance.get("text") or "").strip(),
                    confidence=_float_or_none(utterance.get("confidence")),
                    words=segment_words or None,
                )
            )
    elif isinstance(payload.get("words"), list):
        words = [_assembly_word_to_model(word) for word in payload["words"] if isinstance(word, dict)]
        text = str(payload.get("text") or "").strip()
        segments = [
            TranscriptSegment(
                start_s=words[0].start_s if words else 0,
                end_s=words[-1].end_s if words else fallback_duration,
                speaker="Speaker 1",
                text=text,
                confidence=_float_or_none(payload.get("confidence")),
                words=words or None,
            )
        ]
        speakers = ["Speaker 1"]
    duration = max([segment.end_s for segment in segments], default=fallback_duration or 0)
    return Transcript(
        language=str(payload.get("language_code") or payload.get("language") or "unknown"),
        duration_s=duration,
        speakers=speakers,
        segments=segments,
        words=words or None,
    )


def _assembly_word_to_model(word: dict[str, Any]) -> TranscriptWord:
    speaker = word.get("speaker")
    return TranscriptWord(
        start_s=_ms_to_seconds(word.get("start")),
        end_s=_ms_to_seconds(word.get("end")),
        text=str(word.get("text") or "").strip(),
        confidence=_float_or_none(word.get("confidence")),
        speaker=f"Speaker {speaker}" if speaker is not None else None,
    )


def _ms_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000.0
    return 0.0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None

