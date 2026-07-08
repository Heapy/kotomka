from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from ...config import get_settings
from ...models import Transcript, TranscriptSegment, TranscriptWord, VideoMetadata
from ...utils import write_json
from .base import SttProvider
from .keyterms import extract_keyterms

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# Support for these varies by account and routed speech model; a 400 naming one
# of them triggers a single retry with the minimal request body.
OPTIONAL_REQUEST_KEYS = (
    "speech_models",
    "keyterms_prompt",
    "entity_detection",
    "speakers_expected",
    "language_code",
)


class AssemblyAiSttProvider(SttProvider):
    name = "assemblyai"

    def __init__(self, *, poll_seconds: float = 3.0) -> None:
        self.poll_seconds = poll_seconds

    def transcribe(
        self,
        audio_path: Path,
        metadata: VideoMetadata,
        *,
        speakers_expected: int | None = None,
        raw_path: Path | None = None,
    ) -> Transcript:
        api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is required for stt_provider=assemblyai")
        headers = {"authorization": api_key}
        keyterms = extract_keyterms(metadata, max_terms=get_settings().stt_keyterms_max)
        with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as client:
            with audio_path.open("rb") as handle:
                upload_response = client.post(UPLOAD_URL, headers=headers, content=handle)
            upload_response.raise_for_status()
            audio_url = upload_response.json()["upload_url"]
            request = build_transcription_request(
                audio_url,
                metadata,
                speakers_expected=speakers_expected,
                keyterms=keyterms,
            )
            transcript_id = self._start_transcription(client, headers, request, audio_url)
            while True:
                poll_response = client.get(f"{TRANSCRIPT_URL}/{transcript_id}", headers=headers)
                poll_response.raise_for_status()
                payload = poll_response.json()
                status = payload.get("status")
                if status == "completed":
                    if raw_path is not None:
                        write_json(raw_path, payload)
                    return assemblyai_payload_to_transcript(payload, fallback_duration=metadata.duration_s)
                if status == "error":
                    raise RuntimeError(f"AssemblyAI transcription failed: {payload.get('error')}")
                time.sleep(self.poll_seconds)

    def _start_transcription(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        request: dict[str, Any],
        audio_url: str,
    ) -> str:
        json_headers = {**headers, "content-type": "application/json"}
        response = client.post(TRANSCRIPT_URL, headers=json_headers, json=request)
        if response.status_code == 400 and _mentions_optional_key(response.text):
            response = client.post(
                TRANSCRIPT_URL,
                headers=json_headers,
                json={"audio_url": audio_url, "speaker_labels": True, "language_detection": True},
            )
        response.raise_for_status()
        return response.json()["id"]


def build_transcription_request(
    audio_url: str,
    metadata: VideoMetadata,
    *,
    speakers_expected: int | None,
    keyterms: list[str],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-5-pro", "universal-3-pro", "universal-2"],
        "speaker_labels": True,
        "entity_detection": True,
    }
    language = _normalize_language(metadata.language)
    if language:
        request["language_code"] = language
    else:
        request["language_detection"] = True
    if keyterms:
        request["keyterms_prompt"] = keyterms
    if speakers_expected:
        request["speakers_expected"] = speakers_expected
    return request


def _mentions_optional_key(body: str) -> bool:
    return any(key in body for key in OPTIONAL_REQUEST_KEYS)


def _normalize_language(value: str | None) -> str | None:
    primary = (value or "").strip().split("-")[0].split("_")[0].lower()
    if 2 <= len(primary) <= 3 and primary.isalpha():
        return primary
    return None


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
