import json
from pathlib import Path

import httpx
import respx

from kotomka.models import Chapter, VideoMetadata
from kotomka.providers.stt.assemblyai import (
    AssemblyAiSttProvider,
    assemblyai_payload_to_transcript,
    build_transcription_request,
)
from kotomka.providers.stt.keyterms import extract_keyterms


def make_metadata(**overrides) -> VideoMetadata:
    defaults = dict(source_url="https://example.com/v", title="Talk")
    defaults.update(overrides)
    return VideoMetadata(**defaults)


def test_assemblyai_payload_to_transcript() -> None:
    payload = {
        "language_code": "en",
        "utterances": [
            {
                "speaker": "A",
                "start": 1000,
                "end": 2500,
                "text": "Hello there.",
                "confidence": 0.98,
                "words": [{"text": "Hello", "start": 1000, "end": 1500, "confidence": 0.9, "speaker": "A"}],
            }
        ],
    }
    transcript = assemblyai_payload_to_transcript(payload)
    assert transcript.language == "en"
    assert transcript.duration_s == 2.5
    assert transcript.speakers == ["Speaker A"]
    assert transcript.segments[0].speaker == "Speaker A"
    assert transcript.segments[0].words[0].start_s == 1.0


def test_build_transcription_request_with_language_and_hints() -> None:
    request = build_transcription_request(
        "https://cdn/audio",
        make_metadata(language="en-US"),
        speakers_expected=2,
        keyterms=["FastAPI"],
    )
    assert request["speech_models"] == ["universal-3-pro", "universal-2"]
    assert request["speaker_labels"] is True
    assert request["entity_detection"] is True
    assert request["language_code"] == "en"
    assert "language_detection" not in request
    assert request["keyterms_prompt"] == ["FastAPI"]
    assert request["speakers_expected"] == 2


def test_build_transcription_request_defaults_to_detection() -> None:
    request = build_transcription_request("https://cdn/audio", make_metadata(), speakers_expected=None, keyterms=[])
    assert request["language_detection"] is True
    assert "language_code" not in request
    assert "keyterms_prompt" not in request
    assert "speakers_expected" not in request


def test_extract_keyterms_prioritizes_title_and_filters_noise() -> None:
    metadata = make_metadata(
        title="Scaling PostgreSQL at Yandex Cloud",
        tags=["the", "python"],
        description="We use FastAPI and ClickHouse v23.8 in production. Это доклад про базы данных.",
        chapters=[Chapter(title="Sharding Basics", start_s=0, end_s=60)],
    )
    terms = extract_keyterms(metadata, max_terms=50)
    assert "PostgreSQL" in terms
    assert "Yandex Cloud" in terms
    assert "Sharding Basics" in terms
    assert "FastAPI" in terms
    assert "ClickHouse" in terms
    assert "v23.8" in terms
    assert "the" not in terms
    assert "Это" not in terms
    assert terms.index("PostgreSQL") < terms.index("FastAPI")


def test_extract_keyterms_caps_and_dedupes() -> None:
    metadata = make_metadata(title="Kafka kafka explained", description="Kafka again")
    assert extract_keyterms(metadata, max_terms=1) == ["Kafka"]
    assert extract_keyterms(metadata, max_terms=0) == []


@respx.mock
def test_transcribe_retries_minimal_request_and_saves_raw(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"flac-bytes")
    raw_path = tmp_path / "transcript_raw.json"

    completed = {
        "status": "completed",
        "language_code": "en",
        "utterances": [{"speaker": "A", "start": 0, "end": 1000, "text": "Hi.", "confidence": 0.9, "words": []}],
    }
    respx.post("https://api.assemblyai.com/v2/upload").mock(
        return_value=httpx.Response(200, json={"upload_url": "https://cdn.assemblyai.com/upload/1"})
    )
    start_route = respx.post("https://api.assemblyai.com/v2/transcript").mock(
        side_effect=[
            httpx.Response(400, json={"error": "speech_models is not available for this account"}),
            httpx.Response(200, json={"id": "t-1"}),
        ]
    )
    respx.get("https://api.assemblyai.com/v2/transcript/t-1").mock(return_value=httpx.Response(200, json=completed))

    provider = AssemblyAiSttProvider(poll_seconds=0.01)
    transcript = provider.transcribe(
        audio,
        make_metadata(language="en", title="Kafka Talk"),
        speakers_expected=2,
        raw_path=raw_path,
    )

    assert transcript.segments[0].text == "Hi."
    assert start_route.call_count == 2
    first_body = json.loads(start_route.calls[0].request.content)
    second_body = json.loads(start_route.calls[1].request.content)
    assert first_body["speech_models"] == ["universal-3-pro", "universal-2"]
    assert first_body["language_code"] == "en"
    assert first_body["speakers_expected"] == 2
    assert "Kafka Talk" in first_body["keyterms_prompt"]
    assert second_body == {
        "audio_url": "https://cdn.assemblyai.com/upload/1",
        "speaker_labels": True,
        "language_detection": True,
    }
    assert json.loads(raw_path.read_text(encoding="utf-8"))["status"] == "completed"
