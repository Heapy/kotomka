from types import SimpleNamespace

import pytest

import kotomka.providers.stt.registry as stt_registry
from kotomka.providers.llm import available_llm_providers, get_llm_provider
from kotomka.providers.stt import available_stt_providers, get_stt_provider
from kotomka.providers.stt.whisper_local import whisper_segments_to_transcript


def test_provider_registries() -> None:
    assert "fake" in available_stt_providers()
    assert "fake" in available_llm_providers()
    assert get_stt_provider("fake").name == "fake"
    assert get_llm_provider("fake").name == "fake"


def test_whisper_listed_only_when_installed(monkeypatch) -> None:
    monkeypatch.setattr(stt_registry, "whisper_available", lambda: False)
    assert "whisper" not in stt_registry.available_stt_providers()
    with pytest.raises(ValueError, match="uv sync --extra whisper"):
        stt_registry.get_stt_provider("whisper")

    monkeypatch.setattr(stt_registry, "whisper_available", lambda: True)
    assert "whisper" in stt_registry.available_stt_providers()


def test_whisper_segments_map_to_transcript() -> None:
    segments = [
        SimpleNamespace(
            start=0.0,
            end=2.5,
            text=" Hello there. ",
            avg_logprob=-0.1,
            words=[SimpleNamespace(start=0.0, end=1.0, word=" Hello", probability=0.95)],
        ),
        SimpleNamespace(start=2.5, end=5.0, text="Second part.", avg_logprob=None, words=None),
    ]

    transcript = whisper_segments_to_transcript(iter(segments), language="ru", fallback_duration=10)

    assert transcript.language == "ru"
    assert transcript.duration_s == 5.0
    assert transcript.speakers == ["Speaker 1"]
    assert transcript.segments[0].text == "Hello there."
    assert transcript.segments[0].speaker == "Speaker 1"
    assert transcript.segments[0].confidence == pytest.approx(0.904, abs=0.01)
    assert transcript.segments[0].words[0].text == "Hello"
    assert transcript.segments[1].confidence is None
    assert transcript.words is not None and len(transcript.words) == 1
