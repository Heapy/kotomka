from kotomka.providers.stt.assemblyai import assemblyai_payload_to_transcript


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

