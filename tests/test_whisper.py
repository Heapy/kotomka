import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

import kotomka.providers.stt.whisper_local as whisper
from kotomka.models import VideoMetadata


@pytest.fixture(autouse=True)
def clear_model_cache():
    whisper._load_model.cache_clear()
    yield
    whisper._load_model.cache_clear()


def test_parallel_transcriptions_share_one_model_and_serialize_lazy_inference(monkeypatch) -> None:
    constructions = []
    active = 0
    max_active = 0
    lock = Lock()
    start = Barrier(3)

    class FakeWhisperModel:
        def __init__(self, name, *, compute_type):
            constructions.append((name, compute_type))

        def transcribe(self, audio_path, **kwargs):
            def segments():
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.03)
                    yield SimpleNamespace(start=0, end=1, text=Path(audio_path).stem, words=None, avg_logprob=None)
                finally:
                    with lock:
                        active -= 1
            return segments(), SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    metadata = VideoMetadata(source_url="file:///test.mp4", title="Test", duration_s=1)

    def transcribe(index):
        provider = whisper.WhisperLocalSttProvider(model_name="test-model", compute_type="int8")
        start.wait(timeout=5)
        return provider.transcribe(Path(f"job-{index}.flac"), metadata)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(transcribe, range(3)))

    assert constructions == [("test-model", "int8")]
    assert max_active == 1
    assert [result.segments[0].text for result in results] == ["job-0", "job-1", "job-2"]
