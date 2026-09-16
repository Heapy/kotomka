from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import kotomka.ocr as ocr_module
from kotomka.models import CandidateFrame
from kotomka.ocr import dedupe_ocr_supersets, ocr_available, ocr_image


def frame(frame_id: str, timestamp_s: float, ocr_text: str | None, tmp_path: Path) -> CandidateFrame:
    path = tmp_path / f"{frame_id}.png"
    image = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(image).text((20, 20), ocr_text or "", fill="black")
    image.save(path)
    return CandidateFrame(frame_id=frame_id, timestamp_s=timestamp_s, path=path, ocr_text=ocr_text)


def test_dedupe_ocr_supersets_drops_bullet_builds(tmp_path: Path) -> None:
    frames = [
        frame("f1", 10.0, "Scaling Postgres\n- Read replicas", tmp_path),
        frame("f2", 40.0, "Scaling Postgres\n- Read replicas\n- Sharding by user_id", tmp_path),
        frame("f3", 200.0, "Unrelated slide about Kafka topics", tmp_path),
    ]
    kept = dedupe_ocr_supersets(frames, window_s=90.0)
    assert [item.frame_id for item in kept] == ["f2", "f3"]


def test_dedupe_ocr_supersets_respects_time_window(tmp_path: Path) -> None:
    frames = [
        frame("f1", 10.0, "Recurring footer Conference 2026 Agenda intro", tmp_path),
        frame("f2", 500.0, "Recurring footer Conference 2026 Agenda intro plus much more content here", tmp_path),
    ]
    kept = dedupe_ocr_supersets(frames, window_s=90.0)
    assert [item.frame_id for item in kept] == ["f1", "f2"]


def test_dedupe_ocr_supersets_keeps_frames_without_text(tmp_path: Path) -> None:
    frames = [
        frame("f1", 10.0, None, tmp_path),
        frame("f2", 20.0, "so much text on this slide right here", tmp_path),
        frame("f3", 30.0, "ab", tmp_path),
    ]
    kept = dedupe_ocr_supersets(frames, window_s=90.0)
    assert [item.frame_id for item in kept] == ["f1", "f2", "f3"]


@pytest.mark.parametrize("old,new", [("10", "20"), ("1", "9"), ("1.5", "1.6"), ("not safe", "safe")])
def test_dedupe_preserves_changed_numbers_and_negations(tmp_path, old, new):
    prefix = "Benchmark latency {} ms measured for production database under normal steady traffic load"
    frames = [frame("before", 10, prefix.format(old), tmp_path),
              frame("after", 20, prefix.format(new) + " with retries enabled", tmp_path)]
    assert dedupe_ocr_supersets(frames) == frames


def test_text_superset_does_not_discard_changed_chart(tmp_path):
    frames = [frame("before", 10, "Results latency milliseconds", tmp_path),
              frame("after", 20, "Results latency milliseconds\nWith retries enabled", tmp_path)]
    for item, color in zip(frames, ("red", "blue")):
        with Image.open(item.path) as image:
            ImageDraw.Draw(image).rectangle((50, 100, 300, 200), fill=color)
            image.save(item.path)
    assert dedupe_ocr_supersets(frames) == frames


def test_worker_skips_ocr_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ocr_module, "ocr_available", lambda: False)
    assert ocr_module.ocr_image(tmp_path / "missing.png") is None


@pytest.mark.skipif(not ocr_available(), reason="ocrmac (macOS Vision) not installed")
def test_ocr_image_reads_rendered_text(tmp_path: Path) -> None:
    image = Image.new("RGB", (640, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((40, 80), "Kotomka OCR smoke test", fill=(0, 0, 0))
    path = tmp_path / "text.png"
    image.save(path)
    text = ocr_image(path)
    assert text is not None
    assert "Kotomka" in text
