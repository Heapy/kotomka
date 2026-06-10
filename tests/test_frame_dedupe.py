from pathlib import Path

from PIL import Image

from kotomka.media import dedupe_frames
from kotomka.models import CandidateFrame


def test_dedupe_frames_removes_identical_images(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(first)
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(second)
    frames = [
        CandidateFrame(frame_id="a", timestamp_s=0, path=first),
        CandidateFrame(frame_id="b", timestamp_s=1, path=second),
    ]
    assert [frame.frame_id for frame in dedupe_frames(frames)] == ["a"]

