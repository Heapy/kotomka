from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageFont

from kotomka.media import _DecodedImageCache, dedupe_frames
from kotomka.models import CandidateFrame


def test_similar_slides_decode_once_even_with_a_tiny_cache(tmp_path, monkeypatch):
    frames = []
    for index in range(24):
        slide = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(slide)
        draw.rectangle((0, 0, 1280, 110), fill="black")
        draw.text((40, 40), "Production metrics", font=ImageFont.load_default(size=40), fill="white")
        draw.text((60, 300), f"Latency: {index} ms", font=ImageFont.load_default(size=28), fill="black")
        path = tmp_path / f"metric-{index}.png"
        slide.save(path)
        frames.append(CandidateFrame(frame_id=str(index), timestamp_s=index, path=path))
    opened = []
    original = Image.open
    def tracked(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", tracked)
    assert dedupe_frames(frames, decoded_cache_bytes=1024) == frames
    assert len(opened) == len(frames)


def test_decoded_image_cache_has_a_byte_budget(tmp_path):
    cache = _DecodedImageCache(8 * 8 * 4 * 2)
    for index in range(5):
        cache.put(tmp_path / f"{index}.png", Image.new("RGBA", (8, 8)))
        assert cache.bytes <= cache.budget
    assert [path.name for path in cache.images] == ["3.png", "4.png"]
    cache.put(tmp_path / "too-large.png", Image.new("RGBA", (100, 100)))
    assert cache.bytes <= cache.budget


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


def test_dedupe_keeps_a_completed_slide_build(tmp_path: Path) -> None:
    frames = []
    hashes = []
    for index in range(2):
        slide = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(slide)
        draw.rectangle((0, 0, 1280, 110), fill="#17385c")
        draw.text((45, 30), "Production checklist", font=ImageFont.load_default(size=46), fill="white")
        font = ImageFont.load_default(size=28)
        draw.text((60, 190), "1. Check service health", font=font, fill="black")
        draw.text((60, 270), "2. Confirm replica status", font=font, fill="black")
        if index:
            draw.text((60, 350), "3. Never log access tokens", font=font, fill="black")
        path = tmp_path / f"build-{index}.png"
        slide.save(path)
        hashes.append(imagehash.phash(slide))
        frames.append(CandidateFrame(frame_id=str(index), timestamp_s=10 + index * 20, path=path, source="plateau"))
    assert 1 < hashes[0] - hashes[1] <= 6
    assert [frame.frame_id for frame in dedupe_frames(frames)] == ["0", "1"]


def test_dedupe_keeps_changed_slide_numbers_even_when_phash_collides(tmp_path: Path) -> None:
    frames = []
    hashes = []
    for index, number in enumerate(["1", "9"]):
        slide = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(slide)
        draw.rectangle((0, 0, 1280, 110), fill="black")
        draw.text((40, 40), "Production metrics", font=ImageFont.load_default(size=40), fill="white")
        draw.text((60, 300), f"Latency: {number} ms", font=ImageFont.load_default(size=20), fill="black")
        path = tmp_path / f"metric-{index}.png"
        slide.save(path)
        hashes.append(imagehash.phash(slide))
        frames.append(CandidateFrame(frame_id=str(index), timestamp_s=10 + index * 20, path=path, source="plateau"))
    assert hashes[0] - hashes[1] <= 1
    assert [frame.frame_id for frame in dedupe_frames(frames)] == ["0", "1"]


def test_dedupe_tolerates_small_pixel_noise(tmp_path: Path) -> None:
    first = Image.new("RGB", (320, 180), (120, 150, 180))
    ImageDraw.Draw(first).rectangle((20, 20, 150, 130), fill=(10, 20, 30))
    noisy = first.point(lambda channel: min(255, channel + 3))
    paths = [tmp_path / "original.png", tmp_path / "noisy.png"]
    first.save(paths[0])
    noisy.save(paths[1])
    frames = [CandidateFrame(frame_id=str(i), timestamp_s=i, path=path) for i, path in enumerate(paths)]
    assert [frame.frame_id for frame in dedupe_frames(frames)] == ["0"]
