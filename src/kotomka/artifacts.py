from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Report


@dataclass(frozen=True)
class FrameCleanup:
    files: int = 0
    bytes: int = 0


def prune_frame_files(artifact_dir: Path, report: Report, *, dry_run: bool = False) -> FrameCleanup:
    """Remove generated PNGs only after validating every report image reference.

    The caller must prevent retry/deletion for the duration of this operation.
    Candidate manifests remain as diagnostics; only report frames stay on disk.
    """
    frames_dir = artifact_dir / "frames"
    if frames_dir.is_symlink():
        raise ValueError("Frame directory must not be a symlink")
    root = frames_dir.resolve()
    keep: set[Path] = set()
    for frame in report.frames:
        relative = Path(frame.image_path)
        path = frames_dir / relative
        if (
            relative.is_absolute() or ".." in relative.parts
            or path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(root)
            or any(parent.is_symlink() for parent in path.parents if parent != artifact_dir)
        ):
            raise ValueError("Report contains a missing or unsafe frame reference")
        keep.add(path.resolve())

    files = size = 0
    for path in frames_dir.rglob("*.png"):
        if path.is_symlink() or not path.is_file() or path.resolve() in keep:
            continue
        file_size = path.stat().st_size
        if not dry_run:
            path.unlink()
        files += 1
        size += file_size
    return FrameCleanup(files=files, bytes=size)
