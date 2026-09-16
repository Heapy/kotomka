from pathlib import Path

import pytest
from PIL import Image

import kotomka.pdf as pdf_module
from kotomka.models import (
    AssessmentFlag,
    FrameSelection,
    Report,
    ReportAssessment,
    ReportSection,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)
from kotomka.pdf import _write_reportlab_pdf


@pytest.mark.parametrize("image_size", [(320, 180), (1080, 1920), (800, 8000)])
def test_reportlab_pdf_contains_full_report_shape(tmp_path: Path, image_size) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", image_size, color=(255, 255, 255)).save(frames_dir / "frame.png")
    report = Report(
        video=VideoMetadata(source_url="https://example.com/video", title="Example Video", duration_s=120),
        summary="A useful summary.",
        sections=[
            ReportSection(
                title="Main idea",
                start_s=0,
                end_s=60,
                body="Detailed notes with enough content to render.",
                frame_ids=["frame-1"],
                citations=[0],
            )
        ],
        frames=[
            FrameSelection(
                frame_id="frame-1",
                timestamp_s=12,
                image_path="frame.png",
                caption="A useful frame.",
            )
        ],
        transcript=Transcript(
            language="en",
            duration_s=120,
            speakers=["Speaker A"],
            segments=[
                TranscriptSegment(
                    start_s=0,
                    end_s=120,
                    speaker="Speaker A",
                    text="Transcript text " * 80,
                )
            ],
        ),
        assessment=ReportAssessment(
            verdict="The report replaces watching.",
            originality="Mostly original material.",
            freshness="Current as of the upload date.",
            audience="Engineers.",
            prerequisites=["Basics"],
            actionability="Apply directly.",
            insight_density="High.",
            stale_claims=[AssessmentFlag(claim="Version-specific advice", timestamp_s=30.0, risk="May be outdated")],
        ),
    )
    output = tmp_path / "report.pdf"

    _write_reportlab_pdf(report, output)

    assert output.stat().st_size > 4096


def test_failed_pdf_export_preserves_previous_complete_file(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "report.pdf"
    output.write_bytes(b"previous complete PDF")
    report = Report(
        video=VideoMetadata(source_url="https://example.com/video"),
        summary="Summary", sections=[], frames=[], transcript=Transcript(),
    )

    def broken_renderer(report, destination):
        destination.write_bytes(b"partial PDF")
        raise RuntimeError("render failed")

    monkeypatch.setenv("KOTOMKA_PDF_RENDERER", "reportlab")
    monkeypatch.setattr(pdf_module, "_write_reportlab_pdf", broken_renderer)
    with pytest.raises(RuntimeError, match="render failed"):
        pdf_module.render_pdf(None, report, output)
    assert output.read_bytes() == b"previous complete PDF"
    assert list(tmp_path.iterdir()) == [output]
