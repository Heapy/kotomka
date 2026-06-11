from __future__ import annotations

import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import Request

from .models import Report
from .utils import format_timecode


def render_pdf(request: Request, report: Report, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if os.getenv("KOTOMKA_PDF_RENDERER", "reportlab").strip().lower() != "browser":
        try:
            _write_reportlab_pdf(report, output_path)
        except Exception:
            _write_minimal_pdf(report, output_path)
        return output_path

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.goto(
                    str(request.url_for("job_report", job_id=request.path_params["job_id"])) + "?print=1",
                    wait_until="networkidle",
                )
                page.emulate_media(media="print")
                page.pdf(
                    path=str(output_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "16mm", "right": "14mm", "bottom": "16mm", "left": "14mm"},
                )
            finally:
                browser.close()
        return output_path
    except Exception:
        try:
            _render_with_chrome_cli(request, output_path)
        except Exception:
            try:
                _write_reportlab_pdf(report, output_path)
            except Exception:
                _write_minimal_pdf(report, output_path)
        return output_path


def should_regenerate_pdf(report_path: Path, pdf_path: Path) -> bool:
    if not pdf_path.exists():
        return True
    if pdf_path.stat().st_size < 4096:
        return True
    return report_path.exists() and report_path.stat().st_mtime > pdf_path.stat().st_mtime


def _launch_chromium(playwright):
    try:
        return playwright.chromium.launch(channel="chrome", headless=True)
    except Exception:
        pass
    executable = os.getenv("KOTOMKA_CHROME_EXECUTABLE", "").strip()
    candidates = [Path(executable)] if executable else []
    candidates.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return playwright.chromium.launch(executable_path=str(candidate), headless=True)
    return playwright.chromium.launch(headless=True)


def _render_with_chrome_cli(request: Request, output_path: Path) -> None:
    executable = os.getenv("KOTOMKA_CHROME_EXECUTABLE", "").strip()
    candidates = [Path(executable)] if executable else []
    candidates.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ]
    )
    chrome = next((candidate for candidate in candidates if candidate.exists()), None)
    if chrome is None:
        raise RuntimeError("No system Chrome executable found")
    url = str(request.url_for("job_report", job_id=request.path_params["job_id"])) + "?print=1"
    result = subprocess.run(
        [
            str(chrome),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={output_path}",
            url,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 4096:
        raise RuntimeError(result.stderr or result.stdout or "Chrome CLI PDF generation failed")


def _write_reportlab_pdf(report: Report, output_path: Path) -> None:
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    font_name = _register_reportlab_font(pdfmetrics, TTFont)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="KotomkaTitle", parent=styles["Title"], fontName=font_name, fontSize=22, leading=26))
    styles.add(ParagraphStyle(name="KotomkaH2", parent=styles["Heading2"], fontName=font_name, fontSize=15, leading=19))
    styles.add(ParagraphStyle(name="KotomkaH3", parent=styles["Heading3"], fontName=font_name, fontSize=12, leading=16))
    styles.add(ParagraphStyle(name="KotomkaBody", parent=styles["BodyText"], fontName=font_name, fontSize=9.5, leading=13))
    styles.add(
        ParagraphStyle(
            name="KotomkaMeta",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#68707a"),
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=report.video.title,
    )
    width = A4[0] - 28 * mm
    story: list = []
    story.append(Paragraph(_p(report.video.title), styles["KotomkaTitle"]))
    story.append(
        Paragraph(
            _p(f"{report.video.uploader or 'Video'} · {format_timecode(report.video.duration_s)} · generated {report.generated_at:%Y-%m-%d %H:%M}"),
            styles["KotomkaMeta"],
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph("Summary", styles["KotomkaH2"]))
    story.append(Paragraph(_p(report.summary), styles["KotomkaBody"]))
    story.append(Spacer(1, 6 * mm))

    if report.assessment:
        assessment = report.assessment
        story.append(Paragraph("Assessment", styles["KotomkaH2"]))
        if assessment.verdict:
            story.append(Paragraph(_p(assessment.verdict), styles["KotomkaBody"]))
        scores = f"Originality {assessment.originality_score:.0%} · Freshness {assessment.freshness_score:.0%}"
        if assessment.web_search_used:
            scores += " · web-checked"
        story.append(Paragraph(_p(scores), styles["KotomkaMeta"]))
        for label, value in (
            ("Originality", assessment.originality),
            ("Freshness", assessment.freshness),
            ("Audience", assessment.audience),
            ("Actionability", assessment.actionability),
            ("Insight density", assessment.insight_density),
        ):
            if value:
                story.append(Paragraph(_p(f"{label}. {value}"), styles["KotomkaBody"]))
        if assessment.prerequisites:
            story.append(Paragraph(_p("Prerequisites: " + ", ".join(assessment.prerequisites)), styles["KotomkaBody"]))
        for flag in assessment.stale_claims:
            prefix = f"{format_timecode(flag.timestamp_s)} · " if flag.timestamp_s is not None else ""
            suffix = f" — {flag.risk}" if flag.risk else ""
            story.append(Paragraph(_p(f"Re-check: {prefix}{flag.claim}{suffix}"), styles["KotomkaMeta"]))
        story.append(Spacer(1, 6 * mm))

    frames_by_id = {frame.frame_id: frame for frame in report.frames}
    story.append(Paragraph("Detailed Notes", styles["KotomkaH2"]))
    for section in report.sections:
        story.append(Paragraph(_p(f"{section.title} · {format_timecode(section.start_s)}"), styles["KotomkaH3"]))
        story.append(Paragraph(_p(section.body), styles["KotomkaBody"]))
        for frame_id in section.frame_ids[:2]:
            frame = frames_by_id.get(frame_id)
            if frame:
                _append_frame(
                    story, frame, output_path.parent, min(width, 150 * mm), PILImage, Image, Paragraph, Spacer, styles
                )
        story.append(Spacer(1, 4 * mm))

    if report.frames:
        story.append(PageBreak())
        story.append(Paragraph("Key Frames", styles["KotomkaH2"]))
        for frame in report.frames:
            _append_frame(
                story, frame, output_path.parent, min(width, 150 * mm), PILImage, Image, Paragraph, Spacer, styles
            )

    story.append(PageBreak())
    story.append(Paragraph("Full Transcript", styles["KotomkaH2"]))
    for segment in report.transcript.segments:
        story.append(
            Paragraph(
                _p(f"{format_timecode(segment.start_s)} · {segment.speaker}"),
                styles["KotomkaMeta"],
            )
        )
        story.append(Paragraph(_p(segment.text), styles["KotomkaBody"]))
        story.append(Spacer(1, 3 * mm))
    doc.build(story)


def _register_reportlab_font(pdfmetrics, TTFont) -> str:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("KotomkaSans", str(candidate)))
            return "KotomkaSans"
    return "Helvetica"


def _append_frame(story, frame, job_dir: Path, target_width, PILImage, Image, Paragraph, Spacer, styles) -> None:
    path = job_dir / "frames" / frame.image_path
    if not path.exists():
        return
    with PILImage.open(path) as image:
        image_width, image_height = image.size
    target_height = target_width * image_height / max(1, image_width)
    story.append(Image(str(path), width=target_width, height=target_height))
    caption = f"{format_timecode(frame.timestamp_s)} · {frame.caption or frame.content_type}"
    story.append(Paragraph(_p(caption), styles["KotomkaMeta"]))
    story.append(Spacer(1, 12))


def _p(text: str) -> str:
    return escape(str(text)).replace("\n", "<br/>")


def _write_minimal_pdf(report: Report, output_path: Path) -> None:
    lines = [report.video.title, "", report.summary[:1200]]
    text = "\n".join(lines).encode("latin-1", errors="replace").decode("latin-1")
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 16 Tf 72 760 Td ({escaped}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream".encode("latin-1"),
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    output_path.write_bytes(b"".join(chunks))
