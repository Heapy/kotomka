from __future__ import annotations

from contextlib import asynccontextmanager
import re
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from .config import get_settings
from .models import JobCreate
from .pdf import render_pdf, should_regenerate_pdf
from .providers.llm import available_llm_providers
from .providers.stt import available_stt_providers
from .reporting import CITATION_PATTERN, load_report
from .storage import JobStore
from .utils import format_timecode, read_json
from .worker import JobWorker


PACKAGE_DIR = Path(__file__).parent
settings = get_settings()
store = JobStore(settings.db_path, settings.jobs_dir)
worker = JobWorker(store=store, settings=settings)
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
templates.env.filters["timecode"] = format_timecode
templates.env.globals["timestamp_url"] = lambda url, seconds: _timestamp_url(url, seconds)
templates.env.globals["citation_links"] = lambda text, url: _citation_links(text, url)


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.start()
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(title="Kotomka", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "stt_providers": available_stt_providers(),
            "llm_providers": available_llm_providers(),
            "default_stt": settings.stt_provider,
            "default_llm": settings.llm_provider,
        },
    )


@app.get("/jobs", response_class=HTMLResponse, name="jobs_index")
def jobs_index(request: Request, show_read: bool = False) -> HTMLResponse:
    jobs = store.list_jobs(limit=200, include_read=show_read)
    rows = []
    for job in jobs:
        rows.append({"job": job, "title": _job_display_title(job)})
    return templates.TemplateResponse(request, "jobs.html", {"rows": rows, "show_read": show_read})


@app.post("/api/jobs")
async def create_job_api(payload: JobCreate) -> JSONResponse:
    job = store.create_job(payload)
    worker.enqueue(job.id)
    return JSONResponse({"job_id": job.id})


@app.post("/jobs")
def create_job_form(
    request: Request,
    source_url: str = Form(...),
    output_language: str = Form("ru"),
    stt_provider: str = Form(""),
    llm_provider: str = Form(""),
    cookies_from_browser: str = Form(""),
) -> RedirectResponse:
    payload = JobCreate(
        source_url=source_url,
        output_language=output_language,
        stt_provider=stt_provider or None,
        llm_provider=llm_provider or None,
        cookies_from_browser=cookies_from_browser or None,
    )
    job = store.create_job(payload)
    worker.enqueue(job.id)
    return RedirectResponse(str(request.url_for("job_report", job_id=job.id)), status_code=303)


@app.get("/api/jobs/{job_id}")
def get_job_api(job_id: str) -> JSONResponse:
    job = _get_job_or_404(job_id)
    return JSONResponse(
        {
            "id": job.id,
            "status": job.status,
            "is_read": job.is_read,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "result": job.result,
        }
    )


@app.post("/jobs/{job_id}/retry", name="job_retry")
def retry_job(request: Request, job_id: str, use_current_defaults: bool = Form(False)) -> RedirectResponse:
    job = _get_job_or_404(job_id)
    if job.status not in {"failed", "completed"}:
        return RedirectResponse(str(request.url_for("job_report", job_id=job_id)), status_code=303)
    payload = None
    if use_current_defaults:
        payload = job.input.model_copy(update={"stt_provider": None, "llm_provider": None})
    retried = store.retry_job(job_id, payload=payload)
    worker.enqueue(retried.id)
    return RedirectResponse(str(request.url_for("job_report", job_id=job_id)), status_code=303)


@app.post("/jobs/{job_id}/read", name="job_read")
def set_job_read(
    request: Request,
    job_id: str,
    is_read: bool = Form(True),
    return_to: str = Form("report"),
    show_read: bool = Form(False),
) -> RedirectResponse:
    _get_job_or_404(job_id)
    store.set_job_read(job_id, is_read)
    if return_to == "jobs":
        target = str(request.url_for("jobs_index"))
        if show_read:
            target = f"{target}?{urlencode({'show_read': '1'})}"
        return RedirectResponse(target, status_code=303)
    return RedirectResponse(str(request.url_for("job_report", job_id=job_id)), status_code=303)


@app.post("/jobs/{job_id}/delete", name="job_delete")
def delete_job(request: Request, job_id: str, show_read: bool = Form(False)) -> RedirectResponse:
    job = _get_job_or_404(job_id)
    if job.status not in {"completed", "failed"}:
        return RedirectResponse(str(request.url_for("job_report", job_id=job_id)), status_code=303)
    store.delete_job(job_id)
    target = str(request.url_for("jobs_index"))
    if show_read:
        target = f"{target}?{urlencode({'show_read': '1'})}"
    return RedirectResponse(target, status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse, name="job_report")
def job_report(request: Request, job_id: str) -> HTMLResponse:
    job = _get_job_or_404(job_id)
    report_path = job.artifact_dir / "report.json"
    if job.status != "completed" or not report_path.exists():
        return templates.TemplateResponse(request, "status.html", {"job": job})
    report = load_report(report_path)
    return templates.TemplateResponse(request, "report.html", {"job": job, "report": report})


@app.get("/jobs/{job_id}/assets/{asset_path:path}", name="job_asset")
def job_asset(job_id: str, asset_path: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    root = job.artifact_dir.resolve()
    target = (job.artifact_dir / asset_path).resolve()
    if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/jobs/{job_id}/pdf", name="job_pdf")
def job_pdf(request: Request, job_id: str, force: bool = False) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")
    report_path = job.artifact_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    pdf_path = job.artifact_dir / "report.pdf"
    if force or should_regenerate_pdf(report_path, pdf_path):
        render_pdf(request, load_report(report_path), pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{job.id}.pdf")


def _get_job_or_404(job_id: str):
    try:
        return store.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


def _timestamp_url(url: str | None, seconds: float) -> str:
    if not url:
        return "#"
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={int(seconds)}s"


def _citation_links(text: str, url: str | None) -> Markup:
    escaped = escape(text)
    if not url:
        return Markup(escaped)

    def replace(match: re.Match[str]) -> str:
        values = [value.strip() for value in match.group(1).split(",")]
        links: list[str] = []
        for value in values:
            seconds = float(value)
            href = escape(_timestamp_url(url, seconds))
            label = escape(format_timecode(seconds))
            links.append(f'<a class="time-link" href="{href}">{label}</a>')
        return "[" + ", ".join(links) + "]"

    return Markup(CITATION_PATTERN.sub(replace, str(escaped)))


def _job_display_title(job) -> str:
    report_path = job.artifact_dir / "report.json"
    if report_path.exists():
        try:
            return str(read_json(report_path).get("video", {}).get("title") or job.input.source_url)
        except Exception:
            pass
    source_path = job.artifact_dir / "source.json"
    if source_path.exists():
        try:
            return str(read_json(source_path).get("metadata", {}).get("title") or job.input.source_url)
        except Exception:
            pass
    return job.input.source_url
