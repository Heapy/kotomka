from fastapi.testclient import TestClient

import kotomka.app as app_module
from kotomka.models import JobCreate, Report, ReportAssessment, Transcript, VideoMetadata
from kotomka.reporting import save_report
from kotomka.storage import JobStore


def test_index_renders() -> None:
    with TestClient(app_module.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Video to Presentation" in response.text


def test_jobs_index_renders() -> None:
    with TestClient(app_module.app) as client:
        response = client.get("/jobs")
    assert response.status_code == 200
    assert "Jobs" in response.text


def test_jobs_index_hides_read_jobs_until_requested(tmp_path, monkeypatch) -> None:
    test_store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    unread = test_store.create_job(JobCreate(source_url="https://example.com/unread"))
    read = test_store.create_job(JobCreate(source_url="https://example.com/read"))
    test_store.set_job_read(read.id, True)
    monkeypatch.setattr(app_module, "store", test_store)

    with TestClient(app_module.app) as client:
        response = client.get("/jobs")
        response_with_read = client.get("/jobs?show_read=1")

    assert response.status_code == 200
    assert unread.id in response.text
    assert read.id not in response.text
    assert response_with_read.status_code == 200
    assert unread.id in response_with_read.text
    assert read.id in response_with_read.text


def test_report_page_renders_assessment(tmp_path, monkeypatch) -> None:
    test_store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = test_store.create_job(JobCreate(source_url="https://example.com/v"))
    test_store.update_job(job.id, status="completed", progress=100, message="Completed")
    report = Report(
        video=VideoMetadata(source_url="https://example.com/v", title="Assessed video"),
        summary="Summary text",
        sections=[],
        frames=[],
        transcript=Transcript(language="en", duration_s=10, segments=[]),
        assessment=ReportAssessment(
            verdict="Report replaces watching.",
            originality_score=0.7,
            freshness_score=0.4,
        ),
    )
    save_report(report, job.artifact_dir / "report.json")
    monkeypatch.setattr(app_module, "store", test_store)

    with TestClient(app_module.app) as client:
        response = client.get(f"/jobs/{job.id}")

    assert response.status_code == 200
    assert "Assessment" in response.text
    assert "Report replaces watching." in response.text
    assert "Originality 70%" in response.text


class StubWorker:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)


def test_create_job_form_accepts_speakers_expected(tmp_path, monkeypatch) -> None:
    test_store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    stub_worker = StubWorker()
    monkeypatch.setattr(app_module, "store", test_store)
    monkeypatch.setattr(app_module, "worker", stub_worker)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/jobs",
            data={"source_url": "https://example.com/v", "output_language": "ru", "speakers_expected": "2"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert len(stub_worker.enqueued) == 1
    job = test_store.get_job(stub_worker.enqueued[0])
    assert job.input.speakers_expected == 2


def test_job_read_route_toggles_state_and_preserves_filter(tmp_path, monkeypatch) -> None:
    test_store = JobStore(tmp_path / "app.db", tmp_path / "jobs")
    job = test_store.create_job(JobCreate(source_url="https://example.com/video"))
    monkeypatch.setattr(app_module, "store", test_store)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/jobs/{job.id}/read",
            data={"is_read": "true", "return_to": "jobs"},
            follow_redirects=False,
        )
        response_with_filter = client.post(
            f"/jobs/{job.id}/read",
            data={"is_read": "false", "return_to": "jobs", "show_read": "true"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == "http://testserver/jobs"
    assert response_with_filter.status_code == 303
    assert response_with_filter.headers["location"] == "http://testserver/jobs?show_read=1"
    assert test_store.get_job(job.id).is_read is False
