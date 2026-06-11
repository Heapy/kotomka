from fastapi.testclient import TestClient

import kotomka.app as app_module
from kotomka.models import JobCreate
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
