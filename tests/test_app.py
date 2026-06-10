from fastapi.testclient import TestClient

from kotomka.app import app


def test_index_renders() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Video to Presentation" in response.text


def test_jobs_index_renders() -> None:
    with TestClient(app) as client:
        response = client.get("/jobs")
    assert response.status_code == 200
    assert "Jobs" in response.text
