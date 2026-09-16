import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import httpx
from openai import OpenAI, AuthenticationError

import kotomka.providers.llm.codex_subscription as codex


def token(expiry):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
    return f"test.{payload}.test"


def test_concurrent_credential_resolution_refreshes_once(tmp_path, monkeypatch) -> None:
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(codex, "codex_auth_file", lambda: auth_path)
    codex._save_auth_store({"tokens": {"access_token": token(0), "refresh_token": "old-test-token"}})
    fresh = {"access_token": token(time.time() + 3600), "refresh_token": "new-test-token"}
    calls = []
    ready = Barrier(4)

    def refresh(refresh_token):
        calls.append(refresh_token)
        time.sleep(0.05)
        return fresh.copy()

    def resolve(_):
        ready.wait(timeout=3)
        return codex.resolve_codex_credentials()

    monkeypatch.setattr(codex, "_refresh_codex_oauth", refresh)
    with ThreadPoolExecutor(max_workers=4) as pool:
        credentials = list(pool.map(resolve, range(4)))

    assert calls == ["old-test-token"]
    assert all(item.access_token == fresh["access_token"] for item in credentials)
    assert json.loads(auth_path.read_text())["tokens"] == fresh
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert sorted(path.name for path in tmp_path.iterdir()) == ["auth.json"]


def test_failed_auth_publish_keeps_previous_store_and_removes_temp(tmp_path, monkeypatch) -> None:
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(codex, "codex_auth_file", lambda: auth_path)
    codex._save_auth_store({"version": "old"})

    def fail_replace(*args):
        raise OSError("publish failed")

    monkeypatch.setattr(type(auth_path), "replace", fail_replace)
    with pytest.raises(OSError, match="publish failed"):
        codex._save_auth_store({"version": "new"})
    assert json.loads(auth_path.read_text()) == {"version": "old"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["auth.json"]


def make_provider(tmp_path, monkeypatch, handler):
    monkeypatch.setattr(codex, "codex_auth_file", lambda: tmp_path / "auth.json")
    codex._save_auth_store({"tokens": {"access_token": token(10000), "refresh_token": "refresh-test"}})
    monkeypatch.setattr(codex.time, "time", lambda: 1000)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(codex, "OpenAI", lambda **kwargs: OpenAI(**kwargs, http_client=client, max_retries=0))
    return codex.CodexSubscriptionProvider()


def request(provider):
    return provider._request_json(instructions="test", text="test", images=[], image_detail="low",
                                  schema_name="test", schema={"type": "object"})


def success_response():
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          text='data: {"type":"response.output_text.delta","delta":"{}"}\n\ndata: [DONE]\n\n')


def test_provider_rechecks_expiry_between_requests(tmp_path, monkeypatch):
    used = []
    def handler(req):
        used.append(req.headers["authorization"])
        return success_response()
    provider = make_provider(tmp_path, monkeypatch, handler)
    refreshes = []
    def refresh(old):
        refreshes.append(old)
        return {"access_token": token(20000), "refresh_token": "fresh-test"}
    monkeypatch.setattr(codex, "_refresh_codex_oauth", refresh)
    assert request(provider) == {}
    monkeypatch.setattr(codex.time, "time", lambda: 9999)
    assert request(provider) == {}
    assert used == [f"Bearer {token(10000)}", f"Bearer {token(20000)}"]
    assert refreshes == ["refresh-test"]


@pytest.mark.parametrize("always_rejected", [False, True])
def test_provider_refreshes_once_on_401(tmp_path, monkeypatch, always_rejected):
    calls = []
    def handler(req):
        calls.append(req.headers["authorization"])
        if len(calls) == 1 or always_rejected:
            return httpx.Response(401, json={"error": {"message": "test rejection"}})
        return success_response()
    provider = make_provider(tmp_path, monkeypatch, handler)
    refreshes = []
    def refresh(old):
        refreshes.append(old)
        return {"access_token": token(20000), "refresh_token": "fresh-test"}
    monkeypatch.setattr(codex, "_refresh_codex_oauth", refresh)
    if always_rejected:
        with pytest.raises(AuthenticationError):
            request(provider)
    else:
        assert request(provider) == {}
    assert len(calls) == 2
    assert refreshes == ["refresh-test"]


def test_401_from_old_request_does_not_refresh_already_rotated_credentials(tmp_path, monkeypatch):
    make_provider(tmp_path, monkeypatch, lambda _: success_response())
    calls = []
    def refresh(old):
        calls.append(old)
        return {"access_token": token(20000), "refresh_token": "fresh-test"}
    monkeypatch.setattr(codex, "_refresh_codex_oauth", refresh)
    first = codex.resolve_codex_credentials(rejected_access_token=token(10000))
    second = codex.resolve_codex_credentials(rejected_access_token=token(10000))
    assert first == second
    assert calls == ["refresh-test"]
