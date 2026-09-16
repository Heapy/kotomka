import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

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
