from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from ...config import get_settings
from .json_base import ImageInput, JsonLlmProviderBase, image_data_url
from .json_helpers import parse_json_object


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_ISSUER = "https://auth.openai.com"
CODEX_REFRESH_SKEW_SECONDS = 120


@dataclass(frozen=True)
class CodexCredentials:
    base_url: str
    access_token: str
    refresh_token: str


class CodexAuthError(RuntimeError):
    pass


class CodexSubscriptionProvider(JsonLlmProviderBase):
    name = "codex_subscription"

    def __init__(self, *, model: str | None = None) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.codex_model
        self.scoring_model = self.settings.codex_scoring_model or self.model
        self.creds = resolve_codex_credentials()
        self.client = OpenAI(
            api_key=self.creds.access_token,
            base_url=self.creds.base_url,
            default_headers=codex_default_headers(self.creds.access_token),
        )

    def _scoring_model(self) -> str | None:
        return self.scoring_model

    def _request_json(
        self,
        *,
        instructions: str,
        text: str,
        images: list[ImageInput],
        image_detail: str,
        schema_name: str,
        schema: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        # The Codex backend supports neither strict json_schema nor tools; the
        # schema is embedded as a prompt hint and tool requests are dropped.
        del schema_name, tools
        schema_hint = json.dumps(schema, ensure_ascii=False)
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": f"{text}\n\nReturn a JSON object matching this schema exactly:\n{schema_hint}",
            }
        ]
        for image in images:
            content.append({"type": "input_text", "text": image.label})
            content.append({"type": "input_image", "image_url": image_data_url(image.path), "detail": image_detail})
        events = self.client.responses.create(
            model=model or self.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            store=False,
            reasoning={"effort": "medium", "summary": "auto"},
            stream=True,
        )
        try:
            return parse_json_object(_consume_output_text(events))
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()


def codex_auth_file() -> Path:
    return get_settings().codex_auth_file


def _consume_output_text(event_iter: Any) -> str:
    parts: list[str] = []
    output_items: list[Any] = []
    for event in event_iter:
        event_type = _event_field(event, "type", "")
        if event_type == "error":
            message = _event_field(event, "message", "Codex stream error")
            raise RuntimeError(str(message))
        if event_type == "response.output_text.delta" or "output_text.delta" in str(event_type):
            delta = _event_field(event, "delta", "")
            if isinstance(delta, str):
                parts.append(delta)
            continue
        if event_type == "response.output_item.done":
            item = _event_field(event, "item")
            if item is not None:
                output_items.append(item)
            continue
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            response = _event_field(event, "response")
            error = _event_field(response, "error") if response is not None else None
            if error:
                raise RuntimeError(str(error))
            break
    text = "".join(parts)
    if text:
        return text
    return "".join(_output_item_text(item) for item in output_items)


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name)
    return default if value is None else value


def _output_item_text(item: Any) -> str:
    if _event_field(item, "type", "") != "message":
        return ""
    content = _event_field(item, "content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        part_type = _event_field(part, "type", "")
        if part_type not in {"output_text", "text"}:
            continue
        text = _event_field(part, "text", "")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def codex_auth_exists() -> bool:
    return codex_auth_file().exists()


def run_codex_device_login() -> Path:
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        response = client.post(
            f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/usercode",
            json={"client_id": CODEX_OAUTH_CLIENT_ID},
            headers={"Content-Type": "application/json"},
        )
    response.raise_for_status()
    device_data = response.json()
    user_code = str(device_data.get("user_code") or "").strip()
    device_auth_id = str(device_data.get("device_auth_id") or "").strip()
    poll_interval = max(3, int(device_data.get("interval") or 5))
    if not user_code or not device_auth_id:
        raise CodexAuthError("Device-code response was missing required fields")

    print("Open this URL and enter the code:")
    print(f"  {CODEX_OAUTH_ISSUER}/codex/device")
    print(f"  code: {user_code}")
    print("Waiting for sign-in...")

    authorization_code = ""
    code_verifier = ""
    deadline = time.monotonic() + 15 * 60
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            poll_response = client.post(
                f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/token",
                json={"device_auth_id": device_auth_id, "user_code": user_code},
                headers={"Content-Type": "application/json"},
            )
            if poll_response.status_code in {403, 404}:
                continue
            poll_response.raise_for_status()
            payload = poll_response.json()
            authorization_code = str(payload.get("authorization_code") or "").strip()
            code_verifier = str(payload.get("code_verifier") or "").strip()
            break
    if not authorization_code or not code_verifier:
        raise CodexAuthError("Login timed out or returned no authorization code")

    token_response = httpx.post(
        CODEX_OAUTH_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": f"{CODEX_OAUTH_ISSUER}/deviceauth/callback",
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20.0,
    )
    token_response.raise_for_status()
    tokens = token_response.json()
    _save_auth_store(
        {
            "tokens": {
                "access_token": str(tokens.get("access_token") or ""),
                "refresh_token": str(tokens.get("refresh_token") or ""),
            },
            "base_url": DEFAULT_CODEX_BASE_URL,
            "last_refresh": _utc_now_iso(),
        }
    )
    return codex_auth_file()


def resolve_codex_credentials() -> CodexCredentials:
    store = _load_auth_store()
    tokens = store.get("tokens") if isinstance(store, dict) else None
    if not isinstance(tokens, dict):
        raise CodexAuthError("No Codex subscription credentials. Run `uv run kotomka codex-login`.")
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not access_token or not refresh_token:
        raise CodexAuthError("Codex subscription auth store is missing tokens. Re-run codex-login.")
    if _access_token_is_expiring(access_token, CODEX_REFRESH_SKEW_SECONDS):
        tokens = _refresh_codex_oauth(refresh_token)
        store["tokens"] = tokens
        store["last_refresh"] = _utc_now_iso()
        _save_auth_store(store)
        access_token = str(tokens["access_token"])
        refresh_token = str(tokens["refresh_token"])
    return CodexCredentials(
        base_url=str(store.get("base_url") or DEFAULT_CODEX_BASE_URL).rstrip("/"),
        access_token=access_token,
        refresh_token=refresh_token,
    )


def codex_default_headers(access_token: str) -> dict[str, str]:
    headers = {"User-Agent": "codex_cli_rs/0.0.0 (Kotomka)", "originator": "codex_cli_rs"}
    account_id = _chatgpt_account_id(access_token)
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def _refresh_codex_oauth(refresh_token: str) -> dict[str, str]:
    response = httpx.post(
        CODEX_OAUTH_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CODEX_OAUTH_CLIENT_ID},
        timeout=20.0,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise CodexAuthError("Codex refresh response was missing access_token")
    return {
        "access_token": access_token,
        "refresh_token": str(payload.get("refresh_token") or refresh_token).strip(),
    }


def _load_auth_store() -> dict[str, Any]:
    path = codex_auth_file()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_auth_store(data: dict[str, Any]) -> None:
    path = codex_auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _decode_jwt_claims(token: Any) -> dict[str, Any]:
    if not isinstance(token, str) or token.count(".") < 2:
        return {}
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _access_token_is_expiring(access_token: str, skew_seconds: int) -> bool:
    exp = _decode_jwt_claims(access_token).get("exp")
    return isinstance(exp, (int, float)) and float(exp) <= time.time() + max(0, int(skew_seconds))


def _chatgpt_account_id(access_token: str) -> str | None:
    auth_claims = _decode_jwt_claims(access_token).get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None
