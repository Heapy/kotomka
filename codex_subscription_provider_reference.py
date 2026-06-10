#!/usr/bin/env python3
"""Reference implementation: use OpenAI Codex through ChatGPT subscription auth.

This file is intentionally self-contained.  It extracts the minimum moving
parts needed to use the same kind of route Hermes calls `openai-codex`:

    ChatGPT/Codex OAuth device login
      -> local auth store with access_token + refresh_token
      -> https://chatgpt.com/backend-api/codex
      -> Responses API request using the OpenAI Python SDK

It is not imported by Hermes at runtime.  Treat it as an implementation map for
porting the Codex subscription route into another project, or for reading the
Hermes provider abstraction without jumping across many files.

Official Codex distinction:

* ChatGPT sign-in gives subscription access.  Usage follows the user's ChatGPT
  workspace, plan limits, and Codex entitlements.
* API-key sign-in gives usage-based access.  Usage is billed through OpenAI
  Platform at API rates.

Hermes implements the first option for provider `openai-codex`.  That is why
`openai-codex` does not need OPENAI_API_KEY and why Hermes cost accounting marks
this route as subscription-included instead of token-priced.

Primary Hermes files this reference mirrors:

* plugins/model-providers/openai-codex/__init__.py
* hermes_cli/auth.py
* hermes_cli/runtime_provider.py
* agent/auxiliary_client.py
* agent/transports/codex.py
* agent/codex_responses_adapter.py
* agent/codex_runtime.py
* agent/usage_pricing.py
* agent/account_usage.py

Hermes provider abstraction, compressed:

1. Provider profile
   A provider declares stable metadata: provider id, aliases, `api_mode`,
   auth type, base URL, and optional env vars.  In Hermes this is
   `ProviderProfile` registered by a model-provider plugin.

2. Auth resolution
   The auth layer turns a provider id into usable credentials.  API-key
   providers read env vars or credential-pool entries.  OAuth providers refresh
   access tokens and return a bearer token.  For `openai-codex`, the bearer is a
   ChatGPT/Codex OAuth access token, not an OpenAI Platform API key.

3. Runtime resolution
   The runtime resolver returns a provider-neutral structure:

       RuntimeCredentials(provider, api_mode, base_url, api_key, source)

   `AIAgent` does not need to know how the token was acquired.  It only needs
   the resolved `api_mode`, endpoint, and bearer value.

4. Transport selection
   `api_mode` selects the wire protocol:

       chat_completions    -> OpenAI-compatible /chat/completions
       codex_responses     -> Responses API-style input/tools/events
       anthropic_messages  -> native Anthropic Messages API
       bedrock_converse    -> AWS Bedrock Converse

   `openai-codex` always resolves to `codex_responses`.

5. Request adaptation
   Hermes stores conversation messages in an OpenAI chat-like shape.  The Codex
   transport converts those messages and tool schemas into Responses API input
   items and function tools before calling `client.responses.create(...)`.

6. Billing/accounting
   Billing is provider-route specific.  Hermes classifies `openai-codex` as
   `subscription_included`: usage still consumes Codex/ChatGPT limits, but the
   session cost estimator does not price it as Platform API tokens.

7. Auth-store separation
   Hermes stores its own Codex OAuth state under the Hermes auth store.  It does
   not continuously share `~/.codex/auth.json` with Codex CLI, because OAuth
   refresh tokens rotate and sharing them across clients can invalidate one
   client when the other refreshes.

Run this reference directly:

    python docs/codex_subscription_provider_reference.py login
    python docs/codex_subscription_provider_reference.py chat "Explain this route"
    python docs/codex_subscription_provider_reference.py models
    python docs/codex_subscription_provider_reference.py usage

Dependencies: httpx and openai.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_ISSUER = "https://auth.openai.com"
CODEX_REFRESH_SKEW_SECONDS = 120


class AuthError(RuntimeError):
    """Authentication or entitlement failure for a provider route."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "openai-codex",
        code: str = "auth_error",
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


@dataclass(frozen=True)
class ProviderProfile:
    """Static provider metadata.

    This is the shape Hermes model-provider plugins register.  The profile is
    intentionally not a credential.  It says which protocol and base endpoint a
    provider uses; auth resolution supplies the bearer later.
    """

    name: str
    aliases: tuple[str, ...] = ()
    api_mode: str = "chat_completions"
    base_url: str = ""
    auth_type: str = "api_key"
    env_vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeCredentials:
    """Resolved provider data consumed by the agent transport layer."""

    provider: str
    api_mode: str
    base_url: str
    api_key: str
    source: str
    last_refresh: Optional[str] = None


class ProviderRegistry:
    """Small provider registry mirroring Hermes' provider plugin registry."""

    def __init__(self) -> None:
        self._profiles: dict[str, ProviderProfile] = {}
        self._aliases: dict[str, str] = {}

    def register(self, profile: ProviderProfile) -> None:
        self._profiles[profile.name] = profile
        self._aliases[profile.name] = profile.name
        for alias in profile.aliases:
            self._aliases[alias] = profile.name

    def get(self, requested: str) -> ProviderProfile:
        key = (requested or "").strip().lower()
        canonical = self._aliases.get(key, key)
        try:
            return self._profiles[canonical]
        except KeyError as exc:
            raise ValueError(f"Unknown provider: {requested!r}") from exc


PROVIDERS = ProviderRegistry()
PROVIDERS.register(
    ProviderProfile(
        name="openai-codex",
        aliases=("codex", "openai_codex"),
        api_mode="codex_responses",
        base_url=DEFAULT_CODEX_BASE_URL,
        auth_type="oauth_external",
        env_vars=(),
    )
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _auth_store_path() -> Path:
    """Return the reference auth-store path.

    Hermes uses `~/.hermes/auth.json`.  This reference writes a separate file by
    default so experiments do not mutate a real Hermes installation.  Override
    with CODEX_REFERENCE_AUTH_FILE if you want a different path.
    """

    override = os.getenv("CODEX_REFERENCE_AUTH_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes" / "codex_subscription_reference_auth.json"


def _load_auth_store() -> dict[str, Any]:
    path = _auth_store_path()
    if not path.exists():
        return {"version": 1, "providers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthError(f"Could not read auth store {path}: {exc}", code="auth_store_read_failed") from exc
    if not isinstance(data, dict):
        raise AuthError(f"Auth store {path} is not a JSON object", code="auth_store_invalid")
    providers = data.setdefault("providers", {})
    if not isinstance(providers, dict):
        data["providers"] = {}
    return data


def _save_auth_store(data: dict[str, Any]) -> None:
    path = _auth_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _get_provider_state(provider: str) -> dict[str, Any]:
    store = _load_auth_store()
    state = store.get("providers", {}).get(provider)
    return dict(state) if isinstance(state, dict) else {}


def _set_provider_state(provider: str, state: dict[str, Any]) -> None:
    store = _load_auth_store()
    providers = store.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        store["providers"] = providers
    providers[provider] = state
    store["active_provider"] = provider
    _save_auth_store(store)


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


def _access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


def _chatgpt_account_id(access_token: str) -> Optional[str]:
    claims = _decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def codex_default_headers(access_token: str) -> dict[str, str]:
    """Headers Hermes uses for the ChatGPT Codex backend.

    The Codex backend is not the public Platform API.  In addition to the bearer
    token, Hermes sends a Codex-CLI-like originator and user-agent.  When the
    OAuth JWT includes a ChatGPT account id, include it too.
    """

    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent reference)",
        "originator": "codex_cli_rs",
    }
    account_id = _chatgpt_account_id(access_token)
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def run_codex_device_code_login() -> dict[str, Any]:
    """Run OpenAI Codex device auth and return unsaved token state."""

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install httpx") from exc

    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        response = client.post(
            f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/usercode",
            json={"client_id": CODEX_OAUTH_CLIENT_ID},
            headers={"Content-Type": "application/json"},
        )
    if response.status_code != 200:
        raise AuthError(
            f"Device-code request failed with HTTP {response.status_code}",
            code="device_code_request_failed",
        )

    device_data = response.json()
    user_code = str(device_data.get("user_code") or "").strip()
    device_auth_id = str(device_data.get("device_auth_id") or "").strip()
    poll_interval = max(3, int(device_data.get("interval") or 5))
    if not user_code or not device_auth_id:
        raise AuthError("Device-code response was missing fields", code="device_code_incomplete")

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
            if poll_response.status_code != 200:
                raise AuthError(
                    f"Device-code polling failed with HTTP {poll_response.status_code}",
                    code="device_code_poll_failed",
                )
            payload = poll_response.json()
            authorization_code = str(payload.get("authorization_code") or "").strip()
            code_verifier = str(payload.get("code_verifier") or "").strip()
            break

    if not authorization_code or not code_verifier:
        raise AuthError("Login timed out or returned no authorization code", code="device_code_timeout")

    redirect_uri = f"{CODEX_OAUTH_ISSUER}/deviceauth/callback"
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        token_response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": redirect_uri,
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_response.status_code != 200:
        raise AuthError(
            f"Token exchange failed with HTTP {token_response.status_code}",
            code="token_exchange_failed",
        )

    tokens = token_response.json()
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not access_token or not refresh_token:
        raise AuthError("Token exchange did not return access and refresh tokens", code="token_exchange_incomplete")

    return {
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "account_id": _chatgpt_account_id(access_token),
        },
        "base_url": os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL,
        "auth_mode": "chatgpt",
        "source": "device-code",
        "last_refresh": _utc_now_iso(),
    }


def save_codex_tokens(
    tokens: dict[str, Any],
    *,
    base_url: str = DEFAULT_CODEX_BASE_URL,
    source: str = "device-code",
) -> None:
    state = {
        "tokens": tokens,
        "base_url": base_url.rstrip("/") or DEFAULT_CODEX_BASE_URL,
        "auth_mode": "chatgpt",
        "source": source,
        "last_refresh": _utc_now_iso(),
    }
    _set_provider_state("openai-codex", state)


def read_codex_tokens() -> dict[str, Any]:
    state = _get_provider_state("openai-codex")
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError("No Codex credentials stored. Run the login command.", code="codex_auth_missing")
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not access_token:
        raise AuthError("Codex auth is missing access_token", code="codex_auth_missing_access_token")
    if not refresh_token:
        raise AuthError("Codex auth is missing refresh_token", code="codex_auth_missing_refresh_token")
    return state


def refresh_codex_oauth(access_token: str, refresh_token: str) -> dict[str, str]:
    """Refresh a Codex OAuth access token.

    Refresh-token errors usually require a fresh browser/device login.  Quota
    or usage-limit errors do not become fixable by reauthenticating.
    """

    del access_token
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install httpx") from exc

    with httpx.Client(timeout=httpx.Timeout(20.0), headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code == 429:
        raise AuthError(
            "Codex provider quota exhausted. Credentials are still valid; retry after reset.",
            code="codex_rate_limited",
            relogin_required=False,
        )
    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with HTTP {response.status_code}"
        try:
            body = response.json()
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict):
                code = str(err.get("code") or err.get("type") or code)
                message = str(err.get("message") or message)
            elif isinstance(err, str):
                code = err
                message = str(body.get("error_description") or body.get("message") or message)
        except Exception:
            pass
        raise AuthError(
            message,
            code=code,
            relogin_required=response.status_code in {400, 401, 403}
            or code in {"invalid_grant", "invalid_token", "refresh_token_reused"},
        )

    payload = response.json()
    new_access = str(payload.get("access_token") or "").strip()
    if not new_access:
        raise AuthError("Refresh response was missing access_token", code="codex_refresh_missing_access_token")
    return {
        "access_token": new_access,
        "refresh_token": str(payload.get("refresh_token") or refresh_token).strip(),
        "account_id": _chatgpt_account_id(new_access),
    }


def resolve_codex_runtime_credentials(*, force_refresh: bool = False) -> RuntimeCredentials:
    """Return the provider-neutral runtime credentials for `openai-codex`."""

    profile = PROVIDERS.get("openai-codex")
    state = read_codex_tokens()
    tokens = dict(state["tokens"])
    access_token = str(tokens["access_token"]).strip()

    if force_refresh or _access_token_is_expiring(access_token, CODEX_REFRESH_SKEW_SECONDS):
        tokens = refresh_codex_oauth(access_token, str(tokens.get("refresh_token") or ""))
        save_codex_tokens(
            tokens,
            base_url=str(state.get("base_url") or profile.base_url),
            source=str(state.get("source") or "auth-store-refresh"),
        )
        state = read_codex_tokens()
        access_token = str(tokens["access_token"]).strip()

    return RuntimeCredentials(
        provider=profile.name,
        api_mode=profile.api_mode,
        base_url=str(state.get("base_url") or profile.base_url).rstrip("/"),
        api_key=access_token,
        source=str(state.get("source") or "auth-store"),
        last_refresh=state.get("last_refresh"),
    )


def _content_to_responses_parts(content: Any, *, role: str) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append({"type": text_type, "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"text", "input_text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append({"type": text_type, "text": text})
        elif item_type in {"image_url", "input_image"}:
            image_ref = item.get("image_url")
            detail = item.get("detail")
            if isinstance(image_ref, dict):
                url = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url = image_ref
            if isinstance(url, str) and url:
                image_part = {"type": "input_image", "image_url": url}
                if isinstance(detail, str) and detail:
                    image_part["detail"] = detail
                parts.append(image_part)
    return parts


def chat_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat-style messages into Responses API input items."""

    items: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue
        if role in {"user", "assistant"}:
            content = msg.get("content", "")
            content_parts = _content_to_responses_parts(content, role=role)
            if content_parts:
                items.append({"role": role, "content": content_parts})
            else:
                items.append({"role": role, "content": "" if content is None else str(content)})

            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function") or {}
                    name = fn.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    arguments = fn.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    elif not isinstance(arguments, str):
                        arguments = str(arguments)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": str(tool_call.get("call_id") or tool_call.get("id") or f"call_{len(items)}"),
                            "name": name,
                            "arguments": arguments or "{}",
                        }
                    )
            continue
        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "").strip()
            if not call_id:
                continue
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "" if msg.get("content") is None else str(msg.get("content")),
                }
            )
    return items


def responses_tools(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Convert Chat Completions function schemas to Responses function tools."""

    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for item in tools:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted or None


def build_codex_responses_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    instructions: Optional[str] = None,
    session_id: Optional[str] = None,
    reasoning_effort: str = "medium",
) -> dict[str, Any]:
    """Build the Responses API payload for the Codex backend."""

    payload_messages = list(messages)
    if instructions is None and payload_messages and payload_messages[0].get("role") == "system":
        instructions = str(payload_messages.pop(0).get("content") or "")
    instructions = instructions or "You are a helpful coding assistant."

    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": chat_messages_to_responses_input(payload_messages),
        "store": False,
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }
    converted_tools = responses_tools(tools)
    if converted_tools:
        kwargs["tools"] = converted_tools
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = True
    if session_id:
        kwargs["prompt_cache_key"] = session_id
        kwargs["extra_headers"] = {
            "session_id": session_id,
            "x-client-request-id": session_id,
        }
    return kwargs


@dataclass
class CodexResponse:
    output_text: str
    output: list[Any]
    usage: Any = None
    status: str = "completed"
    error: Any = None


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name)
    return default if value is None else value


def _output_item_text(item: Any) -> str:
    """Best-effort text extraction from a Responses output message item."""

    item_type = _event_field(item, "type", "")
    if item_type != "message":
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


def consume_responses_stream(
    event_iter: Iterable[Any],
    *,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> CodexResponse:
    """Consume Responses SSE events without relying on SDK final reconstruction."""

    output_items: list[Any] = []
    text_parts: list[str] = []
    usage = None
    status = "completed"
    error = None

    for event in event_iter:
        event_type = str(_event_field(event, "type", "") or "")
        if event_type == "error":
            message = str(_event_field(event, "message", "Codex stream error"))
            code = _event_field(event, "code")
            raise RuntimeError(f"{message} ({code})" if code else message)
        if event_type == "response.output_text.delta" or "output_text.delta" in event_type:
            delta = _event_field(event, "delta", "")
            if isinstance(delta, str) and delta:
                text_parts.append(delta)
                if on_text_delta:
                    on_text_delta(delta)
            continue
        if event_type == "response.output_item.done":
            item = _event_field(event, "item")
            if item is not None:
                output_items.append(item)
            continue
        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            response = _event_field(event, "response")
            if response is not None:
                usage = getattr(response, "usage", None)
                if usage is None and isinstance(response, dict):
                    usage = response.get("usage")
                response_status = getattr(response, "status", None)
                if response_status is None and isinstance(response, dict):
                    response_status = response.get("status")
                if isinstance(response_status, str):
                    status = response_status
                response_error = getattr(response, "error", None)
                if response_error is None and isinstance(response, dict):
                    response_error = response.get("error")
                error = response_error
            break

    output_text = "".join(text_parts)
    if not output_text and output_items:
        output_text = "".join(_output_item_text(item) for item in output_items)

    return CodexResponse(
        output_text=output_text,
        output=output_items,
        usage=usage,
        status=status,
        error=error,
    )


def create_codex_openai_client(creds: RuntimeCredentials):
    """Build an OpenAI SDK client pointed at the ChatGPT Codex backend."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install openai") from exc

    return OpenAI(
        api_key=creds.api_key,
        base_url=creds.base_url,
        default_headers=codex_default_headers(creds.api_key),
    )


def codex_chat(
    prompt: str,
    *,
    model: str = "gpt-5.4",
    stream_to_stdout: bool = True,
    session_id: Optional[str] = None,
) -> CodexResponse:
    """Send one text-only prompt through the Codex subscription route."""

    creds = resolve_codex_runtime_credentials()
    client = create_codex_openai_client(creds)
    messages = [{"role": "user", "content": prompt}]
    kwargs = build_codex_responses_kwargs(model=model, messages=messages, session_id=session_id)
    kwargs["stream"] = True

    def _print_delta(delta: str) -> None:
        print(delta, end="", flush=True)

    events = client.responses.create(**kwargs)
    try:
        result = consume_responses_stream(events, on_text_delta=_print_delta if stream_to_stdout else None)
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()
    if stream_to_stdout:
        print()
    return result


def fetch_codex_models() -> list[str]:
    """Return model slugs visible to the authenticated ChatGPT/Codex account."""

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install httpx") from exc

    creds = resolve_codex_runtime_credentials()
    headers = {"Authorization": f"Bearer {creds.api_key}", **codex_default_headers(creds.api_key)}
    with httpx.Client(timeout=10.0) as client:
        response = client.get(
            f"{creds.base_url}/models?client_version=1.0.0",
            headers=headers,
        )
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    result: list[str] = []
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            visibility = str(item.get("visibility") or "").lower()
            if isinstance(slug, str) and slug and visibility not in {"hide", "hidden"}:
                result.append(slug)
    return result


def _codex_usage_url(base_url: str) -> str:
    normalized = (base_url or DEFAULT_CODEX_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    if "/backend-api" in normalized:
        return normalized + "/wham/usage"
    return normalized + "/api/codex/usage"


def fetch_codex_usage() -> dict[str, Any]:
    """Return raw Codex usage/plan payload for the authenticated account."""

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install httpx") from exc

    creds = resolve_codex_runtime_credentials()
    headers = {
        "Authorization": f"Bearer {creds.api_key}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    account_id = _chatgpt_account_id(creds.api_key)
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    with httpx.Client(timeout=15.0) as client:
        response = client.get(_codex_usage_url(creds.base_url), headers=headers)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"raw": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex subscription provider reference")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login", help="Run Codex device-code login and save tokens")
    chat = sub.add_parser("chat", help="Send one prompt through the Codex subscription route")
    chat.add_argument("prompt")
    chat.add_argument("--model", default="gpt-5.4")
    chat.add_argument("--session-id", default=None)
    sub.add_parser("models", help="List visible Codex model slugs")
    sub.add_parser("usage", help="Print raw Codex usage/plan payload")
    args = parser.parse_args()

    if args.command == "login":
        creds = run_codex_device_code_login()
        save_codex_tokens(
            creds["tokens"],
            base_url=creds["base_url"],
            source=creds.get("source", "device-code"),
        )
        print(f"Saved Codex auth state: {_auth_store_path()}")
        return
    if args.command == "chat":
        codex_chat(args.prompt, model=args.model, session_id=args.session_id)
        return
    if args.command == "models":
        for model in fetch_codex_models():
            print(model)
        return
    if args.command == "usage":
        print(json.dumps(fetch_codex_usage(), indent=2, sort_keys=True))
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
