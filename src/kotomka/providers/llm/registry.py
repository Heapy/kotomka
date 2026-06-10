from __future__ import annotations

import os

from .base import LlmProvider
from .codex_subscription import CodexSubscriptionProvider, codex_auth_exists
from .fake import FakeLlmProvider
from .openai_responses import OpenAiResponsesProvider


def available_llm_providers() -> list[str]:
    return ["auto", "fake", "openai", "codex_subscription"]


def get_llm_provider(name: str | None = None) -> LlmProvider:
    provider_name = (name or "auto").strip().lower()
    if provider_name == "auto":
        if codex_auth_exists():
            provider_name = "codex_subscription"
        elif os.getenv("OPENAI_API_KEY", "").strip():
            provider_name = "openai"
        else:
            provider_name = "fake"
    if provider_name == "fake":
        return FakeLlmProvider()
    if provider_name == "openai":
        return OpenAiResponsesProvider()
    if provider_name in {"codex", "openai-codex", "codex_subscription"}:
        return CodexSubscriptionProvider()
    raise ValueError(f"Unknown LLM provider: {provider_name}")

