from __future__ import annotations

from ...config import get_settings
from .assemblyai import AssemblyAiSttProvider
from .base import SttProvider
from .fake import FakeSttProvider


def available_stt_providers() -> list[str]:
    return ["fake", "assemblyai"]


def get_stt_provider(name: str | None = None) -> SttProvider:
    settings = get_settings()
    provider_name = (name or settings.stt_provider or "fake").strip().lower()
    if provider_name == "fake":
        return FakeSttProvider()
    if provider_name == "assemblyai":
        return AssemblyAiSttProvider(poll_seconds=settings.assemblyai_poll_seconds)
    raise ValueError(f"Unknown STT provider: {provider_name}")

