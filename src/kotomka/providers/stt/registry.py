from __future__ import annotations

from ...config import get_settings
from .assemblyai import AssemblyAiSttProvider
from .base import SttProvider
from .fake import FakeSttProvider
from .whisper_local import WhisperLocalSttProvider, whisper_available


def available_stt_providers() -> list[str]:
    providers = ["fake", "assemblyai"]
    if whisper_available():
        providers.append("whisper")
    return providers


def get_stt_provider(name: str | None = None) -> SttProvider:
    settings = get_settings()
    provider_name = (name or settings.stt_provider or "fake").strip().lower()
    if provider_name == "fake":
        return FakeSttProvider()
    if provider_name == "assemblyai":
        return AssemblyAiSttProvider(poll_seconds=settings.assemblyai_poll_seconds)
    if provider_name == "whisper":
        if not whisper_available():
            raise ValueError(
                "STT provider 'whisper' needs faster-whisper; install it with `uv sync --extra whisper`"
            )
        return WhisperLocalSttProvider()
    raise ValueError(f"Unknown STT provider: {provider_name}")
