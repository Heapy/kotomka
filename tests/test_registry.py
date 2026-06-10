from kotomka.providers.llm import available_llm_providers, get_llm_provider
from kotomka.providers.stt import available_stt_providers, get_stt_provider


def test_provider_registries() -> None:
    assert "fake" in available_stt_providers()
    assert "fake" in available_llm_providers()
    assert get_stt_provider("fake").name == "fake"
    assert get_llm_provider("fake").name == "fake"

