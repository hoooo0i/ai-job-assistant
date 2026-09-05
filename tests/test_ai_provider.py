from types import SimpleNamespace

import pytest

from src.ai_provider import (
    OpenAIProvider,
    UnsupportedProviderError,
    create_ai_provider,
    get_model_name,
    get_provider_name,
)
from src.schemas import ResumeProfile


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def empty_resume() -> ResumeProfile:
    return ResumeProfile(
        summary=None,
        education=[],
        experience=[],
        projects=[],
        skills=[],
        languages=[],
        evidence_chunks=[],
    )


def test_openai_provider_preserves_structured_output_safety_settings() -> None:
    client = FakeClient(empty_resume())
    provider = OpenAIProvider(client=client, model="test-model")

    result = provider.parse(
        instructions="Synthetic instructions",
        user_content="Synthetic content",
        schema=ResumeProfile,
    )

    call = client.responses.calls[0]
    assert result == empty_resume()
    assert call["model"] == "test-model"
    assert call["text_format"] is ResumeProfile
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"] == {"verbosity": "low"}


def test_ai_model_setting_takes_priority_over_legacy_setting(monkeypatch) -> None:
    monkeypatch.setenv("AI_MODEL", "new-model")
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")

    assert get_model_name() == "new-model"


def test_provider_defaults_to_openai(monkeypatch) -> None:
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    assert get_provider_name() == "openai"


def test_unsupported_provider_has_clear_error(monkeypatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "local-test")

    with pytest.raises(UnsupportedProviderError, match="暂不支持"):
        create_ai_provider()
