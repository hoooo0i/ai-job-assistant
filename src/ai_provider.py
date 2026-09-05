from __future__ import annotations

import os
from typing import Any, Protocol, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.5"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AiProviderError(RuntimeError):
    """A provider failure that is safe to surface without request contents."""


class MissingProviderCredentialError(AiProviderError):
    pass


class UnsupportedProviderError(AiProviderError):
    pass


class StructuredOutputProvider(Protocol):
    name: str
    model: str

    def parse(
        self,
        *,
        instructions: str,
        user_content: str,
        schema: type[SchemaT],
        max_output_tokens: int = 6_000,
    ) -> SchemaT: ...


def get_provider_name() -> str:
    return os.getenv("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER


def get_model_name() -> str:
    configured = os.getenv("AI_MODEL", "").strip()
    legacy = os.getenv("OPENAI_MODEL", "").strip()
    return configured or legacy or DEFAULT_MODEL


def has_provider_credentials() -> bool:
    if get_provider_name() == "openai":
        return bool(os.getenv("OPENAI_API_KEY", "").strip())
    return False


def create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise MissingProviderCredentialError(
            "尚未配置 OPENAI_API_KEY，无法开始结构化解析。"
        )
    return OpenAI(api_key=api_key, timeout=60.0, max_retries=1)


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self.client = client or create_openai_client()
        self.model = model or get_model_name()

    def parse(
        self,
        *,
        instructions: str,
        user_content: str,
        schema: type[SchemaT],
        max_output_tokens: int = 6_000,
    ) -> SchemaT:
        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=[{"role": "user", "content": user_content}],
                text_format=schema,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
                store=False,
                max_output_tokens=max_output_tokens,
            )
        except OpenAIError as exc:
            raise AiProviderError(
                "OpenAI 服务调用失败，请检查 API 密钥、模型权限和网络后重试。"
            ) from exc

        parsed = response.output_parsed
        if parsed is None:
            raise AiProviderError("模型未返回可用的结构化结果，请重试。")
        return parsed


def create_ai_provider(client: Any | None = None) -> StructuredOutputProvider:
    provider_name = get_provider_name()
    if provider_name == "openai":
        return OpenAIProvider(client=client)
    raise UnsupportedProviderError(
        f"暂不支持 AI_PROVIDER={provider_name}。当前可用值：openai。"
    )
