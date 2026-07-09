"""
Provider adapters for OCR model calls.

The OCR pipeline works with rendered page images and JSON-shaped text results.
This module keeps provider-specific payload formats and usage extraction out of
the PDF/table/output orchestration code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple


@dataclass
class ProviderUsage:
    """Normalized token usage returned by a model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class ProviderResponse:
    """Normalized model response used by the OCR pipeline."""

    text: str
    usage: ProviderUsage
    stop_reason: Optional[str] = None
    truncated: bool = False


class ProviderConfigurationError(ValueError):
    """Raised when provider configuration is missing or invalid."""


class OCRModelProvider(Protocol):
    """Protocol implemented by provider-specific model adapters."""

    name: str
    supports_reasoning_effort: bool
    supports_structured_output: bool

    async def complete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        """Run a multimodal request with one page image."""

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        """Run a text-only request, used for JSON repair."""


def resolve_provider_and_model(provider: str, model: str) -> Tuple[str, str]:
    """
    Resolve provider/model CLI inputs.

    Supports both explicit `--provider anthropic --model claude-...` and compact
    model strings such as `anthropic:claude-...`.
    """
    provider_name = (provider or os.getenv("MODEL_PROVIDER", "openai")).strip().lower()
    model_name = (model or "").strip()

    if ":" in model_name:
        prefix, remainder = model_name.split(":", 1)
        prefix = prefix.strip().lower()
        if prefix in {"openai", "anthropic"}:
            if provider_name not in {"auto", prefix}:
                raise ProviderConfigurationError(
                    f"Provider mismatch: --provider {provider_name!r} with model prefix {prefix!r}."
                )
            provider_name = prefix
            model_name = remainder.strip()

    if provider_name == "auto":
        provider_name = "openai"

    if provider_name not in {"openai", "anthropic"}:
        raise ProviderConfigurationError(
            f"Unsupported provider {provider_name!r}. Use 'openai' or 'anthropic'."
        )
    if not model_name:
        raise ProviderConfigurationError("MODEL or --model must not be empty.")

    return provider_name, model_name


def create_provider(provider: str) -> OCRModelProvider:
    """Create a provider adapter from environment configuration."""
    provider_name = provider.strip().lower()
    if provider_name == "openai":
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY", "").strip())
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())
    raise ProviderConfigurationError(
        f"Unsupported provider {provider_name!r}. Use 'openai' or 'anthropic'."
    )


def _extract_openai_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", "")
            if content_type in {"output_text", "text"}:
                parts.append(str(getattr(content, "text", "")))
    return "\n".join(part for part in parts if part)


def _extract_openai_usage(response: Any) -> ProviderUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ProviderUsage()

    reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)
    output_details = getattr(usage, "output_tokens_details", None)
    if output_details is not None:
        reasoning_tokens = max(
            reasoning_tokens,
            int(getattr(output_details, "reasoning_tokens", 0) or 0),
        )

    return ProviderUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        reasoning_tokens=reasoning_tokens,
    )


def _openai_supports_reasoning(model: str) -> bool:
    model_name = model.lower()
    return model_name.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIProvider:
    """OpenAI Responses API adapter."""

    name = "openai"
    supports_reasoning_effort = True
    supports_structured_output = True

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is not set. Add it to .env or choose another provider."
            )
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)

    async def complete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ]
        return await self._create_response(
            system_prompt=system_prompt,
            input_payload=input_payload,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        input_payload = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        ]
        return await self._create_response(
            system_prompt=system_prompt,
            input_payload=input_payload,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )

    async def _create_response(
        self,
        *,
        system_prompt: str,
        input_payload: list[dict[str, Any]],
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": input_payload,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort and _openai_supports_reasoning(model):
            kwargs["reasoning"] = {"effort": reasoning_effort}

        response = await self.client.responses.create(**kwargs)
        usage = _extract_openai_usage(response)
        stop_reason = str(getattr(response, "status", "") or "")
        return ProviderResponse(
            text=_extract_openai_text(response),
            usage=usage,
            stop_reason=stop_reason,
            truncated=usage.output_tokens >= max_output_tokens,
        )


def _split_data_url_image(image_data_url: str) -> Tuple[str, str]:
    if not image_data_url.startswith("data:image/") or ";base64," not in image_data_url:
        raise ValueError("Expected image data URL like data:image/png;base64,...")

    header, data = image_data_url.split(",", 1)
    media_type = header.removeprefix("data:").removesuffix(";base64")
    return media_type, data


def _extract_anthropic_text(response: Any) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)


def _extract_anthropic_usage(response: Any) -> ProviderUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ProviderUsage()
    return ProviderUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        reasoning_tokens=0,
    )


class AnthropicProvider:
    """Anthropic Messages API adapter."""

    name = "anthropic"
    supports_reasoning_effort = False
    supports_structured_output = True

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderConfigurationError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or choose another provider."
            )
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=api_key)

    async def complete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        media_type, image_data = _split_data_url_image(image_data_url)
        message = await self.client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        usage = _extract_anthropic_usage(message)
        stop_reason = getattr(message, "stop_reason", None)
        return ProviderResponse(
            text=_extract_anthropic_text(message),
            usage=usage,
            stop_reason=stop_reason,
            truncated=stop_reason == "max_tokens",
        )

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        message = await self.client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        )
        usage = _extract_anthropic_usage(message)
        stop_reason = getattr(message, "stop_reason", None)
        return ProviderResponse(
            text=_extract_anthropic_text(message),
            usage=usage,
            stop_reason=stop_reason,
            truncated=stop_reason == "max_tokens",
        )
