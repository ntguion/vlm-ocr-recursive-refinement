import os
import unittest
from types import SimpleNamespace
from unittest import mock

from llm_providers import (
    ProviderConfigurationError,
    _extract_anthropic_text,
    _extract_anthropic_usage,
    _extract_openai_text,
    _extract_openai_usage,
    _openai_supports_reasoning,
    _split_data_url_image,
    create_provider,
    resolve_provider_and_model,
)


class ProviderResolutionTests(unittest.TestCase):
    def test_explicit_provider_and_model(self):
        self.assertEqual(
            resolve_provider_and_model("anthropic", "claude-sonnet-4-6"),
            ("anthropic", "claude-sonnet-4-6"),
        )

    def test_provider_prefixed_model(self):
        self.assertEqual(
            resolve_provider_and_model("auto", "openai:gpt-5.2"),
            ("openai", "gpt-5.2"),
        )

    def test_provider_mismatch_rejected(self):
        with self.assertRaises(ProviderConfigurationError):
            resolve_provider_and_model("openai", "anthropic:claude-sonnet-4-6")

    def test_missing_api_key_rejected_before_sdk_import(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderConfigurationError):
                create_provider("openai")
            with self.assertRaises(ProviderConfigurationError):
                create_provider("anthropic")


class OpenAIExtractionTests(unittest.TestCase):
    def test_extract_output_text_helper(self):
        response = SimpleNamespace(output_text="hello")
        self.assertEqual(_extract_openai_text(response), "hello")

    def test_extract_output_text_fallback(self):
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    content=[
                        SimpleNamespace(type="output_text", text="hello"),
                        SimpleNamespace(type="text", text="world"),
                    ]
                )
            ]
        )
        self.assertEqual(_extract_openai_text(response), "hello\nworld")

    def test_extract_usage_with_reasoning_details(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=20,
                output_tokens_details=SimpleNamespace(reasoning_tokens=5),
            )
        )
        usage = _extract_openai_usage(response)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.reasoning_tokens, 5)

    def test_openai_reasoning_model_detection(self):
        self.assertTrue(_openai_supports_reasoning("gpt-5.2"))
        self.assertTrue(_openai_supports_reasoning("o4-mini"))
        self.assertFalse(_openai_supports_reasoning("gpt-4o"))


class AnthropicExtractionTests(unittest.TestCase):
    def test_split_data_url_image(self):
        media_type, data = _split_data_url_image("data:image/png;base64,abc123")
        self.assertEqual(media_type, "image/png")
        self.assertEqual(data, "abc123")

    def test_extract_text_blocks(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="hello"),
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text="world"),
            ]
        )
        self.assertEqual(_extract_anthropic_text(response), "hello\nworld")

    def test_extract_usage(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=11, output_tokens=22)
        )
        usage = _extract_anthropic_usage(response)
        self.assertEqual(usage.input_tokens, 11)
        self.assertEqual(usage.output_tokens, 22)
        self.assertEqual(usage.reasoning_tokens, 0)


if __name__ == "__main__":
    unittest.main()
