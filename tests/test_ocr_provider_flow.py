import unittest

from llm_providers import ProviderResponse, ProviderUsage
from ocr import call_model_one_page


class FakeProvider:
    name = "fake"
    supports_reasoning_effort = False
    supports_structured_output = False

    async def complete_with_image(
        self,
        *,
        system_prompt,
        user_prompt,
        image_data_url,
        model,
        reasoning_effort,
        max_output_tokens,
    ):
        return ProviderResponse(
            text=(
                '{"page_number": 999, "layout_classification": "letter", '
                '"risk_notes": [], "final_markdown": "Hello world"}'
            ),
            usage=ProviderUsage(input_tokens=10, output_tokens=5),
        )

    async def complete_text(
        self,
        *,
        system_prompt,
        user_prompt,
        model,
        reasoning_effort,
        max_output_tokens,
    ):
        raise AssertionError("repair should not be needed for valid JSON")


class OCRProviderFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_model_one_page_uses_provider_and_normalizes_page_number(self):
        result, usage = await call_model_one_page(
            FakeProvider(),
            model="fake-model",
            image_data_url="data:image/png;base64,abc",
            page_number=3,
            reasoning_effort="high",
            max_output_tokens=100,
        )

        self.assertEqual(result.page_number, 3)
        self.assertEqual(result.layout_classification, "letter")
        self.assertEqual(result.final_markdown, "Hello world")
        self.assertEqual(usage.api_calls, 1)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 5)


if __name__ == "__main__":
    unittest.main()
