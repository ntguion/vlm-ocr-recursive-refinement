import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prompts
from llm_providers import ProviderResponse, ProviderUsage
from ocr import (
    UsageTotals,
    cost_estimate,
    extract_first_json_object,
    parse_args,
    process_document,
    render_document_to_images,
    render_presentation_to_images,
)


class DocumentInputTests(unittest.TestCase):
    def test_extract_json_ignores_braces_inside_strings(self):
        response = 'prefix {"final_markdown": "value with { brace", "risk_notes": []} suffix'
        self.assertEqual(
            extract_first_json_object(response),
            {"final_markdown": "value with { brace", "risk_notes": []},
        )

    def test_reasoning_tokens_are_not_double_counted(self):
        usage = UsageTotals(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            reasoning_tokens=250_000,
        )
        self.assertEqual(usage.total_tokens, 2_000_000)
        self.assertEqual(cost_estimate(usage, 2.0, 3.0), 5.0)

    def test_file_argument_accepts_powerpoint(self):
        with mock.patch.object(sys, "argv", ["ocr.py", "--file", "deck.pptx"]):
            args = parse_args()
        self.assertEqual(args.file, "deck.pptx")
        self.assertIsNone(args.pdf)

    def test_pdf_alias_remains_available(self):
        with mock.patch.object(sys, "argv", ["ocr.py", "--pdf", "document.pdf"]):
            args = parse_args()
        self.assertEqual(args.pdf, "document.pdf")
        self.assertIsNone(args.file)

    def test_unsupported_document_extension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected .pdf, .ppt, or .pptx"):
            render_document_to_images("notes.txt")

    def test_powerpoint_requires_libreoffice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            presentation = Path(temp_dir) / "deck.pptx"
            presentation.write_bytes(b"placeholder")
            with mock.patch("ocr._find_libreoffice", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "requires LibreOffice"):
                    render_presentation_to_images(str(presentation))

    def test_powerpoint_conversion_renders_and_cleans_temporary_pdf(self):
        rendered_paths = []

        def fake_run(command, **kwargs):
            output_dir = Path(command[command.index("--outdir") + 1])
            (output_dir / "deck.pdf").write_bytes(b"%PDF-placeholder")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        def fake_render(pdf_path, dpi):
            rendered_paths.append(pdf_path)
            self.assertTrue(Path(pdf_path).exists())
            self.assertEqual(dpi, 144)
            return [b"page-image"]

        with tempfile.TemporaryDirectory() as temp_dir:
            presentation = Path(temp_dir) / "deck.pptx"
            presentation.write_bytes(b"placeholder")
            with (
                mock.patch("ocr._find_libreoffice", return_value="/usr/bin/soffice"),
                mock.patch("ocr.subprocess.run", side_effect=fake_run),
                mock.patch("ocr.render_pdf_to_images", side_effect=fake_render),
            ):
                images = render_presentation_to_images(str(presentation), dpi=144)

        self.assertEqual(images, [b"page-image"])
        self.assertEqual(len(rendered_paths), 1)
        self.assertFalse(Path(rendered_paths[0]).exists())

    def test_prompts_cover_timeline_and_gantt_layouts(self):
        self.assertIn("timeline", prompts.USER_PROMPT_TEMPLATE)
        self.assertIn("gantt_chart", prompts.USER_PROMPT_TEMPLATE)
        self.assertIn("complex_structure", prompts.SYSTEM_PROMPT)


class FakeProvider:
    name = "fake"
    supports_reasoning_effort = False
    supports_structured_output = False

    async def complete_with_image(self, **kwargs):
        return ProviderResponse(
            text=(
                '{"page_number": 1, "layout_classification": "timeline", '
                '"risk_notes": [], "final_markdown": "| Phase | Q1 |"}'
            ),
            usage=ProviderUsage(
                input_tokens=10,
                output_tokens=5,
                reasoning_tokens=2,
            ),
        )

    async def complete_text(self, **kwargs):
        raise AssertionError("repair should not be needed for valid JSON")


class DocumentMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_document_writes_lean_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_md = Path(temp_dir) / "output.md"
            output_json = Path(temp_dir) / "output.json"
            with (
                mock.patch(
                    "ocr.render_document_to_images",
                    return_value=([b"image"], "powerpoint"),
                ),
                mock.patch("ocr.table_extractor.extract_table_context") as table_context,
            ):
                await process_document(
                    input_path="fixtures/synthetic-deck.pptx",
                    output_md_path=str(output_md),
                    output_json_path=str(output_json),
                    provider_name="fake",
                    provider=FakeProvider(),
                    model="fake-model",
                    reasoning_effort="high",
                    max_output_tokens=100,
                    max_concurrency=1,
                    refinement_passes=2,
                    input_price_per_mtok=1.0,
                    output_price_per_mtok=1.0,
                )

            table_context.assert_not_called()
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["metadata"]["source_file"], "synthetic-deck.pptx")
        self.assertEqual(payload["metadata"]["source_type"], "powerpoint")
        self.assertEqual(payload["metadata"]["usage"]["total_tokens"], 15)
        self.assertEqual(payload["metadata"]["usage"]["reasoning_tokens"], 2)
        self.assertEqual(payload["pages"][0]["refinement_passes_executed"], 0)
        self.assertTrue(payload["pages"][0]["refinement_converged"])


if __name__ == "__main__":
    unittest.main()
