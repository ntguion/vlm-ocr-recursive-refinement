# VLM OCR Recursive Refinement Pipeline

A Python CLI for OCR-style transcription of PDFs using vision-capable language models. The pipeline renders PDF pages to images, extracts table context with `pdfplumber`, sends each page to a configured model provider, and writes both Markdown and JSON outputs.

It supports provider/model selection so the same document workflow can run against compatible OpenAI or Anthropic vision models. Risk notes from the initial pass can feed one or more focused refinement passes.

## What It Does

- Renders each PDF page to a high-resolution PNG with PyMuPDF.
- Detects simple rotated-page cases and crops large whitespace margins.
- Extracts table text with `pdfplumber` and passes it as context for the model.
- Sends rendered page images to a selected provider/model.
- Requests structured JSON containing page-level Markdown, layout classification, and risk notes.
- Runs targeted recursive refinement passes when the model reports transcription risk notes.
- Writes `output.md` for reading and `output.json` for downstream use.
- Tracks provider API calls, token usage, and configurable cost estimates.

## Supported Providers

The CLI supports:

- `openai` through the OpenAI Responses API.
- `anthropic` through the Anthropic Messages API.

Use vision-capable models only. Text-only models cannot process the rendered page images.

Reasoning effort is provider and model specific. OpenAI reasoning models use the configured `--effort` value when the model supports it. Anthropic requests currently ignore `--effort` and rely on the selected model's default behavior.

## Requirements

- Python 3.10 or newer
- An API key for at least one supported provider
- A PDF file to process

## Installation

```bash
git clone https://github.com/ntguion/vlm-ocr-recursive-refinement.git
cd vlm-ocr-recursive-refinement
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add the key for the provider you plan to use:

```bash
MODEL_PROVIDER=openai
MODEL=gpt-5.2
OPENAI_API_KEY=replace-with-openai-key
```

or:

```bash
MODEL_PROVIDER=anthropic
MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=replace-with-anthropic-key
```

## Usage

Check the CLI:

```bash
python ocr.py --help
```

Run with the default provider/model from `.env`:

```bash
python ocr.py --pdf document.pdf
```

Run with OpenAI:

```bash
python ocr.py \
  --provider openai \
  --model gpt-5.2 \
  --pdf document.pdf \
  --out result.md \
  --json result.json \
  --refinement-passes 2
```

Run with Anthropic:

```bash
python ocr.py \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --pdf document.pdf \
  --out result.md \
  --json result.json
```

You can also prefix the model with the provider:

```bash
python ocr.py --model anthropic:claude-sonnet-4-6 --pdf document.pdf
```

## CLI Options

| Option | Description | Default |
|---|---|---|
| `--pdf` | Path to input PDF file | required |
| `--out` | Markdown output path | `output.md` |
| `--json` | JSON output path | `output.json` |
| `--provider` | `openai`, `anthropic`, or `auto` | `openai` |
| `--model` | Vision-capable model name, optionally `provider:model` | `gpt-5.2` |
| `--effort` | Provider/model-specific reasoning effort | `high` |
| `--max-output-tokens` | Maximum response tokens per page | `16384` |
| `--max-concurrency` | Maximum pages processed concurrently | `8` |
| `--refinement-passes` | Maximum issue-targeted refinement passes per page | `1` |

## Output

`output.json` contains page-level records:

```json
{
  "pages": [
    {
      "page_number": 1,
      "layout_classification": "letter",
      "risk_notes": [],
      "final_markdown": "# Document Title\n\nTranscribed content..."
    }
  ],
  "metadata": {
    "source_pdf": "document.pdf",
    "provider": "openai",
    "model": "gpt-5.2"
  }
}
```

`output.md` contains the final page transcriptions separated by page markers.

## Project Structure

```text
.
├── AGENTS.md          # Contributor guidance for local changes
├── ocr.py              # CLI and VLM/OCR refinement orchestration
├── llm_providers.py    # OpenAI and Anthropic provider adapters
├── prompts.py          # OCR and verification prompts
├── table_extractor.py  # pdfplumber table context extraction
├── requirements.txt
├── .env.example
├── tests/
└── README.md
```

## Development Notes

See `AGENTS.md` for repository conventions, validation commands, provider-extension guidance, and data-handling expectations.

## Validation

Run syntax checks:

```bash
python -m py_compile ocr.py llm_providers.py prompts.py table_extractor.py
```

Run unit tests:

```bash
python -m unittest discover -s tests
```

Live OCR tests require a real API key and a synthetic or non-sensitive PDF.

## Limitations

- OCR quality depends on the chosen model, image clarity, document density, and provider image handling.
- The project does not include a benchmark or accuracy guarantee.
- Recursive refinement is model-guided and should not replace human review for high-stakes documents.
- Provider pricing and model support change over time; update `.env` cost settings as needed.
- The CLI sends rendered page images to the selected provider. Do not process sensitive PDFs unless that provider and account configuration are appropriate for the data.
- There is no packaged release, hosted service, access control, queueing, monitoring, or retry persistence.

## Production Hardening Ideas

- Add synthetic PDF fixtures and regression tests for common layouts.
- Add model-by-model evaluation against known transcriptions.
- Add resumable processing for large PDFs.
- Add structured-output mode per provider where available.
- Add safer defaults for concurrency and output directory management.
- Add CI for linting, tests, and secret scanning.

## License

MIT. See `LICENSE`.
