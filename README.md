# VLM OCR Recursive Refinement Pipeline

A Python CLI for OCR-style transcription of PDFs and PowerPoint presentations using vision-capable language models. The pipeline renders document pages to images, extracts table context from source PDFs with `pdfplumber`, sends each page to a configured model provider, and writes both Markdown and JSON outputs.

It supports provider/model selection so the same document workflow can run against compatible OpenAI or Anthropic vision models. Risk notes from the initial pass can feed one or more focused refinement passes.

## What It Does

- Renders each PDF page to a high-resolution PNG with PyMuPDF.
- Converts `.ppt` and `.pptx` files to temporary PDFs with LibreOffice, renders the slides, and removes the temporary files.
- Detects simple rotated-page cases and crops large whitespace margins.
- Extracts table text with `pdfplumber` and passes it as context for the model.
- Sends rendered page images to a selected provider/model.
- Requests structured JSON containing page-level Markdown, layout classification, and risk notes.
- Runs targeted recursive refinement passes when the model reports transcription risk notes.
- Gives timelines and Gantt charts explicit structure-preservation guidance.
- Writes `output.md` for reading and `output.json` for downstream use.
- Records refinement convergence, provider API calls, token usage, and configurable cost estimates.

## Supported Providers

The CLI supports:

- `openai` through the OpenAI Responses API.
- `anthropic` through the Anthropic Messages API.

Use vision-capable models only. Text-only models cannot process the rendered page images.

Reasoning effort is provider and model specific. OpenAI reasoning models use the configured `--effort` value when the model supports it. Anthropic requests currently ignore `--effort` and rely on the selected model's default behavior.

## Requirements

- Python 3.10 or newer
- An API key for at least one supported provider
- A PDF or PowerPoint file to process
- LibreOffice for `.ppt` and `.pptx` input; PDF input does not require it

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

Run a PDF with the default provider/model from `.env`:

```bash
python ocr.py --file document.pdf
```

Run a PowerPoint presentation after installing LibreOffice:

```bash
python ocr.py --file presentation.pptx
```

Run with OpenAI:

```bash
python ocr.py \
  --provider openai \
  --model gpt-5.2 \
  --file document.pdf \
  --out result.md \
  --json result.json \
  --refinement-passes 2
```

Run with Anthropic:

```bash
python ocr.py \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --file document.pdf \
  --out result.md \
  --json result.json
```

You can also prefix the model with the provider:

```bash
python ocr.py --model anthropic:claude-sonnet-4-6 --file document.pdf
```

The older `--pdf document.pdf` form remains available as a deprecated alias.

## CLI Options

| Option | Description | Default |
|---|---|---|
| `--file` | Path to an input `.pdf`, `.ppt`, or `.pptx` file | required unless `--pdf` is used |
| `--pdf` | Deprecated PDF-only alias for `--file` | optional |
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
      "final_markdown": "# Document Title\n\nTranscribed content...",
      "refinement_passes_executed": 0,
      "refinement_converged": true
    }
  ],
  "metadata": {
    "source_file": "document.pdf",
    "source_type": "pdf",
    "provider": "openai",
    "model": "gpt-5.2",
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "reasoning_tokens": 0,
      "total_tokens": 0,
      "api_calls": 0,
      "repair_calls": 0,
      "refinement_calls": 0
    },
    "cost_estimate": 0.0,
    "input_price_per_mtok": 1.75,
    "output_price_per_mtok": 14.0
  }
}
```

`output.md` contains the final page transcriptions separated by page markers.

The JSON stores only the source filename, not its full local path. Cost values are estimates based on the rates configured in `.env`. The example OpenAI rates match the [published `gpt-5.2` token prices](https://developers.openai.com/api/docs/models/gpt-5.2) at the time of this update; change them when selecting another model or when pricing changes.

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

Live OCR tests require a real API key and a synthetic or non-sensitive document.

## Limitations

- OCR quality depends on the chosen model, image clarity, document density, and provider image handling.
- The project does not include a benchmark or accuracy guarantee.
- Recursive refinement is model-guided and should not replace human review for high-stakes documents.
- Provider pricing and model support change over time; update `.env` cost settings as needed.
- The CLI sends rendered page images to the selected provider. Do not process sensitive documents unless that provider and account configuration are appropriate for the data.
- PowerPoint input requires LibreOffice. The pipeline intentionally does not use a text-only fallback because it would discard slide layout and images.
- Table context extraction runs only for source PDFs; PowerPoint slides rely on the rendered slide image.
- There is no packaged release, hosted service, access control, queueing, monitoring, or retry persistence.

## Production Hardening Ideas

- Add synthetic PDF fixtures and regression tests for common layouts.
- Add model-by-model evaluation against known transcriptions.
- Add resumable processing for large PDFs.
- Add structured-output mode per provider where available.
- Add safer defaults for concurrency and output directory management.
- Add secret scanning and dependency auditing to CI.

## License

MIT. See `LICENSE`.
