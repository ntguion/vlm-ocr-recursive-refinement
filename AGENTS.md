# Contributor Notes

## Scope

This repository is a Python CLI for VLM-based OCR and issue-targeted recursive refinement across PDF and PowerPoint input. Keep the core workflow provider-neutral so a page can be processed by compatible OpenAI, Anthropic, or future vision-capable model providers.

## Code Organization

- Keep provider-specific SDK calls inside `llm_providers.py`.
- Keep document conversion/rendering, page orchestration, output writing, and CLI parsing in `ocr.py`.
- Keep prompt text and prompt-format changes in `prompts.py`.
- Keep table-context extraction in `table_extractor.py`.
- Add tests for provider routing, usage parsing, and orchestration behavior when provider logic changes.
- Keep PowerPoint conversion limited to LibreOffice. Do not add a text-only fallback that silently drops slide layout or images.

## Provider Extension Rules

- New providers should implement the `OCRModelProvider` interface in `llm_providers.py`.
- Do not add provider SDK calls directly to `ocr.py`.
- Preserve provider-prefixed model names such as `openai:gpt-5.2` and `anthropic:claude-sonnet-4-6`.
- Use mocked provider tests for default validation. Live provider tests should be opt-in and use non-sensitive PDFs.
- Keep cost estimates configurable through environment variables because provider pricing changes.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` only with local credentials. Do not commit `.env` files or real API keys.

## Validation

Run these checks before opening or merging changes:

```bash
python -m unittest discover -s tests
python -m py_compile ocr.py llm_providers.py prompts.py table_extractor.py tests/*.py
python ocr.py --help
```

For provider-facing changes, also run a live smoke test with a synthetic or otherwise non-sensitive PDF when valid credentials are available.

## Data Handling

- Do not commit PDFs, presentations, rendered page images, logs, local outputs, databases, credentials, or `.env` files.
- Keep examples synthetic and non-sensitive.
- Avoid local filesystem paths in documentation and fixtures.
- Document provider data-flow implications when adding features that transmit additional page context.
