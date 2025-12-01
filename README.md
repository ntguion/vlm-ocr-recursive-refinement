# GPT-5.1 OCR Pipeline

A production-ready OCR pipeline using GPT-5.1 Vision API for high-accuracy document transcription. Automatically handles tables, rotation, callouts, and produces structured JSON output.

## Features

- **High-Accuracy OCR**: Uses GPT-5.1 with high reasoning effort for precise text transcription
- **Automatic Table Extraction**: Extracts tables using pdfplumber and provides as context to improve accuracy
- **Smart Rotation Detection**: Auto-detects and corrects rotated pages
- **Whitespace Cropping**: Automatically crops pages to maximize content resolution
- **Risk Verification**: Optional second-pass verification for pages with identified risks
- **Structured Output**: Produces both JSON (structured) and Markdown (readable) formats
- **Callout Handling**: Preserves annotations, handwritten notes, and arrows with intent-aware formatting
- **Progress Tracking**: Real-time progress bars and detailed terminal output

## Prerequisites

- Python 3.10 or higher
- OpenAI API key with access to GPT-5.1
  - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
  - Ensure your account has access to GPT-5.1 model

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gpt51_ocr_pipeline
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=sk-your-api-key-here
   ```
   
   Optional: Adjust other settings as needed (see `.env.example` for all options).

## Quick Start

**1. Verify your setup:**
```bash
python3 ocr.py --help
```

**2. Run OCR on a PDF:**
```bash
python3 ocr.py --pdf your_document.pdf
```

This will create `output.md` and `output.json` in the current directory.

**3. Check the results:**
- `output.json` - Structured data for programmatic access
- `output.md` - Human-readable markdown document

**With custom output paths:**
```bash
python3 ocr.py --pdf your_document.pdf --out result.md --json result.json
```

**With custom settings:**
```bash
python ocr.py \
  --pdf document.pdf \
  --out result.md \
  --json result.json \
  --model gpt-5.1 \
  --effort high \
  --max-output-tokens 16384 \
  --max-concurrency 8
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pdf` | Path to input PDF file (required) | - |
| `--out` | Path to Markdown output file | `output.md` |
| `--json` | Path to JSON output file | `output.json` |
| `--model` | OpenAI model name | `gpt-5.1` |
| `--effort` | Reasoning effort level (`low`, `medium`, `high`) | `high` |
| `--max-output-tokens` | Maximum tokens per page response | `16384` |
| `--max-concurrency` | Maximum concurrent page processing | `8` |

## Output Formats

### JSON Output (`output.json`)

Structured data containing all page results:

```json
{
  "pages": [
    {
      "page_number": 1,
      "layout_classification": "letter",
      "risk_notes": [
        {
          "type": "small_font",
          "location": "header",
          "description": "Very small text"
        }
      ],
      "final_markdown": "# Document Title\n\nContent..."
    }
  ],
  "metadata": {
    "source_pdf": "document.pdf",
    "model": "gpt-5.1"
  }
}
```

### Markdown Output (`output.md`)

Human-readable document with pages separated by `-----`:

```markdown
### Page 1

# Document Title

Content here...

-----
### Page 2

More content...
```

## How It Works

1. **PDF Rendering**: Converts PDF pages to high-resolution PNG images (300 DPI)
   - Auto-detects vertical text and rotates pages
   - Crops whitespace to maximize content resolution

2. **Table Extraction**: For each page, extracts tables using pdfplumber
   - Provides "source of truth" data to improve LLM accuracy
   - Falls back to text-based extraction for complex layouts

3. **OCR Processing**: Sends page images to GPT-5.1 Vision API
   - Uses high reasoning effort for accuracy
   - Processes multiple pages concurrently (configurable)

4. **Risk Verification**: Optional second pass for pages with identified risks
   - Focuses on specific problem areas
   - Verifies and corrects transcription errors

5. **Output Generation**: Produces both JSON and Markdown formats
   - JSON for programmatic access
   - Markdown for human readability

## API Calls Per Page

- **Normal case**: 1 API call (primary OCR)
- **With risks**: 2 API calls (primary + verification)
- **With JSON repair**: Up to 3 API calls (primary + repair + optional verification)

## Cost Estimation

The pipeline tracks token usage and provides cost estimates based on your `.env` pricing:

```
💰 Estimated cost:       $X.XXXX
   (Input: $1.25/M, Output: $10.0/M)
```

Update `INPUT_PRICE_PER_MTOK` and `OUTPUT_PRICE_PER_MTOK` in `.env` to match current pricing.

## Troubleshooting

**Issue: "OPENAI_API_KEY is not set"**
- Make sure you've created a `.env` file in the project root
- Copy `.env.example` to `.env` and add your API key
- Verify the key starts with `sk-`

**Issue: "Python 3.10 or higher is required"**
- Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
- Verify with: `python3 --version`

**Issue: "PDF file not found"**
- Check that the file path is correct
- Use absolute paths if relative paths don't work
- Ensure the file has `.pdf` extension

**Issue: Pages are rotated incorrectly**
- The pipeline auto-detects rotation, but if issues persist, check the PDF metadata

**Issue: Tables are inaccurate**
- The pipeline uses pdfplumber for table extraction. If tables are still wrong, the PDF may have complex layouts that require manual review

**Issue: Token limit exceeded**
- Increase `--max-output-tokens` (default: 16384)
- Very dense pages may need higher limits

**Issue: Rate limiting**
- Reduce `--max-concurrency` (default: 8)
- The pipeline includes automatic retries with exponential backoff

**Issue: Import errors**
- Make sure you've activated your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

## Project Structure

```
gpt51_ocr_pipeline/
├── ocr.py              # Main pipeline script
├── prompts.py          # LLM prompt templates
├── table_extractor.py  # Table extraction module
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your environment variables (create from .env.example)
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Follow existing code style
2. Add docstrings to all functions
3. Test with various PDF types
4. Update README if adding features

## Support

For issues or questions, please open an issue on GitHub.
