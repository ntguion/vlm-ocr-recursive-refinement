"""
GPT-5.1 OCR Pipeline

A production-ready OCR pipeline using GPT-5.1 Vision API for high-accuracy
document transcription. Features automatic table extraction, rotation detection,
risk verification, and structured JSON output.

Author: Your Team
License: MIT
"""

import os
import re
import sys
import json
import base64
import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import AsyncOpenAI

import prompts
import table_extractor

# ---------- Utility & parsing helpers ----------

def b64_data_url_png(image_bytes: bytes) -> str:
    """
    Convert PNG image bytes to a base64 data URL.
    
    Args:
        image_bytes: Raw PNG image bytes
        
    Returns:
        Data URL string (e.g., "data:image/png;base64,...")
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"

def extract_first_json_object(text: str) -> Optional[dict]:
    """
    Extract the first top-level JSON object from arbitrary text.
    
    Handles cases where the LLM response includes extra text before/after JSON.
    Uses balanced brace matching to find valid JSON boundaries.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    # Quick path: try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: find first balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    # Track braces
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    return None

class JSONRepairError(Exception):
    pass

# ---------- PDF rendering ----------

def render_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[bytes]:
    """
    Render PDF pages to PNG images with automatic rotation and cropping.
    
    Features:
    - Auto-detects vertical text and rotates pages 90° clockwise
    - Crops whitespace to maximize content resolution
    - Renders at specified DPI (default 300)
    
    Args:
        pdf_path: Path to input PDF file
        dpi: Resolution for rendering (default: 300)
        
    Returns:
        List of PNG image bytes, one per page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    images: List[bytes] = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        
        # Detect rotation based on text direction
        rotation = 0
        try:
            # heuristic: check first few text lines for direction
            text_dict = page.get_text("dict")
            vertical_count = 0
            horizontal_count = 0
            content_rect = fitz.Rect(page.rect.width, page.rect.height, 0, 0) # Inverted init
            has_content = False
            
            if "blocks" in text_dict:
                for block in text_dict["blocks"]:
                    if "bbox" in block:
                        # Expand content_rect to include this block
                        has_content = True
                        b = fitz.Rect(block["bbox"])
                        content_rect.x0 = min(content_rect.x0, b.x0)
                        content_rect.y0 = min(content_rect.y0, b.y0)
                        content_rect.x1 = max(content_rect.x1, b.x1)
                        content_rect.y1 = max(content_rect.y1, b.y1)

                    if "lines" not in block: continue
                    for line in block["lines"]:
                        # Check line direction vector
                        d = line.get("dir", (1, 0))
                        if abs(d[0]) > abs(d[1]):
                            horizontal_count += 1
                        else:
                            vertical_count += 1
            
            # If predominantly vertical, rotate 90 degrees clockwise
            if vertical_count > horizontal_count and vertical_count > 5:
                rotation = 90
                print(f"  • Page {page_index + 1}: Detected vertical text ({vertical_count} lines vs {horizontal_count} horiz), applying 90° rotation")
                
            # Crop to content with some padding if we found content
            # Padding: 20 points (~0.28 inch)
            if has_content and content_rect.is_valid and not content_rect.is_empty:
                # Add padding
                pad = 20
                content_rect.x0 = max(0, content_rect.x0 - pad)
                content_rect.y0 = max(0, content_rect.y0 - pad)
                content_rect.x1 = min(page.rect.width, content_rect.x1 + pad)
                content_rect.y1 = min(page.rect.height, content_rect.y1 + pad)
                
                # Only crop if it significantly reduces the area (< 90% of original)
                page_area = page.rect.width * page.rect.height
                crop_area = content_rect.width * content_rect.height
                
                if crop_area < page_area * 0.9:
                    page.set_cropbox(content_rect)
                    print(f"  • Page {page_index + 1}: Auto-cropped to content ({(crop_area/page_area)*100:.1f}% of original size)")

        except Exception as e:
            print(f"  ⚠️ Page {page_index + 1}: Pre-processing error - {e}")
            pass

        # Render at dpi -> scale factor: 72 dpi is 1.0 zoom
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        if rotation:
            mat.prerotate(rotation) # Apply rotation
            
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(pix.tobytes("png"))
    doc.close()
    return images

# ---------- OpenAI call ----------

@dataclass
class PageResult:
    """Result structure for a single processed page."""
    page_number: int
    layout_classification: str
    risk_notes: List[Dict[str, Any]]
    final_markdown: str

@dataclass
class UsageTotals:
    """Tracks API usage statistics across all pages."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0
    repair_calls: int = 0
    verification_calls: int = 0

    def add(self, usage: Any):
        # usage may have fields depending on SDK version
        # Handle both API response usage objects and UsageTotals objects
        if hasattr(usage, "api_calls") and hasattr(usage, "repair_calls"):
            # Merge another UsageTotals object
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.reasoning_tokens += usage.reasoning_tokens
            self.api_calls += usage.api_calls
            self.repair_calls += usage.repair_calls
            self.verification_calls += getattr(usage, "verification_calls", 0)
        else:
            # Handle API response usage object
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            self.reasoning_tokens += int(getattr(usage, "reasoning_tokens", 0) or 0)

    def add_api_call(self, call_type: str = "primary"):
        self.api_calls += 1
        if call_type == "repair":
            self.repair_calls += 1
        elif call_type == "verification":
            self.verification_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

def cost_estimate(usage: UsageTotals, input_price_per_mtok: float, output_price_per_mtok: float) -> float:
    """
    Simple estimate: input tokens billed at input rate, output (visible + reasoning) at output rate.
    """
    in_cost = (usage.input_tokens / 1_000_000.0) * input_price_per_mtok
    out_cost = ((usage.output_tokens + usage.reasoning_tokens) / 1_000_000.0) * output_price_per_mtok
    return in_cost + out_cost

class JsonNotReturnedError(Exception):
    pass

# For rate limits / transient network
class TransientError(Exception):
    pass

# ---------- Model invocation ----------

def build_user_message(page_number: int, table_context: Optional[str] = None) -> List[dict]:
    """
    Build the user message content for the API call.
    
    Args:
        page_number: Page number being processed
        table_context: Optional extracted table data to include as context
        
    Returns:
        List of message content dictionaries for OpenAI API
    """
    text_content = prompts.USER_PROMPT_TEMPLATE.format(page_number=page_number)
    
    if table_context:
        text_content += "\n\n" + table_context
        
    return [
        {"type": "input_text", "text": text_content}
    ]

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=8),
    retry=retry_if_exception_type(TransientError),
)
async def call_model_one_page(
    client: AsyncOpenAI,
    model: str,
    image_data_url: str,
    page_number: int,
    reasoning_effort: str,
    max_output_tokens: int,
    progress_callback: Optional[Callable[[str], None]] = None,
    table_context: Optional[str] = None,
) -> Tuple[PageResult, UsageTotals]:
    """
    Process a single page through GPT-5.1 Vision API.
    
    Handles API calls, JSON parsing, truncation detection, and error recovery.
    Returns PageResult with transcription and usage statistics.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name (e.g., "gpt-5.1")
        image_data_url: Base64-encoded PNG data URL
        page_number: Page number being processed
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        max_output_tokens: Maximum tokens for response
        progress_callback: Optional callback for progress updates
        table_context: Optional extracted table data
        
    Returns:
        Tuple of (PageResult, UsageTotals)
        
    Raises:
        TransientError: For retryable API errors
        JsonNotReturnedError: If JSON parsing fails after repair attempts
    """
    input_list = [
        {
            "role": "user",
            "content": build_user_message(page_number, table_context) + [
                {"type": "input_image", "image_url": image_data_url}
            ],
        }
    ]
    
    if progress_callback:
        progress_callback(f"📤 Page {page_number}: Sending to API...")
    
    try:
        start_time = time.time()
        resp = await client.responses.create(
            model=model,
            reasoning={"effort": reasoning_effort},
            instructions=prompts.SYSTEM_PROMPT,
            input=input_list,
            max_output_tokens=max_output_tokens,
        )
        elapsed = time.time() - start_time
        if progress_callback:
            progress_callback(f"✅ Page {page_number}: API response received ({elapsed:.1f}s)")
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Page {page_number}: API error - {str(e)}")
        raise TransientError(str(e))

    usage = UsageTotals()
    usage.add_api_call(call_type="primary")
    if getattr(resp, "usage", None) is not None:
        usage.add(resp.usage)
    
    # Check for truncation
    output_tokens = int(getattr(resp.usage, "output_tokens", 0) or 0) if getattr(resp, "usage", None) else 0
    is_truncated = output_tokens >= max_output_tokens
    if is_truncated and progress_callback:
        progress_callback(f"⚠️  Page {page_number}: Response may be truncated (hit {max_output_tokens:,} token limit)")

    # Try to parse JSON directly
    text_out = ""
    try:
        # Preferred helper in the SDK
        text_out = resp.output_text  # type: ignore[attr-defined]
    except Exception:
        # Fallback gather
        try:
            # Best effort: accumulate all text parts
            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        parts.append(getattr(c, "text", ""))
            text_out = "\n".join(parts)
        except Exception:
            text_out = ""

    data = extract_first_json_object(text_out) if text_out else None
    
    # If truncated and JSON is incomplete, try to extract partial content
    if is_truncated and text_out:
        if progress_callback:
            progress_callback(f"⚠️  Page {page_number}: Response truncated, attempting to extract partial JSON...")
        # Try to extract final_markdown even if JSON is incomplete
        if not data or not data.get("final_markdown"):
            # Look for final_markdown field in the truncated text
            markdown_match = re.search(r'"final_markdown"\s*:\s*"([^"]*(?:\\.[^"]*)*)', text_out, re.DOTALL)
            if markdown_match:
                # Try to extract as much as possible
                partial_markdown = markdown_match.group(1)
                # Unescape JSON string
                try:
                    partial_markdown = json.loads(f'"{partial_markdown}"')
                except:
                    # If that fails, try basic unescaping
                    partial_markdown = partial_markdown.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                
                if not data:
                    data = {}
                data["final_markdown"] = partial_markdown + "\n\n[TRUNCATED - Response exceeded token limit]"
                data["page_number"] = page_number
                data["layout_classification"] = data.get("layout_classification", "unknown")
                data["risk_notes"] = data.get("risk_notes", [])
                if progress_callback:
                    progress_callback(f"⚠️  Page {page_number}: Extracted partial content ({len(partial_markdown):,} chars)")

    # If we didn't get valid JSON, do a cheap repair pass (no image re-send)
    if not data:
        if progress_callback:
            progress_callback(f"🔧 Page {page_number}: JSON parsing failed, attempting repair...")
        try:
            repair_prompt = (
                "You MUST return a single valid JSON object that matches this schema:\n"
                "{"
                '"page_number": int, '
                '"layout_classification": str, '
                '"risk_notes": list, '
                '"final_markdown": str'
                "}\n"
                "Do not include any text outside the JSON. Repair the following text into valid JSON:\n\n"
                f"{text_out}"
            )
            start_time = time.time()
            resp2 = await client.responses.create(
                model=model,
                reasoning={"effort": "low"},
                instructions="Return valid JSON only. No prose.",
                input=[{"role": "user", "content": [{"type": "input_text", "text": repair_prompt}]}],
                max_output_tokens=1024,
            )
            elapsed = time.time() - start_time
            usage.add_api_call(call_type="repair")
            if getattr(resp2, "usage", None) is not None:
                usage.add(resp2.usage)
            if progress_callback:
                progress_callback(f"✅ Page {page_number}: Repair call completed ({elapsed:.1f}s)")
            repaired = ""
            try:
                repaired = resp2.output_text  # type: ignore[attr-defined]
            except Exception:
                pass
            data = extract_first_json_object(repaired) if repaired else None
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Page {page_number}: Repair failed - {str(e)}")
            raise JsonNotReturnedError(f"Failed to parse and repair JSON for page {page_number}: {e}")

    if not data:
        raise JsonNotReturnedError(f"Model did not return valid JSON for page {page_number}.")

    # Validate minimal keys
    required = ["page_number", "layout_classification", "risk_notes", "final_markdown"]
    for k in required:
        if k not in data:
            raise JsonNotReturnedError(f"Missing key `{k}` in page {page_number} result.")

    # Build PageResult - use actual page_number to prevent LLM errors
    result = PageResult(
        page_number=page_number,  # Always use the actual page number being processed
        layout_classification=str(data["layout_classification"]),
        risk_notes=list(data.get("risk_notes", [])),
        final_markdown=str(data["final_markdown"]),
    )
    
    if progress_callback:
        risk_count = len(result.risk_notes)
        risk_str = f" ({risk_count} risk note{'s' if risk_count != 1 else ''})" if risk_count > 0 else ""
        progress_callback(f"✓ Page {page_number}: Complete - {result.layout_classification}{risk_str}")
    
    return result, usage

async def verify_and_fix_page(
    client: AsyncOpenAI,
    model: str,
    image_data_url: str,
    page_result: PageResult,
    reasoning_effort: str,
    max_output_tokens: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[PageResult, UsageTotals]:
    """
    Verify and fix specific risks identified in the initial OCR pass.
    
    Performs a focused second pass on pages with risk notes, comparing
    the image with the transcription to correct errors in identified areas.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name
        image_data_url: Base64-encoded PNG data URL
        page_result: Initial OCR result with risk notes
        reasoning_effort: Reasoning effort level
        max_output_tokens: Maximum tokens for response
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (updated PageResult, UsageTotals)
        
    Note:
        Returns original result unchanged if no risks or only system errors
    """
    if not page_result.risk_notes:
        return page_result, UsageTotals()

    # Filter for meaningful risks (ignore generic system errors)
    risks_to_check = [r for r in page_result.risk_notes if r.get("type") != "system_error"]
    
    if not risks_to_check:
        return page_result, UsageTotals()

    if progress_callback:
        progress_callback(f"🔍 Page {page_result.page_number}: Verifying {len(risks_to_check)} risk(s)...")

    risk_text = "\n".join([f"- [{r.get('type', 'unknown')}] {r.get('location', 'unknown')}: {r.get('description', '')}" for r in risks_to_check])
    
    prompt_text = prompts.VERIFICATION_PROMPT_TEMPLATE.format(
        page_number=page_result.page_number,
        current_markdown=page_result.final_markdown,
        risk_notes_text=risk_text
    )

    input_list = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": image_data_url}
            ],
        }
    ]

    usage = UsageTotals()
    
    try:
        resp = await client.responses.create(
            model=model,
            reasoning={"effort": "medium"}, # Use medium effort for verification
            instructions="You are a helpful verification assistant.",
            input=input_list,
            max_output_tokens=max_output_tokens,
        )
        
        usage.add_api_call(call_type="verification")
        if getattr(resp, "usage", None) is not None:
            usage.add(resp.usage)

        text_out = ""
        try:
            text_out = resp.output_text # type: ignore
        except Exception:
            try:
                parts = []
                for item in getattr(resp, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                         if getattr(c, "type", "") in ("output_text", "text"):
                             parts.append(getattr(c, "text", ""))
                text_out = "\n".join(parts)
            except:
                text_out = ""
        
        data = extract_first_json_object(text_out)
        if data and "final_markdown" in data:
            new_markdown = str(data["final_markdown"])
            changes = data.get("changes_made", "None")
            status = data.get("verification_status", "verified")
            
            if progress_callback:
                progress_callback(f"✅ Page {page_result.page_number}: Verified ({status}). Changes: {changes}")
            
            page_result.final_markdown = new_markdown
            
    except Exception as e:
        if progress_callback:
            progress_callback(f"⚠️ Page {page_result.page_number}: Verification failed - {e}")
            
    return page_result, usage

# ---------- Orchestrator ----------

async def process_pdf(
    pdf_path: str,
    output_md_path: str,
    output_json_path: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    max_concurrency: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
):
    print("=" * 70)
    print("🚀 GPT-5.1 OCR Pipeline - Starting Processing")
    print("=" * 70)
    print(f"📄 PDF: {pdf_path}")
    print(f"📝 Output: {output_md_path}")
    print(f"🤖 Model: {model}")
    print(f"⚙️  Reasoning Effort: {reasoning_effort}")
    print(f"🔢 Max Concurrency: {max_concurrency}")
    print(f"📊 Max Output Tokens: {max_output_tokens:,}")
    print("-" * 70)
    
    print("\n📖 Step 1/4: Rendering PDF pages to images...")
    start_time = time.time()
    images = render_pdf_to_images(pdf_path, dpi=300)
    total_pages = len(images)
    render_time = time.time() - start_time
    print(f"✅ Rendered {total_pages} page(s) in {render_time:.2f}s (300 DPI)")

    # Initialize OpenAI client (API key already validated in main())
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    client = AsyncOpenAI(api_key=api_key)

    semaphore = asyncio.Semaphore(max_concurrency)
    usage_totals = UsageTotals()
    page_status = {}  # Track status per page
    status_lock = asyncio.Lock()

    results: List[Optional[PageResult]] = [None] * total_pages
    errors: List[Optional[str]] = [None] * total_pages

    def progress_callback(message: str):
        """Thread-safe progress callback"""
        print(f"  {message}")

    async def worker(page_idx: int):
        page_number = page_idx + 1
        data_url = b64_data_url_png(images[page_idx])
        
        # Extract table context if available
        table_context = None
        try:
            table_context = table_extractor.extract_table_context(pdf_path, page_number)
            if table_context and progress_callback:
                progress_callback(f"📋 Page {page_number}: Extracted table context for improved accuracy")
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️  Page {page_number}: Table extraction warning - {e}")

        async with semaphore:
            try:
                page_result, usage = await call_model_one_page(
                    client,
                    model=model,
                    image_data_url=data_url,
                    page_number=page_number,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    progress_callback=progress_callback,
                    table_context=table_context,
                )
                
                # Verification Step
                if page_result.risk_notes:
                    page_result, verify_usage = await verify_and_fix_page(
                        client,
                        model=model,
                        image_data_url=data_url,
                        page_result=page_result,
                        reasoning_effort="medium", # Use medium effort for verification to save costs/time unless high is needed? Let's default to medium or same.
                                                   # Actually, let's stick to high if the main pass was high, or maybe medium is enough. 
                                                   # The prompt says "expert-level", let's keep it robust. 
                                                   # Let's use the passed 'reasoning_effort' but maybe cap it or just use it.
                        # Let's use "medium" for verification to balance cost/speed since we have specific targets.
                        max_output_tokens=max_output_tokens,
                        progress_callback=progress_callback,
                    )
                    usage.add(verify_usage)

                results[page_idx] = page_result
                async with status_lock:
                    usage_totals.add(usage)
                    page_status[page_number] = "success"
            except Exception as e:
                async with status_lock:
                    errors[page_idx] = str(e)
                    page_status[page_number] = f"error: {str(e)}"
                # Even on error, write a minimal placeholder so the document is complete
                placeholder = PageResult(
                    page_number=page_number,
                    layout_classification="error",
                    risk_notes=[{"type": "system_error", "location": "n/a", "description": str(e)}],
                    final_markdown=f"### Page {page_number}\n\n**Error**: {e}\n\n(Original page included in the source PDF.)",
                )
                results[page_idx] = placeholder
                progress_callback(f"❌ Page {page_number}: Failed - {str(e)}")

    print(f"\n📤 Step 2/4: Processing {total_pages} page(s) with API calls...")
    print(f"   (Processing up to {max_concurrency} pages concurrently)\n")
    
    # Enhanced progress bar with more details
    tasks = [asyncio.create_task(worker(i)) for i in range(total_pages)]
    with tqdm(total=total_pages, desc="Pages", unit="page", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for f in asyncio.as_completed(tasks):
            await f
            pbar.update(1)
            # Update progress bar description with current stats
            async with status_lock:
                completed = sum(1 for status in page_status.values() if status == "success")
                errors_count = sum(1 for status in page_status.values() if "error" in status)
                pbar.set_postfix({
                    'done': completed,
                    'err': errors_count,
                    'calls': usage_totals.api_calls,
                    'verif': usage_totals.verification_calls
                })

    # --- Output Generation ---
    
    # 1. Write JSON Output (Structured Data)
    print(f"\n💾 Step 3a/4: Writing JSON output to {output_json_path}...")
    json_output = []
    for i, res in enumerate(results):
        if res:
            json_output.append({
                "page_number": res.page_number,
                "layout_classification": res.layout_classification,
                "risk_notes": res.risk_notes,
                "final_markdown": res.final_markdown
            })
        else:
            json_output.append(None)
            
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"pages": json_output, "metadata": {"source_pdf": pdf_path, "model": model}}, f, indent=2)
    print(f"✅ JSON written successfully")

    # 2. Write Markdown Output (Readable Document)
    print(f"\n💾 Step 3b/4: Writing Markdown output to {output_md_path}...")
    with open(output_md_path, "w", encoding="utf-8") as f:
        for i, res in enumerate(results):
            assert res is not None
            # Ensure numeric page header and separator
            f.write(f"### Page {res.page_number}\n\n")
            f.write(res.final_markdown.strip())
            f.write("\n\n-----\n")
    print(f"✅ Markdown written successfully")

    # Summarize tokens & cost
    total_time = time.time() - start_time
    print(f"\n📊 Step 4/4: Usage Summary")
    print("=" * 70)
    print(f"📄 Pages processed:     {total_pages}")
    print(f"⏱️  Total time:          {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"⚡ Avg time per page:    {total_time/total_pages:.2f}s")
    print()
    print("🔌 API Calls:")
    print(f"   • Total API calls:    {usage_totals.api_calls}")
    print(f"   • Primary calls:      {usage_totals.api_calls - usage_totals.repair_calls - usage_totals.verification_calls}")
    print(f"   • Repair calls:       {usage_totals.repair_calls}")
    print(f"   • Verification calls: {usage_totals.verification_calls}")
    print(f"   • Calls per page:     {usage_totals.api_calls / total_pages:.2f}")
    print()
    print("💬 Token Usage:")
    print(f"   • Input tokens:        {usage_totals.input_tokens:,}")
    print(f"   • Output tokens:       {usage_totals.output_tokens:,}")
    if usage_totals.reasoning_tokens:
        print(f"   • Reasoning tokens:    {usage_totals.reasoning_tokens:,}")
    print(f"   • Total tokens:        {usage_totals.total_tokens:,}")
    print(f"   • Avg tokens/page:     {usage_totals.total_tokens // total_pages:,}")
    print()
    est_cost = cost_estimate(usage_totals, input_price_per_mtok, output_price_per_mtok)
    print(f"💰 Estimated cost:       ${est_cost:,.4f}")
    print(f"   (Input: ${input_price_per_mtok}/M, Output: ${output_price_per_mtok}/M)")
    print("=" * 70)

    # Report any pages with errors
    any_errors = [f"p.{i+1}: {err}" for i, err in enumerate(errors) if err]
    if any_errors:
        print("\n⚠️  Errors encountered on pages:")
        for line in any_errors:
            print(f"   • {line}")
    else:
        print("\n✅ All pages processed successfully!")
    
    print()

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    p = argparse.ArgumentParser(
        description="GPT-5.1 OCR Pipeline: High-accuracy document transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr.py --pdf document.pdf
  python ocr.py --pdf doc.pdf --out result.md --json result.json --effort high
        """
    )
    p.add_argument("--pdf", required=True, help="Path to input PDF file")
    p.add_argument("--out", default="output.md", help="Path to Markdown output file (default: output.md)")
    p.add_argument("--json", default="output.json", help="Path to JSON output file (default: output.json)")
    p.add_argument("--model", default=os.getenv("MODEL", "gpt-5.1"), help="OpenAI model name (default: gpt-5.1)")
    p.add_argument("--effort", default=os.getenv("REASONING_EFFORT", "high"), 
                   choices=["low", "medium", "high"], 
                   help="Reasoning effort level (default: high)")
    p.add_argument("--max-output-tokens", type=int, 
                   default=int(os.getenv("MAX_OUTPUT_TOKENS", "16384")), 
                   help="Maximum output tokens per page (default: 16384)")
    p.add_argument("--max-concurrency", type=int, 
                   default=int(os.getenv("MAX_CONCURRENCY", "8")), 
                   help="Maximum concurrent page processing (default: 8)")
    return p.parse_args()

def main():
    """
    Main entry point for the OCR pipeline.
    
    Loads environment variables, parses arguments, and runs the PDF processing pipeline.
    """
    # Check Python version
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required.", file=sys.stderr)
        print(f"Current version: {sys.version}", file=sys.stderr)
        sys.exit(1)
    
    load_dotenv()
    args = parse_args()

    # Validate PDF file exists
    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    
    if not args.pdf.lower().endswith('.pdf'):
        print(f"WARNING: File does not have .pdf extension: {args.pdf}", file=sys.stderr)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in .env file", file=sys.stderr)
        print("Please create a .env file with your OpenAI API key. See .env.example for reference.", file=sys.stderr)
        sys.exit(1)
    
    if not api_key.startswith("sk-"):
        print("WARNING: API key format may be incorrect (should start with 'sk-')", file=sys.stderr)

    input_price = float(os.getenv("INPUT_PRICE_PER_MTOK", "1.25"))
    output_price = float(os.getenv("OUTPUT_PRICE_PER_MTOK", "10.0"))

    # Ensure output directories exist
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    json_dir = os.path.dirname(os.path.abspath(args.json)) or "."
    os.makedirs(json_dir, exist_ok=True)

    asyncio.run(process_pdf(
        pdf_path=args.pdf,
        output_md_path=args.out,
        output_json_path=args.json,
        model=args.model,
        reasoning_effort=args.effort,
        max_output_tokens=args.max_output_tokens,
        max_concurrency=args.max_concurrency,
        input_price_per_mtok=input_price,
        output_price_per_mtok=output_price,
    ))

if __name__ == "__main__":
    main()
