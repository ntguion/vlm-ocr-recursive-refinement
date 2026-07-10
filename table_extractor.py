"""
Table Extraction Module

Extracts tables from PDF pages using pdfplumber for high-accuracy table data.
Used as "source of truth" context to improve LLM OCR accuracy for tables.
"""

import pdfplumber
from typing import Optional

def extract_table_context(pdf_path: str, page_number: int) -> Optional[str]:
    """
    Extracts tables from a specific PDF page (1-based index) and returns
    them as a formatted Markdown string to be used as context for the LLM.
    Returns None if no tables are found.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # pdfplumber is 0-indexed
            if page_number < 1 or page_number > len(pdf.pages):
                return None
                
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            
            # Fallback: If no tables found, try text-based strategy (helps with vertical/sparse tables)
            if not tables:
                tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
            
            if not tables:
                return None
            
            context_str = "### DETECTED TABLE CONTENT (Source of Truth)\n"
            context_str += "Use the following extracted table data to verify text, numbers, and structure.\n"
            context_str += "Note: The extraction might be imperfect, but the numbers and text are generally reliable.\n\n"
            
            for i, table in enumerate(tables):
                # Clean table data (replace None with empty string)
                clean_table = [[str(cell).replace('\n', ' ') if cell is not None else "" for cell in row] for row in table]
                
                if not clean_table:
                    continue
                
                context_str += f"#### Extracted Table {i+1}\n"
                
                # Construct Markdown table
                if len(clean_table) > 0:
                    headers = clean_table[0]
                    # Header row
                    context_str += "| " + " | ".join(headers) + " |\n"
                    # Separator row
                    context_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    # Data rows
                    for row in clean_table[1:]:
                        context_str += "| " + " | ".join(row) + " |\n"
                
                context_str += "\n"
                
            return context_str
            
    except Exception as e:
        print(f"Warning: Table extraction failed for page {page_number}: {e}")
        return None
