import fitz
import re
from pathlib import Path

PDF_DIR = Path("data/pdfs")
TXT_DIR = Path("data/text")
TXT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_HEADERS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
]

def cut_references(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip().lower()
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in REFERENCE_HEADERS):
            break
        kept.append(line)
    return "\n".join(kept).strip()

for pdf_file in PDF_DIR.glob("*.pdf"):
    txt_file = TXT_DIR / (pdf_file.stem + ".txt")

    text_parts = []
    try:
        doc = fitz.open(pdf_file)
        for page in doc:
            text_parts.append(page.get_text())
        raw_text = "\n".join(text_parts)
        cleaned_text = cut_references(raw_text)
        txt_file.write_text(cleaned_text, encoding="utf-8")
        print(f"Extracted: {pdf_file.name}")
    except Exception as e:
        print(f"Failed on {pdf_file.name}: {e}")
