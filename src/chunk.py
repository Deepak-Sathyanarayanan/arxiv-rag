import json
from pathlib import Path

TXT_DIR = Path("data/text")
OUT_FILE = Path("data/chunks.jsonl")

def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

with OUT_FILE.open("w", encoding="utf-8") as out:
    for txt_file in TXT_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            record = {
                "doc_id": txt_file.stem,
                "chunk_id": i,
                "text": chunk
            }
            out.write(json.dumps(record) + "\n")

print(f"Saved chunks to {OUT_FILE}")
