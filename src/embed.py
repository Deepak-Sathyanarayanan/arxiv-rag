import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
INDEX_FILE = Path("data/index/faiss.index")
META_FILE = Path("data/index/metadata.json")

INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = []
metadata = []

with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])
        metadata.append(item)

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
embeddings = embeddings.astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(INDEX_FILE))
META_FILE.write_text(json.dumps(metadata), encoding="utf-8")

print("Saved FAISS index and metadata.")
