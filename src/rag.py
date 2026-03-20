import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


INDEX_FILE = Path("data/index/faiss.index")
META_FILE = Path("data/index/metadata.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def load_embedder() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


def load_llm():
    print(f"Loading LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    # Force deterministic greedy decoding
    model.generation_config.do_sample = False

    # Remove sampling-only settings that cause warnings during greedy decoding
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return tokenizer, model


def load_index_and_metadata() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_FILE}")

    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, metadata


embed_model = load_embedder()
tokenizer, llm = load_llm()
print("Generation config:")
print("do_sample:", llm.generation_config.do_sample)
print("temperature:", llm.generation_config.temperature)
print("top_p:", llm.generation_config.top_p)
print("top_k:", llm.generation_config.top_k)

index, metadata = load_index_and_metadata()


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        item["rank"] = rank + 1
        item["distance"] = float(distances[0][rank])
        results.append(item)
    return results


def build_context(contexts: List[Dict[str, Any]]) -> str:
    parts = []
    for c in contexts:
        parts.append(f"[Source: {c['doc_id']} | Chunk {c['chunk_id']}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    contexts = retrieve(query, k=k)
    if not contexts:
        return "I could not find any relevant context.", []

    context_text = build_context(contexts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict retrieval-based research assistant. "
                "Answer only from the provided context. "
                "Do not use outside knowledge. "
                "Do not infer beyond what is explicitly supported. "
                "If the context does not directly support an answer, say: "
                "'I do not know based on the provided context.' "
                "Every factual claim must be followed by one or more exact source labels "
                "from the context in this format: "
                "[Source: DOC_ID | Chunk CHUNK_ID]. "
                "Do not use numbered references like [1] or [22]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Context:\n{context_text}\n\n"
                "Instructions:\n"
                "1. Answer only with facts directly stated in the context.\n"
                "2. Do not paraphrase unsupported assumptions from the question.\n"
                "3. If support is incomplete, explicitly say what is not supported.\n"
                "4. Keep the answer concise.\n"
                "5. Cite every sentence with exact source labels.\n"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return result, contexts


def print_sources(contexts: List[Dict[str, Any]], full_text: bool = False, preview_chars: int = 700) -> None:
    print("\nRetrieved sources:")
    for c in contexts:
        text = c["text"] if full_text else c["text"][:preview_chars].replace("\n", " ").strip() + "..."
        print("\n" + "=" * 100)
        print(f"Rank     : {c['rank']}")
        print(f"Doc ID   : {c['doc_id']}")
        print(f"Chunk ID : {c['chunk_id']}")
        print(f"Distance : {c['distance']:.4f}")
        print(f"Label    : [Source: {c['doc_id']} | Chunk {c['chunk_id']}]")
        print(f"Text     : {text}")


def main():
    print("\nRAG system ready.")
    print("Type a question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        answer, contexts = answer_question(query, k=5)

        print_sources(contexts, full_text=False)
        print("\n" + "=" * 100)
        print("Answer:\n")
        print(answer)
        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
