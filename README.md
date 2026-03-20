# AI Research Assistant (RAG over arXiv Papers)

This project builds a Retrieval-Augmented Generation (RAG) system over arXiv papers.

## Features

- Downloads papers from arXiv
- Extracts and chunks text
- Creates embeddings using Sentence Transformers
- Stores embeddings in FAISS
- Answers questions using a local LLM
- Returns answers with source citations

## Run

```bash
python src/download.py
python src/extract.py
python src/chunk.py
python src/embed.py
uvicorn src.api:app --reload
