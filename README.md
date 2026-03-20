# AI Research Assistant (RAG over arXiv Papers)

A GPU-accelerated Retrieval-Augmented Generation (RAG) system that ingests arXiv papers and answers questions with grounded source citations using a local LLM.

---

## Features

- End-to-end RAG pipeline (PDF → embeddings → FAISS → LLM)
- Local inference using GPU
- Grounded answers with source citations
- FastAPI backend with interactive Swagger UI
- Handles unsupported queries without hallucination

---

## Architecture

arXiv PDFs → Text Extraction → Chunking → Embeddings → FAISS → Retriever → LLM → Answer + Sources

---

## Dataset

This system was built on a diverse collection of 50+ arXiv papers across:

- Machine Learning and LLMs 
- Robotics and Autonomous Systems 
- Computer Vision and 3D Generation 
- Physics and Mathematics 
- Optimization and Systems 

Full list: [`assets/papers.txt`](./assets/papers.txt)

---

## Screenshots

### API Running
![Root](./assets/root.png)

---

### Valid Query (Input)
![Ask Good](./assets/api-ask-good.png)

---

### Valid Query (Answer with Sources)
![Good Response](./assets/api-good-response.png)

---

Valid Response: [`assets/valid-response.json`](./assets/valid-response.json)

---

### Unsupported Query (Input)
![Ask Bad](./assets/api-ask-bad.png)

---

### Unsupported Query (Handled Correctly)
![Bad Response](./assets/api-bad-response.png)

---

Unsupported Response: [`assets/unsupported-response.json`](./assets/unsupported-response.json)

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* FAISS
* FastAPI
* Sentence Transformers

---

## How to Run

```bash
# Clone repository
git clone https://github.com/Deepak-Sathyanarayanan/arxiv-rag.git
cd arxiv-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn src.api:app --reload
```

Open: http://127.0.0.1:8000/docs

---

## Key Highlights

* Built a complete RAG pipeline using local LLMs
* Implemented semantic search using FAISS
* Enabled GPU-based inference for efficient processing
* Designed system to avoid hallucinations using retrieval constraints
* Evaluated on a diverse multi-domain research dataset
