from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "arXiv RAG API is running"}

@app.post("/ask")
def ask(query: Query):
    answer, contexts = answer_question(query.question)
    return {
        "question": query.question,
        "answer": answer,
        "sources": [
            {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "rank": c["rank"],
                "distance": c["distance"],
            }
            for c in contexts
        ],
    }
