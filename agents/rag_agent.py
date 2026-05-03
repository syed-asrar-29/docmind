import os
from typing import Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq

from pathlib import Path

MODEL_NAME = "llama-3.1-8b-instant"
ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "docmind"


def _get_collection():
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_func
    )


def answer_subquestion(subquestion: str, top_k: int = 4) -> Dict:
    collection = _get_collection()
    results = collection.query(
        query_texts=[subquestion],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    sources: List[str] = []
    context_chunks: List[str] = []

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
        if source not in sources:
            sources.append(source)
        context_chunks.append(f"Source: {source}\n{doc}")

    context = "\n\n".join(context_chunks)

    prompt = (
        "You are a RAG agent. Use the provided context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {subquestion}\n"
        "Answer:"
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content

    return {"answer": answer, "sources": sources}
