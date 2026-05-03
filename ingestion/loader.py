import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "docmind"


def _read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_csv(file_path: Path) -> str:
    df = pd.read_csv(file_path)
    return df.to_csv(index=False)


def _read_json(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return json.dumps(data, indent=2, ensure_ascii=True)


def _read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    documents: list[dict] = []
    if not data_dir.exists():
        return documents

    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                text = _read_pdf(file_path)
            elif ext == ".csv":
                text = _read_csv(file_path)
            elif ext == ".json":
                text = _read_json(file_path)
            elif ext in {".txt", ".md"}:
                text = _read_txt(file_path)
            else:
                continue
        except Exception:
            continue

        if text.strip():
            documents.append({"text": text, "source": file_path.name})

    return documents


def split_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[dict] = []

    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for index, chunk in enumerate(pieces):
            chunks.append(
                {
                    "text": chunk,
                    "source": doc["source"],
                    "chunk": index,
                }
            )

    return chunks


def index_documents(data_dir: Path = DATA_DIR) -> int:
    documents = load_documents(data_dir)
    if not documents:
        return 0

    chunks = split_documents(documents)
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        raw_id = f"{chunk['source']}-{chunk['chunk']}-{chunk['text']}"
        chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
        ids.append(chunk_id)
        texts.append(chunk["text"])
        metadatas.append({"source": chunk["source"], "chunk": chunk["chunk"]})

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    return len(chunks)


def main() -> None:
    count = index_documents(DATA_DIR)
    print(f"Indexed {count} chunks into ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
