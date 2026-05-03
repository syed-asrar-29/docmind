import re
import sqlite3
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "docmind_memory.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {word for word in words if len(word) > 2}


def check_memory(query: str, threshold: float = 0.5) -> Optional[str]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return None

    conn = _connect()
    cursor = conn.execute("SELECT query, answer FROM memory")

    best_score = 0.0
    best_answer = None

    for row in cursor.fetchall():
        stored_query, stored_answer = row
        stored_tokens = _tokenize(stored_query)
        if not stored_tokens:
            continue

        overlap = query_tokens.intersection(stored_tokens)
        score = len(overlap) / max(len(query_tokens), 1)

        if score > best_score and len(overlap) >= 2:
            best_score = score
            best_answer = stored_answer

    conn.close()

    if best_score >= threshold:
        return best_answer

    return None


def save_memory(query: str, answer: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO memory (query, answer) VALUES (?, ?)", (query, answer)
    )
    conn.commit()
    conn.close()
