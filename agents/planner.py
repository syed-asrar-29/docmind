import json
import os
from typing import List

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"


def _parse_subquestions(raw: str) -> List[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            items = [str(item).strip() for item in data if str(item).strip()]
            return items
    except json.JSONDecodeError:
        pass

    lines = [line.strip("- ").strip() for line in raw.splitlines()]
    return [line for line in lines if line]


def plan_subquestions(question: str, max_subquestions: int = 3) -> List[str]:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = (
        "You are a planning agent. Break the user question into 2-3 concise, "
        "focused sub-questions that can be answered from documents. "
        "Return only a JSON array of strings.\n\n"
        f"User question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content
    items = _parse_subquestions(raw)

    if not items:
        return [question]

    return items[:max_subquestions]
