import os
from typing import Dict, List

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"


def synthesize_answer(question: str, sub_answers: List[Dict]) -> Dict:
    sources: List[str] = []
    answer_blocks: List[str] = []

    for index, item in enumerate(sub_answers, start=1):
        sources.extend(item.get("sources", []))
        answer_blocks.append(
            f"Sub-answer {index}:\n{item.get('answer', '').strip()}"
        )

    dedup_sources = sorted({src for src in sources if src})
    combined = "\n\n".join(answer_blocks)

    prompt = (
        "You are a synthesis agent. Combine the sub-answers into a single, "
        "coherent response for the user. Avoid repeating yourself.\n\n"
        f"Original question: {question}\n\n"
        f"Sub-answers:\n{combined}\n\n"
        "Final answer:"
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    final_answer = response.choices[0].message.content

    return {"final_answer": final_answer, "all_sources": dedup_sources}
