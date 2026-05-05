
"""
Creates fine-tuning data from your Indian Kanoon corpus.
Run this locally first, then upload finetune_data.jsonl to Colab.
"""
import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw/raw_judgements.jsonl")
OUTPUT_PATH = Path("data/finetune_data.jsonl")

QUESTION_TEMPLATES = [
    "What are the legal rights of a person in this situation: {topic}?",
    "Explain in plain language what this court judgement means for ordinary citizens.",
    "What should a first-generation litigant know about {topic}?",
    "Summarise this legal judgement in simple language that a non-lawyer can understand.",
]

def create_qa_pairs(doc: dict) -> list[dict]:
    """Generate instruction-answer pairs from a document."""
    text = doc["text"]
    topic = doc.get("query_topic", "their legal situation")

    if len(text) < 300:
        return []

    pairs = []

    # Type 1: Summarisation
    pairs.append({
        "instruction": "Explain this court judgement in plain language for someone who has never been to court.",
        "input": text[:800],
        "output": f"This case from {doc.get('court', 'an Indian court')} is about {topic}. " +
                  "Based on this judgement: " + text[:300] + "...",
    })

    # Type 2: Rights explanation
    if topic:
        pairs.append({
            "instruction": f"What legal rights does a person have regarding {topic} in India?",
            "input": text[:600],
            "output": f"According to this court judgement, regarding {topic}: " + text[:400] + "...",
        })

    return pairs

docs = []
with open(RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

all_pairs = []
for doc in docs:
    pairs = create_qa_pairs(doc)
    all_pairs.extend(pairs)

# Shuffle and save
import random
random.shuffle(all_pairs)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"Created {len(all_pairs)} fine-tuning examples → {OUTPUT_PATH}")
print("Upload this to Colab and run finetune_qlora.py")
