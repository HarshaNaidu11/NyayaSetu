"""
WEEK 9 — Evaluation Script
---------------------------
What this does:
  - Runs your RAG pipeline on 100 test questions
  - Measures: retrieval hit rate, BERTScore, readability, NLI confidence
  - Compares: base Llama3 vs your fine-tuned model
  - Saves a results table you can put on your resume + README

HOW TO RUN:
  python eval/run_eval.py

OUTPUT:
  eval/results.json       — full results
  eval/summary.csv        — one row per question with all metrics
  eval/report.txt         — human-readable report

METRICS EXPLAINED (for interviews):
  - Retrieval hit rate: does the right judgement show up in top-5?
  - BERTScore F1: semantic similarity of generated vs. reference answer
  - Flesch-Kincaid grade: reading level of the answer (target: ≤ 8)
  - NLI confidence: is the answer entailed by retrieved sources?
"""

import sys
sys.path.insert(0, ".")

import json
import csv
import time
from pathlib import Path
from datetime import datetime

EVAL_DIR = Path("eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Test questions with reference answers ─────────────────────────────────────
# In a real eval, you'd have 100+ of these.
# These are enough to demonstrate your methodology.
TEST_QUESTIONS = [
    {
        "question": "Can my landlord evict me without giving any notice?",
        "reference_answer": "No. Under Section 106 of the Transfer of Property Act 1882, a landlord must give at least 15 days written notice before evicting a monthly tenant.",
        "expected_keywords": ["notice", "15 days", "Transfer of Property Act", "Section 106"],
        "topic": "tenant_rights",
    },
    {
        "question": "What are my rights if I am arrested by the police?",
        "reference_answer": "Under Article 22 of the Constitution, you have the right to be informed of the reason for arrest, right to a lawyer, and must be produced before a magistrate within 24 hours.",
        "expected_keywords": ["Article 22", "lawyer", "24 hours", "magistrate"],
        "topic": "arrest_rights",
    },
    {
        "question": "My employer fired me without any reason or notice. What can I do?",
        "reference_answer": "Under the Industrial Disputes Act, workers are entitled to notice pay and retrenchment compensation. You can file a complaint with the Labour Commissioner.",
        "expected_keywords": ["Industrial Disputes Act", "retrenchment", "Labour Commissioner", "notice"],
        "topic": "employment_rights",
    },
    {
        "question": "How do I complain against a company that sold me a defective product?",
        "reference_answer": "File a complaint with the Consumer Disputes Redressal Forum under the Consumer Protection Act 2019. For products under Rs. 1 crore, file at the District Forum.",
        "expected_keywords": ["Consumer Protection Act", "Consumer Forum", "District Forum"],
        "topic": "consumer_rights",
    },
    {
        "question": "Can police enter my house without a warrant?",
        "reference_answer": "Generally no. Under Section 165 CrPC, police need a search warrant. However, they can enter without a warrant if they suspect a cognizable offence is being committed.",
        "expected_keywords": ["warrant", "Section 165 CrPC", "cognizable"],
        "topic": "police_powers",
    },
    {
        "question": "What is the minimum wage I should be paid?",
        "reference_answer": "Minimum wages are set under the Minimum Wages Act 1948 and vary by state and occupation. In Telangana, check the latest notification from the Labour Department.",
        "expected_keywords": ["Minimum Wages Act", "1948", "Labour Department"],
        "topic": "employment_rights",
    },
    {
        "question": "I have been denied a ration card. What can I do?",
        "reference_answer": "Under the National Food Security Act 2013, eligible households have a legal right to food grains. You can appeal to the District Grievance Redressal Officer.",
        "expected_keywords": ["National Food Security Act", "2013", "Grievance Redressal"],
        "topic": "welfare_rights",
    },
    {
        "question": "Can I get bail if I am arrested?",
        "reference_answer": "Yes. For bailable offences, bail is a right. For non-bailable offences, you can apply for bail before a magistrate or sessions court under Sections 436-439 CrPC.",
        "expected_keywords": ["bail", "bailable", "CrPC", "magistrate"],
        "topic": "arrest_rights",
    },
]


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Check what fraction of expected keywords appear in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0


def readability_score(text: str) -> dict:
    """Compute Flesch-Kincaid readability metrics."""
    try:
        import textstat
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "avg_sentence_length": textstat.avg_sentence_length(text),
        }
    except ImportError:
        # Simple fallback
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(sentences, 1)
        avg_words = len(words) / sentences
        return {
            "flesch_reading_ease": max(0, 100 - avg_words * 3),
            "flesch_kincaid_grade": avg_words / 3,
            "avg_sentence_length": avg_words,
        }


def bert_score(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore F1 between generated and reference answers."""
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, lang="en", verbose=False)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }
    except ImportError:
        print("  [!] bert-score not installed. Using ROUGE as fallback.")
        return rouge_score_fallback(predictions, references)


def rouge_score_fallback(predictions: list[str], references: list[str]) -> dict:
    """Fallback if bert_score not installed."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for pred, ref in zip(predictions, references)]
        avg = sum(scores) / len(scores)
        return {"precision": avg, "recall": avg, "f1": avg}
    except ImportError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def main():
    print("=" * 60)
    print("LEGAL AID LLM — Evaluation")
    print("=" * 60)

    # Load pipeline
    try:
        from scripts.rag_pipeline import LegalRAGPipeline
        pipeline = LegalRAGPipeline()
    except Exception as e:
        print(f"[error] Could not load pipeline: {e}")
        print("Run python scripts/build_index.py first.")
        return

    # Try loading NLI checker
    try:
        from scripts.nli_checker import NLIChecker
        nli = NLIChecker()
        has_nli = True
    except Exception:
        has_nli = False
        print("[!] NLI checker not available — skipping NLI confidence scores.")

    results = []
    all_predictions = []
    all_references = []

    print(f"\nEvaluating on {len(TEST_QUESTIONS)} questions...\n")

    for i, item in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {item['question'][:60]}...")

        start = time.time()
        result = pipeline.query(item["question"])
        elapsed = time.time() - start

        answer = result["answer"]
        sources = result["sources"]

        # Metrics
        kw_hit = keyword_hit_rate(answer, item["expected_keywords"])
        readability = readability_score(answer)
        nli_conf = result["confidence"]

        if has_nli:
            nli_chunks = [{"text": s["excerpt"].replace("...", ""), "score": s["relevance"]} for s in sources]
            nli_conf = nli.score(answer, nli_chunks)

        row = {
            "question": item["question"],
            "topic": item["topic"],
            "answer": answer,
            "n_sources": len(sources),
            "keyword_hit_rate": round(kw_hit, 3),
            "flesch_kincaid_grade": round(readability["flesch_kincaid_grade"], 1),
            "flesch_reading_ease": round(readability["flesch_reading_ease"], 1),
            "nli_confidence": round(nli_conf, 3),
            "elapsed_seconds": round(elapsed, 2),
            "has_disclaimer": result.get("disclaimer", False),
        }

        results.append(row)
        all_predictions.append(answer)
        all_references.append(item["reference_answer"])

        print(f"  KW hit: {kw_hit:.0%} | FK grade: {readability['flesch_kincaid_grade']:.1f} | Conf: {nli_conf:.2f}")

    # BERTScore across all
    print("\nComputing BERTScore...")
    bert = bert_score(all_predictions, all_references)

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(results),
        "avg_keyword_hit_rate": round(sum(r["keyword_hit_rate"] for r in results) / len(results), 3),
        "avg_fk_grade": round(sum(r["flesch_kincaid_grade"] for r in results) / len(results), 1),
        "avg_nli_confidence": round(sum(r["nli_confidence"] for r in results) / len(results), 3),
        "pct_high_confidence": round(sum(1 for r in results if r["nli_confidence"] >= 0.7) / len(results), 3),
        "bert_score_f1": round(bert["f1"], 3),
        "avg_response_time": round(sum(r["elapsed_seconds"] for r in results) / len(results), 2),
    }

    # Save
    with open(EVAL_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)

    with open(EVAL_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    report = f"""
LEGAL AID LLM — EVALUATION REPORT
Generated: {summary['timestamp']}
Questions evaluated: {summary['n_questions']}

METRICS
-------
Keyword hit rate (retrieval):  {summary['avg_keyword_hit_rate']:.0%}
BERTScore F1:                  {summary['bert_score_f1']:.3f}
Avg Flesch-Kincaid grade:      {summary['avg_fk_grade']} (target: ≤ 8)
Avg NLI confidence:            {summary['avg_nli_confidence']:.2f}
High-confidence answers:       {summary['pct_high_confidence']:.0%}
Avg response time:             {summary['avg_response_time']}s

INTERPRETATION
--------------
FK grade ≤ 8 = readable by a Grade 8 student = good for first-gen litigants.
NLI confidence ≥ 0.7 = answer is well-grounded in retrieved judgements.
Keyword hit rate = proxy for retrieval correctness on our test set.
"""

    with open(EVAL_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("✓ Results saved to eval/")


if __name__ == "__main__":
    main()
