"""
WEEK 8 — Hallucination Detection via NLI
-----------------------------------------
What this does:
  - Takes a generated answer and the retrieved source chunks
  - Uses a DeBERTa NLI model to check if the answer is ENTAILED by sources
  - Returns a confidence score (0.0 to 1.0)
  - Flags answers that are not grounded in the retrieved text

HOW IT WORKS (this is your CS depth talking point):
  NLI = Natural Language Inference. The model predicts one of:
    ENTAILMENT   — the answer is supported by the source
    NEUTRAL      — answer is not contradicted but not supported
    CONTRADICTION — answer directly contradicts the source

  We treat ENTAILMENT probability as our confidence score.
  If confidence < 0.6, we show a disclaimer instead of a confident answer.

  This is different from simple RAG — most RAG systems just retrieve and generate.
  Adding an NLI verification gate makes it a trustworthy system, not just a demo.

HOW TO INTEGRATE:
  In rag_pipeline.py, replace the mock check_confidence() with:
    from scripts.nli_checker import NLIChecker
    checker = NLIChecker()
    confidence = checker.score(answer, chunks)
"""

import numpy as np
import re


class NLIChecker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.ENTAILMENT_IDX = 0
        self.NEUTRAL_IDX = 1
        self.CONTRADICTION_IDX = 2

    def _lazy_load(self):
        if self._loaded:
            return

        print("Loading DeBERTa NLI model...")
        print("(Downloads ~700MB on first run)\n")

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "cross-encoder/nli-deberta-v3-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        print(f"✓ NLI model loaded on {self.device}")

    def _predict_single(self, premise: str, hypothesis: str) -> dict:
        import torch

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        return {
            "entailment": float(probs[self.ENTAILMENT_IDX]),
            "neutral": float(probs[self.NEUTRAL_IDX]),
            "contradiction": float(probs[self.CONTRADICTION_IDX]),
        }

    def score(self, answer: str, chunks: list, top_k: int = 3) -> float:
        self._lazy_load()

        if not chunks:
            return 0.0

        # Split answer into individual sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]

        if not sentences:
            return 0.0

        top_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        all_scores = []
        for chunk in top_chunks:
            premise = chunk["text"][:1000]
            for sentence in sentences[:5]:
                result = self._predict_single(premise, sentence)
                all_scores.append(result["entailment"])

        if not all_scores:
            return 0.0

        # Average of top 30% scores
        all_scores.sort(reverse=True)
        top_scores = all_scores[:max(1, len(all_scores) // 3)]
        return round(float(np.mean(top_scores)), 4)

    def detailed_score(self, answer: str, chunks: list) -> dict:
        self._lazy_load()

        results = []
        for i, chunk in enumerate(chunks):
            premise = chunk["text"][:512]
            scores = self._predict_single(premise, answer[:256])
            results.append({
                "chunk_index": i,
                "chunk_title": chunk.get("metadata", {}).get("title", f"Chunk {i}"),
                "entailment": scores["entailment"],
                "neutral": scores["neutral"],
                "contradiction": scores["contradiction"],
                "verdict": max(scores, key=scores.get),
            })

        final_confidence = max(r["entailment"] for r in results)

        return {
            "confidence": round(final_confidence, 4),
            "is_grounded": final_confidence >= 0.6,
            "show_disclaimer": final_confidence < 0.6,
            "chunk_scores": results,
            "interpretation": self._interpret(final_confidence),
        }

    def _interpret(self, score: float) -> str:
        if score >= 0.8:
            return "High confidence — answer is well-supported by retrieved judgements."
        elif score >= 0.6:
            return "Moderate confidence — answer is mostly supported."
        elif score >= 0.4:
            return "Low confidence — answer may not be fully grounded. Verify with a lawyer."
        else:
            return "Very low confidence — retrieved sources don't clearly support this answer."


def main():
    print("=" * 60)
    print("LEGAL AID LLM — NLI Hallucination Checker Test")
    print("=" * 60)

    checker = NLIChecker()

    chunks_grounded = [{
        "text": "Under Section 106 of the Transfer of Property Act, a landlord must give 15 days notice before evicting a monthly tenant.",
        "score": 0.92,
        "metadata": {"title": "Sharma vs Kumar 2019"},
    }]
    answer_grounded = "Your landlord must give you at least 15 days notice before evicting you."
    score_g = checker.score(answer_grounded, chunks_grounded)
    print(f"\n[GROUNDED] Score: {score_g}")

    chunks_hallucinated = [{
        "text": "Under Section 106 of the Transfer of Property Act, a landlord must give 15 days notice before evicting a monthly tenant.",
        "score": 0.92,
        "metadata": {"title": "Sharma vs Kumar 2019"},
    }]
    answer_hallucinated = "Landlords in India are not required to give any notice and can evict immediately."
    score_h = checker.score(answer_hallucinated, chunks_hallucinated)
    print(f"\n[HALLUCINATED] Score: {score_h}")

    print(f"\nGrounded={score_g:.2f} vs Hallucinated={score_h:.2f}")
    print("The gap between these scores is your model's trustworthiness metric.")


if __name__ == "__main__":
    main()