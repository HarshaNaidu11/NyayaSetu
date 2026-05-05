import pickle
import numpy as np
import faiss
from pathlib import Path

EMBED_DIR = Path("embeddings")
TOP_K = 5
OLLAMA_MODEL = "gemma2:2b"


class LegalRAGPipeline:
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadatas = []
        self.embed_model = None
        self.llm = None
        self._load()

    def _load(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(EMBED_DIR / "legal_index.faiss"))

        print("Loading chunk store...")
        with open(EMBED_DIR / "chunks.pkl", "rb") as f:
            store = pickle.load(f)
            self.texts = store["texts"]
            self.metadatas = store["metadatas"]

        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._load_ollama()

        print("\n✓ Pipeline ready.\n")

    def _load_ollama(self):
        try:
            from langchain_ollama import OllamaLLM as Ollama
            self.llm = Ollama(model=OLLAMA_MODEL)
            self.llm.invoke("Hi")
            print(f"  ✓ Connected to {OLLAMA_MODEL} via Ollama")
        except Exception as e:
            print(f"  [!] Ollama not available: {e}")
            print("  Run: ollama serve")
            print("  Run: ollama pull gemma2:2b")
            self._load_hf_fallback()

    def _load_hf_fallback(self):
        print("  Loading TinyLlama as fallback...")
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=256,
        )

        class HFWrapper:
            def invoke(self, prompt):
                out = pipe(prompt)[0]["generated_text"]
                return out[len(prompt):]

        self.llm = HFWrapper()
        print("  ✓ Fallback loaded")

    def retrieve(self, query: str, k: int = TOP_K) -> list:
        query_vec = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "score": float(score),
            })
        return results

    def build_prompt(self, query: str, chunks: list) -> str:
        context = "\n\n".join([
            f"Source {i+1} [{c['metadata']['title']}]:\n{c['text'][:400]}"
            for i, c in enumerate(chunks[:3])
        ])

        prompt = f"""You are a free legal helper in India for poor citizens.

LEGAL SOURCES:
{context}

QUESTION: {query}

STRICT RULES:
- Write ONLY 5 to 8 sentences total
- Each sentence: maximum 8 words
- Start each point with a number: 1. 2. 3.
- Use the simplest possible words
- Name the exact Indian law (example: Section 106, Transfer of Property Act)
- Last line must be: "Go to [specific place] for free help."
- NEVER write more than 8 sentences
- NEVER use words like: pursuant, aforementioned, hereinafter, thereof

ANSWER:"""

        return prompt

    def check_confidence(self, answer: str, chunks: list) -> float:
        avg_chunk_score = np.mean([c["score"] for c in chunks])
        confidence = min(avg_chunk_score, 1.0)
        return float(confidence)

    def query(self, user_query: str) -> dict:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.translate import detect_language, to_english, from_english

        # Step 1 - detect language and translate query to English
        src_lang = detect_language(user_query)
        query_en, _ = to_english(user_query)

        # Step 2 - retrieve chunks using English query
        chunks = self.retrieve(query_en)
        if not chunks:
            return {
                "answer": "I could not find relevant legal information for your question.",
                "sources": [],
                "confidence": 0.0,
                "disclaimer": True,
                "detected_language": src_lang,
            }

        # Step 3 - generate answer in English
        prompt = self.build_prompt(query_en, chunks)
        answer = self.llm.invoke(prompt)

        # Step 4 - clean up answer
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        elif "<|assistant|>" in answer:
            answer = answer.split("<|assistant|>")[-1].strip()

        # Step 5 - translate answer back to user's language
        if src_lang != "en":
            answer = from_english(answer, src_lang)

        # Step 6 - confidence + sources
        confidence = self.check_confidence(answer, chunks)

        sources = [
            {
                "title": c["metadata"]["title"],
                "court": c["metadata"]["court"],
                "date": c["metadata"]["date"],
                "relevance": round(c["score"], 3),
                "excerpt": c["text"][:300] + "...",
            }
            for c in chunks
        ]

        disclaimer = confidence < 0.6
        if disclaimer:
            answer += "\n\n⚠️ Note: This answer may not be fully supported by available judgements. Please consult a qualified lawyer for your specific situation."

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
            "disclaimer": disclaimer,
            "detected_language": src_lang,
        }


def main():
    print("=" * 60)
    print("LEGAL AID LLM — RAG Pipeline Test")
    print("=" * 60)

    pipeline = LegalRAGPipeline()

    test_questions = [
        "Can my landlord evict me without any notice?",
        "I was arrested. What are my rights in police custody?",
        "My employer fired me without any reason. What can I do?",
        "How do I file a consumer complaint against a company?",
    ]

    print("Running test queries:\n")
    for q in test_questions:
        print(f"Q: {q}")
        result = pipeline.query(q)
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {len(result['sources'])} judgements retrieved")
        print(f"Answer preview: {result['answer'][:300]}...")
        print("-" * 50)

    print("\n\nEntering interactive mode. Type 'quit' to exit.\n")
    while True:
        user_input = input("Your legal question: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = pipeline.query(user_input)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Detected language: {result.get('detected_language', 'en')}")
        print(f"Based on {len(result['sources'])} court judgements")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()