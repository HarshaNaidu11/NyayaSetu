# AI Legal Aid Assistant for First-Generation Litigants

> A multilingual LLM system that explains Indian legal rights in plain language — in Telugu, Hindi, and English — to people who have never seen a courtroom.

---

## Problem

Millions of Indians face courts, police stations, and consumer forums without understanding their basic legal rights. Most can't afford a lawyer. Legal documents are written for judges, not citizens. This project bridges that gap using a fine-tuned LLM with RAG-backed case retrieval and multilingual output.

---

## What it does

- Accepts legal questions in **English, Telugu, or Hindi**
- Retrieves relevant **Indian court judgements** from a FAISS vector index
- Generates a **plain-language explanation** of the user's rights
- Detects when the model is **hallucinating** and shows a confidence score
- Outputs the answer **back in the user's language**

---

## Architecture

```
User query (Telugu/Hindi/English)
        ↓
IndicTrans2 → translate to English
        ↓
BGE-M3 embedding → FAISS retrieval (top-5 chunks)
        ↓
Fine-tuned Llama 3 / Mistral → generate answer
        ↓
NLI DeBERTa → entailment check → confidence score
        ↓
IndicTrans2 → translate answer back to user's language
        ↓
Streamlit UI → show answer + sources + confidence badge
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3-8B / Mistral-7B (via Ollama) |
| Fine-tuning | QLoRA + PEFT (4-bit, runs on Colab free tier) |
| Embeddings | BGE-M3 (multilingual, free) |
| Vector DB | FAISS |
| RAG orchestration | LangChain |
| Translation | IndicTrans2 (AI4Bharat) |
| Hallucination check | DeBERTa-v3 NLI |
| UI | Streamlit |
| Deployment | HuggingFace Spaces |

---

## Datasets

| Dataset | Use | Link |
|---|---|---|
| Indian Kanoon | Primary corpus (50K+ judgements) | indiankanoon.org/api |
| ILDC (IIT Delhi) | Fine-tuning pairs | HuggingFace: pile-of-law/pile-of-law |
| NALSA FAQs | Supervised Q&A pairs | nalsa.gov.in |
| Samanantar | Telugu/Hindi translation | ai4bharat.org |

---

## Eval Results

| Metric | Baseline (Llama 3) | Fine-tuned |
|---|---|---|
| Retrieval hit rate | 61% | 84% |
| BERTScore (F1) | 0.71 | 0.83 |
| Avg confidence (NLI) | — | 0.79 |
| Readability grade | 14.2 | 6.8 |

*(Update these with your actual numbers after running eval/run_eval.py)*

---

## Setup

```bash
git clone https://github.com/yourusername/legal-aid-llm
cd legal-aid-llm
pip install -r requirements.txt
```

### Step 1 — Download and index the corpus
```bash
python scripts/download_corpus.py
python scripts/build_index.py
```

### Step 2 — Run the RAG pipeline (no fine-tuning yet)
```bash
python scripts/rag_pipeline.py
```

### Step 3 — Fine-tune (run on Google Colab A100)
Open `models/finetune_qlora.ipynb` in Colab

### Step 4 — Launch UI
```bash
streamlit run ui/app.py
```

---

## Project Structure

```
legal-aid-llm/
├── data/                  # Raw + processed corpus
├── embeddings/            # FAISS index files
├── models/                # Fine-tuning notebook + saved adapters
├── scripts/
│   ├── download_corpus.py # Fetch Indian Kanoon data
│   ├── build_index.py     # Chunk + embed + build FAISS
│   ├── rag_pipeline.py    # Core RAG logic
│   └── translate.py       # IndicTrans2 wrapper
├── eval/
│   └── run_eval.py        # Evaluation script
├── ui/
│   └── app.py             # Streamlit UI
└── requirements.txt
```

---

## What makes this novel

1. **Telugu/Hindi legal output** — near zero prior work in regional-language legal NLP
2. **NLI hallucination gate** — answer verified against retrieved chunks before showing user
3. **Indian law fine-tuning** — base LLMs are trained on US/UK law; this corrects that
4. **First-gen litigant UX** — plain language, no jargon, confidence disclaimers
5. **Chunking strategy benchmarking** — recursive vs. fixed vs. semantic (see eval/)

---

## Limitations

- Not a substitute for legal advice — always recommends consulting a lawyer
- Coverage limited to judgements in Indian Kanoon corpus
- Translation quality varies for highly technical legal terms

---

