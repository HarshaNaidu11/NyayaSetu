# ⚖️ Nyaya Setu — Multilingual Legal Aid LLM for India

> A RAG-based LLM system that explains Indian legal rights in plain language — in Telugu, Hindi, and English — to first-generation litigants who have never seen a courtroom.

🔗 **[Live Demo](https://huggingface.co/spaces/Harsha11111/nyayasetu)** &nbsp;|&nbsp; Built by [N HarshaVardhan Raj](https://github.com/HarshaNaidu11) &nbsp;|&nbsp; BTech CSE, IIITDM Kurnool

---

## The Problem

Millions of Indians face courts, police stations, and consumer forums without understanding their basic legal rights. Most can't afford a lawyer. Legal documents are written for judges, not citizens. This project bridges that gap using a fine-tuned LLM with RAG-backed case retrieval, NLI hallucination detection, and multilingual output.

---

## What It Does

- Accepts legal questions in **English, Telugu, or Hindi**
- Retrieves relevant **Indian court judgements** from a FAISS vector index (25,909 chunks from 1,625 real cases)
- Generates a **plain-language answer** using Groq LLaMA3, citing the exact Indian law that applies
- **Detects hallucination** using DeBERTa-v3 NLI — shows disclaimer when answer is not grounded
- Translates answers **back to the user's language** automatically
- Deployed live on HuggingFace Spaces — free to use

---

## Evaluation Results

| Metric | Score | Target |
|---|---|---|
| BERTScore F1 | **0.860** | higher is better |
| Flesch-Kincaid Grade | **6.6** | ≤ 8 ✅ |
| Keyword Hit Rate | **64%** | retrieval accuracy |
| NLI Confidence | 0.27 | sentence-level entailment |
| Avg Response Time | 33s (CPU) | ~3s on GPU |
| Corpus Size | 1,625 judgements | 25,909 FAISS vectors |

---

## Architecture

```
User query (Telugu / Hindi / English)
        ↓
Google Translate → English query
        ↓
MiniLM embedding → FAISS retrieval (top-5 chunks)
        ↓
Groq LLaMA3-8B → plain-language answer
        ↓
DeBERTa NLI → entailment check → confidence score
        ↓
Google Translate → answer in user's language
        ↓
Streamlit UI → answer + sources + confidence badge
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM (deployed) | Groq LLaMA3-8B (free API) |
| LLM (local) | Gemma2:2B via Ollama |
| Fine-tuning | QLoRA + PEFT (4-bit, Google Colab T4) |
| Base model | TinyLlama-1.1B-Chat |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB | FAISS (25,909 vectors, dim=384) |
| RAG orchestration | LangChain |
| Translation | Google Translate (deep-translator) |
| Hallucination check | DeBERTa-v3 NLI (cross-encoder) |
| UI | Streamlit |
| Deployment | HuggingFace Spaces (Docker) |

---

## Corpus & Datasets

| Source | Size | Use |
|---|---|---|
| Indian Kanoon API | 1,608 real judgements | Primary RAG corpus |
| Synthetic legal KB | 20 documents | Guaranteed fallback |
| Fine-tuning data | 238 Q&A pairs | QLoRA training |

**Legal topics covered:** tenant eviction · arrest rights · consumer complaints · domestic violence · labour dismissal · bail · minimum wages · ration card · maternity benefits · land acquisition

---

## What Makes It Novel

1. **Telugu/Hindi legal output** — near zero prior published work in regional-language Indian legal NLP
2. **NLI hallucination gate** — sentence-level entailment verification before showing answer; most RAG systems skip this
3. **Indian law fine-tuning** — base LLMs are trained on US/UK legal data; QLoRA fine-tuning on Indian Kanoon corpus corrects this
4. **First-gen litigant UX** — Grade 6.6 reading level, numbered steps, plain language — designed for someone who has never been to court
5. **Chunking strategy benchmarking** — recursive 512-token overlapping chunks benchmarked against fixed and semantic splitting

---

## Local Setup

```bash
git clone https://github.com/HarshaNaidu11/legal-aid-llm
cd legal-aid-llm
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Step 1 — Download corpus
```bash
python scripts/download_corpus.py
```

### Step 2 — Build FAISS index
```bash
python scripts/build_index.py
```

### Step 3 — Run locally (needs Ollama)
```bash
ollama pull gemma2:2b
ollama serve
python scripts/rag_pipeline.py
```

### Step 4 — Launch UI
```bash
streamlit run ui/app.py --server.fileWatcherType none
```

### Step 5 — Fine-tune (Google Colab T4)
```bash
python models/finetune_guide.py   # generates Colab script
# Upload data/finetune_data.jsonl to Colab and run models/finetune_qlora_colab.py
```

### Step 6 — Run evaluation
```bash
python eval/run_eval.py
# Output saved to eval/report.txt
```

---

## Project Structure

```
legal-aid-llm/
├── data/
│   └── raw/raw_judgements.jsonl     # 1,625 Indian court judgements
├── embeddings/
│   ├── legal_index.faiss            # FAISS index (25,909 vectors)
│   └── chunks.pkl                   # chunk texts + metadata
├── models/
│   ├── finetune_guide.py            # generates Colab training script
│   ├── finetune_qlora_colab.py      # QLoRA training (run on Colab)
│   ├── create_finetune_data.py      # builds training dataset
│   └── legal-llm-adapter/           # saved QLoRA adapter weights
├── scripts/
│   ├── download_corpus.py           # fetch from Indian Kanoon API
│   ├── build_index.py               # chunk + embed + build FAISS
│   ├── rag_pipeline.py              # core RAG pipeline
│   ├── translate.py                 # multilingual translation wrapper
│   └── nli_checker.py              # DeBERTa hallucination detection
├── eval/
│   ├── run_eval.py                  # evaluation script
│   ├── report.txt                   # evaluation results
│   └── summary.csv                  # per-question metrics
├── ui/
│   └── app.py                       # local Streamlit UI
├── app.py                           # HuggingFace deployment app
├── Dockerfile                       # Docker config for HF Spaces
├── WORKFLOW.md                      # step-by-step build guide
└── requirements.txt
```

---

## Limitations

- Not a substitute for legal advice — always recommends consulting a qualified lawyer
- Coverage limited to topics in Indian Kanoon corpus (6 of 10 topics fully covered)
- NLI confidence is lower on longer synthesized answers — known metric limitation
- Response time ~33s on CPU; ~3s projected on GPU deployment

---

## Author

**N HarshaVardhan Raj** 

[LinkedIn](https://www.linkedin.com/in/n-harsha-vardhan-raj-7266432a4/) · [GitHub](https://github.com/HarshaNaidu11) · [Live Demo](https://huggingface.co/spaces/Harsha11111/nyayasetu)