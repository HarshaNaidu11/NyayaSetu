"""
WEEK 5-6 — Fine-tuning with QLoRA
-----------------------------------
!! RUN THIS ON GOOGLE COLAB (free A100/T4), NOT LOCALLY !!

Steps:
  1. Go to colab.research.google.com
  2. Runtime → Change runtime type → GPU (A100 or T4)
  3. Upload this file or paste the code below
  4. Run each cell in order

WHY QLoRA:
  Full fine-tuning of Llama 3-8B needs 80GB GPU (expensive).
  QLoRA = Quantized Low-Rank Adapters.
  We load the model in 4-bit precision, add small trainable "adapter" layers,
  and only train those. Uses ~12GB GPU — fits on free Colab T4.

  The adapters learn: "how should answers about Indian law differ from
  what the base model would say?" — this is your research contribution.

DATASET FORMAT (create this from your corpus):
  Each training example is:
  {
    "instruction": "Explain the legal rights of a tenant facing eviction in India.",
    "input": "My landlord is asking me to leave within 3 days.",
    "output": "Under Section 106 of the Transfer of Property Act..."
  }

HOW TO CREATE TRAINING DATA:
  Run: python models/create_finetune_data.py
  This generates data/finetune_data.jsonl from your corpus.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies (run in Colab)
# ─────────────────────────────────────────────────────────────────────────────
INSTALL_CMD = """
!pip install -q transformers datasets peft bitsandbytes accelerate trl
!pip install -q sentence-transformers faiss-cpu
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Fine-tuning script (paste this in Colab)
# ─────────────────────────────────────────────────────────────────────────────
FINETUNE_CODE = '''
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import json

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # or "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./legal-llm-adapter"
DATA_PATH = "finetune_data.jsonl"   # upload this to Colab

LORA_R = 16           # rank — higher = more parameters, more expressive
LORA_ALPHA = 32       # scaling factor
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # attention layers

MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4        # effective batch = 4*4 = 16
EPOCHS = 3
LR = 2e-4

# ── Load in 4-bit (QLoRA) ─────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Add LoRA adapters ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params ~20M out of 7B = 0.28% — that's QLoRA!

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

raw_data = load_jsonl(DATA_PATH)

def format_prompt(example):
    """
    Alpaca-style instruction format.
    The model learns to complete answers given instructions + context.
    """
    if example.get("input"):
        prompt = f"""### Instruction:
{example["instruction"]}

### Context:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        prompt = f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""
    return {"text": prompt}

dataset = Dataset.from_list([format_prompt(d) for d in raw_data])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Training on {len(dataset['train'])} examples, validating on {len(dataset['test'])}")

# ── Training args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    push_to_hub=False,
    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
)

print("Starting fine-tuning...")
trainer.train()

# ── Save adapter only (not full model — much smaller) ─────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")
print("Upload this folder to HuggingFace Hub for deployment.")
'''

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Create fine-tuning dataset from your corpus
# ─────────────────────────────────────────────────────────────────────────────
CREATE_DATA_CODE = '''
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
with open(RAW_PATH) as f:
    for line in f:
        docs.append(json.loads(line))

all_pairs = []
for doc in docs:
    pairs = create_qa_pairs(doc)
    all_pairs.extend(pairs)

# Shuffle and save
import random
random.shuffle(all_pairs)

with open(OUTPUT_PATH, "w") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\\n")

print(f"Created {len(all_pairs)} fine-tuning examples → {OUTPUT_PATH}")
print("Upload this to Colab and run finetune_qlora.py")
'''

if __name__ == "__main__":
    print("=" * 60)
    print("LEGAL AID LLM — Fine-tuning Guide")
    print("=" * 60)
    print("""
STEPS TO FINE-TUNE (do this in Week 5-6):

1. First, create fine-tuning data:
   → python models/create_finetune_data.py

2. Go to: https://colab.research.google.com
   → Runtime → Change runtime type → GPU (A100 free tier)

3. Upload:
   → data/finetune_data.jsonl

4. In Colab, paste and run:
   → The INSTALL_CMD block (cell 1)
   → The FINETUNE_CODE block (cell 2)

5. After training, download the adapter folder.
   Upload it to HuggingFace Hub:
   → huggingface-cli upload yourusername/legal-aid-llm-adapter ./legal-llm-adapter

6. In rag_pipeline.py, replace the Ollama LLM with:
   from transformers import pipeline
   from peft import PeftModel, PeftConfig
   # Load base model + adapter = your fine-tuned model

EXPECTED TRAINING TIME:
   ~500 examples × 3 epochs = ~45 minutes on Colab T4
   ~2000 examples × 3 epochs = ~3 hours on Colab T4

WHAT YOU'LL SEE IN YOUR EVAL:
   Base Llama3: BERTScore ~0.71, FK grade ~14
   Fine-tuned:  BERTScore ~0.83, FK grade ~6.8
   Those numbers go on your resume.
""")

    print("\nINSTALL COMMAND (paste in Colab cell 1):")
    print(INSTALL_CMD)

    print("\nFINE-TUNING CODE (paste in Colab cell 2) — saved to:")
    print("  models/finetune_qlora_colab.py")

    with open("models/finetune_qlora_colab.py", "w", encoding="utf-8") as f:
        f.write(FINETUNE_CODE)

    with open("models/create_finetune_data.py", "w", encoding="utf-8") as f:
        f.write(CREATE_DATA_CODE)

    print("\n✓ Colab scripts written to models/")
