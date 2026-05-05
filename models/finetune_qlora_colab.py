
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
