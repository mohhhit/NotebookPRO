# Qwen3-1.7B Fine-Tuning in Google Colab (LoRA + 4bit)

Deprecated for your current workflow. Use the Unsloth notebook instead:

- QWEN3_UNSLOTH_GGUF_COLAB.ipynb

The new notebook includes Unsloth training plus merged GGUF export for local llama.cpp usage.

This notebook-style guide fine-tunes `Qwen/Qwen3-1.7B` using your dataset at:

- `training_data/qwen_golden_dataset.jsonl`

It supports both:
- `{"text": "<|im_start|>..."}` records (your current format)
- `{"messages": [{"role": ..., "content": ...}, ...]}` records

---

## 1) Install dependencies

```python
!pip -q install -U transformers datasets trl peft accelerate bitsandbytes sentencepiece huggingface_hub
```

---

## 2) (Optional) Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 3) Config

```python
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen3-1.7B"
# Update this path based on where you put your JSONL in Colab/Drive
DATA_PATH = "/content/drive/MyDrive/NoteBookPRO/training_data/qwen_golden_dataset.jsonl"
OUTPUT_DIR = "/content/qwen3_1_7b_lora"

MAX_SEQ_LEN = 2048
TRAIN_EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 8
VAL_SPLIT = 0.02
SEED = 42
```

---

## 4) Load and normalize dataset into chat messages

```python
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def parse_qwen_chatml(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    roles = {}
    for role in ("system", "user", "assistant"):
        marker = f"{IM_START}{role}\n"
        i = text.find(marker)
        if i < 0:
            return None
        i += len(marker)
        j = text.find(IM_END, i)
        if j < 0:
            return None
        roles[role] = text[i:j].strip()

    if not all(roles.values()):
        return None

    return [
        {"role": "system", "content": roles["system"]},
        {"role": "user", "content": roles["user"]},
        {"role": "assistant", "content": roles["assistant"]},
    ]


def to_messages(example):
    msgs = example.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 3:
        return {"messages": msgs}

    txt = example.get("text")
    if isinstance(txt, str):
        parsed = parse_qwen_chatml(txt)
        if parsed is not None:
            return {"messages": parsed}

    return {"messages": None}


raw = load_dataset("json", data_files=DATA_PATH, split="train")
raw_count = len(raw)

normalized = raw.map(to_messages)
normalized = normalized.filter(lambda x: x["messages"] is not None)

print(f"Raw samples: {raw_count}")
print(f"Usable samples: {len(normalized)}")
```

---

## 5) Build tokenizer and training text with chat template

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_for_training(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text + tokenizer.eos_token}


dataset = normalized.map(format_for_training, remove_columns=normalized.column_names)
split = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)

print(split)
print("Example sample:\n", split["train"][0]["text"][:700])
```

---

## 6) Load Qwen3-1.7B in 4-bit and attach LoRA

```python
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## 7) Train

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    fp16=not use_bf16,
    bf16=use_bf16,
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=True,
)

trainer.train()
```

---

## 8) Save adapter

```python
adapter_dir = f"{OUTPUT_DIR}/adapter"
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("Saved adapter:", adapter_dir)
```

---

## 9) Quick inference test with adapter

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
ft_model.eval()

messages = [
    {"role": "system", "content": "You are an expert academic tutor."},
    {
        "role": "user",
        "content": "Explain in simple terms why small-angle approximation turns pendulum motion into SHM."
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(ft_model.device)

with torch.no_grad():
    outputs = ft_model.generate(
        input_ids=inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

---

## 10) (Optional) Push adapter to Hugging Face Hub

```python
from huggingface_hub import notebook_login
notebook_login()

repo_id = "your-username/qwen3-1.7b-notebookpro-lora"
trainer.model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

---

## Notes for stability

- If Colab RAM/VRAM is tight, lower:
  - `MAX_SEQ_LEN` from `2048` to `1536` or `1024`
  - `BATCH_SIZE` from `2` to `1`
  - increase `GRAD_ACCUM` to keep effective batch size similar
- If loss is unstable, try:
  - `LEARNING_RATE = 1e-4`
  - `lora_dropout = 0.1`
- For best final quality, do 2 to 3 epochs and monitor eval loss.
