import json
import time
import gc
from pathlib import Path
from PIL import Image

from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def clear_memory():
    for name in ["inputs", "model", "processor", "trainer", "bnb_config"]:
        if name in globals():
            del globals()[name]

    time.sleep(1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    if torch.cuda.is_available():
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


clear_memory()


def to_vlm_example(example):
    messages = example["messages"]
    image_paths = []
    normalized_messages = []

    for msg in messages:
        normalized_msg = {"role": msg["role"]}
        content = msg.get("content")

        if isinstance(content, list):
            normalized_content = []
            for part in content:
                if part.get("type") == "image":
                    image_path = part.get("image")
                    if image_path:
                        image_paths.append(image_path)
                    # Keep an image placeholder in chat content; actual paths go in top-level "images".
                    normalized_content.append({"type": "image"})
                elif part.get("type") == "text":
                    normalized_content.append({"type": "text", "text": part.get("text", "")})
            normalized_msg["content"] = normalized_content
        else:
            normalized_msg["content"] = content

        normalized_messages.append(normalized_msg)

    return {"messages": normalized_messages, "images": image_paths}


def _load_images_from_example(example):
    imgs = []
    for p in example.get("images", []):
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def _prompt_messages(messages):
    # Keep only non-assistant turns to compute where assistant target begins.
    return [m for m in messages if m.get("role") != "assistant"]


def vlm_assistant_only_collator(features):
    """
    Build batch with labels masked to assistant tokens only:
    - system/user/template prefix => -100
    - padding => -100
    """
    texts_full = []
    texts_prompt = []
    images_batch = []

    for ex in features:
        messages = ex["messages"]
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = processor.apply_chat_template(
            _prompt_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

        texts_full.append(full_text)
        texts_prompt.append(prompt_text)
        images_batch.append(_load_images_from_example(ex))

    # Tokenize full inputs (targets included)
    batch_full = processor(
        text=texts_full,
        images=images_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    # Tokenize prompt-only to find assistant start index per sample
    batch_prompt = processor(
        text=texts_prompt,
        images=images_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    input_ids = batch_full["input_ids"]
    attention_mask = batch_full["attention_mask"]
    labels = input_ids.clone()

    bsz, seq_len = labels.shape
    for i in range(bsz):
        # Number of non-pad prompt tokens in this sample
        prompt_len = int(batch_prompt["attention_mask"][i].sum().item())
        prompt_len = min(prompt_len, seq_len)
        labels[i, :prompt_len] = -100
        labels[i, attention_mask[i] == 0] = -100

    batch_full["labels"] = labels
    return batch_full


model_name = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dataset_path = Path(__file__).with_name("data.jsonl")
full_dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")
full_dataset = full_dataset.map(to_vlm_example)

train_test = full_dataset.train_test_split(test_size=0.2, seed=42)
val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = train_test["train"]
eval_dataset = val_test["train"]
test_dataset = val_test["test"]

training_args = SFTConfig(
    output_dir="qwen3-8b-instruct-fine-tune-V3",
    num_train_epochs=3,
    max_steps=-1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=4096,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    logging_first_step=True,
    logging_steps=1,
    eval_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    bf16=True,
    max_grad_norm=0.3,
    warmup_steps=1,
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=vlm_assistant_only_collator,
)

# Quick one-batch sanity check
sample_batch = vlm_assistant_only_collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])
num_target = (sample_batch["labels"] != -100).sum().item()
print(f"Sanity check: supervised tokens in sample batch = {num_target}")

trainer.train()
trainer.save_model(training_args.output_dir)