import argparse
import json
import os
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
import yaml
import torch


def load_jsonl(path: str) -> List[Dict[str, str]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            data.append({"text": rec["prompt"] + rec["response"]})
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_sft.yaml")
    ap.add_argument("--out_dir", type=str, default="outputs/sft_lora")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[SFT] Loading {cfg['model_name']}...")
    print(f"[SFT] Available GPUs: {torch.cuda.device_count()}")

    # QLoRA: 4-bit quantization on single GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map={"": 0},  # Single GPU
        trust_remote_code=True,
    )
    
    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA configuration
    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SFT] Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load data
    train_data = load_jsonl(cfg["data"]["train_jsonl"])
    val_path = cfg["data"].get("val_jsonl")
    val_data = load_jsonl(val_path) if val_path else None
    
    print(f"[SFT] Train samples: {len(train_data)}")
    if val_data:
        print(f"[SFT] Val samples: {len(val_data)}")

    train_ds = datasets.Dataset.from_list(train_data)
    val_ds = datasets.Dataset.from_list(val_data) if val_data else None

    max_len = cfg.get("seq_len", 1024)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_len, padding=False)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    if val_ds:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Training arguments - single GPU optimized
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=cfg.get("per_device_batch", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        num_train_epochs=cfg.get("epochs", 2),
        learning_rate=cfg.get("lr", 2e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
        dataloader_num_workers=cfg.get("dataloader_workers", 4),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=0.3,  # Gradient clipping for stability
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tok,
    )
    
    print(f"[SFT] Starting training...")
    trainer.train()
    
    print(f"[SFT] Saving model to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("[SFT] Done!")


if __name__ == "__main__":
    main()
