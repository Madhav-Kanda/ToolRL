import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_dpo.yaml")
    ap.add_argument("--out_dir", type=str, default="outputs/dpo_lora")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA config
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Load preference pairs as HuggingFace Dataset
    pairs = []
    with open(cfg["data"]["pairs_jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pairs.append({
                "prompt": rec["prompt"],
                "chosen": rec["chosen"],
                "rejected": rec["rejected"],
            })
    train_dataset = Dataset.from_list(pairs)

    # DPO training config
    training_args = DPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=cfg.get("epochs", 1),
        learning_rate=cfg.get("lr", 5e-5),
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("seq_len", 1024),
        max_prompt_length=cfg.get("seq_len", 1024) // 2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )

    trainer = DPOTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tok,
    )
    trainer.train()
    peft_model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()


