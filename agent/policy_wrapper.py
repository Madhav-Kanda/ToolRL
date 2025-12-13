import json
import re
from typing import Dict, Any
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


class Policy:
    def __init__(self, cfg_path: str = "configs/model.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        print(f"[Model] Loading {cfg['model_name']} (4bit={cfg.get('load_in_4bit', False)}) ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
        kwargs = {}
        if cfg.get("load_in_4bit", False):
            kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto", **kwargs)
        self.model.eval()
        print("[Model] Loaded.", flush=True)

    def decide(self, prompt: str) -> Dict[str, Any]:
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.get("max_new_tokens", 256),
            do_sample=True,
            temperature=self.cfg.get("temperature", 0.2),
            top_p=self.cfg.get("top_p", 0.9),
        )
        text = self.tok.decode(gen[0], skip_special_tokens=True)
        prompt_text = self.tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
        completion = text[len(prompt_text) :]
        m = re.search(r"\{[\s\S]*\}", completion)
        if not m:
            return {"tool_name": "run_tests", "args": {"timeout_s": 60}}
        try:
            obj = json.loads(m.group(0))
            if "tool_name" in obj and "args" in obj and isinstance(obj["args"], dict):
                return obj
        except Exception:
            pass
        return {"tool_name": "run_tests", "args": {"timeout_s": 60}}


