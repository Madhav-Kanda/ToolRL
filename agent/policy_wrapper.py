import json
import re
import os
import logging
from typing import Dict, Any
import yaml

# Suppress verbose vLLM logging
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
logging.getLogger("vllm").setLevel(logging.WARNING)


class Policy:
    """
    Unified policy wrapper that supports both vLLM (fast inference) and HuggingFace (training/adapters).
    """
    def __init__(self, cfg_path: str = "configs/model.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.backend = cfg.get("backend", "hf").lower()
        
        if self.backend == "vllm":
            self._init_vllm()
        else:
            self._init_hf()
    
    def _init_vllm(self):
        """Initialize vLLM for fast inference."""
        from vllm import LLM, SamplingParams
        
        model_name = self.cfg["model_name"]
        tp_size = self.cfg.get("tensor_parallel_size", 1)
        gpu_mem = self.cfg.get("gpu_memory_utilization", 0.85)
        
        print(f"[Model] Loading {model_name} with vLLM (tp={tp_size}, gpu_mem={gpu_mem}) ...", flush=True)
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=True,
            dtype="half",  # FP16 is faster than 4-bit on vLLM
        )
        self.sampling_params = SamplingParams(
            temperature=self.cfg.get("temperature", 0.2),
            top_p=self.cfg.get("top_p", 0.9),
            max_tokens=self.cfg.get("max_new_tokens", 256),
        )
        print("[Model] vLLM loaded.", flush=True)
    
    def _init_hf(self):
        """Initialize HuggingFace transformers (for training or when adapters are needed)."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        model_name = self.cfg["model_name"]
        print(f"[Model] Loading {model_name} with HF (4bit={self.cfg.get('load_in_4bit', False)}) ...", flush=True)
        
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        kwargs = {}
        if self.cfg.get("load_in_4bit", False):
            kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs)
        
        adapters = (self.cfg.get("adapters_path") or "").strip()
        if adapters:
            print(f"[Model] Loading adapters from {adapters} ...", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapters)
        self.model.eval()
        print("[Model] HF loaded.", flush=True)

    def decide(self, prompt: str) -> Dict[str, Any]:
        """Generate a tool call decision from the prompt."""
        if self.backend == "vllm":
            return self._decide_vllm(prompt)
        else:
            return self._decide_hf(prompt)
    
    def _decide_vllm(self, prompt: str) -> Dict[str, Any]:
        """Generate using vLLM."""
        outputs = self.llm.generate([prompt], self.sampling_params)
        completion = outputs[0].outputs[0].text
        return self._parse_tool_call(completion)
    
    def _decide_hf(self, prompt: str) -> Dict[str, Any]:
        """Generate using HuggingFace."""
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
        completion = text[len(prompt_text):]
        return self._parse_tool_call(completion)
    
    def _parse_tool_call(self, completion: str) -> Dict[str, Any]:
        """Parse JSON tool call from model output."""
        m = re.search(r"\{[\s\S]*\}", completion)
        if not m:
            return {"tool_name": "final_answer", "args": {"answer": completion.strip()}}
        try:
            obj = json.loads(m.group(0))
            if "tool_name" in obj and "args" in obj and isinstance(obj["args"], dict):
                return obj
        except Exception:
            pass
        return {"tool_name": "final_answer", "args": {"answer": completion.strip()}}
