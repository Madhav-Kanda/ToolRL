# RL Code Repair (SWE-bench Lite)

An agent that reads an issue/task spec, searches a codebase, proposes unified diff patches, runs tests (`pytest`), and iterates. The project targets SWE-bench Lite and includes scaffolding for SFT → DPO → light PPO/GRPO with QLoRA on Qwen2.5-Coder-7B-Instruct.

## Quickstart

```bash
# 1) Setup environment
bash scripts/setup_env.sh

# 2) (Optional) Prepare SWE-bench Lite normalized specs
bash scripts/prepare_swe_lite.sh

# 3) Run a baseline agent evaluation on a small subset
bash scripts/run_baseline.sh
```

## Repository Layout

- configs/: model/env/train configs
- agent/: prompts, policy wrapper, tools, agent loop
- envs/swebench/: dataset loader, harness, verifier, sandbox
- policy/: model loading & training scripts (SFT/DPO/PPO)
- eval/: evaluation runner and metrics
- scripts/: helper bash scripts
- data/: caches and generated artifacts

## Notes

- You will need Git, Python 3.10+, and optionally a GPU with ~24GB VRAM for fastest runs (QLoRA).
- SWE-bench Lite may require network to clone repos for episodes; you can pre-cache repositories to avoid repeated downloads.


