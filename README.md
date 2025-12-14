# Tool-Use Agent (ToolBench-style)

An agent that solves natural-language tasks by calling offline tools (calculator, retriever, calendar, weather, SQLite). Includes scaffolding for SFT → DPO → light PPO/GRPO with QLoRA on Qwen2.5-Coder-7B-Instruct.

## Quickstart

```bash
# 1) Setup
bash scripts/setup_env.sh

# 2) Prepare tool-use demo data
bash scripts/prepare_tool_data.sh

# 3) Run baseline on tool-use subset
bash scripts/run_baseline_tool.sh
```

## Repository Layout

- configs/: model/tool configs
- agent/: prompts, policy wrapper, memory, parsing helpers
- envs/tooluse/: dataset loader, tool registry, validators, harness, trajectory
- policy/: model loading & training scripts (SFT/DPO/PPO)
- eval/: evaluation runners and metrics
- scripts/: helper bash scripts
- data/: datasets, corpus, outputs

## Notes

- You will need Python 3.10+ and a GPU with ~24GB VRAM recommended (QLoRA).
- Tool-use is fully offline and deterministic for PPO rewards.


