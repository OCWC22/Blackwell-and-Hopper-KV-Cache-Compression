# KV Cache Runtime Research

This repository is now split into two deliberate tracks:

## Tracks

- `Blackwell/`: the hackathon execution lane. This is where we validate a vLLM + LMCache compatible tiered KV runtime with FP8 KV cache as the stable hot tier, LMCache as the cold/warm reusable KV layer, and KVTC as a cold-tier codec candidate, optimizing serving economics (sessions, HBM, TTFT) and quality over a 24-hour window.
- `Hopper/`: the longer-horizon research lane. This is where we study Hopper `H100` and `H200` paths that emulate some of the Blackwell KV economics with FP4-like packed storage, direct FP8 reconstruction, and aggressive quality protection.

## Operating Principle

The two tracks are related, but they are not the same project:

- `Blackwell/` is the product-shaped demo path for the hackathon.
- `Hopper/` is the systems-research path and longer-term moat.

If you blur them together, the repo will start making false hardware assumptions and the experiments will stop being credible.

## Where To Start

- If the goal is a weekend deliverable, start in `Blackwell/`.
- If the goal is long-term R&D on deployed Hopper fleets, start in `Hopper/`.

Each track has its own:

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `QUICKSTART_PROMPTS.md`
- track-specific context brief
- `.agents/skills/*`
- `.codex/agents/*`
- `.claude/agents/*`

The repo also includes:

- `AGENT_HARNESS_BEST_PRACTICES.md`: best practices for Codex `AGENTS.md`, repo skills, Claude Code subagents, and handoff structure

## One-Time Setup Per Track

Run the skill sync script inside the track you are working in:

```bash
cd Blackwell
bash scripts/sync_skills.sh

# or

cd Hopper
bash scripts/sync_skills.sh
```

## Source Priorities

When behavior or hardware claims matter, prefer:

- official NVIDIA docs and blogs
- official vLLM docs
- official LMCache docs
- primary papers or OpenReview / arXiv links

## Primary References

- NVIDIA Blackwell NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA Blackwell KV cache optimization blog: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- NVIDIA H100: <https://www.nvidia.com/en-us/data-center/h100/>
- NVIDIA H200: <https://www.nvidia.com/en-us/data-center/h200/>
- NVIDIA Hopper architecture: <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- LMCache docs: <https://docs.lmcache.ai/>
- vLLM quantized KV cache docs: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- vLLM production-stack KV cache docs: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- KVTC paper: <https://arxiv.org/pdf/2511.01815>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
