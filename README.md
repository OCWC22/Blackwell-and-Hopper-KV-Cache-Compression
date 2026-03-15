# KV Cache Runtime Research

LLM serving at scale is bottlenecked by KV cache memory, not compute. A single 30B-parameter model at 64K context can consume 8-16 GB of KV cache per session in FP16, and with multiple concurrent users GPU HBM fills before the SMs are fully utilized. This repository builds and validates a tiered KV cache architecture on NVIDIA Blackwell B200 GPUs that addresses this by combining NVFP4 — Blackwell's native 4-bit floating-point format (E2M1 with FP8 E4M3 per-group scales) — as the hot KV tier on GPU, with host-memory offload as a warm tier for evicted blocks, and KVTC (KV-Cache Compression with Learned Transform Coding, ICLR 2026) as an optional cold-tier codec achieving 20x+ compression ratios. We measured the offload/restore roofline on B200 hardware (~57 GB/s sustained PCIe, sub-0.03ms per-block restore latency) and proved via both empirical benchmarks and first-principles FLOP analysis on H200 that loading evicted KV blocks from host memory is 48-5,222x faster than recomputing them — validating an offload-first policy for large MoE models. The benchmark infrastructure includes a 45-row eval matrix covering four scenarios (longer context, more sessions, both on one GPU, node-level), ten runtime variants (TensorRT-LLM BF16/FP8/NVFP4 with and without offload, tiered demand/eager promotion, vLLM FP8 with LMCache), and a complete harness pipeline for end-to-end evaluation of serving economics: concurrent sessions, peak HBM, TTFT on prefix reuse, throughput, and quality delta versus BF16 baseline. Key limitations: the recompute-vs-load crossover analysis was validated only for Qwen3-30B-A3B (128-expert MoE) on H200 in PyTorch eager mode with 32-token blocks — dense models, fused-kernel runtimes like TensorRT-LLM, and different block sizes may yield different ratios and should be measured independently.

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
