# KV Cache Runtime Research

LLM serving at scale is bottlenecked by KV cache memory, not compute. A single 30B-parameter model at 64K context can consume 8-16 GB of KV cache per session in FP16, and with multiple concurrent users GPU HBM fills before the SMs are fully utilized. This repository builds and validates a tiered KV cache architecture on NVIDIA Blackwell B200 GPUs that addresses this by combining NVFP4 — Blackwell's native 4-bit floating-point format (E2M1 with FP8 E4M3 per-group scales) — as the hot KV tier on GPU, with host-memory offload as a warm tier for evicted blocks, and KVTC (KV-Cache Compression with Learned Transform Coding, ICLR 2026) as an optional cold-tier codec achieving 20x+ compression ratios. The tiered policy depends on a critical cost asymmetry: loading an evicted KV block from host memory over PCIe is dramatically cheaper than recomputing it via a forward pass. We validated this with three complementary approaches on real hardware:

1. **B200 offload/restore roofline** (`docs/research/kv_block_loading_cost_model.md`, `results/kv_block_offload_restore_*.json`): measured all four transfer paths (GPU-to-host, host-to-GPU, GPU-to-disk, disk-to-GPU) across block sizes 1-8192 tokens in BF16/FP8/FP4 formats. PCIe saturates at ~57 GB/s above 256 tokens with sub-0.03ms per-block restore latency. FP8 cast overhead is 0.034ms round-trip versus FP4's 0.265ms, making FP8 the preferred format for frequently promoted blocks.

2. **H200 recompute-vs-load crossover** (`Blackwell/docs/research/recompute_vs_load_crossover_h200.md`, `Blackwell/scripts/sweep_recompute_cost.py`, `Blackwell/scripts/sweep_load_cost.py`): swept recompute cost across batch sizes B=1-1024 and context positions 32-131K tokens for Qwen3-30B-A3B on H200, simultaneously sweeping load cost across PCIe concurrency C=1-128. At typical serving conditions (B=8, ctx=8K), loading is 5,222x faster. Even at the best achievable recompute (B=1024, ctx=32: 2.3ms/block), loading still wins by 48x (0.048ms worst-case). The crossover matrix shows loading wins at every tested operating point.

3. **First-principles FLOP validation** (`Blackwell/docs/research/theoretical_flop_validation_recompute_vs_load.md`): computed theoretical recompute cost from model architecture (48 layers, 128 MoE experts, 3B active params/token, 182-1830 GFLOP per 32-token block depending on context position). Even at a physically impossible 100% MFU (1 PFLOP/s), recompute takes 0.18ms at minimum — still 3.8x slower than worst-case PCIe load. This confirms the gap is hardware-fundamental, not a framework artifact.

The benchmark infrastructure includes a 45-row eval matrix covering four scenarios (longer context, more sessions, both on one GPU, node-level), ten runtime variants (TensorRT-LLM BF16/FP8/NVFP4 with and without offload, tiered demand/eager promotion, vLLM FP8 with LMCache), and a complete harness pipeline for end-to-end evaluation of serving economics: concurrent sessions, peak HBM, TTFT on prefix reuse, throughput, and quality delta versus BF16 baseline.

**Limitations.** The "always offload, never recompute" policy was validated only for Qwen3-30B-A3B (a 128-expert MoE model) on H200 in PyTorch eager mode with 32-token blocks in the decode regime. This scoping matters for several reasons: (1) MoE models have high kernel launch overhead (~29 kernels/layer/token) that inflates recompute cost — dense models like LLaMA-70B or Mistral-7B would have lower recompute cost; (2) recompute was measured in PyTorch eager mode only, not with TensorRT-LLM fused kernels despite TRT-LLM being the claimed primary runtime; (3) the `sweep_recompute_cost.py` dtype map silently falls back to BF16 for FP8/NVFP4, so quantized recompute was never actually measured; (4) only 32-token blocks were tested (vLLM defaults to 16); (5) load cost assumes uncontested PCIe with pre-allocated pinned buffers — real multi-GPU serving has PCIe contention from other GPUs, NVMe, and NICs; (6) the analysis uses OOM as evidence for offload ("high-batch recompute is unreachable due to KV memory pressure, making offload even more necessary") which is circular — it proves the comparison is constrained, not that offload is universally better; (7) chunked prefill and full prefill workloads were not tested; (8) KVTC decompression cost, promotion policy overhead, and HBM fragmentation effects were not measured. For large MoE models on single-GPU H200/B200 with uncontested PCIe in decode workloads, always offload. For other configurations — dense models, fused-kernel runtimes, multi-GPU, prefill workloads — measure first.

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
