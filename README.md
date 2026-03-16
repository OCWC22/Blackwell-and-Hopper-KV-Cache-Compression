# KV Cache Runtime Research

LLM serving at scale is bottlenecked by KV cache memory, not compute. This repository builds and validates a tiered KV cache architecture on NVIDIA Blackwell B200 GPUs that addresses this by combining NVFP4 — Blackwell's native 4-bit floating-point format (E2M1 with FP8 E4M3 per-group scales) — as the hot KV tier on GPU, with host-memory offload as a warm tier for evicted blocks, and KVTC (KV-Cache Compression with Learned Transform Coding, ICLR 2026) as an optional cold-tier codec achieving 20x+ compression ratios. The tiered policy depends on a critical cost asymmetry: loading an evicted KV block from host memory over PCIe is dramatically cheaper than recomputing it via a forward pass. We validated this with three complementary approaches on real hardware:

1. **B200 offload/restore roofline** (`docs/research/kv_block_loading_cost_model.md`, `results/kv_block_offload_restore_*.json`): measured all four transfer paths (GPU-to-host, host-to-GPU, GPU-to-disk, disk-to-GPU) across block sizes 1-8192 tokens in BF16/FP8/FP4 formats. PCIe saturates at ~57 GB/s above 256 tokens with sub-0.03ms per-block restore latency. FP8 cast overhead is 0.034ms round-trip versus FP4's 0.265ms, making FP8 the preferred format for frequently promoted blocks.

2. **H200 recompute-vs-load crossover** (`Blackwell/docs/research/recompute_vs_load_crossover_h200.md`, `Blackwell/scripts/sweep_recompute_cost.py`, `Blackwell/scripts/sweep_load_cost.py`): swept recompute cost across batch sizes B=1-1024 and context positions 32-131K tokens for Qwen3-30B-A3B on H200, simultaneously sweeping load cost across PCIe concurrency C=1-128. At typical serving conditions (B=8, ctx=8K), loading is 5,222x faster. Even at the best achievable recompute (B=1024, ctx=32: 2.3ms/block), loading still wins by 48x (0.048ms worst-case). The crossover matrix shows loading wins at every tested operating point.

3. **First-principles FLOP validation** (`Blackwell/docs/research/theoretical_flop_validation_recompute_vs_load.md`): computed theoretical recompute cost from model architecture (48 layers, 128 MoE experts, 3B active params/token, 182-1830 GFLOP per 32-token block depending on context position). Even at a physically impossible 100% MFU (1 PFLOP/s), recompute takes 0.18ms at minimum — still 3.8x slower than worst-case PCIe load. This confirms the gap is hardware-fundamental, not a framework artifact.

The benchmark infrastructure includes a 45-row eval matrix covering four scenarios (longer context, more sessions, both on one GPU, node-level), ten runtime variants (TensorRT-LLM BF16/FP8/NVFP4 with and without offload, tiered demand/eager promotion, vLLM FP8 with LMCache), and a complete harness pipeline for end-to-end evaluation of serving economics: concurrent sessions, peak HBM, TTFT on prefix reuse, throughput, and quality delta versus BF16 baseline.

**Limitations.** The "always offload, never recompute" policy was validated only for Qwen3-30B-A3B (a 128-expert MoE model) on H200 in PyTorch eager mode with 32-token blocks in the decode regime. This scoping matters for several reasons: (1) MoE models have high kernel launch overhead (~29 kernels/layer/token) that inflates recompute cost — dense models like LLaMA-70B or Mistral-7B would have lower recompute cost; (2) recompute was measured in PyTorch eager mode only, not with TensorRT-LLM fused kernels despite TRT-LLM being the claimed primary runtime; (3) the `sweep_recompute_cost.py` dtype map silently falls back to BF16 for FP8/NVFP4, so quantized recompute was never actually measured; (4) only 32-token blocks were tested (vLLM defaults to 16); (5) load cost assumes uncontested PCIe with pre-allocated pinned buffers — real multi-GPU serving has PCIe contention from other GPUs, NVMe, and NICs; (6) the analysis uses OOM as evidence for offload ("high-batch recompute is unreachable due to KV memory pressure, making offload even more necessary") which is circular — it proves the comparison is constrained, not that offload is universally better; (7) chunked prefill and full prefill workloads were not tested; (8) KVTC decompression cost, promotion policy overhead, and HBM fragmentation effects were not measured. For large MoE models on single-GPU H200/B200 with uncontested PCIe in decode workloads, always offload. For other configurations — dense models, fused-kernel runtimes, multi-GPU, prefill workloads — measure first.

**Why Recomputation Can Be Competitive.** Despite the findings above, recomputation has become a viable eviction strategy in modern inference systems for several reasons:

1. **vLLM V1 defaults to RECOMPUTE over SWAP** — the default preemption mode in vLLM V1 is recomputation because it has lower overhead in the V1 architecture [[vLLM Optimization Docs](https://docs.vllm.ai/en/stable/configuration/optimization/)].

2. **Compute vs. I/O bandwidth crossover** — the "Cake" paper (arXiv 2410.03065) demonstrates that computing KV cache can be faster than loading from storage in many scenarios. Loading a 25 GB KV cache (10K tokens for Llama3-70B) takes ~8 seconds even with optimal NVMe bandwidth (~3 GB/s), while "computing KV cache with 100% A100 GPU power is comparable to loading from SSD or network" [[arXiv:2410.03065](https://arxiv.org/html/2410.03065v1)].

3. **Quadratic vs. linear scaling** — recompute cost grows quadratically with sequence length (attention mechanism), while load cost grows linearly with KV cache size. This makes recomputation competitive for short-medium sequences, while caching wins for very long contexts [[arXiv:2410.03065 §3](https://arxiv.org/html/2410.03065v1)].

4. **Storage tier speed is critical** — "if the storage tier is too slow, the overhead of transferring KV data back to the GPU may negate the benefits" [[BentoML KV Cache Offloading](https://bentoml.com/llm/inference-optimization/kv-cache-offloading)]. The transfer cost must be lower than recomputing from scratch.

5. **Modern GPU compute abundance** — with A100/H100/H200/B200 GPUs, compute capacity often exceeds I/O bandwidth for many workloads, making recomputation increasingly competitive.

**Why KV Cache Offload Remains Critical Despite Hardware Advances.** Even with Blackwell Ultra's 288 GB HBM3e (3.6x H100 capacity) and NVIDIA's continued memory scaling, the memory bottleneck persists for fundamental structural reasons:

1. **Compute-memory scaling gap** — AI chip compute power grew 80x over the last decade while memory bandwidth grew only 17x, creating a 4.7x differential. "Most LLM inference workloads [are] memory bandwidth-bound rather than compute bound" [[SemiAnalysis HBM Roadmap](https://newsletter.semianalysis.com/p/scaling-the-memory-wall-the-rise-and-roadmap-of-hbm)]. Current GPUs/TPUs were designed for training workloads, leaving inference "starved for memory bandwidth while compute units sit idle" [[WinBuzzer Memory Bottleneck](https://winbuzzer.com/2026/01/26/memory-bottleneck-llm-inference-hardware-challenge-xcxwbn/)].

2. **HBM supply-demand imbalance** — TrendForce estimates HBM demand surged 130%+ YoY in 2025 with 70%+ growth expected in 2026, straining supply and driving server DRAM prices up 50% [[WinBuzzer](https://winbuzzer.com/2026/01/26/memory-bottleneck-llm-inference-hardware-challenge-xcxwbn/)]. "HBM is getting more expensive per GB and per GB/s over time while standard DDR keeps getting cheaper" — a pricing inversion that favors tiered offload architectures.

3. **Context window expansion outpaces HBM** — Production models now support 1M+ token contexts (Claude Sonnet 4: 1M [[Anthropic](https://claude.com/blog/1m-context)], Qwen2.5-1M [[Qwen](https://qwenlm.github.io/blog/qwen2.5-1m/)], Gemini models with 1M+ [[Google](https://ai.google.dev/gemini-api/docs/long-context)]). KV cache size is model-dependent: for example, Llama 3.1-405B requires ~61 GB for a 128K-token context [[VAST Dynamo](https://www.vastdata.com/blog/nvidia-dynamo-vast-scalable-optimized-inference)]. Even Blackwell Ultra's 288 GB HBM3e supports only a limited number of concurrent long-context sessions before accounting for model weights.

4. **Reasoning models explode memory pressure** — "As models improve, they have increased in horizon lengths... Deep Research from OpenAI can think for tens of minutes at a time, while GPT-4 mustered mere tens of seconds. As models can now think and reason over a long period of time, the pressure on memory capacity explodes as context lengths regularly exceed hundreds of thousands of tokens" [[SemiAnalysis HBM](https://newsletter.semianalysis.com/p/scaling-the-memory-wall-the-rise-and-roadmap-of-hbm)].

5. **NVIDIA Dynamo validates tiered offload** — NVIDIA's production inference framework implements a 4-tier KV cache hierarchy: G1 (GPU HBM hot tier), G2 (CPU memory warm tier), G3 (local/pooled SSD cold tier), G4 (shared network storage for persistence) [[VAST + Dynamo](https://www.vastdata.com/blog/nvidia-dynamo-vast-scalable-optimized-inference)]. This architecture "allows Dynamo to handle memory pressure gracefully" and confirms that offload is not a workaround but a core design principle.

6. **Capital efficiency** — "Organizations often buy GPUs for memory aggregation rather than compute power. This leads to expensive GPUs sitting idle while being used primarily for their VRAM" [[WinBuzzer](https://winbuzzer.com/2026/01/26/memory-bottleneck-llm-inference-hardware-challenge-xcxwbn/)]. Tiered offload decouples memory capacity from GPU count, improving utilization.

7. **KV cache offload is already production-standard** — "Today, KVCache offloading is already commonly used. Nvidia has a framework for this called Dynamo Distributed KVCache Manager... A well-optimized system keeps all currently used KVs in HBM, infrequently used KV in DDR, and very rarely used KV in NVMe" [[SemiAnalysis](https://newsletter.semianalysis.com/p/scaling-the-memory-wall-the-rise-and-roadmap-of-hbm)].

**Implication for this research:** The tiered KV cache architecture (NVFP4 hot tier + host memory warm tier + KVTC cold tier) addresses a structural bottleneck that hardware scaling alone cannot solve. Even with Blackwell Ultra's 288 GB HBM3e [[NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)] — or B200's 180 GB HBM3E [[NVIDIA HGX B200](https://developer.nvidia.com/blog/nvidia-hgx-b200-reduces-embodied-carbon-emissions-intensity/)] — the combination of longer contexts, reasoning models, multi-turn agentic conversations, and HBM supply constraints ensures that offload remains economically and technically necessary.

**Further Research Required.** The recomputation vs. offload tradeoff needs additional investigation for future inference workloads:

1. **Prefill vs. decode crossover analysis** — the current measurements focus on decode regime. Prefill workloads have different compute/memory characteristics and may shift the crossover point significantly.

2. **Dense model recompute cost** — MoE models have high kernel launch overhead. Dense models (LLaMA-70B, Mistral-7B) need separate measurement to establish their crossover points.

3. **Fused-kernel runtime impact** — TensorRT-LLM and other fused-kernel runtimes reduce kernel launch overhead, potentially making recompute more competitive. This was not measured.

4. **Multi-GPU PCIe contention** — real serving systems have multiple GPUs, NVMe, and NICs competing for PCIe bandwidth. The uncontested PCIe assumption needs validation.

5. **Chunked prefill scheduling** — vLLM's chunked prefill interleaves compute-bound prefill with memory-bound decode. How does this interact with eviction/recompute decisions?

6. **Dynamic compute-I/O hybrid** — the "Cake" approach of simultaneously computing and loading KV cache [[arXiv:2410.03065](https://arxiv.org/html/2410.03065v1)] could be integrated with tiered KV cache systems.

7. **Attention sink awareness** — a user-reported concern notes that vLLM's eviction policy may evict attention sink tokens, potentially causing quality degradation [[vLLM Issue #36311](https://github.com/vllm-project/vllm/issues/36311)]. Smart eviction that preserves sinks while recomputing less important tokens needs study.

8. **SGLang HiCache hierarchical policies** — SGLang's HiCache achieves 6× throughput improvement with hierarchical GPU/CPU/disk caching [[SGLang Blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/)]. Integration with NVFP4/KVTC compression needs investigation.

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
