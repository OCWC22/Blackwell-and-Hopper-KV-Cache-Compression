# Blackwell KV Runtime Agent Guide

## Mission

Validate that a Blackwell-first TensorRT-LLM + NVFP4 + offload tiered KV runtime improves reuse-heavy long-context serving.

Primary measurable outcomes:
- lower peak HBM
- better TTFT on repeated-prefix traffic
- more concurrent sessions at fixed p95 target
- longer effective context

Primary runtime path:
- TensorRT-LLM as the hackathon inference engine
- NVFP4 hot KV cache on GPU
- Secondary memory offload (host RAM) via TRT-LLM KV reuse / eviction / offload
- KVTC as compression on the offloaded secondary tier

Follow-up compatibility / productization path:
- vLLM with FP8 KV cache as stable hot tier
- LMCache as cold/warm reusable KV layer

Runtime honesty rule:
- treat TRT-LLM + NVFP4 as the primary Blackwell hackathon path
- treat vLLM + LMCache as the follow-up compatibility path, not the main runtime
- treat NVFP4 hot KV as a support-gated capability (verify via env_probe before assuming)
- treat KVTC as a cold-tier codec candidate, not a same-day blocker
- treat Kimi K2.5 as a stretch / node-level target only

This repo is not building a new inference engine.
This repo is validating one concrete systems hypothesis: that NVFP4 hot KV + secondary offload improves serving capacity on Blackwell.

## Four Benchmark Scenarios

The four scenarios are the organizing structure for this repo. Scenario 3 is the primary target.

| Scenario | Question | KV Pressure | scenario_id |
|----------|----------|-------------|-------------|
| **1** — Longer context on one GPU | How far can one GPU go in context length? | KV bytes per session | `scenario_1_longer_context_gpu` |
| **2** — More sessions on one GPU | At fixed context, how many concurrent sessions? | Number of live KV replicas | `scenario_2_more_sessions_gpu` |
| **3** — Longer context + more sessions on one GPU | Can one GPU serve many users with long prompts? (PRIMARY) | KV scales with both context and sessions | `scenario_3_longer_context_more_sessions_gpu` |
| **4** — Longer context + more sessions on one node | Does the same idea improve serving at node level? | Node-level aggregate | `scenario_4_longer_context_more_sessions_node` |

**Repo focus:** Scenario 3 is the primary single-GPU target. Scenario 4 is the node-level follow-up. Scenarios 1 and 2 are explanatory baselines.

**Success = serving-facing KPI, not compression ratio.** At least one of: >=20% lower peak HBM, materially better TTFT on reused prefixes, >=25% more concurrent sessions at fixed p95, or materially longer effective context.

## Models

| Role | Model |
|------|-------|
| Primary | `Qwen/Qwen3-30B-A3B` (or `Qwen/Qwen3-32B`) |
| Smoke test | `Qwen/Qwen3-8B-Instruct` |
| Stretch / node-level only | `moonshotai/Kimi-K2.5` |

Do NOT make Kimi K2.5 the first runnable target. The public vLLM recipe says the provided Kimi-K2.5 serving configuration has been verified on 8x H200 GPUs. Try it only after the primary single-GPU proof is stable.

## Execution Ladder

Run this ladder in order. Do not skip steps.

1. **Environment probe** — run `scripts/env_probe.sh`, write `results/env_probe.json`
2. **Support gate** — determine TRT-LLM NVFP4/FP8 KV support from probe results
3. **Stable JSON baseline harness** — verify `run_baseline.py` emits correct schema with `--engine tensorrt_llm`
4. **TRT-LLM BF16 / default baseline** — BF16 at 8k and 32k (Scenarios 1 & 2)
5. **TRT-LLM FP8 baseline** — FP8 at 8k and 32k
6. **TRT-LLM NVFP4 hot-KV baseline** — NVFP4 at 8k and 32k
7. **TRT-LLM + secondary offload tier** — host memory offload with NVFP4 hot KV (Scenario 3)
8. **TRT-LLM + secondary offload + KVTC candidate** — KVTC compression on offloaded tier
9. **One promotion-policy ablation** — demand vs eager promotion
10. **One-node run** — only after single-GPU success (Scenario 4)
11. **Decision memo** — comparison table + bottleneck summary + recommendation

Each step answers a concrete question about serving capacity, not about compression.
Do not start with vLLM + LMCache as the primary runtime path.
Do not start with Kimi K2.5.

## Non-Negotiable Constraints

- Target Blackwell / B200 first.
- **TensorRT-LLM is the primary hackathon runtime.** NVFP4 hot KV is the primary thesis.
- **vLLM + LMCache is the follow-up path**, not the main runtime.
- KVTC is a cold-tier codec candidate, not the cold tier itself. Secondary memory offload via TRT-LLM is the cold tier.
- Kimi K2.5 is a stretch / node-level target only. Do not block the single-GPU proof on it.
- Prefer Slurm-safe execution. Keep long jobs off login nodes.
- Store machine-readable outputs under `results/` and job logs under `logs/`.
- Do not couple KVTC integration risk with tiering risk on day zero.
- Do not run multi-node before single-GPU and one-node success.

## Hardware Context

- NVFP4 is native on Blackwell SM120 and should be treated as a real hardware primitive, not an emulation.
- The interesting question is not "can we compress KV?" The interesting question is "can we reduce HBM pressure without giving the latency back during promotion?"
- If a proposal only improves storage ratio but regresses p95 decode latency, it loses.
- NVIDIA documents NVFP4 as a Blackwell-native format with KV cache support in TensorRT-LLM (`[R7]`, `[R8]`).

## Tiered Architecture

See `TIERED_KV_ARCHITECTURE.md` for the full specification organized into three buckets: upstream facts, repo hypotheses, and must-be-measured metrics.

## Final Runtime Architecture

Primary hackathon path:
- TensorRT-LLM as inference engine
- NVFP4 KV cache as hot tier on GPU
- Host memory offload via TRT-LLM KV reuse / eviction
- KVTC as compression candidate on offloaded tier

Follow-up compatibility path:
- vLLM as serving engine with FP8 KV cache
- LMCache as cold/warm reusable KV layer
- KVTC as codec candidate in LMCache

Compression path:
- KVTC is a candidate codec for the secondary offloaded tier
- Do not make perfect KVTC integration a blocker for the first serving-capacity result

## Source Of Truth Files

Read these first:

- `README.md`
- `TIERED_KV_ARCHITECTURE.md`
- `blackwell_kv_hackathon_context.md`
- `PROMPT_BLACKWELL_NVFP4_KVTC.md`
- `BLACKWELL_24H_PRD.md`
- `B200_SLURM_EVAL_RUNBOOK.md`
- `scripts/env_probe.sh`
- `scripts/run_baseline.py`
- `scripts/run_tiered_experiment.py`
- `scripts/compare_results.py`
- `scripts/baseline_single_gpu.sbatch`
- `scripts/baseline_one_node.sbatch`

## Shared Repo Skills

These repo-local skills live under `.agents/skills/`.

- `slurm-hpc-executor`: safe cluster usage, batch job patterns, environment handling
- `kv-cache-research`: experiment design, baselines, result schemas, and fair comparisons
- `blackwell-b200-optimizer`: Blackwell-specific performance guidance and hardware-aware tuning
- `nvfp4-kvtc-runtime`: hot-tier NVFP4 in TRT-LLM, KVTC on secondary tier, and promotion-path runtime design
- `latency-quality-eval`: decode latency, throughput, memory, and quality retention evaluation
- `nsys-ncu-profiler`: Nsight Systems and Nsight Compute profiling workflow
- `kvtc-codec`: KVTC calibration, PCA basis, compression ratios, entropy coding, secondary tier codec integration
- `modelopt-nvfp4-quantization`: ModelOpt quantization recipes, NVFP4 KV cache config, TensorRT-LLM export
- `promotion-policy`: tier promotion design, protected tokens, eager vs demand promotion, hit/miss logging
- `b200-architecture`: B200 hardware specs, ISA details, tensor core throughput, memory hierarchy, first-principles reasoning for KV cache
- `nvfp4-format-reference`: E2M1 bit layout, quantization and dequantization formulas, microscaling math, NVFP4 vs MXFP4 comparison, error analysis, limitations
- `kvtc-algorithm-reference`: PCA/SVD calibration procedure, DP bit allocation algorithm, entropy coding, protected token ablation, quality results, paper limitations

## Repo-Local Agent Roles

Codex role files live in `.codex/agents/`. Claude subagents live in `.claude/agents/`.

- `repo-explorer`: identify the next concrete repo change that shortens time to a real Blackwell run
- `slurm-operator`: make execution safe and reproducible on the cluster
- `kv-cache-researcher`: shape experiment plans, metrics, and ablations
- `blackwell-runtime-optimizer`: guide the TRT-LLM + NVFP4 + offload tiered KV runtime work
- `eval-guard`: keep latency and quality evaluation honest
- `kvtc-codec-engineer`: implement KVTC calibration, compression, decompression, and secondary tier codec integration
- `modelopt-quantizer`: run ModelOpt NVFP4-KV quantization and export checkpoints for TensorRT-LLM
- `promotion-policy-designer`: design and ablate tier promotion policies for TRT-LLM KV offload / reuse
- `b200-hardware-advisor`: ground KV cache decisions in B200 architecture first principles and ISA details

## Routing Guidance

- Use `slurm-hpc-executor` when the task touches `sbatch`, `srun`, environment setup, filesystem layout, or batch safety.
- Use `kv-cache-research` when the task is about benchmark design, ablation structure, or interpreting memory-versus-latency tradeoffs.
- Use `blackwell-b200-optimizer` when the task is about hardware-aware tuning, memory layout, or Blackwell-specific considerations.
- Use `nvfp4-kvtc-runtime` when the task is about hot-tier NVFP4 in TRT-LLM, KVTC, promotion policy, or tiering decisions.
- Use `latency-quality-eval` when the task is about benchmarking, quality retention, seeds, or result schema.
- Use `nsys-ncu-profiler` when the task is about tracing kernels, promotion overhead, or memory traffic.
- Use `kvtc-codec` when the task is about KVTC calibration, PCA basis computation, compression ratios, entropy coding, or secondary tier codec integration.
- Use `modelopt-nvfp4-quantization` when the task is about ModelOpt quantization, NVFP4 KV config, calibration datasets, or TensorRT-LLM checkpoint export.
- Use `promotion-policy` when the task is about tier promotion design, protected token policies, eager vs demand strategies, or hit/miss rate logging.
- Use `b200-architecture` when the task needs hardware specs, bandwidth budgets, KV cache sizing estimates, promotion cost calculations, or first-principles reasoning about B200 performance.
- Use `nvfp4-format-reference` when the task needs E2M1 representable values, quantization or dequantization formulas, microscaling math, NVFP4 vs MXFP4 first-principles comparison, or error analysis.
- Use `kvtc-algorithm-reference` when the task needs PCA calibration details, DP bit allocation algorithm math, compression ratio analysis, quality benchmarks from the paper, or understanding KVTC limitations.

## Research And Source Rules

- If the latest behavior matters, browse official docs and upstream repos.
- Prefer NVIDIA TRT-LLM docs, NVIDIA NVFP4 docs, vLLM docs, LMCache docs, OpenReview, and upstream GitHub sources over tertiary summaries.
- When a claim is likely to drive implementation, include the exact URL and retrieval date in the file you edit or reference.
- Do not cite the KVTC paper as evidence for TRT-LLM behavior. Keep compression sources and runtime sources separate.
- Do not cite NVIDIA NVFP4 docs as proof of vLLM hot-KV support.
- Be exact and technically honest.

## Working Style

- Prefer the smallest edit that gets us to a credible Blackwell run sooner.
- Keep CLI flags explicit and output formats stable.
- Be skeptical of any idea that does not name the exact baseline it must beat.
- Treat KVTC as a tiering primitive first, not automatically as the always-hot decode representation.
- Follow the B200 Slurm runbook when adding or changing Slurm entrypoints.

## What Good Progress Looks Like

- The repo can run reproducible BF16, FP8, and NVFP4 baselines on TensorRT-LLM.
- The repo can describe and evaluate a TRT-LLM NVFP4 + host offload tiered KV path.
- The repo can fall back to vLLM FP8 + LMCache as a follow-up comparison.
- Kimi K2.5 is documented as a stretch target but does not block the primary proof.
- Slurm scripts are safe to submit without manual patching.
- Results make reuse benefit, memory savings, and quality tradeoffs easy to compare.
