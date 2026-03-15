# Blackwell KV Runtime Agent Guide

## Mission

Validate that a Blackwell-first, vLLM + LMCache compatible tiered KV runtime improves reuse-heavy long-context serving.

Primary measurable outcomes:
- lower peak HBM
- better TTFT on repeated-prefix traffic
- more concurrent sessions at fixed p95 target
- longer effective context

Primary runtime path:
- vLLM hot KV path
- LMCache cold/warm reusable KV path

Runtime honesty rule:
- treat vLLM FP8 KV cache as the stable documented hot-tier path
- treat NVFP4 as a Blackwell-aware support-gated enhancement, not a guaranteed public vLLM hot-KV capability
- treat KVTC as a cold-tier codec candidate, not a same-day blocker

This repo is not building a new inference engine.
This repo is testing a KV-lifecycle controller around existing serving stacks (vLLM, LMCache).

## Execution Ladder

Run this ladder in order. Do not skip steps.

1. **Support gate** — run `scripts/env_probe.sh`, determine FP8/NVFP4 KV support
2. vLLM BF16 baseline
3. vLLM FP8 KV baseline
4. vLLM + LMCache cold-tier reuse
5. One promotion / reuse-policy ablation
6. One-node run only after single-GPU success

Each step answers a concrete question about serving capacity, not about compression.
Do not block on undocumented NVFP4 hot-KV assumptions in vLLM.

## Non-Negotiable Constraints

- Target Blackwell / B200 first.
- FP8 KV cache is the stable documented hot-tier path in vLLM.
- NVFP4 is a Blackwell-aware optional enhancement — only use if runtime support is explicitly verified.
- `KVTC` is a cold-tier codec candidate, not the cold tier itself. `LMCache` is the cold/warm tier software layer.
- vLLM is the serving runtime. LMCache is the KV reuse/offload layer.
- Prefer Slurm-safe execution. Keep long jobs out of login nodes.
- Store machine-readable outputs under `results/` and job logs under `logs/`.
- Do not block on undocumented NVFP4 hot-KV assumptions in vLLM.
- Do not couple KVTC integration risk with tiering risk on day zero.

## Hardware Context

- `NVFP4` is native on Blackwell SM120 and should be treated as a real hardware primitive, not an emulation.
- The interesting question is not "can we compress KV?" The interesting question is "can we reduce HBM pressure without giving the latency back during promotion?"
- If a proposal only improves storage ratio but regresses `p95` decode latency, it loses.
- NVIDIA documents NVFP4 as a Blackwell-native format, but KV-cache support is stack-dependent.

## Tiered Architecture

See `TIERED_KV_ARCHITECTURE.md` for the full specification organized into three buckets: upstream facts, repo hypotheses, and must-be-measured metrics.

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
- `nvfp4-kvtc-runtime`: hot-tier `NVFP4`, warm-tier `KVTC`, and promotion-path runtime design
- `latency-quality-eval`: decode latency, throughput, memory, and quality retention evaluation
- `nsys-ncu-profiler`: Nsight Systems and Nsight Compute profiling workflow
- `kvtc-codec`: KVTC calibration, PCA basis, compression ratios, entropy coding, LMCache codec integration
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
- `blackwell-runtime-optimizer`: guide the vLLM + LMCache tiered KV runtime work
- `eval-guard`: keep latency and quality evaluation honest
- `kvtc-codec-engineer`: implement KVTC calibration, compression, decompression, and LMCache serde integration
- `modelopt-quantizer`: run ModelOpt NVFP4-KV quantization and export checkpoints for TensorRT-LLM
- `promotion-policy-designer`: design and ablate tier promotion policies for NVFP4 plus KVTC
- `b200-hardware-advisor`: ground KV cache decisions in B200 architecture first principles and ISA details

## Routing Guidance

- Use `slurm-hpc-executor` when the task touches `sbatch`, `srun`, environment setup, filesystem layout, or batch safety.
- Use `kv-cache-research` when the task is about benchmark design, ablation structure, or interpreting memory-versus-latency tradeoffs.
- Use `blackwell-b200-optimizer` when the task is about hardware-aware tuning, memory layout, or Blackwell-specific considerations.
- Use `nvfp4-kvtc-runtime` when the task is about hot-tier `NVFP4`, `KVTC`, promotion policy, or tiering decisions.
- Use `latency-quality-eval` when the task is about benchmarking, quality retention, seeds, or result schema.
- Use `nsys-ncu-profiler` when the task is about tracing kernels, promotion overhead, or memory traffic.
- Use `kvtc-codec` when the task is about KVTC calibration, PCA basis computation, compression ratios, entropy coding, or LMCache serde integration.
- Use `modelopt-nvfp4-quantization` when the task is about ModelOpt quantization, NVFP4 KV config, calibration datasets, or TensorRT-LLM checkpoint export.
- Use `promotion-policy` when the task is about tier promotion design, protected token policies, eager vs demand strategies, or hit/miss rate logging.
- Use `b200-architecture` when the task needs hardware specs, bandwidth budgets, KV cache sizing estimates, promotion cost calculations, or first-principles reasoning about B200 performance.
- Use `nvfp4-format-reference` when the task needs E2M1 representable values, quantization or dequantization formulas, microscaling math, NVFP4 vs MXFP4 first-principles comparison, or error analysis.
- Use `kvtc-algorithm-reference` when the task needs PCA calibration details, DP bit allocation algorithm math, compression ratio analysis, quality benchmarks from the paper, or understanding KVTC limitations.

## Research And Source Rules

- If the latest behavior matters, browse official docs and upstream repos.
- Prefer NVIDIA, vLLM, LMCache, OpenReview, and upstream GitHub sources over tertiary summaries.
- When a claim is likely to drive implementation, include the exact URL and retrieval date in the file you edit or reference.
- Do not cite the `KVTC` paper as evidence for `LMCache` or `vLLM` behavior. Keep compression sources and runtime sources separate.

## Working Style

- Prefer the smallest edit that gets us to a credible Blackwell run sooner.
- Keep CLI flags explicit and output formats stable.
- Be skeptical of any idea that does not name the exact baseline it must beat.
- Treat `KVTC` as a tiering primitive first, not automatically as the always-hot decode representation.
- Follow the multi-node `B200` runbook when adding or changing Slurm entrypoints.

## What Good Progress Looks Like

- The repo can run reproducible `BF16` and `FP8` baselines on vLLM.
- The repo can describe and evaluate a vLLM + LMCache tiered KV reuse path.
- NVFP4 baselines are included only if runtime support is verified.
- Slurm scripts are safe to submit without manual patching.
- Results make reuse benefit, memory savings, and quality tradeoffs easy to compare.
