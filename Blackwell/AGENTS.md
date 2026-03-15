# Blackwell KV Runtime Agent Guide

This directory is the 24-hour Blackwell hackathon harness.

The immediate job is to validate a concrete runtime thesis:

- hot KV in native `NVFP4`
- warm or cold KV in `KVTC`
- low-latency promotion back into the active decode path
- quality protection that keeps the idea honest

## Mission

Run this ladder in order:

1. `BF16` or default KV baseline
2. `FP8` KV baseline
3. native `NVFP4` KV baseline
4. optional `LMCache` or raw host-path baseline
5. `NVFP4 + KVTC` tiering path
6. promotion and protection ablations

If the runtime cannot defend itself against `FP8` or native `NVFP4`, it is not ready.

## Non-Negotiable Assumptions

- Target Blackwell first.
- `NVFP4` is the hot or active-KV format.
- `KVTC` is a warm or cold representation unless hot-path latency proves acceptable.
- `vLLM` remains the active serving baseline.
- `LMCache` remains the offload or storage baseline when relevant.
- Prefer Slurm-safe execution. Keep long jobs out of login nodes.
- Store machine-readable outputs under `results/` and job logs under `logs/`.

## Hardware Facts That Must Influence Decisions

- `NVFP4` is native on Blackwell and should be treated as a real hardware primitive, not an emulation.
- The interesting question is not "can we compress KV?" The interesting question is "can we reduce HBM pressure without giving the latency back during promotion?"
- If a proposal only improves storage ratio but regresses `p95` decode latency, it loses.

## Tiered Architecture

We are not choosing between KVTC and NVFP4. We are building a two-tier system:

- **Tier 0 (GPU HBM):** NVFP4 active KV — Blackwell-native, decode consumes from here
- **Tier 1 (host/SSD/remote):** KVTC bitstream — warm/cold reusable KV at 20×+ compression
- **Promotion:** KVTC decode → FP8 staging → NVFP4 repack → active blocks (cost paid once on reuse)
- **Protection:** 4 attention sink tokens + 128 recent tokens always uncompressed

See `TIERED_KV_ARCHITECTURE.md` for full specification including ModelOpt recipe, KVTC calibration workflow, promotion policies, and success criteria.

## Source Of Truth Files

Read these first:

- `README.md`
- `TIERED_KV_ARCHITECTURE.md`
- `blackwell_kv_hackathon_context.md`
- `PROMPT_BLACKWELL_NVFP4_KVTC.md`
- `BLACKWELL_24H_PRD.md`
- `B200_SLURM_EVAL_RUNBOOK.md`
- `scripts/check_cluster.sh`
- `scripts/check_gpu.sh`
- `scripts/baseline_inference.sh`
- `scripts/run_baseline.py`

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

## Repo-Local Agent Roles

Codex role files live in `.codex/agents/`. Claude subagents live in `.claude/agents/`.

- `repo-explorer`: identify the next concrete repo change that shortens time to a real Blackwell run
- `slurm-operator`: make execution safe and reproducible on the cluster
- `kv-cache-researcher`: shape experiment plans, metrics, and ablations
- `blackwell-runtime-optimizer`: guide the `NVFP4 + KVTC` runtime work
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

- The repo can run reproducible `BF16`, `FP8`, and `NVFP4` baselines.
- The repo can describe and evaluate an `NVFP4 + KVTC` tiering path.
- Slurm scripts are safe to submit without manual patching.
- Results make promotion cost, memory savings, and quality tradeoffs easy to compare.
