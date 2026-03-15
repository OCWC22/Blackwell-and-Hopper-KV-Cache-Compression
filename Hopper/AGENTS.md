# Hopper KV Runtime Research Agent Guide

This directory is the Hopper research harness.

The immediate job is not "pretend Hopper is Blackwell." The immediate job is to determine whether a Hopper-compatible packed FP4-like KV path can keep enough of the Blackwell memory and quality economics to matter in production fleets.

## Mission

Run this sequence in order:

1. `BF16` or default KV baseline on a single `H100` or `H200`
2. `FP8` KV baseline on the same model and prompts
3. first packed FP4-like storage path
4. direct `FP8` reconstruction path
5. quality-protection ablations
6. optional `LMCache` cold-tier integration after the hot decode path is understood

If the active decode path is still unclear, offload integration is premature.

## Non-Negotiable Assumptions

- Target NVIDIA Hopper `H100` and `H200` first.
- Treat FP4 as a storage and transport format on Hopper. Do not claim native Hopper FP4 or `NVFP4` compute.
- Use `FP8`, `BF16`, or `FP16` for compute after reconstruction unless the code clearly proves another path.
- Prefer Slurm-safe execution. Keep long jobs out of login nodes.
- Store machine-readable outputs under `results/` and job logs under `logs/`.
- If an optimization is speculative or hardware-conditional, label it clearly and gate it behind a fallback.

## Hardware Facts That Must Influence Decisions

- H100 SXM HBM bandwidth is about `3.35 TB/s`.
- H200 HBM3e bandwidth is about `4.8 TB/s`.
- PCIe Gen5 x16 is only about `63 GB/s` theoretical per direction, even though the bidirectional headline is about `128 GB/s`.

Practical implication:

- refill traffic from host memory is tens of times slower than HBM access
- decode-path wins require low dequant overhead, not just small bytes on paper
- compression work should be justified by active-path economics, not by elegance

## Source Of Truth Files

Read these first:

- `README.md`
- `hopper_kv_compression_hackathon_context.md`
- `PROMPT_HOPPER_HFP4_RESEARCH.md`
- `HOPPER_RESEARCH_PRD.md`
- `scripts/check_cluster.sh`
- `scripts/check_gpu.sh`
- `scripts/baseline_inference.sh`
- `scripts/run_baseline.py`

## Shared Repo Skills

These repo-local skills live under `.agents/skills/`.

- `slurm-hpc-executor`: safe cluster usage, batch job patterns, environment handling
- `kv-cache-research`: experiment design, ablations, result schemas, and fair comparisons
- `hopper-h100-h200-optimizer`: Hopper-specific performance guidance without Blackwell-only assumptions
- `fp4-emulation-runtime`: FP4-like packing and reconstruction on Hopper with higher-precision compute
- `latency-quality-eval`: decode latency, throughput, memory, and quality retention evaluation
- `nsys-ncu-profiler`: Nsight Systems and Nsight Compute profiling workflow

## Repo-Local Agent Roles

Codex role files live in `.codex/agents/`. Claude subagents live in `.claude/agents/`.

- `repo-explorer`: identify the next concrete repo change that shortens time to a real research run
- `slurm-operator`: make execution safe and reproducible on the cluster
- `kv-cache-researcher`: shape experiment plans, metrics, and ablations
- `hopper-kernel-optimizer`: guide Hopper-specific runtime work once baseline data exists
- `eval-guard`: keep latency and quality evaluation honest

## Routing Guidance

- Use `slurm-hpc-executor` when the task touches `sbatch`, `srun`, environment setup, filesystem layout, or batch safety.
- Use `kv-cache-research` when the task is about benchmark design, ablation structure, or interpreting tradeoffs.
- Use `hopper-h100-h200-optimizer` when the task is about memory layout, bottlenecks, vectorization, TMA overlap, or Hopper limits.
- Use `fp4-emulation-runtime` when the task is about packed FP4-like storage, direct `FP8` reconstruction, or scale design.
- Use `latency-quality-eval` when the task is about benchmarking, quality retention, seeds, or report schema.
- Use `nsys-ncu-profiler` when the task is about tracing kernels, dequant overhead, or memory traffic.

## Research And Source Rules

- If the latest behavior matters, browse official docs and upstream repos.
- Prefer NVIDIA, LMCache, vLLM, PCI-SIG, OpenReview, arXiv, and upstream GitHub sources over blog summaries.
- When a claim is likely to drive implementation, include the exact URL and date in the file you edit or reference.
- Do not cite the `KVTC` paper as evidence for `LMCache` or `vLLM` behavior. Keep compression sources and runtime sources separate.

## Working Style

- Prefer the smallest edit that gets us to a real run sooner.
- When adding experiments, keep CLI flags explicit and output formats stable.
- If the repo is missing a piece, add the missing file instead of writing a long plan without code.
- Be skeptical of any proposal that cannot name the exact baseline it will beat.
- Keep hardware claims precise. Hopper supports `FP8` well; native `NVFP4` compute claims are wrong here.

## What Good Progress Looks Like

- The repo can run reproducible `BF16` and `FP8` baselines.
- The repo can measure a packed FP4-like path against those baselines.
- Slurm scripts are safe to submit without manual patching.
- Results make decode latency, memory footprint, and quality tradeoffs easy to compare.
