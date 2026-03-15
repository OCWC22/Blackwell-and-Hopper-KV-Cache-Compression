# Claude Code Guide For The Hopper Track

Use this file as the track-level instruction source for Claude Code. Shared skills are mirrored from `.agents/skills/` into `.claude/skills/` by `bash scripts/sync_skills.sh`.

## Primary Goal

Treat this as the Hopper research repo.

Run this ladder in order:

1. `BF16` or default KV baseline
2. `FP8` KV baseline
3. packed FP4-like storage path
4. direct `FP8` reconstruction
5. quality-protection and grouping ablations
6. optional cold-tier integration after the hot path is real

## Core Constraints

- Hopper `H100` and `H200` are the real targets.
- FP4 is a storage format on Hopper, not native Hopper tensor-core compute.
- Compute should stay in `FP8`, `BF16`, or `FP16` after unpacking or dequantization.
- Keep long-running work Slurm-safe and reproducible.
- Save benchmark artifacts in `results/` and batch logs in `logs/`.
- Be explicit about the host-path bottleneck: PCIe Gen5 x16 is about `63 GB/s` one way, far below Hopper HBM bandwidth.

## What To Read First

- `README.md`
- `hopper_kv_compression_hackathon_context.md`
- `PROMPT_HOPPER_HFP4_RESEARCH.md`
- `HOPPER_RESEARCH_PRD.md`
- `QUICKSTART_PROMPTS.md`
- `scripts/check_cluster.sh`
- `scripts/check_gpu.sh`
- `scripts/baseline_inference.sh`
- `scripts/run_baseline.py`

## Shared Skills

Mirrored skills should exist in `.claude/skills/` after running the sync script:

- `slurm-hpc-executor`
- `kv-cache-research`
- `hopper-h100-h200-optimizer`
- `fp4-emulation-runtime`
- `latency-quality-eval`
- `nsys-ncu-profiler`

## Subagents

Claude subagents live under `.claude/agents/`.

- `repo-explorer`
- `slurm-operator`
- `kv-cache-researcher`
- `hopper-kernel-optimizer`
- `eval-guard`

## Guidance

- Start from the existing repo state instead of assuming hidden infrastructure.
- Prefer code changes over broad strategy notes.
- When the latest behavior matters, use official docs and upstream repos.
- Keep `LMCache` sources, `vLLM` sources, and FP4 or paper sources separate.
- If you mention FP4 on Hopper, frame it as emulation, packing, or storage unless code proves otherwise.
- If you propose profiling, keep it scoped so an allocated GPU can actually run it.
- When updating shared skills, run `bash scripts/sync_skills.sh` so `.claude/skills/` stays in sync.
