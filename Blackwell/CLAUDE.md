# Claude Code Guide For The Blackwell Track

Use this file as the track-level instruction source for Claude Code. Shared skills are mirrored from `.agents/skills/` into `.claude/skills/` by `bash scripts/sync_skills.sh`.

## Primary Goal

Treat this as the Blackwell hackathon execution repo.

Run this ladder in order:

1. `BF16` or default KV baseline
2. `FP8` KV baseline
3. native `NVFP4` KV baseline
4. optional `LMCache` or raw host-path baseline
5. `NVFP4 + KVTC` tiering follow-up

## Core Constraints

- Blackwell is the real target.
- `NVFP4` is the hot active-KV format.
- `KVTC` is a warm or cold tier unless a hot-path use proves acceptable.
- Keep long-running work Slurm-safe and reproducible.
- Save benchmark artifacts in `results/` and batch logs in `logs/`.
- Optimize for latency and quality first, then memory savings.

## What To Read First

- `README.md`
- `blackwell_kv_hackathon_context.md`
- `PROMPT_BLACKWELL_NVFP4_KVTC.md`
- `BLACKWELL_24H_PRD.md`
- `B200_SLURM_EVAL_RUNBOOK.md`
- `QUICKSTART_PROMPTS.md`
- `scripts/check_cluster.sh`
- `scripts/check_gpu.sh`
- `scripts/baseline_inference.sh`
- `scripts/run_baseline.py`

## Shared Skills

Mirrored skills should exist in `.claude/skills/` after running the sync script:

- `slurm-hpc-executor`
- `kv-cache-research`
- `blackwell-b200-optimizer`
- `nvfp4-kvtc-runtime`
- `latency-quality-eval`
- `nsys-ncu-profiler`

## Subagents

Claude subagents live under `.claude/agents/`.

- `repo-explorer`
- `slurm-operator`
- `kv-cache-researcher`
- `blackwell-runtime-optimizer`
- `eval-guard`

## Guidance

- Start from the existing repo state instead of assuming hidden infrastructure.
- Prefer code changes over broad strategy notes.
- When the latest behavior matters, use official docs and upstream repos.
- Keep `LMCache` sources, `vLLM` sources, and `KVTC` or `NVFP4` sources separate.
- If you propose a new hot-path representation, prove the `p95` decode latency story.
- If you propose profiling, keep it scoped so an allocated GPU can actually run it.
- When updating shared skills, run `bash scripts/sync_skills.sh` so `.claude/skills/` stays in sync.
