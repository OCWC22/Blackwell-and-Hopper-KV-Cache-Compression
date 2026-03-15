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
- The hot active-KV format depends on stack support (NVFP4 if verified, FP8 as fallback).
- `KVTC` is a warm or cold tier unless a hot-path use proves acceptable.
- Keep long-running work Slurm-safe and reproducible.
- Save benchmark artifacts in `results/` and batch logs in `logs/`.
- Optimize for serving efficiency first (sessions, HBM, TTFT), then memory savings.

## Support Gate (Run Before Building Thesis)

Before implementing the tiered runtime, run `scripts/env_probe.sh` and verify:

1. **GPU model** — must contain B200 or B100 (reject if Hopper or older)
2. **Driver version** — >= 570.x for Blackwell support
3. **CUDA version** — >= 12.8
4. **Runtime version** — vLLM and/or TensorRT-LLM version
5. **Hot-tier KV support** — determine which path is available:
   - **NVFP4 hot KV**: check if vLLM accepts `kv_cache_dtype="nvfp4"` or TRT-LLM loads NVFP4-KV checkpoint
   - **FP8 hot KV**: check if vLLM accepts `kv_cache_dtype="fp8"` (baseline gate)
   - **Unsupported / unclear**: neither path works cleanly

**Decision tree:**

| Gate result | Action |
|-------------|--------|
| NVFP4 hot-KV supported | Run full ladder: BF16 → FP8 → NVFP4 → tiered |
| FP8 hot-KV only | Run: BF16 → FP8 → tiered with FP8 hot tier |
| Neither supported | Stop. Report env_probe.json. Do not proceed. |

If NVFP4 hot-KV is not clearly supported in the chosen stack, **do not block the hackathon**.
Pivot to the nearest stable hot-tier baseline and preserve the serving-capacity experiment.

NVIDIA clearly documents NVFP4 as a Blackwell-native format, but the KV-cache support story
is stack-dependent. Treat NVFP4 hot-KV as a support gate, not an assumed fact.

The probe result is written to `results/env_probe.json` and must exist before any benchmark runs.

## Blackwell KV Runtime: Core Thesis

We are not choosing between KVTC and NVFP4. We are building:

- **Tier 0** hot KV in NVFP4 on Blackwell (GPU HBM, active decode)
- **Tier 1** warm/cold reusable KV in KVTC format (host RAM / SSD / remote)
- **Promotion path:** KVTC decode → FP8 staging → NVFP4 repack → active cache

Success criteria: NVFP4 + KVTC beats raw offload or plain NVFP4 on HBM efficiency, TTFT/hit-rate, or concurrency — without breaking quality.

Rules: do not replace the inference engine, do not use KVTC as always-hot representation first, keep sink tokens and recent window protected, treat pre-RoPE vs post-RoPE as explicit research question, pay decode/reconstruction cost on promotion not every token.

See `TIERED_KV_ARCHITECTURE.md` for the full architectural specification.

## What To Read First

- `README.md`
- `TIERED_KV_ARCHITECTURE.md`
- `blackwell_kv_hackathon_context.md`
- `PROMPT_BLACKWELL_NVFP4_KVTC.md`
- `BLACKWELL_24H_PRD.md`
- `B200_SLURM_EVAL_RUNBOOK.md`
- `QUICKSTART_PROMPTS.md`
- `scripts/env_probe.sh`
- `scripts/run_baseline.py`
- `scripts/run_tiered_experiment.py`
- `scripts/compare_results.py`
- `scripts/baseline_single_gpu.sbatch`
- `scripts/baseline_one_node.sbatch`

## Shared Skills

Mirrored skills should exist in `.claude/skills/` after running the sync script:

- `slurm-hpc-executor`
- `kv-cache-research`
- `blackwell-b200-optimizer`
- `nvfp4-kvtc-runtime`
- `latency-quality-eval`
- `nsys-ncu-profiler`
- `kvtc-codec`
- `modelopt-nvfp4-quantization`
- `promotion-policy`
- `b200-architecture`
- `nvfp4-format-reference`
- `kvtc-algorithm-reference`

## Subagents

Claude subagents live under `.claude/agents/`.

- `repo-explorer`
- `slurm-operator`
- `kv-cache-researcher`
- `blackwell-runtime-optimizer`
- `eval-guard`
- `kvtc-codec-engineer`
- `modelopt-quantizer`
- `promotion-policy-designer`
- `b200-hardware-advisor`

## Guidance

- Start from the existing repo state instead of assuming hidden infrastructure.
- Prefer code changes over broad strategy notes.
- When the latest behavior matters, use official docs and upstream repos.
- Keep `LMCache` sources, `vLLM` sources, and `KVTC` or `NVFP4` sources separate.
- If you propose a new hot-path representation, prove the `p95` decode latency story.
- If you propose profiling, keep it scoped so an allocated GPU can actually run it.
- When updating shared skills, run `bash scripts/sync_skills.sh` so `.claude/skills/` stays in sync.
