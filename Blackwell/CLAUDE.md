# Claude Code Guide For The Blackwell Track

Use this file as the track-level instruction source for Claude Code. Shared skills are mirrored from `.agents/skills/` into `.claude/skills/` by `bash scripts/sync_skills.sh`.

## Primary Goal

Treat this as the Blackwell hackathon execution repo.

## Four Benchmark Scenarios

| Scenario | Question | Primary Metrics | scenario_id |
|----------|----------|----------------|-------------|
| **1** — Longer context, one GPU | How far can one GPU go in context length? | peak HBM, TTFT, max effective context | `scenario_1_longer_context_gpu` |
| **2** — More sessions, one GPU | At fixed context, how many concurrent sessions? | max sessions at p95, throughput, TTFT under reuse | `scenario_2_more_sessions_gpu` |
| **3** — Both, one GPU **(PRIMARY)** | Can one GPU serve many users with long prompts? | max sessions at large context, peak HBM, p95/p99 TPOT, TTFT on reused prefixes, quality delta | `scenario_3_longer_context_more_sessions_gpu` |
| **4** — Both, one node | Does the same idea improve serving at node level? | aggregate sessions/node, aggregate throughput, aggregate HBM | `scenario_4_longer_context_more_sessions_node` |

## Execution Order (8 steps)

1. **Environment probe** — `scripts/env_probe.sh` → `results/env_probe.json`
2. **Support gate** — determine FP8/NVFP4 KV support from probe
3. **Stable JSON baseline harness** — verify `run_baseline.py` schema
4. **Aligned single-GPU baselines** — BF16, FP8 at 8k and 32k
5. **First vLLM + LMCache result** — `run_tiered_experiment.py` + `serve_and_bench.py` (Scenario 3)
6. **One policy ablation** — demand vs eager promotion
7. **One-node run** — only after single-GPU success (Scenario 4)
8. **Decision memo** — comparison table + bottleneck summary + recommendation

## Success Criteria (KPI-based)

Achieve at least one of:
- >=20% lower peak HBM vs best non-tiered baseline
- Materially better TTFT on repeated-prefix traffic
- >=25% more concurrent sessions at fixed p95 target
- Materially longer effective context at same HBM budget

While keeping:
- p95 TPOT regression <= 10% vs best non-tiered baseline
- p99 TPOT regression <= 15%
- Quality delta <= 1% vs bf16 baseline

## Core Constraints

- Blackwell is the real target.
- vLLM FP8 KV cache is the stable documented hot-tier path.
- NVFP4 is a Blackwell-aware optional enhancement — only use if runtime support is explicitly verified.
- `LMCache` is the cold/warm reusable KV layer. `KVTC` is a cold-tier codec candidate.
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

This repo validates that on Blackwell/B200, a hot/cold KV lifecycle can improve serving economics for reuse-heavy long-context inference. The hot decode path uses the fastest practical supported KV representation in the serving runtime, and the cold reusable tier stores stale prefixes for later restore. The metric of success is not compression ratio alone, but lower HBM pressure, more concurrent sessions, better TTFT under reuse, or longer effective context.

- **Tier 0 — Hot KV:** vLLM FP8 KV cache (stable documented path)
- **Tier 0b — Experimental Blackwell enhancement:** NVFP4-aware hot path only if runtime support is explicitly verified
- **Tier 1 — Cold / reusable KV:** LMCache-managed storage, host RAM first, optional compression via KVTC candidate path

**Goal:** Improve serving economics via KV reuse and better lifecycle management, not compression ratio alone.

Rules: do not replace the inference engine, do not block on undocumented NVFP4 hot-KV assumptions in vLLM, keep sink tokens and recent window protected, treat pre-RoPE vs post-RoPE as explicit research question, pay decode/reconstruction cost on promotion not every token.

See `TIERED_KV_ARCHITECTURE.md` for the full architectural specification.

## Execution Priority

Follow the 8-step execution order above. Do not skip steps. Do not run multi-node before single-GPU and one-node are stable. TensorRT-LLM / NVFP4 comparison only if time and support permit. Do not block on undocumented NVFP4 hot-KV assumptions in vLLM.

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
