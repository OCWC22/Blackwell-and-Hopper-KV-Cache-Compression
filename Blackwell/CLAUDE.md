# Claude Code Guide For The Blackwell Track

Use this file as the track-level instruction source for Claude Code. Shared skills are mirrored from `.agents/skills/` into `.claude/skills/` by `bash scripts/sync_skills.sh`.

## Primary Goal

Treat this as the Blackwell hackathon execution repo. TensorRT-LLM is the primary runtime. NVFP4 hot KV is the primary Blackwell thesis. vLLM + LMCache is the follow-up compatibility / productization path.

## Four Benchmark Scenarios

| Scenario | Question | Primary Metrics | scenario_id |
|----------|----------|----------------|-------------|
| **1** — Longer context, one GPU | How far can one GPU go in context length? | peak HBM, TTFT, max effective context | `scenario_1_longer_context_gpu` |
| **2** — More sessions, one GPU | At fixed context, how many concurrent sessions? | max sessions at p95, throughput, TTFT under reuse | `scenario_2_more_sessions_gpu` |
| **3** — Both, one GPU **(PRIMARY)** | Can one GPU serve many users with long prompts? | max sessions at large context, peak HBM, p95/p99 TPOT, TTFT on reused prefixes, quality delta | `scenario_3_longer_context_more_sessions_gpu` |
| **4** — Both, one node | Does the same idea improve serving at node level? | aggregate sessions/node, aggregate throughput, aggregate HBM | `scenario_4_longer_context_more_sessions_node` |

## Execution Order (11 steps)

1. **Environment probe** — `scripts/env_probe.sh` → `results/env_probe.json`
2. **Support gate** — determine TRT-LLM NVFP4/FP8 KV support from probe
3. **Stable JSON baseline harness** — verify `run_baseline.py` schema with `--engine tensorrt_llm`
4. **TRT-LLM BF16 / default baseline** — BF16 at 8k and 32k
5. **TRT-LLM FP8 baseline** — FP8 at 8k and 32k
6. **TRT-LLM NVFP4 hot-KV baseline** — NVFP4 at 8k and 32k
7. **TRT-LLM + secondary offload tier** — host memory offload with NVFP4 hot KV
8. **TRT-LLM + secondary offload + KVTC candidate** — KVTC compression on offloaded tier
9. **One promotion-policy ablation** — demand vs eager promotion
10. **One-node run** — only after single-GPU success (Scenario 4)
11. **Decision memo** — comparison table + bottleneck summary + recommendation

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
- **TensorRT-LLM is the primary hackathon runtime.** Do not start with vLLM + LMCache as the primary path.
- **NVFP4 is the primary Blackwell hot-tier thesis.** If NVFP4 is not supported in TRT-LLM, fall back to FP8.
- **Secondary memory offload** (host RAM) is the warm/cold tier. KVTC is the cold-tier codec candidate.
- **vLLM + LMCache** is the follow-up compatibility / productization path. Document it separately.
- **Kimi K2.5** is a stretch / node-level target only. Do not block the single-GPU proof on it.
- Keep long-running work Slurm-safe and reproducible.
- Save benchmark artifacts in `results/` and batch logs in `logs/`.
- Optimize for serving efficiency first (sessions, HBM, TTFT), then memory savings.
- **Do not start multi-node before single-GPU and one-node success.**

## Support Gate (Run Before Building Thesis)

Before implementing the tiered runtime, run `scripts/env_probe.sh` and verify:

1. **GPU model** — must contain B200 or B100 (reject if Hopper or older)
2. **Driver version** — >= 570.x for Blackwell support
3. **CUDA version** — >= 12.8
4. **TensorRT-LLM version** — must be installed
5. **ModelOpt version** — if used for quantization
6. **Hot-tier KV support** — determine which path is available:
   - **NVFP4 hot KV in TRT-LLM**: check if TRT-LLM supports `kv_cache_type` with NVFP4 or loads NVFP4-KV checkpoint
   - **FP8 hot KV in TRT-LLM**: check if TRT-LLM supports FP8 KV cache
   - **FP8 hot KV in vLLM**: fallback check for follow-up path
7. **Offload support** — whether TRT-LLM KV host offload is available

**Decision tree:**

| Gate result | Action |
|-------------|--------|
| TRT-LLM NVFP4 hot-KV supported | Run full ladder: BF16 → FP8 → NVFP4 → NVFP4+offload → +KVTC |
| TRT-LLM FP8 only | Run: BF16 → FP8 → FP8+offload. Report inability to validate NVFP4 |
| Neither supported in TRT-LLM | Try vLLM FP8 as fallback. Report env_probe.json |
| Nothing works | Stop. Report env_probe.json. Do not proceed |

If NVFP4 hot-KV is not supported in TRT-LLM, **do not block the hackathon**. Fall back to FP8 and preserve the serving-capacity experiment.

The probe result is written to `results/env_probe.json` and must exist before any benchmark runs.

## Blackwell KV Runtime: Core Thesis

This repo validates that on Blackwell/B200, an NVFP4 hot KV tier plus an offloaded/compressed secondary KV tier can let the same GPU or node serve more concurrent sessions, longer effective context, or both.

- **Tier 0 — Hot KV:** TensorRT-LLM NVFP4 KV cache (primary Blackwell thesis)
- **Tier 0 fallback:** TensorRT-LLM FP8 KV cache (if NVFP4 not supported)
- **Tier 1 — Secondary offload:** Host memory via TRT-LLM KV offload / eviction / reuse
- **Tier 1 compression:** KVTC codec candidate on the offloaded tier

**Follow-up compatibility path:** vLLM with FP8 KV cache + LMCache cold/warm reusable KV layer.

**Goal:** Improve serving economics via NVFP4 hot-tier efficiency and KV reuse/offload, not compression ratio alone.

Rules: do not replace the inference engine, do not start with vLLM + LMCache as primary, keep sink tokens and recent window protected, pay decode/reconstruction cost on promotion not every token.

See `TIERED_KV_ARCHITECTURE.md` for the full architectural specification.

## Execution Priority

Follow the 11-step execution order above. Do not skip steps. Do not run multi-node before single-GPU and one-node are stable. Do not start with Kimi K2.5. Do not start with vLLM + LMCache as the primary runtime path.

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
- Keep TRT-LLM sources, vLLM sources, LMCache sources, KVTC sources, and NVFP4 sources separate.
- If you propose a new hot-path representation, prove the p95 decode latency story.
- If you propose profiling, keep it scoped so an allocated GPU can actually run it.
- When updating shared skills, run `bash scripts/sync_skills.sh` so `.claude/skills/` stays in sync.
- Do not use KVTC paper as evidence for TRT-LLM behavior.
- Do not use NVIDIA NVFP4 docs as proof of vLLM hot-KV support.
