# Blackwell KV Runtime

This directory is the hackathon execution lane.

This repo validates that on Blackwell/B200, a hot/cold KV lifecycle can improve serving economics for reuse-heavy long-context inference. The hot decode path uses the fastest practical supported KV representation in the serving runtime, and the cold reusable tier stores stale prefixes for later restore. The metric of success is not compression ratio alone, but lower HBM pressure, more concurrent sessions, better TTFT under reuse, or longer effective context.

The architecture:

1. **Hot tier:** vLLM FP8 KV cache (stable documented path), with optional NVFP4 Blackwell enhancement if runtime support is verified
2. **Cold / reusable tier:** LMCache-managed storage (host RAM first), with optional KVTC compression candidate
3. **Promotion path:** restore cold KV to hot tier only on reuse
4. **Goal:** improve serving economics via KV reuse and lifecycle management

We are not replacing `vLLM` or `LMCache` — they are the runtime and the reuse layer respectively.

## 24-Hour Validation Ladder

Run the work in this order:

1. vLLM BF16 baseline
2. vLLM FP8 KV baseline
3. vLLM + LMCache cold-tier reuse
4. one promotion / reuse-policy ablation (demand vs eager)
5. one-node run only after single-GPU success

Do not block on undocumented NVFP4 hot-KV assumptions in vLLM.

If we cannot beat or match the practical baselines on latency and quality, there is no reason to keep going.

## Why Blackwell First

This is the highest-leverage path for the weekend because:

- Blackwell/B200 has the HBM bandwidth and capacity to make KV reuse economically interesting
- vLLM FP8 KV cache is a stable, documented hot-tier path
- LMCache integrates with vLLM and supports KV lookup/inject/offload/sharing
- NVFP4 is a Blackwell-native format that could provide additional hot-tier efficiency if runtime support is verified

The weekend work focuses on proving the vLLM + LMCache tiered lifecycle, not on undocumented NVFP4 assumptions.

## What Success Looks Like

One defendable sentence:

> On the same B200 hardware, our Blackwell-first vLLM + LMCache tiered KV setup served more reuse-heavy long-context traffic by reducing repeated prefill work and/or lowering peak KV memory pressure, while keeping latency and quality within acceptable bounds.

Concretely:
- vLLM BF16 and FP8 baselines run cleanly
- vLLM + LMCache cold-tier reuse shows measurable TTFT improvement or HBM reduction
- results capture p50/p95 decode latency, throughput, HBM footprint, cache hit rate, and quality deltas
- the repo is ready for a one-shot agent handoff to Codex or Claude Code

## AI Harness

This track includes a Codex and Claude Code harness tuned for the Blackwell hackathon path.

- `AGENTS.md`: track-level guidance for Codex and AGENTS-aware tools
- `CLAUDE.md`: track-level guidance for Claude Code
- `.agents/skills/*`: shared Blackwell skills
- `.codex/agents/*`: Codex role definitions
- `.claude/agents/*`: Claude subagents
- `QUICKSTART_PROMPTS.md`: one-shot prompts for the Blackwell track
- `blackwell_kv_hackathon_context.md`: execution brief, validation ladder, and sources
- `PROMPT_BLACKWELL_NVFP4_KVTC.md`: onboarding prompt for engineers and coding agents
- `BLACKWELL_24H_PRD.md`: structured task list for the weekend
- `B200_SLURM_EVAL_RUNBOOK.md`: exact Slurm handoff for multi-node `B200` evals
- `scripts/sync_skills.sh`: mirrors shared skills into `.claude/skills/`

Run this once after pulling changes:

```bash
bash scripts/sync_skills.sh
```

## Quick Start

```bash
# 1. Sync mirrored skills for Claude Code
bash scripts/sync_skills.sh

# 2. Verify the environment or cluster assumptions
bash scripts/check_cluster.sh

# 3. Prepare the environment
bash scripts/setup_env.sh

# 4. Validate one allocated GPU
srun --gpus=1 --time=00:10:00 --pty bash
bash scripts/check_gpu.sh
exit

# 5. Run the baseline harness
sbatch scripts/baseline_inference.sh

# 6. Read the Blackwell-specific brief
sed -n '1,220p' blackwell_kv_hackathon_context.md
```

## Repo Map

```text
README.md                              High-level Blackwell plan
AGENTS.md                              Codex and AGENTS-aware operating rules
CLAUDE.md                              Claude Code operating rules
QUICKSTART_PROMPTS.md                  One-shot prompts for concrete work
blackwell_kv_hackathon_context.md      Blackwell execution brief and source notes
PROMPT_BLACKWELL_NVFP4_KVTC.md         Agent onboarding prompt
BLACKWELL_24H_PRD.md                   Weekend PRD with tasks and subtasks
B200_SLURM_EVAL_RUNBOOK.md             Multi-node B200 Slurm runbook
scripts/check_cluster.sh               Login-node checks
scripts/check_gpu.sh                   Compute-node checks
scripts/setup_env.sh                   Environment bootstrap
scripts/baseline_inference.sh          Baseline Slurm job
scripts/run_b200_eval.sh               Shared metadata-capturing Slurm helper
scripts/b200_eval.sbatch               Generic multi-node B200 eval harness
scripts/b200_eval_array.sbatch         Generic B200 array harness
scripts/run_baseline.py                Baseline benchmark
configs/                               Config templates and eval matrix
results/                               Machine-readable outputs
logs/                                  Slurm stdout and stderr
src/                                   Runtime code
```

## Ground Rules

- Blackwell is the primary hardware target here.
- FP8 is the stable hot active-KV format in vLLM. NVFP4 is a Blackwell enhancement if verified.
- `LMCache` is the cold/warm reusable KV layer. `KVTC` is a cold-tier codec candidate.
- Do not run GPU workloads on login nodes.
- Prefer official docs and upstream repos when the latest behavior matters.
- Optimize for serving economics (sessions, HBM, TTFT), not for compression ratio in isolation.

## Primary Sources

- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA Blackwell KV cache optimization blog: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- vLLM quantized KV cache docs: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- vLLM production-stack KV cache docs: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- LMCache docs: <https://docs.lmcache.ai/>
- KVTC paper and discussion: <https://openreview.net/forum?id=tMiBQXQ0Cm>
