# Blackwell KV Runtime

This directory is the hackathon execution lane.

This repo validates that on Blackwell/B200, an NVFP4 hot KV tier plus an offloaded/compressed secondary KV tier can let the same GPU or node serve more concurrent sessions, longer effective context, or both — while keeping latency and quality within acceptable bounds.

The architecture:

1. **Primary runtime:** TensorRT-LLM as the hackathon inference engine
2. **Hot tier:** NVFP4 KV cache on GPU (primary Blackwell thesis)
3. **Secondary tier:** Host memory offload via TRT-LLM KV reuse / eviction / offload
4. **Compression:** KVTC as compression on the offloaded secondary tier
5. **Follow-up path:** vLLM + LMCache as the compatibility / productization path (documented separately)

We are not building a new inference engine. We are validating one concrete systems hypothesis using TensorRT-LLM's documented KV reuse, host offload, and NVFP4 support on Blackwell.

## Four Benchmark Scenarios

| Scenario | Question | Primary |
|----------|----------|---------|
| 1 — Longer context, one GPU | How far can one GPU go in context length? | No |
| 2 — More sessions, one GPU | How many concurrent sessions at fixed context? | No |
| **3 — Both, one GPU** | **Can one GPU serve many users with long prompts?** | **Yes** |
| 4 — Both, one node | Does the same idea improve at node level? | Follow-up |

## Models

| Role | Model |
|------|-------|
| Primary | `Qwen/Qwen3-30B-A3B` (or `Qwen/Qwen3-32B`) |
| Smoke test | `Qwen/Qwen3-8B-Instruct` |
| Stretch / node-level only | `moonshotai/Kimi-K2.5` |

**Why Kimi K2.5 is stretch only:** The public vLLM recipe for Kimi-K2.5 says the provided serving configuration has been verified on 8x H200 GPUs. It is a node-level relevance demo, not the first single-GPU Blackwell proof. Do not block the primary hackathon on Kimi K2.5.

## 5-Hour Execution Plan

1. Environment probe + support gate (30 min)
2. Aligned single-GPU baselines — TRT-LLM BF16, FP8, NVFP4 at 8k/32k (60 min)
3. First TRT-LLM NVFP4 + offload result — Scenario 3 (90 min)
4. One policy ablation — demand vs eager promotion (45 min)
5. One-node run — only after single-GPU Scenario 3 success (45 min)
6. Decision memo + artifacts (30 min)

Do not start with vLLM + LMCache as the primary runtime path. Do not run multi-node before single-GPU success. Do not start with Kimi K2.5.

## Why TensorRT-LLM First

This is the highest-leverage path for the hackathon because:

- TensorRT-LLM publicly documents KV reuse, host offload, and restore from host memory (`[R2]`, `[R8]`)
- Recent TRT-LLM release notes validate Qwen3-32B and Qwen3-30B-A3B with NVFP4 (`[R7]`)
- NVFP4 is a Blackwell-native format that provides ~50% KV memory reduction vs FP8 (`[R1]`)
- TRT-LLM's KV cache system supports configurable eviction, block reuse, and host cache (`[R8]`)

The hackathon focuses on proving the TRT-LLM + NVFP4 + offload tiered lifecycle. vLLM + LMCache is the follow-up compatibility path.

## What Success Looks Like

One defendable sentence:

> On the same B200 hardware, our Blackwell-first TensorRT-LLM NVFP4 hot KV tier plus host-offloaded secondary tier served more reuse-heavy long-context traffic by reducing KV memory pressure and/or improving TTFT on repeated prefixes, while keeping latency and quality within acceptable bounds.

Concretely:
- TRT-LLM BF16, FP8, and NVFP4 baselines run cleanly
- TRT-LLM NVFP4 + host offload shows measurable HBM reduction or TTFT improvement
- Results capture p50/p95 decode latency, throughput, HBM footprint, cache hit rate, and quality deltas
- The repo is ready for a one-shot agent handoff to Codex or Claude Code

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
- `B200_SLURM_EVAL_RUNBOOK.md`: exact Slurm handoff for B200 evals
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

# 5. Run the environment probe and support gate
bash scripts/env_probe.sh

# 6. Run the first TRT-LLM baseline
python scripts/run_baseline.py --engine tensorrt_llm --kv-mode bf16 --context-length 8192 --requests 2

# 7. Read the Blackwell-specific brief
sed -n '1,220p' blackwell_kv_hackathon_context.md
```

## Repo Map

```text
README.md                              High-level Blackwell plan
AGENTS.md                              Codex and AGENTS-aware operating rules
CLAUDE.md                              Claude Code operating rules
TIERED_KV_ARCHITECTURE.md              Tiered architecture specification
QUICKSTART_PROMPTS.md                  One-shot prompts for concrete work
blackwell_kv_hackathon_context.md      Blackwell execution brief and source notes
PROMPT_BLACKWELL_NVFP4_KVTC.md         Agent onboarding prompt
BLACKWELL_24H_PRD.md                   Weekend PRD with tasks and subtasks
B200_SLURM_EVAL_RUNBOOK.md             B200 Slurm runbook
scripts/env_probe.sh                   Environment probe and support gate
scripts/check_cluster.sh               Login-node checks
scripts/check_gpu.sh                   Compute-node checks
scripts/setup_env.sh                   Environment bootstrap
scripts/run_baseline.py                Baseline benchmark (TRT-LLM primary)
scripts/run_tiered_experiment.py       Tiered KV experiment (TRT-LLM + offload)
scripts/serve_and_bench.py             Online serving benchmark
scripts/compare_results.py             Result comparison and analysis
scripts/kv_bench_utils.py              Shared benchmark utilities
configs/                               Config templates and eval matrix
results/                               Machine-readable outputs
logs/                                  Slurm stdout and stderr
```

## Ground Rules

- Blackwell is the primary hardware target.
- TensorRT-LLM is the primary hackathon runtime. vLLM + LMCache is the follow-up compatibility path.
- NVFP4 is the primary hot KV tier on Blackwell. FP8 is the stable fallback.
- Secondary memory offload (host RAM) is the warm/cold tier. KVTC is a cold-tier codec candidate.
- Do not run GPU workloads on login nodes.
- Do not start with Kimi K2.5. It is a stretch / node-level target only.
- Prefer official docs and upstream repos when the latest behavior matters.
- Optimize for serving economics (sessions, HBM, TTFT), not for compression ratio in isolation.

## Primary Sources

- `[R1]` NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- `[R2]` NVIDIA TRT-LLM KV Cache Early Reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- `[R3]` vLLM quantized KV cache docs: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- `[R4]` vLLM production-stack KV cache docs: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- `[R5]` LMCache docs: <https://docs.lmcache.ai/>
- `[R6]` KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- `[R7]` NVIDIA NVFP4 KV Cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- `[R8]` TRT-LLM KV Cache Reuse: <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
- `[R8]` TRT-LLM KV Cache System: <https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html>
- `[R9]` ModelOpt PTQ examples: <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
