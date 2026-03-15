# Blackwell KV Runtime

This directory is the hackathon execution lane.

The thesis is specific:

1. keep hot or active KV in native `NVFP4`
2. move stale, reusable, or oversized KV into a `KVTC` warm or cold tier
3. optimize the promotion path from `KVTC` back into `NVFP4`
4. preserve quality while reducing HBM pressure and keeping decode latency low

We are not trying to replace `vLLM` or `LMCache`.

- `vLLM` remains the active serving engine and baseline quantized-KV stack
- `LMCache` remains the baseline for offload and storage lifecycle
- our wedge is the Blackwell-native KV runtime and policy layer that decides what stays resident, what gets transform-coded, and when promotion happens

## 24-Hour Validation Ladder

Run the work in this order:

1. clean `BF16` or default KV baseline
2. clean `FP8` KV baseline
3. clean native `NVFP4` KV baseline
4. optional `LMCache` or raw host-path offload baseline
5. `NVFP4` hot tier plus `KVTC` warm or cold tier
6. promotion-policy ablations:
   - eager promotion
   - demand fetch
   - protected recent window
   - protected sink or pivot tokens

If we cannot beat or match the practical baselines on latency and quality, there is no reason to keep going.

## Why Blackwell First

This is the highest-leverage path for the weekend because the hardware already gives us the primitive we need:

- Blackwell natively supports `NVFP4`
- NVIDIA positions `NVFP4` as a practical low-precision path for accurate inference
- NVIDIA also documents Blackwell KV-cache optimizations for longer context windows and faster reuse behavior

That means the weekend work can focus on policy, tiering, promotion, and profiling instead of pretending Hopper has native `NVFP4`.

## What Success Looks Like

- native `NVFP4` baseline runs cleanly
- the repo can benchmark `BF16`, `FP8`, and `NVFP4` on the same model and prompt set
- `KVTC` is integrated or at least scaffolded as a warm or cold tier with a clear promotion story
- results capture `p50/p95` decode latency, throughput, HBM footprint, and quality deltas
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
- `NVFP4` is the hot active-KV format.
- `KVTC` should be treated as a warm or cold tier unless the hot-path latency is proven acceptable.
- Do not run GPU workloads on login nodes.
- Prefer official docs and upstream repos when the latest behavior matters.
- Optimize for latency plus quality, not for compression ratio in isolation.

## Primary Sources

- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA Blackwell KV cache optimization blog: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- vLLM quantized KV cache docs: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- vLLM production-stack KV cache docs: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- LMCache docs: <https://docs.lmcache.ai/>
- KVTC paper and discussion: <https://openreview.net/forum?id=tMiBQXQ0Cm>
