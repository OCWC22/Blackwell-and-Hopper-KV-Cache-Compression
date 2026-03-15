# Hopper KV Runtime Research

This directory is the longer-horizon research lane.

The thesis is also specific:

1. target deployed `H100` and `H200` fleets, not hypothetical hardware
2. emulate some of the Blackwell KV economics with packed FP4-like storage
3. reconstruct directly into `FP8`-oriented compute on Hopper
4. protect quality with recent-window and sink or pivot-token policies
5. integrate cold-tier storage only after the active decode path is credible

We are not trying to claim native `NVFP4` on Hopper.

## Research Question

Can Hopper achieve something close to Blackwell-like 4-bit KV quality and memory savings by using:

- FP4-like packed storage
- direct `FP8` reconstruction
- hardware-aware grouping
- protected token windows
- profiling-driven decode-path optimization

That is the wedge.

## Why Hopper Still Matters

The bottleneck is still structural.

- H100 exposes about `3.35 TB/s` HBM bandwidth on SXM systems.
- H200 exposes about `4.8 TB/s` HBM3e bandwidth.
- PCIe Gen5 x16 is only about `63 GB/s` theoretical per direction, even though the headline bidirectional number is about `128 GB/s`.

Practical implication:

- host refill bandwidth is roughly `53x` lower than H100 SXM HBM bandwidth
- host refill bandwidth is roughly `76x` lower than H200 HBM bandwidth
- if the decode path pays too much dequant or refill overhead, the storage savings do not matter

## Research Ladder

Run the work in this order:

1. clean `BF16` or default KV baseline
2. clean `FP8` KV baseline
3. first packed FP4-like KV storage path
4. direct `FP8` reconstruction path
5. quality protection ablations:
   - protected recent window
   - protected sink or pivot tokens
   - inner-dimension grouping versus naive grouping
6. optional `LMCache` cold-tier integration once the hot decode path is real

If the packed representation saves bytes but loses on `p95` decode latency or quality, it is not the right path.

## AI Harness

This track includes a Codex and Claude Code harness tuned for Hopper research and R&D.

- `AGENTS.md`: track-level guidance for Codex and AGENTS-aware tools
- `CLAUDE.md`: track-level guidance for Claude Code
- `.agents/skills/*`: shared Hopper skills
- `.codex/agents/*`: Codex role definitions
- `.claude/agents/*`: Claude subagents
- `QUICKSTART_PROMPTS.md`: one-shot prompts for the Hopper track
- `hopper_kv_compression_hackathon_context.md`: research brief, experiment ladder, and sources
- `PROMPT_HOPPER_HFP4_RESEARCH.md`: onboarding prompt for engineers and coding agents
- `HOPPER_RESEARCH_PRD.md`: structured task list for the long-horizon research lane
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

# 4. Validate one allocated Hopper GPU
srun --gpus=1 --time=00:10:00 --pty bash
bash scripts/check_gpu.sh
exit

# 5. Run the baseline harness
sbatch scripts/baseline_inference.sh

# 6. Read the Hopper research brief
sed -n '1,260p' hopper_kv_compression_hackathon_context.md
```

## Repo Map

```text
README.md                              High-level Hopper research plan
AGENTS.md                              Codex and AGENTS-aware operating rules
CLAUDE.md                              Claude Code operating rules
QUICKSTART_PROMPTS.md                  One-shot prompts for concrete work
hopper_kv_compression_hackathon_context.md
                                       Hopper research brief and source notes
PROMPT_HOPPER_HFP4_RESEARCH.md         Agent onboarding prompt
HOPPER_RESEARCH_PRD.md                 Research PRD with tasks and kill criteria
scripts/check_cluster.sh               Login-node checks
scripts/check_gpu.sh                   Compute-node checks
scripts/setup_env.sh                   Environment bootstrap
scripts/baseline_inference.sh          Baseline Slurm job
scripts/run_baseline.py                Baseline benchmark
configs/                               Config templates
results/                               Machine-readable outputs
logs/                                  Slurm stdout and stderr
src/                                   Runtime code
```

## Ground Rules

- Hopper `H100` and `H200` are the real targets here.
- FP4 is a storage and transport idea on Hopper, not a native Hopper compute claim.
- Compute should stay in `FP8`, `BF16`, or `FP16` after reconstruction unless code proves otherwise.
- Do not run GPU workloads on login nodes.
- Use `/mnt/sharefs` for shared state and `/tmp` or `/local_scratch` for scratch when the cluster provides it.
- Prefer official docs and primary papers when the latest behavior matters.

## Primary Sources

- NVIDIA H100: <https://www.nvidia.com/en-us/data-center/h100/>
- NVIDIA H200: <https://www.nvidia.com/en-us/data-center/h200/>
- NVIDIA Hopper architecture: <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- LMCache docs: <https://docs.lmcache.ai/>
- vLLM quantized KV cache docs: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- vLLM production-stack KV cache docs: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- PCI-SIG webinar deck with PCIe 5.0 bandwidth table: <https://pcisig.com/sites/default/files/files/PCIe%206.0%20Webinar_Final_.pdf>
