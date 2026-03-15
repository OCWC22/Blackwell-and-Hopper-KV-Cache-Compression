# Quickstart Prompts

Use these prompts with either Codex or Claude Code. They are written for a research-minded, skeptical reader who wants one-shot Hopper experiments and clear kill criteria.

## 1. Hopper Repo Recon

```text
Work inside this repository's Hopper/ directory.
Read README.md, AGENTS.md, CLAUDE.md, QUICKSTART_PROMPTS.md, hopper_kv_compression_hackathon_context.md, PROMPT_HOPPER_HFP4_RESEARCH.md, and scripts/.

Then answer three questions with concrete file edits:
1. What can this repo run today on H100 or H200?
2. What is the shortest path to aligned BF16 and FP8 baselines?
3. What is the next missing piece for a credible packed FP4-like path with direct FP8 reconstruction?

Constraints:
- Hopper only
- no native NVFP4 claims
- focus on the active decode path before colder tiers
- prefer the smallest useful set of edits over a long memo
```

## 2. Baseline Hardening

```text
Improve the existing baseline harness in scripts/run_baseline.py and scripts/baseline_inference.sh so it is a trustworthy Hopper research reference point.

Focus on:
- stable CLI flags
- better output metadata
- timing and memory reporting that is honest about limitations
- safer dependency handling
- clear JSON results under results/

Read:
- README.md
- hopper_kv_compression_hackathon_context.md
- scripts/run_baseline.py
- scripts/baseline_inference.sh

Make the edits, then explain exactly how to run BF16 and FP8 baselines on the cluster.
```

## 3. Packed FP4-Like Scaffold

```text
Turn the repo into a clean first-pass Hopper packed-KV research harness.

Read:
- README.md
- hopper_kv_compression_hackathon_context.md
- PROMPT_HOPPER_HFP4_RESEARCH.md
- scripts/check_gpu.sh
- scripts/setup_env.sh
- scripts/
- src/

Deliver:
- one packed FP4-like storage scaffold
- clear correctness checks
- clear success and failure signals
- output that makes BF16, FP8, and packed-KV easy to compare

Constraints:
- compute stays FP8, BF16, or FP16
- no native NVFP4 language
- do not hide scale assumptions
```

## 4. Direct FP8 Reconstruction Path

```text
Add or improve the direct FP8 reconstruction path for Hopper.

Read:
- README.md
- hopper_kv_compression_hackathon_context.md
- PROMPT_HOPPER_HFP4_RESEARCH.md
- scripts/
- src/

Requirements:
- avoid unnecessary BF16 round-trips
- make grouping and scale placement explicit
- keep logs and outputs easy to inspect
- explain what to watch for if dequant overhead erases the memory win
```

## 5. Quality-Protection Ablations

```text
Prepare this repo for the first Hopper quality-retention sweep.

Read:
- README.md
- AGENTS.md
- hopper_kv_compression_hackathon_context.md
- scripts/
- src/

Add or improve:
- protected recent-window knobs
- protected sink or pivot-token knobs
- result naming
- machine-readable comparisons against the FP8 baseline

Do not add more algorithmic complexity before these ablations are measurable.
```

## 6. Cold-Tier Follow-Up

```text
Only after the active packed-KV path is understood, plan the first colder-tier integration.

Read:
- hopper_kv_compression_hackathon_context.md
- PROMPT_HOPPER_HFP4_RESEARCH.md
- README.md
- scripts/
- src/

Requirements:
- Hopper target only
- FP4-like storage plus direct FP8 reconstruction remains the center
- explain why the colder tier is worth doing after the active-path data, not before it
```

## 7. Skeptical Review

```text
Review the current Hopper track like a skeptical CTO who wants proof, not ambition.

Read every Markdown file and the runnable scripts.

Call out:
- any place where the docs still imply native NVFP4 on Hopper
- any stale or weak source link
- any place where host-path and dequant economics are under-explained
- any benchmark plan that skips BF16 or FP8 before packed FP4-like work

Then make the concrete documentation edits, not just a review summary.
```
