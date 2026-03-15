# Quickstart Prompts

Use these prompts with either Codex or Claude Code. They are written for a skeptical engineer who wants a one-shot Blackwell task with a clear benchmark ladder.

## 1. Blackwell Repo Recon

```text
Work inside this repo's Blackwell/ directory.
Read README.md, AGENTS.md, CLAUDE.md, QUICKSTART_PROMPTS.md, blackwell_kv_hackathon_context.md, PROMPT_BLACKWELL_NVFP4_KVTC.md, and scripts/.

Then answer three questions with concrete file edits:
1. What can this repo run today on Blackwell?
2. What is the shortest path to aligned BF16, FP8, and NVFP4 baselines?
3. What is the next missing piece for a credible NVFP4 plus KVTC tiering experiment?

Constraints:
- Blackwell first
- native NVFP4 hot KV
- KVTC warm or cold tier
- prefer the smallest useful set of edits over a long memo
```

## 2. Baseline Hardening

```text
Improve the existing baseline harness in scripts/run_baseline.py and scripts/baseline_inference.sh so it is a trustworthy Blackwell reference point.

Focus on:
- stable CLI flags
- explicit precision metadata
- timing and memory reporting that is honest about limitations
- safer dependency handling
- clear JSON results under results/

Read:
- README.md
- blackwell_kv_hackathon_context.md
- scripts/run_baseline.py
- scripts/baseline_inference.sh

Make the edits, then explain exactly how to run BF16, FP8, and NVFP4 baselines on the target hardware.
```

## 3. NVFP4 Baseline Path

```text
Turn the repo into a clean NVFP4 baseline validation harness.

Read:
- README.md
- blackwell_kv_hackathon_context.md
- PROMPT_BLACKWELL_NVFP4_KVTC.md
- scripts/check_gpu.sh
- scripts/setup_env.sh
- scripts/

Deliver:
- one runnable NVFP4 path
- clear prerequisites
- clear success and failure signals
- output that makes BF16, FP8, and NVFP4 easy to compare

Constraints:
- use official NVIDIA and vLLM behavior
- keep Slurm-safe execution
- do not jump to KVTC until the NVFP4 baseline is real
```

## 4. KVTC Warm-Tier Scaffold

```text
Add or improve the first NVFP4 plus KVTC scaffold for the hackathon.

Read:
- README.md
- blackwell_kv_hackathon_context.md
- PROMPT_BLACKWELL_NVFP4_KVTC.md
- scripts/
- src/

Requirements:
- treat KVTC as a warm or cold tier by default
- make promotion-path assumptions explicit
- keep logs and outputs easy to inspect
- explain what evidence would justify making KVTC more hot-path-visible later
```

## 5. Blackwell Sweep

```text
Prepare this repo for a safe Blackwell validation sweep.

Read:
- README.md
- AGENTS.md
- blackwell_kv_hackathon_context.md
- scripts/

Add or improve:
- submission ergonomics
- environment assumptions
- result naming
- simple commands for repeated single-GPU jobs and optional multi-GPU follow-ups

Do not jump to distributed inference unless the single-GPU path is already stable.
```

## 6. Promotion And Protection Policy

```text
Plan and scaffold the first runtime policy pass for NVFP4 plus KVTC.

Read:
- blackwell_kv_hackathon_context.md
- PROMPT_BLACKWELL_NVFP4_KVTC.md
- README.md
- scripts/
- src/

Requirements:
- define what stays resident in NVFP4
- define what moves into KVTC
- define how to protect recent or sink-sensitive token windows if accuracy drops
- optimize for latency plus quality, not codec elegance
```

## 7. Skeptical Review

```text
Review the current Blackwell track like a skeptical CTO who wants proof, not ambition.

Read every Markdown file and the runnable scripts.

Call out:
- any place where the docs blur Blackwell and Hopper assumptions
- any stale or weak source link
- any place where the baseline ladder skips BF16, FP8, or NVFP4
- any plan that treats KVTC as a hot-path win without proving latency

Then make the concrete documentation edits, not just a review summary.
```
