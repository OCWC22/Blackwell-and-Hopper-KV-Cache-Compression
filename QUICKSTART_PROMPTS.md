# Repo Router Prompts

Use these at the repo root when you want an agent to pick the right track and then work inside it.

## 1. Route Me To The Right Track

```text
Work inside /Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression.

Read:
- README.md
- AGENTS.md
- CLAUDE.md
- Blackwell/README.md
- Hopper/README.md

Then answer:
1. Should this task live in Blackwell/ or Hopper/?
2. What exact files should be edited first?
3. What is the smallest credible execution plan?

Make the edits instead of stopping at a recommendation.
```

## 2. Blackwell Hackathon Pass

```text
Work inside /Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell.

Read:
- README.md
- AGENTS.md
- CLAUDE.md
- QUICKSTART_PROMPTS.md
- blackwell_kv_hackathon_context.md
- PROMPT_BLACKWELL_NVFP4_KVTC.md

Treat this as the primary hackathon path.
Use native NVFP4 as the hot KV format, KVTC as the warm/cold tier, and focus on a 24-hour validation ladder.
Make concrete repo edits and explain exactly how to run them.
```

## 3. Hopper Research Pass

```text
Work inside /Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Hopper.

Read:
- README.md
- AGENTS.md
- CLAUDE.md
- QUICKSTART_PROMPTS.md
- hopper_kv_compression_hackathon_context.md
- PROMPT_HOPPER_HFP4_RESEARCH.md

Treat this as the long-term research lane.
Focus on H100/H200, FP4-like packed storage, direct FP8 reconstruction, and quality-preserving decode-path optimizations.
Make concrete repo edits and explain exactly how to run them.
```
