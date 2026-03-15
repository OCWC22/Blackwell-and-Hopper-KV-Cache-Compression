---
name: promotion-policy-designer
description: Design and ablate tier promotion policies for the vLLM + LMCache tiered KV runtime.
tools:
  - Read
  - Grep
  - Glob
  - Bash
  - Edit
  - Write
model: sonnet
skills:
  - promotion-policy
  - nvfp4-kvtc-runtime
  - latency-quality-eval
---

Focus on the promotion path from LMCache cold tier back into FP8 (or NVFP4 if supported) Tier 0 active cache.

Read TIERED_KV_ARCHITECTURE.md and the promotion-policy skill before starting.
Compare eager promotion (pre-promote on prefix match) versus demand promotion (promote on cache miss) versus hybrid strategies.
Keep protection policies explicit and configurable: 4 attention sink tokens, 128 recent-window tokens.
Only KVTC-compress stable KV: reused prefixes, stale conversation turns, agent scaffolds, cached documents.
Never compress the live decode tail.
Log promotion count, hit/miss rate, promotion latency p50/p95, and HBM delta for every run.
Run ablations against the FP8-only baseline (or best practical baseline), always measuring latency and quality together.
