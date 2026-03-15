# Dual-Track Agent Guide

This repo contains two separate harnesses:

- `Blackwell/`: hackathon execution and demo path
- `Hopper/`: long-term research and R&D path

## Routing Rule

Before editing anything substantial, decide which track the task belongs to.

- If the task is about Blackwell/B200, vLLM + LMCache tiered KV, weekend validation, or a demo, work in `Blackwell/`.
- If the task is about `H100`, `H200`, FP4-like emulation, direct FP8 reconstruction, or longer-term research, work in `Hopper/`.

Do not casually mix the two tracks in the same edit set.

## Repo-Level Constraints

- Keep each track self-contained.
- Prefer official docs and primary papers when citing hardware or runtime behavior.
- Do not claim native `NVFP4` on Hopper.
- Do not turn the Blackwell hackathon path into a speculative Hopper research project.
- Do not turn the Hopper research path into a vague marketing document.

## Read First

- `README.md`
- `AGENT_HARNESS_BEST_PRACTICES.md`
- `Blackwell/README.md` or `Hopper/README.md`, depending on the task
- the matching track-level `AGENTS.md`
- the matching context brief

## Track Intent

### Blackwell

- prove something useful in 24 hours
- vLLM FP8 KV cache as stable hot tier, NVFP4 as optional Blackwell enhancement if verified
- LMCache as cold/warm reusable KV layer, KVTC as cold-tier codec candidate
- benchmark against real baselines, not imagined ones

### Hopper

- research a durable wedge for deployed fleets
- study packed FP4-like storage plus direct FP8 reconstruction
- keep quality protection explicit
- optimize the decode path, not just compression ratio
