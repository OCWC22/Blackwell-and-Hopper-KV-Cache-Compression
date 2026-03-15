# Claude Code Guide For The Dual-Track Repo

Use the repo root only as a router.

The real work happens in one of two subdirectories:

- `Blackwell/` for the hackathon execution path
- `Hopper/` for the longer-term Hopper research path

## Workflow

1. decide the target track first
2. read `AGENT_HARNESS_BEST_PRACTICES.md`
3. read that track's `README.md`, `AGENTS.md`, `CLAUDE.md`, and context brief
4. keep edits inside that track unless the root routing docs also need updating

## Hard Rules

- Do not assume the Blackwell strategy applies unchanged to Hopper.
- Do not claim native `NVFP4` support on Hopper.
- Do not dilute the Blackwell hackathon track with off-mission research detours.
- Do not strip the Hopper track of its research references and experiments.

## Track Summary

### Blackwell

- native `NVFP4` hot KV
- `KVTC` warm or cold tier
- promotion latency, memory footprint, and quality are the key metrics

### Hopper

- FP4-like packed storage
- direct FP8 reconstruction
- protected token windows and hardware-aware grouping
- decode latency versus quality is the key tradeoff
