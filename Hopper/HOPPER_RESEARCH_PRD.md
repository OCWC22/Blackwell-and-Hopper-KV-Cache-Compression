# Hopper Research PRD

## Objective

Evaluate whether a Hopper-compatible packed FP4-like KV runtime can retain enough of the Blackwell memory and quality economics to matter for deployed `H100` and `H200` fleets.

## Non-Goals

- do not claim native `NVFP4` support on Hopper
- do not replace the serving engine
- do not prioritize colder-tier integration before the active path is measured
- do not judge success by compression ratio alone

## Deliverables

1. aligned `BF16` and `FP8` baseline results
2. first packed FP4-like path result
3. first direct `FP8` reconstruction result
4. one quality-protection ablation result
5. one decision memo: continue, pivot, or kill

## Workstreams

### Workstream 0: environment and baseline

1. verify `H100` or `H200`
2. harden the baseline harness
3. run aligned `BF16` and `FP8` baselines

### Workstream 1: packed format

1. define the packed representation
2. define scale granularity and metadata layout
3. add correctness checks

### Workstream 2: direct reconstruction

1. reconstruct directly into `FP8`-oriented compute
2. measure unpack and reconstruct overhead
3. compare against `FP8`

### Workstream 3: quality protection

1. recent-window protection
2. sink or pivot-token protection
3. grouping-strategy comparison

### Workstream 4: optional colder-tier follow-up

1. decide whether `LMCache` or another colder tier is worth integrating
2. only continue if the active path still looks promising

## Acceptance Criteria

- all key results are machine-readable
- all comparisons are against the aligned `FP8` baseline
- quality and latency are recorded together
- the next engineer can continue without additional oral context
