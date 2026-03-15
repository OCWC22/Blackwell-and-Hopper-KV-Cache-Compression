Good. Here’s the actual PRD you can hand to the coding agent.

This assumes the repo pivot is now:
	•	TRT-LLM primary
	•	NVFP4 hot KV primary
	•	secondary-memory offload as warm/cold tier
	•	KVTC as compression on the offloaded tier
	•	Scenario 3 = main target
	•	Scenario 4 = follow-up  ￼

⸻

Blackwell Hackathon PRD

Objective

In 4–5 hours, produce one credible Blackwell/B200 result showing that an NVFP4 hot KV tier plus an offloaded secondary KV tier can improve serving capacity for long-context inference.

We win if we show at least one of:
	•	>=20% lower peak HBM
	•	>=25% more concurrent sessions at fixed p95 target
	•	materially longer effective context
	•	materially better TTFT on repeated-prefix traffic

while keeping:
	•	p95 TPOT regression <= 10%
	•	p99 TPOT regression <= 15%
	•	quality delta <= 1%

This is a serving-capacity experiment, not a codec-only experiment.

Primary thesis

Blackwell-specific KV hierarchy:
	•	Tier 0 hot KV on GPU = NVFP4
	•	Tier 1 secondary memory = host/offloaded KV
	•	Tier 1 compression = KVTC if feasible
	•	restore/promotion only on reuse

The thing we are proving is:

On the same B200, this hierarchy can support more sessions and/or longer context than baseline.

⸻

Deliverables

By the end, the agent must leave:
	1.	results/env_probe.json
	2.	baseline result JSONs
	3.	NVFP4 result JSONs
	4.	NVFP4 + offload result JSONs
	5.	optional NVFP4 + offload + KVTC result JSONs
	6.	results/comparison.md
	7.	results/benchmark_summary.md
	8.	results/bottleneck_summary.md
	9.	exact rerun commands
	10.	single-GPU Slurm script
	11.	one-node Slurm script

⸻

Success criteria

Primary success

One sentence we can defend:

On the same Blackwell GPU, NVFP4 hot KV plus secondary-memory offload improved reuse-heavy long-context serving capacity.

Stronger success

Same sentence, but for one node too.

⸻

Benchmark scenarios

Scenario 1 — longer context on one GPU

Purpose:
Isolate KV bytes per session.

Question:
How much farther can one Blackwell GPU go in context length?

Metrics:
	•	peak HBM
	•	TTFT
	•	p95/p99 TPOT
	•	max effective context

Scenario 2 — more sessions on one GPU

Purpose:
Isolate live KV replicas across sessions.

Question:
How many concurrent sessions can one GPU sustain at fixed p95?

Metrics:
	•	max concurrent sessions
	•	throughput
	•	TTFT
	•	cache hit rate

Scenario 3 — longer context + more sessions on one GPU

Purpose:
This is the main target.

Question:
Can one B200 serve many users with long prompts at once?

Metrics:
	•	max sessions at large context
	•	peak HBM
	•	TTFT
	•	p95/p99 TPOT
	•	quality delta

Scenario 4 — longer context + more sessions on one node

Purpose:
Follow-up if Scenario 3 is stable.

Question:
Does this improve node-level serving capacity too?

Metrics:
	•	aggregate sessions/node
	•	aggregate throughput
	•	aggregate HBM
	•	node-level power efficiency

⸻

Models

Primary
	•	Qwen3-30B-A3B
or
	•	Qwen3-32B

Smoke test
	•	Qwen3-8B-Instruct

Stretch only
	•	Kimi K2.5

Rule:
Do not block the hackathon on Kimi. Use it only if Scenario 3 is already working.

⸻

Workloads

Primary workload

repeated_prefix

Definition:
	•	same long system prompt / scaffold / doc prefix
	•	many requests with short differing suffixes

This is the best workload for proving reuse / restore value.

Secondary workloads if time
	•	multi_turn_reuse
	•	repeated_long_doc

⸻

Benchmark matrix

Scenario 1

Context sweep:
	•	8k
	•	32k
	•	64k

Concurrency:
	•	1

Compare:
	•	BF16/default
	•	FP8
	•	NVFP4
	•	NVFP4 + offload
	•	optional NVFP4 + offload + KVTC

Scenario 2

Context:
	•	8k

Concurrency sweep:
	•	1,2,4,8,16,32

Compare:
	•	baseline
	•	NVFP4
	•	NVFP4 + offload
	•	optional + KVTC

Scenario 3 (PRIMARY)

Contexts:
	•	8k
	•	32k

Concurrency:
	•	4,8,16

Compare:
	•	baseline
	•	NVFP4
	•	NVFP4 + offload
	•	optional NVFP4 + offload + KVTC

Scenario 4

Same as Scenario 3, but on one full node.

Only run after Scenario 3 is stable.

⸻

Metrics to record in every JSON
	•	run_id
	•	timestamp_utc
	•	runtime.engine
	•	runtime.version
	•	gpu_name
	•	gpu_count
	•	node_count
	•	model.name
	•	context_length
	•	scenario_id
	•	workload.type
	•	requests
	•	concurrency
	•	ttft_ms_p50
	•	ttft_ms_p95
	•	tpot_ms_p50
	•	tpot_ms_p95
	•	tpot_ms_p99
	•	throughput_tokens_per_s
	•	peak_hbm_gb
	•	gpu_power_w_avg
	•	cache_hit_rate
	•	promotion_latency_ms_p50
	•	promotion_latency_ms_p95
	•	quality_metric_name
	•	quality_metric_value
	•	quality_delta_vs_best_baseline
	•	notes.support_gate_result

For offload runs, also record:
	•	bytes offloaded
	•	bytes restored
	•	restore count
	•	restore bandwidth if measurable

⸻

4-hour execution plan

0:00–0:20 — environment + support gate

Tasks:
	•	verify B200
	•	verify TRT-LLM runtime
	•	verify NVFP4 path
	•	verify offload path
	•	write results/env_probe.json

Subtasks:
	•	run env probe
	•	capture driver/CUDA/runtime versions
	•	confirm NVFP4 is runnable
	•	confirm offload path is runnable
	•	stop immediately if neither path works

Output:
	•	results/env_probe.json

0:20–1:00 — aligned baselines

Tasks:
	•	run BF16/default
	•	run FP8
	•	run NVFP4

Subtasks:
	•	use smoke-test model if needed first
	•	then run primary model
	•	use Scenario 1 and Scenario 2 first
	•	make sure JSON outputs are stable

Output:
	•	baseline JSONs
	•	first comparison table draft

1:00–2:15 — NVFP4 + offload

Tasks:
	•	stand up offloaded secondary tier
	•	measure restore timing
	•	run Scenario 3

Subtasks:
	•	run repeated_prefix workload
	•	contexts 8k / 32k
	•	concurrency 4 / 8 / 16
	•	compare baseline vs NVFP4 vs NVFP4 + offload
	•	record peak HBM and TTFT

Output:
	•	NVFP4 + offload JSONs
	•	scenario 3 comparison

2:15–3:00 — KVTC integration if feasible

Tasks:
	•	add compression to offloaded tier
	•	compare raw offload vs compressed offload

Subtasks:
	•	wire KVTC to secondary tier path
	•	run one compressed tier experiment
	•	measure:
	•	compression gain
	•	restore latency
	•	impact on serving KPI

Rule:
If this blocks, stop and keep raw offload result.

Output:
	•	optional NVFP4 + offload + KVTC JSONs

3:00–3:30 — one policy ablation

Tasks:
	•	compare demand promotion vs eager promotion

Subtasks:
	•	run both on Scenario 3
	•	same model / same context / same concurrency
	•	compare p95, HBM, TTFT

Output:
	•	ablation JSONs
	•	one short note on better policy

3:30–4:00 — one-node follow-up OR summary

Tasks:
	•	if stable, run Scenario 4 on one node
	•	otherwise write final summary

Subtasks:
	•	submit one-node Slurm run
	•	same Scenario 3 matrix
	•	if time runs out, skip node and finish docs

Output:
	•	one-node JSONs OR
	•	final benchmark summary

⸻

Kill criteria

Stop or pivot if:
	1.	NVFP4 path is broken
	2.	offload path is broken
	3.	restore cost dominates so badly that reuse no longer helps
	4.	p95/p99 blow up without enough capacity gain
	5.	quality drops clearly and simple protection does not fix it
	6.	one-node run is unstable and wastes the remaining time

If blocked:
	•	keep the baseline table
	•	keep the NVFP4 proof if possible
	•	keep the offload proof if possible
	•	drop KVTC before dropping the core result

⸻

Scope cutting order

If time is tight, cut in this order:
	1.	Kimi K2.5
	2.	one-node run
	3.	full KVTC integration
	4.	extra context sizes

Never cut:
	•	aligned baseline table
	•	NVFP4 baseline
	•	offload restore timing
	•	one serving-facing KPI

⸻

Tasks for the coding agent

Task 1 — support gate

Build / run:
	•	env probe
	•	support gate
	•	output JSON

Task 2 — baseline harness

Build / harden:
	•	benchmark runner
	•	stable JSON output
	•	single-GPU Slurm script

Task 3 — baseline benchmarks

Run:
	•	BF16/default
	•	FP8
	•	NVFP4

Task 4 — offload tier

Implement / verify:
	•	secondary-memory offload
	•	restore/promotion timing
	•	run Scenario 3

Task 5 — KVTC

If feasible:
	•	integrate KVTC compression on secondary tier
	•	compare raw vs compressed offload

Task 6 — policy ablation

Run:
	•	demand promotion
	•	eager promotion

Task 7 — one-node follow-up

Only if single-GPU path is stable.

Task 8 — summary

Produce:
	•	comparison table
	•	benchmark summary
	•	bottleneck summary
	•	exact rerun commands

⸻

One-paragraph handoff to the agent

Use this too:

Build the Blackwell hackathon around one proof: NVFP4 hot KV plus secondary-memory offload should let the same B200 serve more sessions and/or longer context. Use TensorRT-LLM as the primary runtime. Benchmark BF16, FP8, NVFP4, NVFP4+offload, and optional NVFP4+offload+KVTC. Focus on Scenario 3 first: longer context + more sessions on one GPU. Use Qwen3-30B-A3B or Qwen3-32B as the primary model, repeated_prefix as the primary workload, and contexts 8k / 32k with concurrency 4 / 8 / 16. Kimi K2.5 is stretch only. Deliver env_probe.json, baseline JSONs, offload JSONs, optional KVTC JSONs, a comparison table, bottleneck summary, rerun commands, and Slurm scripts.

That’s the actual PRD.