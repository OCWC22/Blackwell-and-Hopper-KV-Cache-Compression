# Agent Harness Best Practices

Retrieved and updated on `2026-03-14`.

This file captures the best-practice rules we want for the repo-native Codex and Claude Code harnesses, plus the way those rules map onto the `Blackwell/` and `Hopper/` tracks.

## Why This Exists

The fastest way to make coding agents useless is to dump everything into a giant top-level instruction file and hope for the best.

The current best practice is narrower and more operational:

- keep the persistent root instructions short
- push detailed procedures into versioned repo docs
- make subagents and skills focused, specific, and composable
- make evals, logs, and outputs legible enough that an agent can steer itself

That matches how OpenAI describes effective `AGENTS.md` usage and how Anthropic describes effective Claude Code subagents and skills. See `[R1]`, `[R2]`, `[R3]`, and `[R4]`.

## Codex Best Practices

### 1. Keep `AGENTS.md` short and routing-oriented

The root `AGENTS.md` should act as a router and constraint surface, not as a full design doc.

Good:

- repo purpose
- directory routing rules
- safety constraints
- what to read next

Bad:

- long paper summaries
- full experiment matrices
- repeated boilerplate from deeper files

That is why this repo uses:

- root `AGENTS.md` as a dual-track router
- `Blackwell/AGENTS.md` for the weekend execution lane
- `Hopper/AGENTS.md` for the research lane

### 2. Put deep instructions in versioned files the agent can open

OpenAI's guidance emphasizes that agents do better when the environment is legible and the task is structured like a clear issue or work order. In practice that means:

- context brief
- PRD
- runbook
- prompt templates
- result schema

instead of one giant instruction blob.

### 3. Use repo-local skills for recurring workflows

Skills should be:

- focused on one workflow
- short enough to load fast
- operational rather than philosophical
- directly tied to concrete files and deliverables

That is why both tracks use a small, explicit skill set rather than a single mega-skill.

### 4. Make tasks issue-shaped

Codex works best when prompts look like:

- objective
- exact files
- constraints
- deliverable
- success criteria

The track-specific `QUICKSTART_PROMPTS.md` files and `PROMPT_*.md` files follow that pattern.

### 5. Expose the eval system

Agents improve faster when logs, metrics, scripts, and artifacts are easy to discover.

That means:

- deterministic output directories
- machine-readable result files
- clear Slurm job names
- saved cluster metadata
- exact benchmark ladders

OpenAI's "Harness engineering" advice points in the same direction: make the environment easier to navigate and the loop easier to close. See `[R2]`.

## Claude Code Best Practices

### 1. Use focused subagents with explicit job boundaries

Anthropic recommends that subagents be specific and detailed so Claude can delegate cleanly. A good subagent has:

- one job
- one crisp description
- clear inputs
- clear output expectations
- only the tools it actually needs when practical
- preloaded skills when the domain knowledge is stable and repetitive

That is why both tracks use:

- `repo-explorer`
- `slurm-operator`
- `kv-cache-researcher`
- one hardware or runtime optimizer
- `eval-guard`

### 2. Use project instructions and project skills under version control

Anthropic recommends putting project-level guidance in repo files so the team can maintain and review it. In this repo that means:

- track-level `CLAUDE.md`
- versioned `.claude/agents/*.md`
- mirrored `.claude/skills/*`

The Anthropic docs also support:

- `tools` and `disallowedTools` in subagent frontmatter
- `skills` to preload full skill content into a subagent
- `model` to route focused work to faster or cheaper models
- `permissionMode` to keep read-only or plan-only agents constrained

This repo uses those features where they improve predictability and context hygiene. See `[R3]` and `[R4]`.

### 3. Use XML-style structure for complex prompts

Anthropic's prompting guidance explicitly recommends XML tags for separating instructions, constraints, and context in more complex requests. That is why the handoff prompts can be adapted into tagged structures when needed for Claude Code. See `[R5]`.

### 4. Keep subagents concrete and tool-aware

Subagents should not be motivational posters. They should say:

- what to read
- what not to do
- what metrics matter
- what counts as success or failure

## How This Repo Implements Those Rules

### Root

- `README.md`: explain the dual-track structure
- `AGENTS.md`: route work into `Blackwell/` or `Hopper/`
- `CLAUDE.md`: same routing for Claude Code
- `AGENT_HARNESS_BEST_PRACTICES.md`: deeper harness guidance

### Blackwell

- product-shaped hackathon lane
- native `NVFP4` plus `KVTC`
- B200 Slurm runbook and 24-hour PRD
- focused Blackwell skills and subagents with preloaded skills where helpful

### Hopper

- long-term research lane
- packed FP4-like storage plus direct `FP8` reconstruction
- deeper paper stack
- focused Hopper skills and subagents with preloaded skills where helpful

## Handoff Rules For Engineers

When handing work to Codex or Claude Code:

1. name the track explicitly
2. point at the exact runbook or PRD
3. point at the exact benchmark ladder
4. name the exact baseline that must be beaten
5. require concrete edits and exact run commands

That keeps the agent out of "high-level bullshit" mode.

## Skill Design Rules We Follow

Anthropic's skills docs are more concrete than most teams realize. The useful takeaways are:

- keep `SKILL.md` compact and move bulk detail into supporting files when needed
- decide whether a skill is reference knowledge or an action workflow
- use `disable-model-invocation: true` for side-effectful workflows that should only be run manually
- use `allowed-tools` if the skill should operate inside a constrained tool envelope
- use `context: fork` and `agent:` only when a skill truly belongs in a separate subagent context

For this repo, that means:

- reference-heavy skills stay short and domain-specific
- side-effectful Slurm or deploy-style skills remain explicit and human-triggered
- large research briefs live in normal repo docs, not inside `SKILL.md`

## References

- `[R1]` OpenAI, "Custom instructions with AGENTS.md"
  - <https://developers.openai.com/codex/guides/agents-md/>
- `[R2]` OpenAI, "Harness engineering"
  - <https://openai.com/index/introducing-codex/>
- `[R3]` Anthropic, "Claude Code subagents"
  - <https://docs.anthropic.com/en/docs/claude-code/sub-agents>
- `[R4]` Anthropic, "Claude Code skills"
  - <https://docs.anthropic.com/en/docs/claude-code/skills>
- `[R5]` Anthropic, "Use XML tags"
  - <https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags>
