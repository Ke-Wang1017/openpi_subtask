# agents.md — Codex Agent Guide (LLM + Vision-Language-Action / VLA projects)

This file tells Codex (and other coding agents) how to work in this repo: what to prioritize, how to navigate, how to run/evaluate, and how to produce safe, reviewable changes.

---

## 0) Prime directive

**Ship correct, reproducible improvements with minimal risk.**  
Prefer small, test-backed PRs over large refactors. Every change should be:
- **Reproducible** (one command to run / train / eval)
- **Measurable** (metrics and/or golden tests updated)
- **Reviewable** (clear diffs; no drive-by formatting)

---

## 1) Project goals (edit these to match your repo)

This repository builds and evaluates:
- **LLMs**: finetuning, inference, tool use
- **VLMs**: image/video encoders + language decoders
- **VLA**: policy learning for action (robotics/UI agents), including perception → plan → action loops
- **Infra**: dataset pipelines, training loops, eval harness, deployment

Primary outcomes:
- Improve task success rate on target benchmarks
- Reduce latency/cost at fixed quality
- Increase robustness (OOD images, lighting, UI changes, partial observability)
- Improve safety and controllability (action constraints, audit logs)

---

## 2) Repo map (keep current)

Key directories (update as needed):
- `src/` — core library code
  - `src/models/` — model definitions (LLM/VLM/VLA)
  - `src/policies/` — action/policy heads, controllers, planners
  - `src/data/` — datasets, transforms, dataloaders
  - `src/eval/` — evaluation harness + metrics
  - `src/tools/` — tool-use / action adapters (APIs, simulators, UI automation)
- `configs/` — YAML/JSON configs for training/eval
- `scripts/` — runnable entrypoints (train, eval, export, data prep)
- `tests/` — unit + integration tests
- `docs/` — documentation
- `runs/` or `outputs/` — artifacts (gitignored)

If something is missing, **do not invent paths**—search first.

---

## 3) Default agent behavior

When asked to implement a feature or fix:
1. **Locate the relevant code** (`ripgrep`, `fd`, “search first”).
2. **Add/extend tests** (unit or golden) before or alongside changes.
3. **Implement minimally**; avoid large refactors unless requested.
4. **Run the smallest correct validation** (targeted tests + lint + minimal eval).
5. **Summarize changes** with:
   - what changed
   - how to run/test
   - expected metric impact
   - any risks or follow-ups

When uncertain, prefer:
- adding instrumentation/logging
- writing a failing test
- making a small change and measuring

---

## 4) Ground rules for LLM/VLM/VLA code

### 4.1 Determinism & reproducibility
- Seed everything (python, numpy, torch, dataloader workers).
- Record: git commit, config, dataset version/hash, environment (CUDA, driver).
- Training scripts must support `--dry-run` and `--limit-batches` (or equivalent).
- Save checkpoints with **schema versioning** and metadata.

### 4.2 Clean separation of concerns
- Keep **model** (forward pass) separate from:
  - loss/optimization
  - data transforms
  - evaluation
  - environment/simulator
- Policy rollouts should be “pure” where possible; isolate side effects behind interfaces.

### 4.3 VLA specifics
- Explicitly represent:
  - **observations**: (images/video, proprioception/state, text, history)
  - **actions**: discrete tokens / continuous vectors / structured actions
  - **constraints**: action bounds, forbidden actions, rate limits, safety filters
- Log every action with:
  - timestamp / step
  - observation IDs
  - model outputs (pre/post constraints)
  - final executed action
- Prefer **action schemas** (typed dataclasses/pydantic) over ad-hoc dicts.

### 4.4 Safety-by-construction for action models
- Add guardrails in code, not just prompts:
  - action range clamps
  - workspace limits
  - “deadman switch” / abort
  - collision checks (sim)
  - allowlist for UI actions
- Any new action-capable tool must have:
  - a sandbox mode
  - verbose logging
  - explicit confirmation gates where appropriate (configurable)

---

## 5) Coding standards

- Language: Python (3.10+), and/or TypeScript/JS if present.
- Type hints required for new public APIs.
- Use `dataclasses` or `pydantic` for structured inputs/outputs.
- Avoid global state; thread config via explicit objects.
- No hardcoded paths; use config and environment variables.
- Keep functions small; prefer composable components.

### Error handling
- Fail fast with clear messages.
- For training/eval scripts: exit non-zero on invalid configs.
- Do not swallow exceptions in rollouts; capture and report.

### Logging
- Use a consistent logger (`logging` / `loguru` / etc. as repo standard).
- Include run ID, step, and metric names.
- For VLA: log action trace in structured format (JSONL recommended).

---

## 6) Prompts, tool use, and agent loops

### Prompt files
- Store prompts in `prompts/` (or repo-standard location).
- Version prompts and include:
  - purpose
  - expected input schema
  - output schema
  - examples
- If prompts change behavior, add a **golden test**.

### Tool-use interfaces
- Tools must have:
  - stable function signatures
  - input validation
  - timeouts
  - retry policy (bounded)
  - idempotency notes (what happens on retry)
- Prefer structured outputs (JSON schema / typed objects).
- Never let model-generated strings directly execute shell commands without parsing + allowlists.

### Memory / history
- Keep a strict budget:
  - summarize history into structured state
  - store embeddings only if needed and tested
- Avoid leaking secrets into logs or prompts.

---

## 7) Data & datasets

### Dataset versioning
- Every dataset must have:
  - a unique version tag
  - a manifest (counts, splits, checksums)
  - license/source documentation
- Add a “data card” under `docs/data/` describing:
  - collection method
  - known biases
  - privacy constraints
  - intended use

### Vision data
- Store transformations in one place; document:
  - resize/crop policy
  - normalization
  - augmentations (with probabilities)
- For video: document fps, sampling strategy, clip length.

### Action data
- Clearly define action encoding:
  - discretization bins (if any)
  - normalization ranges
  - coordinate frames (robotics)
- Include a conversion test:
  - encode → decode round-trip

---

## 8) Evaluation (must-have)

### Minimum eval standard for changes
Every PR that touches model, data, or policy must include at least one:
- unit test (shape, schema, conversion)
- integration test (single batch train/eval)
- benchmark run (small subset)

### Benchmarks
Maintain:
- `eval/benchmarks/*.yaml` with named suites
- A quick suite (`smoke`)
- A full suite (`full`) for CI/nightly

Metrics (examples):
- success rate / completion
- reward/return
- action validity rate
- latency (p50/p95)
- cost/token usage
- safety violations / constraint triggers
- calibration (ECE) if classification-like

Golden tests:
- snapshot model outputs for fixed seeds on a few examples
- allow small tolerance for floating point drift

---

## 9) Performance & scaling

- Avoid unnecessary copies between CPU/GPU.
- Prefer batched operations.
- Use mixed precision carefully; include overflow checks when needed.
- For inference:
  - support streaming where relevant
  - measure throughput and tail latency
- Use profiling tools (torch profiler) when optimizing; attach results.

---

## 10) Security & secrets

- Never commit secrets.
- Do not print API keys, tokens, or credentials.
- Redact sensitive env vars in logs.
- If adding integrations, use:
  - `.env.example`
  - config validation that fails when missing

---

## 11) CI expectations

Before marking work done:
- `pytest -q` (or repo equivalent)
- `ruff`/`black`/`mypy` (or repo equivalents)
- run at least one smoke eval if model/policy changed

If CI is slow, add a local smoke target:
- `make test`
- `make smoke`
- `make lint`

---

## 12) How to run (using uv)

> This repo uses **uv** for fast, reproducible Python env + dependency management.  
> Keep `pyproject.toml` and `uv.lock` up to date with any dependency changes.

### Install uv
```bash
# recommended (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# ensure uv is on PATH (restart shell if needed)
uv --version
