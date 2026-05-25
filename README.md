<p align="center">
  <img src="docs/hibt-logo.svg" alt="HiBT logo" width="360">
</p>

<h1 align="center">HiBT: From Hierarchical Concepts to Behavior Trees for Reliable Robot Task Planning</h1>

<p align="center">
  <a href="https://github.com/anonymous-2026/HiBT">
    <img alt="Artifact" src="https://img.shields.io/badge/artifact-anonymous--release-1f6f96">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="Behavior Trees" src="https://img.shields.io/badge/output-behavior%20trees-2aa7b8">
  <img alt="Repo size" src="https://img.shields.io/badge/repo-%3C50MB-2f9e8f">
  <img alt="License" src="https://img.shields.io/badge/license-see%20notices-lightgrey">
</p>

<p align="center">
  A compact, reusable artifact for converting symbolic robotic assembly goals
  and world states into executable behavior trees.
</p>

## Overview

HiBT is a research codebase for language-conditioned robotic assembly planning.
It maps a symbolic task goal and an initial world state into an executable
behavior tree through a hierarchical concept-planning interface and
deterministic symbolic realization.

The repository focuses on task planning. It does not include real-robot
deployment, motion planning, low-level control, or hardware-specific code.

## What This Repository Provides

- deterministic plan-to-behavior-tree compilation
- lightweight behavior-tree runtime and symbolic simulator
- one-step behavior-tree generation backend
- action-sequence generation followed by deterministic compilation
- concept-planning backend with latent decoding, symbolic repair, and BT
  compilation
- compact plan-pyramid schemas, examples, request files, and training data
- configuration files for Builder/Predictor experiments
- artifact benchmark request pools, including the 60-task split and held-out
  12-task split used by the paper experiments
- reusable LIBERO planning-only and VLA rollout harnesses
- quickstart scripts that write generated outputs outside Git

Large model weights, generated checkpoints, local logs, and result records are
intentionally excluded.

## Repository Layout

```text
artifact/
  configs/        Builder/Predictor and dataset-adapter configs
  data/           compact schemas, examples, prompts, requests, datasets
  eval/           generation, compilation, and evaluation entry points
  planning/       concept backend, latent decoder, dataset/bank builders
  runtime/        minimal behavior-tree runtime and simulator
planner/          Builder/Predictor model code and training utilities
scripts/          public command-line entry points
experiments/      optional benchmark harnesses, including LIBERO
analysis/         aggregation scripts for generated run directories
docs/             configuration and quickstart notes
examples/         runnable shell examples
```

## Method at a Glance

HiBT supports three planning regimes:

| Backend | Pipeline |
| --- | --- |
| `local` | `target + world_state -> one-step BT generation` |
| `actionseq` | `target + world_state -> action sequence -> deterministic BT compiler` |
| `concept` | `target + world_state -> latent concept interface -> symbolic repair -> plan pyramid -> deterministic BT compiler` |

The `concept` backend separates the learned planning interface from executable
realization. Latent concepts provide a structured interface; deterministic
symbolic synthesis and repair construct an executable plan pyramid; the compiler
then expands dependency-aware method records into a behavior tree.

## Plan Pyramid

The intermediate symbolic plan has four levels:

1. `goal`: the final grounded target predicate
2. `stable_subgoals`: intermediate conditions used during BT execution
3. `methods`: dependency-aware achievement records with `achieves`, `action`,
   and `requires`
4. `actions`: a readable linear action trace

The compiler consumes the `methods` layer and recursively constructs
selector/sequence behavior-tree nodes with target checks, precondition checks,
and grounded action leaves.

## Installation

```bash
pip install -r requirements.txt
```

For language-model based generation, provide a local model path or Hugging Face
model ID:

```bash
export BT_MODEL=Qwen/Qwen3-8B
```

Runtime checkpoints should be placed under `artifact/data/runtime/`, which is
ignored by Git.

## Quick Start

Run a dependency-light smoke test that compiles a structured plan into a
behavior tree and evaluates it with the included simulator:

```bash
bash examples/quick_compile_and_eval.sh
```

Expected output includes:

```text
compiled_samples=1
[sk_sim_result] success
success_rate: 1.0
```

Run a small language-model generation example after setting `BT_MODEL`:

```bash
bash examples/quick_lm_eval.sh
```

Generated outputs are written under `outputs/`, which is ignored by Git.

## Common Commands

Compile and evaluate a hand-labeled plan:

```bash
python3 artifact/eval/compile_plan_to_bt.py \
  --examples-file artifact/data/pyramids/plan_examples_v1.json \
  --sample-id gearset_insert_gear3_shaft2 \
  --evaluate \
  --output-file outputs/compiled_bt.json \
  --report-file outputs/compile_report.json \
  --summary-file outputs/compile_summary.json
```

Evaluate one-step generation with a local or Hugging Face model:

```bash
export BT_MODEL=Qwen/Qwen3-8B
python3 scripts/run_eval.py \
  --backend local \
  --requests-file artifact/data/requests/sample_requests.json \
  --model "$BT_MODEL" \
  --output-dir outputs/local_eval \
  --summary-output outputs/local_summary.json
```

Run the action-sequence baseline:

```bash
python3 scripts/run_eval.py \
  --backend actionseq \
  --requests-file artifact/data/requests/sample_requests.json \
  --model "$BT_MODEL" \
  --output-dir outputs/actionseq_eval \
  --summary-output outputs/actionseq_summary.json
```

Check the main artifact inputs and optional checkpoint/model paths:

```bash
bash scripts/check_setup.sh
```

Run the 60-task planning benchmark with any supported backend:

```bash
python3 scripts/run_eval.py \
  --backend actionseq \
  --requests-file artifact/data/requests/test_requests_60.json \
  --model "$BT_MODEL" \
  --output-dir outputs/test60_actionseq \
  --summary-output outputs/test60_actionseq_summary.json
```

LIBERO planning-only and rollout harnesses are under `experiments/libero/`.
They require a separate LIBERO installation and optional VLA carrier
checkpoints. See `experiments/libero/README.md`.

## Concept Backend

The concept backend requires Builder/Predictor checkpoints. These are not
included in the repository.

Train concept components from repository data:

```bash
bash scripts/train_concept_from_scratch.sh \
  --split v3 \
  --storage-root ./artifact/data/runtime/repro_concept \
  --device cuda:0
```

The script exports predictor data, trains the Builder, builds a prototype bank,
trains the Predictor, and prints an evaluation command for the resulting
artifacts.

## Configuration

See `docs/configuration.md` for:

- model path and device settings
- output directory conventions
- concept-backend checkpoint options
- optional API backend configuration

See `docs/quickstart.md` for a shorter run guide.

## Data Assets

Compact reusable assets are included under:

- `artifact/data/pyramids/`
- `artifact/data/datasets/`
- `artifact/data/requests/`
- `artifact/data/examples/`
- `artifact/data/prompts/`

The repository intentionally excludes:

- foundation model weights
- Builder/Predictor checkpoints
- generated result records
- local runtime folders
- benchmark run directories
- Python caches and logs

## Evaluation Metrics

The evaluation scripts report:

- `Exec`: generated tree can be compiled and evaluated without runtime error
- `LC`: logical coherence under the symbolic evaluator
- `SR`: simulator success rate
- `GD`: average generation duration in seconds

Metric implementations live under `artifact/eval/`.

## Third-Party Notices

See `THIRD_PARTY_NOTICES.md`.
