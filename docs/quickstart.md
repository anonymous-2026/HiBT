# Quickstart

This quickstart uses only repository-local inputs. It does not require bundled
model checkpoints.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Compile and Evaluate a Structured Plan

```bash
bash examples/quick_compile_and_eval.sh
```

This command compiles one plan from `artifact/data/pyramids/plan_examples_v1.json`
into a behavior tree and evaluates it with the lightweight simulator.

Outputs are written to:

```text
outputs/quick_compile/
```

## 3. Run a Small LM-Based Evaluation

Set a model path or Hugging Face model ID:

```bash
export BT_MODEL=Qwen/Qwen3-8B
```

Then run:

```bash
bash examples/quick_lm_eval.sh
```

This evaluates the `local` and `actionseq` backends on
`artifact/data/requests/sample_requests.json`.

## 4. Train Concept Components

Concept checkpoints are not included. To reproduce them from repository data:

```bash
bash scripts/train_concept_from_scratch.sh \
  --split v3 \
  --storage-root ./artifact/data/runtime/repro_concept \
  --device cuda:0
```

The script writes checkpoints to `artifact/data/runtime/`, which is ignored by
Git.
