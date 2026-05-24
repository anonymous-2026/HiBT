# Configuration Guide

This repository is designed to keep source code and compact reproducibility
assets in Git, while generated outputs and large checkpoints stay outside Git.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Runtime Outputs

Generated files should be written under `outputs/`:

```bash
mkdir -p outputs
```

`outputs/` is ignored by Git.

## Model Configuration

The local language-model backends use `BT_MODEL`.

```bash
export BT_MODEL=Qwen/Qwen3-8B
```

You may also pass a model explicitly:

```bash
python3 scripts/run_eval.py \
  --backend local \
  --requests-file artifact/data/requests/sample_requests.json \
  --model Qwen/Qwen3-8B
```

Useful runtime options:

- `--device cuda:0` or `--device cpu`
- `--torch-dtype bfloat16`
- `--attn-implementation sdpa`
- `--max-new-tokens 2048`
- `--prompt-profile full` or `compact`

## Concept Backend Configuration

The concept backend requires Builder/Predictor checkpoints. These checkpoints
are not included in the repository.

Default paths:

- planner config: `artifact/configs/planner/train_predictor_Qwen3-8B_planlocal_4level_shared_v2_smoke.yml`
- storage root: `artifact/data/runtime/planner_v2`
- prototype bank: `artifact/data/pyramids/plan_bank_v2.pt`

Override them with:

```bash
python3 scripts/run_eval.py \
  --backend concept \
  --requests-file artifact/data/requests/sample_requests.json \
  --planner-config artifact/configs/planner/train_predictor_Qwen3-8B_planlocal_4level_shared_v3_heldout.yml \
  --planner-storage-root artifact/data/runtime/repro_concept \
  --plan-bank artifact/data/pyramids/plan_bank_v3.pt \
  --planner-builder-ckpt artifact/data/runtime/repro_concept/path/to/builder.pt \
  --planner-predictor-ckpt artifact/data/runtime/repro_concept/path/to/predictor.pt
```

## Training Storage

When training Builder/Predictor components, place checkpoints under:

```text
artifact/data/runtime/
```

This path is ignored by Git and is the intended location for large generated
artifacts.

## API Backend

The optional DeepSeek backend reads the API key from `DEEPSEEK_API_KEY` by
default:

```bash
export DEEPSEEK_API_KEY=...
python3 scripts/run_eval.py \
  --backend deepseek \
  --requests-file artifact/data/requests/sample_requests.json
```

Use `--api-key-env` and `--base-url` to change the environment variable or API
endpoint.
