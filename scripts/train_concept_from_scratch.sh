#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

SPLIT="v3"
MODEL_PATH="${BT_MODEL:-Qwen/Qwen3-8B}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"

slugify_model() {
  local raw="$1"
  local base
  base="$(basename "$raw")"
  base="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
  base="${base%-instruct}"
  base="${base%-chat}"
  base="${base//./_}"
  base="$(printf '%s' "$base" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
  printf '%s' "$base"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/train_concept_from_scratch.sh [--split v2|v3] [--storage-root PATH] [--model PATH_OR_HF_ID] [--device DEVICE]

Environment variables:
  BT_MODEL      Model path or Hugging Face model id. Default: Qwen/Qwen3-8B
  STORAGE_ROOT  Output root for builder/predictor checkpoints and logs.
  DEVICE        Torch device, e.g. cuda:0 or cpu.
  TRAIN_RATIO   Train split ratio used when regenerating datasets. Default: 0.8

What this script does:
  1. Regenerates predictor dataset JSONL files.
  2. Trains the builder from scratch.
  3. Builds a prototype bank from the frozen builder.
  4. Trains the predictor from scratch.
  5. Prints the exact --backend concept evaluation command to reuse the trained artifacts.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --storage-root)
      STORAGE_ROOT="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export BT_MODEL="$MODEL_PATH"
MODEL_TAG="${MODEL_TAG:-$(slugify_model "$BT_MODEL")}"
STORAGE_ROOT="${STORAGE_ROOT:-$REPO_ROOT/artifact/data/runtime/planner_${MODEL_TAG}}"

case "$SPLIT" in
  v2)
    DATASET_NAME="plan_predictor_v2"
    EXAMPLES_JSON="artifact/data/pyramids/plan_examples_v2.json"
    DATASET_DIR="artifact/data/datasets/${DATASET_NAME}"
    SPLIT_SPEC=""
    BUILDER_CONFIG="artifact/configs/planner/train_builder_Qwen3-8B_planlocal_4level_v2_smoke.yml"
    PREDICTOR_CONFIG="artifact/configs/planner/train_predictor_Qwen3-8B_planlocal_4level_shared_v2_smoke.yml"
    PLAN_BANK="artifact/data/pyramids/plan_bank_${MODEL_TAG}_v2.pt"
    BANK_SOURCE_JSONL="${DATASET_DIR}/all.jsonl"
    ;;
  v3)
    DATASET_NAME="plan_predictor_v3"
    EXAMPLES_JSON="artifact/data/pyramids/plan_examples_v2.json"
    DATASET_DIR="artifact/data/datasets/${DATASET_NAME}"
    SPLIT_SPEC="artifact/data/datasets/plan_predictor_v3_split.json"
    BUILDER_CONFIG="artifact/configs/planner/train_builder_Qwen3-8B_planlocal_4level_v3_heldout.yml"
    PREDICTOR_CONFIG="artifact/configs/planner/train_predictor_Qwen3-8B_planlocal_4level_shared_v3_heldout.yml"
    PLAN_BANK="artifact/data/pyramids/plan_bank_${MODEL_TAG}_v3.pt"
    BANK_SOURCE_JSONL="${DATASET_DIR}/train.jsonl"
    ;;
  *)
    echo "Unsupported split: ${SPLIT}. Use v2 or v3." >&2
    exit 1
    ;;
esac

echo "[INFO] repo_root      = $REPO_ROOT"
echo "[INFO] split          = $SPLIT"
echo "[INFO] model_tag      = $MODEL_TAG"
echo "[INFO] storage_root   = $STORAGE_ROOT"
echo "[INFO] model          = $BT_MODEL"
echo "[INFO] device         = $DEVICE"

DATASET_EXPORT_CMD=(
  python artifact/planning/export_plan_dataset.py
  --input "$EXAMPLES_JSON"
  --output-dir "$DATASET_DIR"
  --dataset-name "$DATASET_NAME"
  --train-ratio "$TRAIN_RATIO"
)
if [[ -n "$SPLIT_SPEC" ]]; then
  DATASET_EXPORT_CMD+=(--split-spec "$SPLIT_SPEC")
fi

echo "[STEP 1/4] Export predictor dataset"
"${DATASET_EXPORT_CMD[@]}"

echo "[STEP 2/4] Train builder"
python planner/train_builder.py \
  -c "$BUILDER_CONFIG" \
  -s "$STORAGE_ROOT"

BUILDER_CKPT="$(python - <<'PY' "$STORAGE_ROOT" "$BUILDER_CONFIG"
from pathlib import Path
import sys
from planner.config_io import load_config, apply_storage_root

storage_root = sys.argv[1]
config_path = sys.argv[2]
cfg = load_config(config_path)
apply_storage_root(cfg, storage_root)
checkpoint_dir = Path(cfg["log"]["checkpoint_path"])
patterns = ("checkpoint_best_eval-*.pt", "checkpoint_best-*.pt", "checkpoint*.pt")
candidates = []
for pattern in patterns:
    candidates.extend(checkpoint_dir.glob(pattern))
def step_key(path: Path) -> tuple[int, str]:
    name = path.name
    step = 0
    if "-step" in name:
        try:
            step = int(name.split("-step", 1)[1].split(".pt", 1)[0])
        except Exception:
            step = 0
    return (step, name)
if not candidates:
    raise SystemExit(f"No builder checkpoint found under {checkpoint_dir}")
print(sorted(candidates, key=step_key)[-1])
PY
)"
echo "[INFO] builder_ckpt   = $BUILDER_CKPT"

echo "[STEP 3/4] Build prototype bank"
python artifact/planning/build_plan_bank.py \
  --builder-config "$BUILDER_CONFIG" \
  --builder-checkpoint "$BUILDER_CKPT" \
  --dataset-jsonl "$BANK_SOURCE_JSONL" \
  --output "$PLAN_BANK" \
  --device "$DEVICE"

echo "[STEP 4/4] Train predictor"
python planner/train_predictor.py \
  -c "$PREDICTOR_CONFIG" \
  -s "$STORAGE_ROOT"

PREDICTOR_CKPT="$(python - <<'PY' "$STORAGE_ROOT" "$PREDICTOR_CONFIG"
from pathlib import Path
import sys
from planner.config_io import load_config, apply_storage_root

storage_root = sys.argv[1]
config_path = sys.argv[2]
cfg = load_config(config_path)
apply_storage_root(cfg, storage_root)
checkpoint_dir = Path(cfg["log"]["checkpoint_path"])
patterns = ("checkpoint_best_eval-*.pt", "checkpoint_best-*.pt", "checkpoint*.pt")
candidates = []
for pattern in patterns:
    candidates.extend(checkpoint_dir.glob(pattern))
def step_key(path: Path) -> tuple[int, str]:
    name = path.name
    step = 0
    if "-step" in name:
        try:
            step = int(name.split("-step", 1)[1].split(".pt", 1)[0])
        except Exception:
            step = 0
    return (step, name)
if not candidates:
    raise SystemExit(f"No predictor checkpoint found under {checkpoint_dir}")
print(sorted(candidates, key=step_key)[-1])
PY
)"
echo "[INFO] predictor_ckpt = $PREDICTOR_CKPT"

cat <<EOF

[DONE] Concept training assets are ready.

Use them with:
python scripts/run_eval.py \\
  --backend concept \\
  --requests-file artifact/data/requests/full_request_single.json \\
  --device ${DEVICE} \\
  --planner-config ${PREDICTOR_CONFIG} \\
  --planner-storage-root ${STORAGE_ROOT} \\
  --plan-bank ${PLAN_BANK} \\
  --planner-predictor-ckpt ${PREDICTOR_CKPT} \\
  --planner-builder-ckpt ${BUILDER_CKPT}

EOF
