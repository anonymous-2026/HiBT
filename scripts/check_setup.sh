#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-all}"

DEFAULT_MODEL="Qwen/Qwen3-8B"
LOCAL_MODEL="${BT_LOCAL_MODEL:-${BT_MODEL:-$DEFAULT_MODEL}}"
ACTIONSEQ_MODEL="${BT_ACTIONSEQ_MODEL:-${BT_LOCAL_MODEL:-${BT_MODEL:-$DEFAULT_MODEL}}}"
CONCEPT_CONFIG="${ROOT_DIR}/artifact/configs/planner/train_predictor_Qwen3-8B_planlocal_4level_shared_v3_heldout.yml"
CONCEPT_BANK="${ROOT_DIR}/artifact/data/pyramids/plan_bank_v3.pt"
CONCEPT_RUNTIME="${ROOT_DIR}/artifact/data/runtime/planner_qwen3_8b"
MAIN_BENCHMARK="${ROOT_DIR}/artifact/data/requests/benchmark_main.json"
HELDOUT_BENCHMARK="${ROOT_DIR}/artifact/data/requests/benchmark_heldout.json"

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    printf '[ok]   %s: %s\n' "$label" "$path"
  elif [[ "$path" != /* && "$path" == */* ]]; then
    printf '[cfg]  %s: %s\n' "$label" "$path"
  else
    printf '[miss] %s: %s\n' "$label" "$path"
  fi
}

echo "== Paper Artifact Setup Check =="
echo "repo: ${ROOT_DIR}"
echo

check_path "main benchmark" "$MAIN_BENCHMARK"
check_path "held-out benchmark" "$HELDOUT_BENCHMARK"
echo

if [[ "$BACKEND" == "all" || "$BACKEND" == "local" ]]; then
  echo "== local =="
  check_path "resolved local model" "$LOCAL_MODEL"
  echo "example:"
  echo "  python scripts/run_eval.py --backend local --requests-file artifact/data/requests/benchmark_main.json --device cuda:0"
  echo
fi

if [[ "$BACKEND" == "all" || "$BACKEND" == "actionseq" ]]; then
  echo "== actionseq =="
  check_path "resolved actionseq model" "$ACTIONSEQ_MODEL"
  echo "example:"
  echo "  python scripts/run_eval.py --backend actionseq --requests-file artifact/data/requests/benchmark_main.json --device cuda:0"
  echo
fi

if [[ "$BACKEND" == "all" || "$BACKEND" == "concept" ]]; then
  echo "== concept =="
  check_path "planner config" "$CONCEPT_CONFIG"
  check_path "prototype bank" "$CONCEPT_BANK"
  check_path "runtime root" "$CONCEPT_RUNTIME"
  echo "note: concept additionally requires builder/predictor checkpoints under the runtime root."
  echo "reproduce from scratch:"
  echo "  export BT_MODEL=/path/to/Qwen3-8B"
  echo "  bash scripts/train_concept_from_scratch.sh --split v3 --storage-root ./artifact/data/runtime/planner_qwen3_8b --device cuda:0"
  echo
fi
