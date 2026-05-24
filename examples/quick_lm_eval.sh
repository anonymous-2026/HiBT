#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${BT_MODEL:-Qwen/Qwen3-8B}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-outputs/quick_lm_eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" scripts/run_eval.py \
  --backend local \
  --requests-file artifact/data/requests/sample_requests.json \
  --model "$MODEL" \
  --device "$DEVICE" \
  --output-dir "$OUT_DIR/local_records" \
  --summary-output "$OUT_DIR/local_summary.json"

"$PYTHON_BIN" scripts/run_eval.py \
  --backend actionseq \
  --requests-file artifact/data/requests/sample_requests.json \
  --model "$MODEL" \
  --device "$DEVICE" \
  --output-dir "$OUT_DIR/actionseq_records" \
  --summary-output "$OUT_DIR/actionseq_summary.json"

echo "Wrote quick LM evaluation outputs to $OUT_DIR"
