#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-outputs/quick_compile}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" artifact/eval/compile_plan_to_bt.py \
  --examples-file artifact/data/pyramids/plan_examples_v1.json \
  --sample-id gearset_insert_gear3_shaft2 \
  --evaluate \
  --output-file "$OUT_DIR/compiled_bt.json" \
  --report-file "$OUT_DIR/report.json" \
  --summary-file "$OUT_DIR/summary.json"

echo "Wrote quick compile outputs to $OUT_DIR"
