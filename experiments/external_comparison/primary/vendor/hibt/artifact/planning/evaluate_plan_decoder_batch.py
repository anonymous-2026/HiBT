#!/usr/bin/env python3
"""Batch-evaluate predictor artifacts with the v2 decoder.

Pipeline:
  predictor sample artifacts -> v2 decoder -> plan -> compiler -> execution

The script writes per-sample decoded artifacts plus a metrics summary across
all samples. It reports:
  - Exec / LC / SR with the repository metric definitions
  - adjusted LC that de-duplicates repeated prerequisite subtree actions
  - Seen/unseen split summaries using the predictor dataset manifest
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import ARTIFACT_DATA_DIR, ARTIFACT_EVAL_DIR, bootstrap_runtime

bootstrap_runtime()
if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))

from compile_plan_to_bt import PyramidCompiler
from decode_plan_latents import (
    _load_tensor_list,
    _parse_input_payload,
    decode_with_slot_retrieval_and_repair,
)
from evaluate_bt_metric_common import passes_exec, passes_lc, passes_sr
from runtime import run_sk_simulation

import torch


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _extract_action_nodes(node: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    name = node.get("name", "")
    if isinstance(name, str) and name.startswith("action:"):
        actions.append(name.split(":", 1)[1].strip().replace(" ", ""))
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            actions.extend(_extract_action_nodes(child))
    return actions


def _dedup_first(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _passes_ram_lc(record: dict[str, Any]) -> tuple[bool, str]:
    exec_ok, reason = passes_exec(record)
    if not exec_ok:
        return False, reason
    action_sequence = [
        item.replace(" ", "")
        for item in (record.get("generation_result") or {}).get("action_sequence", [])
    ]
    behavior_tree = (record.get("generation_result") or {}).get("behavior_tree")
    if not isinstance(behavior_tree, dict):
        return False, "missing_behavior_tree"
    tree_actions = _dedup_first(_extract_action_nodes(behavior_tree))
    if action_sequence != tree_actions:
        return False, "dedup_action_sequence_mismatch"
    if (record.get("evaluation_result") or {}).get("result") != "success":
        return False, (record.get("evaluation_result") or {}).get("result", "not_success")
    return True, "ok"


def _summarize(records: list[dict[str, Any]], metric_name: str, metric_fn):
    passes = 0
    reason_counts: dict[str, int] = {}
    for record in records:
        ok, reason = metric_fn(record)
        record[f"{metric_name}_pass"] = ok
        record[f"{metric_name}_reason"] = reason
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if ok:
            passes += 1
    return {
        "metric": metric_name,
        "passes": passes,
        "total": len(records),
        "pass_rate": passes / len(records) if records else 0.0,
        "reason_counts": reason_counts,
    }


def _summarize_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "exec": _summarize(copy.deepcopy(records), "exec", passes_exec),
        "lc": _summarize(copy.deepcopy(records), "lc", passes_lc),
        "sr": _summarize(copy.deepcopy(records), "sr", passes_sr),
        "concept_lc": _summarize(copy.deepcopy(records), "concept_lc", _passes_ram_lc),
    }


def _load_manifest(manifest_path: Path) -> dict[str, set[str]]:
    manifest = json.loads(manifest_path.read_text())
    return {
        "train": set(manifest["splits"]["train"]),
        "eval": set(manifest["splits"]["eval"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate predictor artifacts with the v2 decoder."
    )
    parser.add_argument(
        "--predictor-output-root",
        required=True,
        help="Directory containing sample_* subfolders from eval_predictor.py.",
    )
    parser.add_argument(
        "--prototype-bank",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_bank_v1.pt"),
        help="Prototype bank .pt path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(ARTIFACT_DATA_DIR / "datasets" / "plan_predictor_v1" / "manifest.json"),
        help="Dataset manifest path for train/eval split reporting.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Where to write decoded per-sample outputs and summary.",
    )
    parser.add_argument(
        "--concept-source",
        choices=("predicted", "gt"),
        default="predicted",
        help="Which tensors to decode from concepts.pt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval candidates to keep in decode reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor_output_root = Path(args.predictor_output_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bank = torch.load(Path(args.prototype_bank).expanduser().resolve(), map_location="cpu")
    split_sets = _load_manifest(Path(args.manifest).expanduser().resolve())

    records: list[dict[str, Any]] = []
    for sample_dir in sorted(predictor_output_root.glob("sample_*")):
        concepts_file = sample_dir / "concepts.pt"
        input_file = sample_dir / "input.json"
        predicted_levels, concept_payload = _load_tensor_list(concepts_file, args.concept_source)
        input_payload = _parse_input_payload(input_file)

        decoded, report = decode_with_slot_retrieval_and_repair(
            predicted_levels=predicted_levels,
            bank=bank,
            input_payload=input_payload,
            top_k=args.top_k,
        )
        behavior_tree = PyramidCompiler(decoded).compile()
        evaluation_result = run_sk_simulation(
            copy.deepcopy(decoded["input"]["initial_state"]),
            copy.deepcopy(behavior_tree),
        )

        out_dir = output_root / sample_dir.name
        _write_json(out_dir / "decoded_pyramid.json", decoded)
        _write_json(
            out_dir / "decode_report.json",
            {
                **report,
                "concepts_file": str(concepts_file),
                "concept_source": args.concept_source,
                "prototype_bank": str(Path(args.prototype_bank).expanduser().resolve()),
                "level_lengths_from_predictor": concept_payload.get("level_lengths"),
            },
        )
        _write_json(out_dir / "compiled_bt.json", behavior_tree)
        _write_json(out_dir / "compile_eval.json", evaluation_result)

        actions = [
            f"{item['name']}({', '.join(item['args'])})"
            for item in decoded["pyramid"][3]["items"]
        ]
        problem_id = decoded.get("problem_id")
        split = "other"
        if problem_id in split_sets["train"]:
            split = "train"
        elif problem_id in split_sets["eval"]:
            split = "eval"

        records.append(
            {
                "sample_id": sample_dir.name,
                "problem_id": problem_id,
                "split": split,
                "generation_result": {
                    "action_sequence": actions,
                    "behavior_tree": behavior_tree,
                },
                "evaluation_result": evaluation_result,
                "generation_error": None,
                "evaluation_error": None,
                "decode_report": report,
            }
        )

    overall = _summarize_subset(copy.deepcopy(records))
    train_records = [record for record in records if record["split"] == "train"]
    eval_records = [record for record in records if record["split"] == "eval"]

    summary = {
        "decoder_version": "plan-slot-retrieval-v2",
        "concept_source": args.concept_source,
        "num_records": len(records),
        "overall": overall,
        "by_split": {
            "train": _summarize_subset(copy.deepcopy(train_records)),
            "eval": _summarize_subset(copy.deepcopy(eval_records)),
        },
        "records": records,
    }
    _write_json(output_root / "metrics_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
