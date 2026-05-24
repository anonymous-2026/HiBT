#!/usr/bin/env python3
"""Export hand-labeled plan examples to predictor training format.

The training loader expects each sample to expose at least:

- question
- cot_answer
- groundtruth
- main_id

This exporter converts the hand-labeled plan examples into a JSONL
dataset with those fields, plus a few extra metadata fields kept for debugging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = THIS_DIR.parent / "data" / "pyramids" / "plan_examples_v1.json"
DEFAULT_OUTPUT_DIR = THIS_DIR.parent / "data" / "datasets" / "plan_predictor_v1"


def _call(name: str, args: list[str]) -> str:
    return f"{name}({', '.join(args)})"


def _render_question(instance: dict[str, Any]) -> str:
    payload = {
        "target": instance["input"]["target"],
        "initial_state": instance["input"]["initial_state"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_pyramid_text(instance: dict[str, Any]) -> str:
    levels = instance["pyramid"]
    lines: list[str] = []

    for level in levels:
        level_idx = level["level"]
        level_name = level["name"]
        lines.append(f"L{level_idx}: {level_name}")

        items = level["items"]
        if not items:
            lines.append("- <empty>")
            lines.append("")
            continue

        for item in items:
            item_type = item["type"]
            if item_type in {"goal", "subgoal"}:
                call = _call(item["predicate"], item["args"])
                supports = item.get("supports", [])
                if supports:
                    lines.append(
                        f"- {item['id']} | {item_type}: {call} | supports: {', '.join(supports)}"
                    )
                else:
                    lines.append(f"- {item['id']} | {item_type}: {call}")
            elif item_type == "method":
                action = _call(item["action"]["name"], item["action"]["args"])
                requires = item.get("requires", [])
                if requires:
                    lines.append(
                        f"- {item['id']} | method: achieves {item['achieves']} via {action} | requires: {', '.join(requires)}"
                    )
                else:
                    lines.append(
                        f"- {item['id']} | method: achieves {item['achieves']} via {action}"
                    )
            elif item_type == "action":
                action = _call(item["name"], item["args"])
                lines.append(
                    f"- step {item['step']} | action: {action} | derived_from: {item['derived_from']}"
                )
            else:
                raise ValueError(f"Unsupported pyramid item type: {item_type}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_groundtruth(instance: dict[str, Any]) -> str:
    action_level = instance["pyramid"][3]["items"]
    if not action_level:
        return "<goal_already_satisfied>"
    return "\n".join(_call(item["name"], item["args"]) for item in action_level)


def _to_record(example: dict[str, Any]) -> dict[str, Any]:
    instance = example["instance"]
    return {
        "main_id": example["sample_id"],
        "question": _render_question(instance),
        "cot_answer": _render_pyramid_text(instance),
        "groundtruth": _render_groundtruth(instance),
        "problem_id": instance.get("problem_id", example["sample_id"]),
        "source_path": example.get("source_path"),
        "target": instance["input"]["target"],
        "pyramid_json": instance["pyramid"],
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_split_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Split spec must be a JSON object: {path}")
    return payload


def export_dataset(
    input_path: Path,
    output_dir: Path,
    dataset_name: str,
    train_ratio: float,
    split_spec_path: Path | None = None,
) -> None:
    examples = json.loads(input_path.read_text(encoding="utf-8"))["examples"]
    all_records = [_to_record(example) for example in examples]
    records_by_id = {record["main_id"]: record for record in all_records}

    if split_spec_path is not None:
        split_spec = _load_split_spec(split_spec_path)
        non_test_pool = list(split_spec.get("non_test_pool", []))
        if not non_test_pool:
            raise ValueError(
                f"Split spec at {split_spec_path} is missing a non-empty non_test_pool"
            )
        missing = [item for item in non_test_pool if item not in records_by_id]
        if missing:
            raise ValueError(
                f"Split spec references sample ids missing from input examples: {missing}"
            )
        ordered_pool = [records_by_id[item] for item in non_test_pool]
        if len(ordered_pool) < 2:
            raise ValueError("Need at least 2 non-test records to create train/eval splits")
        split_index = max(
            1, min(len(ordered_pool) - 1, int(round(len(ordered_pool) * train_ratio)))
        )
        train_records = ordered_pool[:split_index]
        eval_records = ordered_pool[split_index:]
        test_ids = list(split_spec.get("test", []))
        test_records = [records_by_id[item] for item in test_ids if item in records_by_id]
        records = ordered_pool + test_records
        manifest_extra = {
            "split_spec": str(split_spec_path),
            "test_count": len(test_records),
            "splits": {
                "train": [record["main_id"] for record in train_records],
                "eval": [record["main_id"] for record in eval_records],
                "test": [record["main_id"] for record in test_records],
            },
        }
    else:
        records = all_records
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        split_index = max(1, min(len(records) - 1, int(round(len(records) * train_ratio))))
        train_records = records[:split_index]
        eval_records = records[split_index:]
        manifest_extra = {
            "splits": {
                "train": [record["main_id"] for record in train_records],
                "eval": [record["main_id"] for record in eval_records],
            }
        }

    _write_jsonl(output_dir / "all.jsonl", records)
    _write_jsonl(output_dir / "train.jsonl", train_records)
    _write_jsonl(output_dir / "eval.jsonl", eval_records)

    manifest = {
        "dataset_name": dataset_name,
        "format": "jsonl",
        "fields_required_by_backend": [
            "main_id",
            "question",
            "cot_answer",
            "groundtruth",
        ],
        "counts": {
            "all": len(records),
            "train": len(train_records),
            "eval": len(eval_records),
        },
    }
    manifest.update(manifest_extra)
    _write_manifest(output_dir / "manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export hand-labeled plan examples to predictor JSONL format."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to plan_examples_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the exported JSONL files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="plan_predictor_v1",
        help="Dataset name to write into manifest.json.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of ordered samples to keep in train split.",
    )
    parser.add_argument(
        "--split-spec",
        default="",
        help=(
            "Optional JSON split spec with non_test_pool/test ids. "
            "When set, only non_test_pool samples are used to form train/eval."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    export_dataset(
        Path(args.input).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.dataset_name,
        args.train_ratio,
        Path(args.split_spec).expanduser().resolve() if args.split_spec else None,
    )


if __name__ == "__main__":
    main()
