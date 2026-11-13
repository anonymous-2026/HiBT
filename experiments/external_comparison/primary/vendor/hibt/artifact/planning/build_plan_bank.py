#!/usr/bin/env python3
"""Build a prototype bank from a frozen builder checkpoint.

This bridges the continuous concept space and the discrete plan schema. The
script runs the frozen builder over each hand-labeled sample, captures the
ground-truth concept tensors, and pairs them with the corresponding discrete
plan JSON.

The resulting bank is later consumed by ``decode_plan_latents.py`` for
nearest-neighbor latent decoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import (
    ARTIFACT_CONFIGS_DIR,
    ARTIFACT_DATA_DIR,
    bootstrap_planner_runtime,
)

bootstrap_planner_runtime()

from planner.concept_builder import ConceptPyramidBuilder
from planner.data_loader import BuilderInput
from planner.config_io import load_config


LOGGER = logging.getLogger("build_plan_bank")


def _align_builder_runtime_module_dtypes(builder: ConceptPyramidBuilder, device: str) -> None:
    if not device.startswith("cuda"):
        return
    model_dtype = builder.reason_model.dtype
    for module in (
        builder.input_proj,
        builder.input_proj_norm,
        builder.level_projs,
        builder.back_proj,
    ):
        module.to(device=device, dtype=model_dtype)
    for param in builder.concept_queries:
        param.data = param.data.to(device=device, dtype=model_dtype)
    builder.temperature.data = builder.temperature.data.to(device=device, dtype=model_dtype)


def _resolve_checkpoint(config_path: Path, checkpoint_path: str) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_absolute():
        return checkpoint
    return config_path.resolve().parents[3] / checkpoint


def _load_frozen_builder(
    builder_config_path: Path,
    builder_checkpoint_path: Path,
    device: str,
) -> ConceptPyramidBuilder:
    builder_cfg = load_config(str(builder_config_path))
    builder = ConceptPyramidBuilder(builder_cfg)
    builder.to(device)
    _align_builder_runtime_module_dtypes(builder, device)
    checkpoint = torch.load(builder_checkpoint_path, map_location="cpu")
    builder.load_state_dict(checkpoint["model_state_dict"], strict=False)
    for param in builder.parameters():
        param.requires_grad = False
    builder.eval()
    return builder


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_input(row: dict[str, Any]) -> BuilderInput:
    return BuilderInput(
        questions=[row["question"]],
        cot_answers=[row["cot_answer"]],
        solutions=[],
        main_ids=[row["main_id"]],
    )


def build_bank(
    rows: list[dict[str, Any]],
    builder: ConceptPyramidBuilder,
    device: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in rows:
            batch = _build_input(row)
            pyramid = builder(batch)
            gt_concepts = [concept.detach().cpu().squeeze(0) for concept in pyramid.concepts]
            entries.append(
                {
                    "main_id": row["main_id"],
                    "problem_id": row.get("problem_id", row["main_id"]),
                    "target": row.get("target"),
                    "source_path": row.get("source_path"),
                    "question": row["question"],
                    "cot_answer": row["cot_answer"],
                    "groundtruth": row.get("groundtruth"),
                    "pyramid_json": row["pyramid_json"],
                    "template_lengths": [
                        len(level.get("items", [])) for level in row["pyramid_json"]
                    ],
                    "gt_concepts": gt_concepts,
                }
            )

    return {
        "bank_version": "plan-bank-v1",
        "schema_version": "plan-schema-v1",
        "device_used": device,
        "num_entries": len(entries),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prototype bank from frozen Builder gt_concepts."
    )
    parser.add_argument(
        "--builder-config",
        default=str(
            ARTIFACT_CONFIGS_DIR
            / "planner"
            / "train_builder_Qwen3-8B_planlocal_4level_smoke.yml"
        ),
        help="Path to the Builder YAML config.",
    )
    parser.add_argument(
        "--builder-checkpoint",
        default=str(
            ARTIFACT_DATA_DIR
            / "runtime"
            / "planner"
            / "EXPERIMENT/planner/builder/Planner_Qwen3-8B_4level_smoke/checkpoints/checkpoint_best_eval-epoch0-step5.pt"
        ),
        help="Path to the Builder checkpoint.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        default=str(ARTIFACT_DATA_DIR / "datasets" / "plan_predictor_v1" / "all.jsonl"),
        help="Path to the exported predictor dataset JSONL.",
    )
    parser.add_argument(
        "--output",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_bank_qwen3_8b_v1.pt"),
        help="Where to write the prototype bank (.pt).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for running the frozen Builder.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    builder_config_path = Path(args.builder_config)
    builder_checkpoint_path = Path(args.builder_checkpoint)
    dataset_jsonl_path = Path(args.dataset_jsonl)
    output_path = Path(args.output)

    rows = _read_jsonl(dataset_jsonl_path)
    LOGGER.info("Loaded %d dataset rows from %s", len(rows), dataset_jsonl_path)

    builder = _load_frozen_builder(
        builder_config_path=builder_config_path,
        builder_checkpoint_path=builder_checkpoint_path,
        device=args.device,
    )

    bank = build_bank(rows=rows, builder=builder, device=args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path)
    LOGGER.info("Saved prototype bank with %d entries to %s", bank["num_entries"], output_path)


if __name__ == "__main__":
    main()
