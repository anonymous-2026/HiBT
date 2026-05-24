#!/usr/bin/env python3
"""Concept-backed behavior-tree generator.

This module turns a single ``target + world_state`` request into:

    question -> predictor -> predicted_concepts -> decoder -> BT skeleton

It is the online/runtime counterpart of the existing offline evaluation chain.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import (
    ARTIFACT_PLANNING_DIR,
    ARTIFACT_CONFIGS_DIR,
    ARTIFACT_DATA_DIR,
    REPO_ROOT,
    bootstrap_all,
)

bootstrap_all()

from compile_plan_to_bt import PyramidCompiler
from decode_plan_latents import decode_with_slot_retrieval_and_repair
from planner.eval_predictor import (
    _find_best_predictor_checkpoint,
    _inherit_pyramid_from_builder,
    _load_frozen_builder,
    _load_predictor,
    _resolve_checkpoint_path,
    _resolve_config_path,
)
from planner.env_tools import get_device
from planner.data_loader import BuilderInput
from planner.config_io import apply_storage_root, load_config


LOGGER = logging.getLogger("concept_backend")

DEFAULT_PREDICTOR_CONFIG = str(
    ARTIFACT_CONFIGS_DIR
    / "planner"
    / "train_predictor_Qwen3-8B_planlocal_4level_shared_v2_smoke.yml"
)
DEFAULT_STORAGE_ROOT = str(ARTIFACT_DATA_DIR / "runtime" / "planner_v2")
DEFAULT_PROTOTYPE_BANK = str(ARTIFACT_DATA_DIR / "pyramids" / "plan_bank_v2.pt")
DEFAULT_INFERENCE_DTYPE = "bfloat16"


def _resolve_repo_or_vendor_config(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return _resolve_config_path(raw, REPO_ROOT).resolve()


def _render_question(target: str, world_state: dict[str, Any]) -> str:
    payload = {
        "target": target,
        "initial_state": world_state,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


class ConceptBehaviorTreeGenerator:
    def __init__(
        self,
        predictor_config_path: str = DEFAULT_PREDICTOR_CONFIG,
        storage_root: str = DEFAULT_STORAGE_ROOT,
        prototype_bank_path: str = DEFAULT_PROTOTYPE_BANK,
        predictor_checkpoint_path: str | None = None,
        builder_checkpoint_path: str | None = None,
        device: str = "auto",
        top_k: int = 3,
        inference_dtype: str = DEFAULT_INFERENCE_DTYPE,
        enable_repair: bool = True,
        stable_targets: bool = True,
    ) -> None:
        self.predictor_config_path = Path(predictor_config_path).expanduser().resolve()
        self.storage_root = storage_root
        self.prototype_bank_path = Path(prototype_bank_path).expanduser().resolve()
        self.predictor_checkpoint_path = predictor_checkpoint_path
        self.builder_checkpoint_path = builder_checkpoint_path
        self.top_k = top_k
        self.inference_dtype = inference_dtype
        self.enable_repair = enable_repair
        self.stable_targets = stable_targets

        load_start = time.perf_counter()
        self.device = str(get_device(device))
        self.predictor_config = load_config(str(self.predictor_config_path))
        apply_storage_root(self.predictor_config, self.storage_root)

        builder_config_raw = self.predictor_config["model"]["builder"]["config_path"]
        self.builder_config_path = _resolve_repo_or_vendor_config(builder_config_raw)
        self.builder_config = load_config(str(self.builder_config_path))
        apply_storage_root(self.builder_config, self.storage_root)
        _inherit_pyramid_from_builder(self.predictor_config, self.builder_config)
        self._override_inference_dtype()

        if self.builder_checkpoint_path:
            builder_ckpt = Path(self.builder_checkpoint_path).expanduser().resolve()
        else:
            builder_ckpt_raw = self.predictor_config["model"]["builder"]["checkpoint_path"]
            builder_ckpt = _resolve_checkpoint_path(builder_ckpt_raw, self.storage_root).resolve()

        if self.predictor_checkpoint_path:
            predictor_ckpt = Path(self.predictor_checkpoint_path).expanduser().resolve()
        else:
            checkpoint_dir = Path(self.predictor_config["log"]["checkpoint_path"])
            predictor_ckpt = _find_best_predictor_checkpoint(checkpoint_dir)

        bootstrap_logger = logging.getLogger("concept_backend.bootstrap")
        self.builder = _load_frozen_builder(
            self.builder_config, builder_ckpt, self.device, bootstrap_logger
        )
        self.predictor = _load_predictor(
            self.predictor_config, self.builder, predictor_ckpt, self.device, bootstrap_logger
        )
        self._align_runtime_module_dtypes()
        self.max_length = self.predictor_config["model"]["pyramid"]["max_seq_len"]
        self.prototype_bank = torch.load(self.prototype_bank_path, map_location="cpu")
        self.load_duration_sec = time.perf_counter() - load_start

    def _override_inference_dtype(self) -> None:
        if not self.device.startswith("cuda"):
            return
        reason_cfg = self.builder_config.get("model", {}).get("reason_model", {})
        if reason_cfg:
            reason_cfg["torch_dtype"] = self.inference_dtype

    def _align_runtime_module_dtypes(self) -> None:
        if not self.device.startswith("cuda"):
            return
        model_dtype = self.predictor.reason_model.dtype
        for module in (
            self.builder.back_proj,
            self.predictor.level_embeddings,
            self.predictor.position_embeddings,
            self.predictor.concept_head,
        ):
            module.to(device=self.device, dtype=model_dtype)

    def _predict_concepts(self, question: str) -> list[torch.Tensor]:
        tokenizer = self.builder.tokenizer
        tokenized = tokenizer(
            [question],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        q_ids = tokenized["input_ids"].to(self.device)
        q_mask = tokenized["attention_mask"].to(self.device)
        with torch.inference_mode():
            output = self.predictor(
                question_ids=q_ids,
                question_attention_mask=q_mask,
            )
        return [tensor.squeeze(0).float().cpu() for tensor in output.predicted_concepts]

    def generate(
        self, target: str, world_state: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        question = _render_question(target, world_state)
        start = time.perf_counter()
        predicted_levels = self._predict_concepts(question)
        decoded, report = decode_with_slot_retrieval_and_repair(
            predicted_levels=predicted_levels,
            bank=self.prototype_bank,
            input_payload={
                "main_id": "<adhoc>",
                "target": target,
                "initial_state": copy.deepcopy(world_state),
            },
            top_k=self.top_k,
            enable_repair=self.enable_repair,
            stable_targets=self.stable_targets,
        )
        behavior_tree = PyramidCompiler(decoded).compile()
        duration = time.perf_counter() - start
        generation = {
            "thought": "Generated by predictor + decoder + deterministic compiler.",
            "action_sequence": [
                f"{item['name']}({', '.join(item['args'])})"
                for item in decoded["pyramid"][3]["items"]
            ],
            "behavior_tree": behavior_tree,
            "decoded_plan": decoded,
            "decode_report": report,
            "simulation_world_state": copy.deepcopy(
                decoded["input"]["initial_state"]
            ),
        }
        return generation, duration
