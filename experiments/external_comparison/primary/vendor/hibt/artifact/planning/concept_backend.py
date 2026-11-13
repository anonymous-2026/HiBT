#!/usr/bin/env python3
"""Concept-backed behavior-tree generator.

This module turns a single ``target + world_state`` request into:

    question -> predictor -> predicted_concepts -> decoder -> BT skeleton

It is the online/runtime counterpart of the existing offline evaluation chain.
"""

from __future__ import annotations

import copy
import hashlib
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
DEFAULT_STORAGE_ROOT = str(ARTIFACT_DATA_DIR / "runtime" / "planner")
DEFAULT_PROTOTYPE_BANK = str(
    ARTIFACT_DATA_DIR / "pyramids" / "plan_bank_qwen3_8b_v2.pt"
)
DEFAULT_INFERENCE_DTYPE = "bfloat16"
DEFAULT_ABLATION_CONFIDENCE_THRESHOLD = 0.01


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
        repeated_tick_closure_boost: bool = True,
        weak_terminal_closure: bool = False,
        disable_guided_prefix_closure: bool = False,
        ablation_mode: str = "normal",
        compare_to_normal: bool = True,
        ablation_confidence_threshold: float = DEFAULT_ABLATION_CONFIDENCE_THRESHOLD,
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
        self.repeated_tick_closure_boost = repeated_tick_closure_boost
        self.weak_terminal_closure = weak_terminal_closure
        self.disable_guided_prefix_closure = disable_guided_prefix_closure
        self.ablation_mode = ablation_mode
        self.compare_to_normal = compare_to_normal
        self.ablation_confidence_threshold = ablation_confidence_threshold
        if self.ablation_mode not in {"normal", "ruleonly", "shuffle", "random"}:
            raise ValueError(f"Unsupported concept ablation mode: {self.ablation_mode}")

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

    @staticmethod
    def _stable_seed(text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _clone_levels(levels: list[torch.Tensor]) -> list[torch.Tensor]:
        return [level.clone() for level in levels]

    def _zero_levels_like_bank(self) -> list[torch.Tensor]:
        template = self.prototype_bank["entries"][0]["gt_concepts"]
        return [torch.zeros_like(level).float().cpu() for level in template]

    def _apply_concept_ablation(
        self, predicted_levels: list[torch.Tensor], question: str
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if self.ablation_mode == "normal":
            return self._clone_levels(predicted_levels), {
                "mode": "normal",
                "source": "predictor",
                "seed": None,
            }

        if self.ablation_mode == "ruleonly":
            ablated_levels = self._zero_levels_like_bank()
            return ablated_levels, {
                "mode": "ruleonly",
                "source": "zero_bank_template",
                "seed": None,
            }

        seed = self._stable_seed(f"{self.ablation_mode}:{question}")
        if self.ablation_mode == "shuffle":
            bank_entries = self.prototype_bank["entries"]
            entry = bank_entries[seed % len(bank_entries)]
            return [level.float().cpu().clone() for level in entry["gt_concepts"]], {
                "mode": self.ablation_mode,
                "source": "prototype_bank_entry",
                "seed": seed,
                "main_id": entry.get("main_id"),
            }

        ablated_levels: list[torch.Tensor] = []
        for level_idx, level in enumerate(predicted_levels):
            level_cpu = level.float().cpu()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + level_idx)
            if self.ablation_mode == "random":
                ablated_levels.append(
                    torch.randn(
                        level_cpu.shape,
                        generator=generator,
                        dtype=level_cpu.dtype,
                    )
                )
            else:
                raise ValueError(f"Unsupported concept ablation mode: {self.ablation_mode}")

        return ablated_levels, {
            "mode": self.ablation_mode,
            "source": "predictor_ablated",
            "seed": seed,
        }

    @staticmethod
    def _compute_concept_deltas(
        reference_levels: list[torch.Tensor], compared_levels: list[torch.Tensor]
    ) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for level_idx, (reference, compared) in enumerate(
            zip(reference_levels, compared_levels, strict=True)
        ):
            if reference.numel() == 0:
                l2_delta = 0.0
                mean_abs_delta = 0.0
            else:
                diff = compared.float() - reference.float()
                l2_delta = float(diff.norm().item())
                mean_abs_delta = float(diff.abs().mean().item())
            deltas.append(
                {
                    "level": level_idx,
                    "shape": list(reference.shape),
                    "l2_delta": l2_delta,
                    "mean_abs_delta": mean_abs_delta,
                }
            )
        return deltas

    @staticmethod
    def _fingerprint_payload(payload: Any) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def _augment_decode_report(
        self,
        report: dict[str, Any],
        decoded: dict[str, Any],
        ablation_details: dict[str, Any],
        concept_deltas: list[dict[str, Any]],
        normal_decoded: dict[str, Any] | None = None,
        normal_report: dict[str, Any] | None = None,
    ) -> None:
        plan_fingerprint = self._fingerprint_payload(decoded["pyramid"])
        action_fingerprint = self._fingerprint_payload(report["repaired_action_sequence"])
        report["concept_ablation"] = ablation_details
        report["concept_delta_vs_normal"] = concept_deltas
        report["repaired_plan_fingerprint"] = plan_fingerprint
        report["repaired_action_sequence_fingerprint"] = action_fingerprint

        if normal_decoded is None or normal_report is None:
            return

        normal_plan_fingerprint = self._fingerprint_payload(normal_decoded["pyramid"])
        normal_action_fingerprint = self._fingerprint_payload(
            normal_report["repaired_action_sequence"]
        )
        report["comparison_to_normal"] = {
            "enabled": True,
            "normal_repaired_plan_fingerprint": normal_plan_fingerprint,
            "normal_repaired_action_sequence_fingerprint": normal_action_fingerprint,
            "same_repaired_plan": normal_plan_fingerprint == plan_fingerprint,
            "same_repaired_action_sequence": (
                normal_action_fingerprint == action_fingerprint
            ),
            "changed_vs_normal": normal_plan_fingerprint != plan_fingerprint,
            "normal_repaired_action_sequence": normal_report["repaired_action_sequence"],
        }

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
        baseline_levels = self._predict_concepts(question)
        decoded_levels, ablation_details = self._apply_concept_ablation(
            baseline_levels, question
        )
        concept_deltas = self._compute_concept_deltas(baseline_levels, decoded_levels)

        decoded, report = decode_with_slot_retrieval_and_repair(
            predicted_levels=decoded_levels,
            bank=self.prototype_bank,
            input_payload={
                "main_id": "<adhoc>",
                "target": target,
                "initial_state": copy.deepcopy(world_state),
            },
            top_k=self.top_k,
            enable_repair=self.enable_repair,
            stable_targets=self.stable_targets,
            repeated_tick_closure_boost=self.repeated_tick_closure_boost,
            strict_guidance=self.ablation_mode != "normal",
            use_target_rerank=self.ablation_mode == "normal",
            weak_terminal_closure=self.weak_terminal_closure,
            disable_guided_prefix_closure=self.disable_guided_prefix_closure,
        )
        if (
            self.ablation_mode != "normal"
            and self.enable_repair
            and report.get("template_matches")
        ):
            top_score = float(report["template_matches"][0].get("mean_score", 0.0))
            if top_score <= self.ablation_confidence_threshold:
                decoded, report = decode_with_slot_retrieval_and_repair(
                    predicted_levels=decoded_levels,
                    bank=self.prototype_bank,
                    input_payload={
                        "main_id": "<adhoc>",
                        "target": target,
                        "initial_state": copy.deepcopy(world_state),
                    },
                    top_k=self.top_k,
                    enable_repair=False,
                    stable_targets=self.stable_targets,
                    repeated_tick_closure_boost=self.repeated_tick_closure_boost,
                    strict_guidance=True,
                    use_target_rerank=False,
                    weak_terminal_closure=self.weak_terminal_closure,
                    disable_guided_prefix_closure=self.disable_guided_prefix_closure,
                )
                report.setdefault("repair_notes", []).append(
                    f"ablation_confidence_gate_triggered:{top_score:.6f}"
                )
        normal_decoded: dict[str, Any] | None = None
        normal_report: dict[str, Any] | None = None
        if self.compare_to_normal and self.ablation_mode != "normal":
            normal_decoded, normal_report = decode_with_slot_retrieval_and_repair(
                predicted_levels=baseline_levels,
                bank=self.prototype_bank,
                input_payload={
                    "main_id": "<adhoc>",
                    "target": target,
                    "initial_state": copy.deepcopy(world_state),
                },
                top_k=self.top_k,
                enable_repair=self.enable_repair,
                stable_targets=self.stable_targets,
                repeated_tick_closure_boost=self.repeated_tick_closure_boost,
                strict_guidance=False,
                use_target_rerank=True,
                weak_terminal_closure=self.weak_terminal_closure,
                disable_guided_prefix_closure=self.disable_guided_prefix_closure,
            )
        self._augment_decode_report(
            report=report,
            decoded=decoded,
            ablation_details=ablation_details,
            concept_deltas=concept_deltas,
            normal_decoded=normal_decoded,
            normal_report=normal_report,
        )
        behavior_tree = PyramidCompiler(decoded).compile()
        duration = time.perf_counter() - start
        generation = {
            "thought": (
                "Generated by predictor + decoder + deterministic compiler."
                if self.ablation_mode == "normal"
                else (
                    "Generated by concept ablation backend "
                    f"({self.ablation_mode}) + decoder + deterministic compiler."
                )
            ),
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
