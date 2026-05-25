#!/usr/bin/env python3
"""Decode predictor concept tensors into a discrete plan.

Version 2 keeps latent retrieval explicit, but no longer relies on whole-sample
template lookup alone. The pipeline is:

1. Read ``predicted_concepts`` (or ``gt_concepts``) from eval artifacts.
2. Retrieve per-level / per-slot nearest neighbors from the prototype bank.
3. Apply structural repair with the local domain model and the sample input
   state/target to synthesize a legal action sequence and pyramid.
4. Optionally compile the pyramid into a behavior tree and run ``sk_sim_run``.

The repair stage is intentionally deterministic. It separates:
  - whether the latent representation contains task signal, from
  - whether the final plan obeys the local action/state semantics.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import ARTIFACT_DATA_DIR, ARTIFACT_EVAL_DIR, bootstrap_runtime

bootstrap_runtime()
if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))

from compile_plan_to_bt import PyramidCompiler, run_sk_simulation
from runtime.minimal_bt import WorldState, ground_action


LOGGER = logging.getLogger("decode_plan_latents")

GOAL_TO_ACTION = {
    "is_inserted_to": "insert",
    "is_placed_to": "place",
    "is_screwed_to": "screw",
}


def _load_tensor_list(
    path: Path, concept_source: str
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    key = "predicted_concepts" if concept_source == "predicted" else "gt_concepts"
    concepts = [tensor.squeeze(0).float().cpu() for tensor in payload[key]]
    return concepts, payload


def _flatten_level(level_tensor: torch.Tensor) -> torch.Tensor:
    return level_tensor.reshape(-1)


def _flatten_slot(slot_tensor: torch.Tensor) -> torch.Tensor:
    return slot_tensor.reshape(-1)


def _cosine_score(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_flat = _flatten_level(lhs)
    rhs_flat = _flatten_level(rhs)
    return float(F.cosine_similarity(lhs_flat, rhs_flat, dim=0).item())


def _cosine_slot_score(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(_flatten_slot(lhs), _flatten_slot(rhs), dim=0).item()
    )


def _parse_input_payload(input_path: Path | None) -> dict[str, Any] | None:
    if input_path is None:
        return None
    raw = json.loads(input_path.read_text())
    question = raw.get("question")
    if not isinstance(question, str):
        return None
    try:
        parsed = json.loads(question)
    except json.JSONDecodeError:
        return None
    return {
        "main_id": raw.get("main_id"),
        "target": parsed.get("target"),
        "initial_state": parsed.get("initial_state"),
    }


def _parse_call(call: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", call)
    if not match:
        raise ValueError(f"Invalid grounded call: {call}")
    name = match.group(1)
    args_part = match.group(2).strip()
    args = [] if not args_part else [arg.strip() for arg in args_part.split(",")]
    return name, args


def _call(name: str, args: list[str]) -> str:
    return f"{name}({', '.join(args)})"


def _find_relation(
    state: dict[str, Any], source: str, name: str, target: str | None = None
) -> bool:
    for relation in state.get("relations", []):
        if relation.get("source") != source or relation.get("name") != name:
            continue
        if target is None or relation.get("target") == target:
            return True
    return False


def _find_constraint(
    state: dict[str, Any], source: str, name: str, target: str | None = None
) -> bool:
    for relation in state.get("constraints", []):
        if relation.get("source") != source or relation.get("name") != name:
            continue
        if target is None or relation.get("target") == target:
            return True
    return False


def _find_property(state: dict[str, Any], obj_name: str, prop: str) -> bool:
    for obj in state.get("objects", []):
        if obj.get("name") != obj_name:
            continue
        return prop in obj.get("properties", [])
    return False


def _set_property(
    state: dict[str, Any], obj_name: str, prop: str, enabled: bool = True
) -> None:
    for obj in state.get("objects", []):
        if obj.get("name") != obj_name:
            continue
        props = obj.setdefault("properties", [])
        if enabled:
            if prop not in props:
                props.append(prop)
        else:
            if prop in props:
                props.remove(prop)
        return


def _tool_names(state: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for relation in state.get("constraints", []):
        if relation.get("name") == "can_manipulate":
            source = relation.get("source")
            if isinstance(source, str):
                names.add(source)
    for obj in state.get("objects", []):
        name = obj.get("name")
        if isinstance(name, str) and "gripper" in name:
            names.add(name)
    return names


def _hand_names(state: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for obj in state.get("objects", []):
        name = obj.get("name")
        if isinstance(name, str) and "hand" in name:
            names.add(name)
    return names


def _holding_target(state: dict[str, Any], source: str) -> str | None:
    for relation in state.get("relations", []):
        if relation.get("source") == source and relation.get("name") == "hold":
            return relation.get("target")
    return None


def _normalize_state_for_kios(state: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(state)
    tool_names = _tool_names(normalized)
    hand_names = _hand_names(normalized)

    held_by_hand: dict[str, str] = {}
    held_targets: set[str] = set()
    for relation in normalized.get("relations", []):
        if relation.get("name") != "hold":
            continue
        source = relation.get("source")
        target = relation.get("target")
        if source in hand_names and isinstance(target, str):
            held_by_hand[target] = source
        if isinstance(target, str):
            held_targets.add(target)

    for hand in hand_names:
        _set_property(normalized, hand, "is_empty", _holding_target(normalized, hand) is None)

    for tool in tool_names:
        tool_payload = _holding_target(normalized, tool)
        _set_property(normalized, tool, "is_empty", tool_payload is None)
        equippable = tool_payload is None and tool not in held_by_hand
        _set_property(normalized, tool, "is_equippable", equippable)

    return normalized


def _is_call_satisfied(state: dict[str, Any], call: str) -> bool:
    name, args = _parse_call(call)
    if name in {
        "hold",
        "is_inserted_to",
        "is_placed_to",
        "is_screwed_to",
    }:
        return _find_relation(state, args[0], name, args[1])
    if name in {"can_insert_to", "can_place_to", "can_screw_to", "can_manipulate"}:
        return _find_constraint(state, args[0], name, args[1])
    if name in {"is_empty", "is_equippable"}:
        return _find_property(state, args[0], name)
    return False


def _get_current_tool(state: dict[str, Any], hand: str) -> str | None:
    for relation in state.get("relations", []):
        if relation.get("source") == hand and relation.get("name") == "hold":
            return relation.get("target")
    return None


def _get_part_held_by_tool(state: dict[str, Any], tool: str) -> str | None:
    for relation in state.get("relations", []):
        if relation.get("source") == tool and relation.get("name") == "hold":
            return relation.get("target")
    return None


def _infer_hand(state: dict[str, Any]) -> str:
    object_names = [obj.get("name") for obj in state.get("objects", [])]
    if "left_hand" in object_names:
        return "left_hand"
    for name in object_names:
        if name and "hand" in name:
            return name
    return "left_hand"


def _infer_tool_for_part(state: dict[str, Any], part: str) -> str | None:
    for relation in state.get("constraints", []):
        if relation.get("name") == "can_manipulate" and relation.get("target") == part:
            return relation.get("source")
    return None


def _stable_target_from_action(name: str, args: list[str]) -> str | None:
    if name == "put_down":
        return _call("is_empty", [args[1]])
    if name == "unload_tool":
        return _call("is_equippable", [args[1]])
    if name == "load_tool":
        return _call("hold", [args[0], args[1]])
    if name in {"pick_up", "pull", "detach", "unscrew"}:
        return _call("hold", [args[1], args[2]])
    if name == "change_tool":
        return _call("hold", [args[0], args[2]])
    return None


def _naive_target_from_action(name: str, args: list[str]) -> str | None:
    if name == "put_down":
        return _call("is_empty", [args[1]])
    if name == "unload_tool":
        return _call("is_empty", [args[0]])
    if name == "load_tool":
        return _call("hold", [args[0], args[1]])
    if name in {"pick_up", "pull", "detach", "unscrew"}:
        return _call("hold", [args[1], args[2]])
    if name == "change_tool":
        return _call("hold", [args[0], args[2]])
    return None


def _op_to_call(op: Any) -> str:
    args = [op.object_name]
    if op.property_value is not None:
        args.append(op.property_value)
    return _call(op.property_name, args)


def _get_bank_entry(bank: dict[str, Any], main_id: str) -> dict[str, Any]:
    return next(entry for entry in bank["entries"] if entry["main_id"] == main_id)


def _extract_template_action_names(entry: dict[str, Any]) -> list[str]:
    return [
        item["name"]
        for item in entry.get("pyramid_json", [None, None, None, {"items": []}])[3].get(
            "items", []
        )
        if isinstance(item, dict) and item.get("name")
    ]


def _extract_slot_action_names(slot_retrievals: list[dict[str, Any]]) -> list[str]:
    for level_report in slot_retrievals:
        if level_report.get("level") != 3:
            continue
        names: list[str] = []
        for slot in level_report.get("slots", []):
            topk = slot.get("topk") or []
            if not topk:
                continue
            item = topk[0].get("item") or {}
            name = item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return names
    return []


def _resolve_retrieved_action_hints(
    template_action_names: list[str], slot_action_names: list[str]
) -> list[str]:
    support_actions = {"put_down", "change_tool", "unload_tool", "load_tool"}
    if not template_action_names:
        return list(slot_action_names)
    resolved = list(template_action_names)
    if not any(name in support_actions for name in resolved):
        support_prefix: list[str] = []
        for name in slot_action_names:
            if name in support_actions and name not in support_prefix:
                support_prefix.append(name)
        if support_prefix:
            return support_prefix + resolved
    return resolved


def _prefer_change_tool_from_hints(action_hints: list[str]) -> bool:
    if "change_tool" not in action_hints:
        return False
    change_idx = action_hints.index("change_tool")
    unload_idx = (
        min(
            idx
            for idx, name in enumerate(action_hints)
            if name in {"unload_tool", "load_tool"}
        )
        if any(name in {"unload_tool", "load_tool"} for name in action_hints)
        else len(action_hints) + 1
    )
    return change_idx < unload_idx


def _retrieve_template_scores(
    predicted_levels: list[torch.Tensor], bank: dict[str, Any]
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for entry in bank["entries"]:
        level_scores: list[float] = []
        for predicted_level, prototype_level in zip(
            predicted_levels, entry["gt_concepts"], strict=True
        ):
            proto = prototype_level.float().cpu()
            level_scores.append(_cosine_score(predicted_level, proto))
        scored.append(
            {
                "main_id": entry["main_id"],
                "problem_id": entry.get("problem_id"),
                "target": entry.get("target"),
                "source_path": entry.get("source_path"),
                "template_lengths": entry.get("template_lengths"),
                "level_scores": level_scores,
                "mean_score": sum(level_scores) / len(level_scores) if level_scores else float("-inf"),
            }
        )
    scored.sort(key=lambda item: item["mean_score"], reverse=True)
    return scored


def _rerank_template_scores_for_target(
    template_scores: list[dict[str, Any]], bank: dict[str, Any], target_call: str
) -> list[dict[str, Any]]:
    target_predicate, target_args = _parse_call(target_call)
    target_argc = len(target_args)
    reranked: list[dict[str, Any]] = []
    for item in template_scores:
        entry = _get_bank_entry(bank, item["main_id"])
        entry_target = entry.get("target")
        compatibility_bonus = 0.0
        if isinstance(entry_target, str):
            entry_predicate, entry_args = _parse_call(entry_target)
            if entry_predicate == target_predicate:
                compatibility_bonus += 2.0
                if len(entry_args) == target_argc:
                    compatibility_bonus += 0.2
            action_names = _extract_template_action_names(entry)
            terminal_action = GOAL_TO_ACTION.get(target_predicate)
            if terminal_action and action_names and action_names[-1] == terminal_action:
                compatibility_bonus += 0.5
            if target_predicate == "hold" and any(
                name in {"change_tool", "load_tool"} for name in action_names
            ):
                compatibility_bonus += 0.2
        reranked.append(
            {
                **item,
                "compatibility_bonus": compatibility_bonus,
                "reranked_score": item["mean_score"] + compatibility_bonus,
            }
        )
    reranked.sort(key=lambda entry: entry["reranked_score"], reverse=True)
    return reranked


def _build_slot_prototypes(
    bank: dict[str, Any], target_main_id: str | None = None
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    prototypes: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in bank["entries"]:
        if target_main_id is not None and entry["main_id"] == target_main_id:
            continue
        for level_idx, (level_json, level_vec) in enumerate(
            zip(entry["pyramid_json"], entry["gt_concepts"], strict=True)
        ):
            for slot_idx, item in enumerate(level_json.get("items", [])):
                prototypes[level_idx][slot_idx].append(
                    {
                        "main_id": entry["main_id"],
                        "problem_id": entry.get("problem_id"),
                        "vector": level_vec[slot_idx].float().cpu(),
                        "item": copy.deepcopy(item),
                    }
                )
    return prototypes


def _retrieve_slot_candidates(
    predicted_levels: list[torch.Tensor],
    prototypes: dict[int, dict[int, list[dict[str, Any]]]],
    level_lengths: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    retrievals: list[dict[str, Any]] = []
    for level_idx, slot_count in enumerate(level_lengths):
        level_report = {"level": level_idx, "slots": []}
        for slot_idx in range(slot_count):
            predicted_slot = predicted_levels[level_idx][slot_idx]
            candidates = prototypes[level_idx].get(slot_idx, [])
            scored = []
            for candidate in candidates:
                score = _cosine_slot_score(predicted_slot, candidate["vector"])
                scored.append(
                    {
                        "score": score,
                        "main_id": candidate["main_id"],
                        "problem_id": candidate.get("problem_id"),
                        "item": candidate["item"],
                    }
                )
            scored.sort(key=lambda item: item["score"], reverse=True)
            level_report["slots"].append(
                {
                    "slot": slot_idx,
                    "topk": scored[:top_k],
                }
            )
        retrievals.append(level_report)
    return retrievals


def _build_goal_only_pyramid(
    target_call: str, initial_state: dict[str, Any], problem_id: str
) -> dict[str, Any]:
    predicate, args = _parse_call(target_call)
    return {
        "schema_version": "plan-schema-v1",
        "domain": "assembly_planning",
        "problem_id": problem_id,
        "input": {
            "target": target_call,
            "initial_state": initial_state,
        },
        "pyramid": [
            {
                "level": 0,
                "name": "goal",
                "items": [
                    {
                        "id": "g0",
                        "type": "goal",
                        "predicate": predicate,
                        "args": args,
                    }
                ],
            },
            {"level": 1, "name": "stable_subgoals", "items": []},
            {"level": 2, "name": "methods", "items": []},
            {"level": 3, "name": "actions", "items": []},
        ],
    }


def _synthesize_action_sequence(
    target_call: str,
    initial_state: dict[str, Any],
    prefer_change_tool: bool = False,
    guided_action_hints: set[str] | None = None,
    enforce_guided_support: bool = False,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    initial_state = _normalize_state_for_kios(initial_state)

    if _is_call_satisfied(initial_state, target_call):
        return [], ["goal_already_satisfied"]

    goal_predicate, goal_args = _parse_call(target_call)
    hand = _infer_hand(initial_state)
    support_actions = {"put_down", "change_tool", "unload_tool", "load_tool"}

    def _hinted(action_name: str) -> bool:
        if not enforce_guided_support and action_name in support_actions:
            return True
        return guided_action_hints is None or action_name in guided_action_hints

    if goal_predicate == "hold" and len(goal_args) == 2:
        source, target = goal_args
        current_tool = _get_current_tool(initial_state, hand)
        current_payload = (
            _get_part_held_by_tool(initial_state, current_tool) if current_tool else None
        )
        actions: list[tuple[str, list[str]]] = []
        notes: list[str] = []

        tool_names = _tool_names(initial_state)
        if source == hand and target in tool_names:
            if current_tool and current_tool != target:
                if current_payload:
                    if not prefer_change_tool and not _hinted("put_down"):
                        return [], ["guided_missing_put_down_for_hold_goal"]
                    actions.append(("put_down", [hand, current_tool, current_payload]))
                    notes.append("put_down_current_tool_payload")
                if prefer_change_tool:
                    if not _hinted("change_tool"):
                        return [], ["guided_missing_change_tool_for_hold_goal"]
                    actions.append(("change_tool", [hand, current_tool, target]))
                    notes.append("change_wrong_tool")
                else:
                    if not _hinted("unload_tool"):
                        return [], ["guided_missing_unload_tool_for_hold_goal"]
                    actions.append(("unload_tool", [hand, current_tool]))
                    notes.append("unload_wrong_tool")
            if current_tool != target:
                if not (prefer_change_tool and current_tool and current_tool != target):
                    if current_tool is not None and not _hinted("load_tool"):
                        return [], ["guided_missing_load_tool_for_hold_goal"]
                    actions.append(("load_tool", [hand, target]))
                    notes.append("load_required_tool_for_hold_goal")
            return actions, notes or ["hold_goal_already_loaded"]

        if source in tool_names:
            required_tool = source
            if current_tool and current_tool != required_tool:
                if current_payload:
                    if not prefer_change_tool and not _hinted("put_down"):
                        return [], ["guided_missing_put_down_for_pickup_goal"]
                    actions.append(("put_down", [hand, current_tool, current_payload]))
                    notes.append("put_down_current_tool_payload")
                if prefer_change_tool:
                    if not _hinted("change_tool"):
                        return [], ["guided_missing_change_tool_for_pickup_goal"]
                    actions.append(("change_tool", [hand, current_tool, required_tool]))
                    notes.append("change_wrong_tool")
                else:
                    if not _hinted("unload_tool"):
                        return [], ["guided_missing_unload_tool_for_pickup_goal"]
                    actions.append(("unload_tool", [hand, current_tool]))
                    notes.append("unload_wrong_tool")
            if current_tool != required_tool:
                if not (
                    prefer_change_tool
                    and current_tool
                    and current_tool != required_tool
                ):
                    if current_tool is not None and not _hinted("load_tool"):
                        return [], ["guided_missing_load_tool_for_pickup_goal"]
                    actions.append(("load_tool", [hand, required_tool]))
                    notes.append("load_required_tool_for_pickup_goal")
            held_part = _get_part_held_by_tool(initial_state, required_tool)
            if held_part and held_part != target:
                if not _hinted("put_down"):
                    return [], ["guided_missing_put_down_for_wrong_part"]
                actions.append(("put_down", [hand, required_tool, held_part]))
                notes.append("put_down_wrong_part_from_required_tool")
            if not _find_relation(initial_state, required_tool, "hold", target):
                actions.append(("pick_up", [hand, required_tool, target]))
                notes.append("pick_up_target_for_hold_goal")
            return actions, notes

    if goal_predicate == "is_empty" and len(goal_args) == 1:
        tool = goal_args[0]
        held_part = _get_part_held_by_tool(initial_state, tool)
        if held_part is None:
            return [], ["normalize_empty_goal"]
        current_tool = _get_current_tool(initial_state, hand)
        notes: list[str] = []
        if current_tool and current_tool != tool:
            return [], [f"is_empty_goal_tool_not_loaded:{tool}"]
        actions = [("put_down", [hand, tool, held_part])]
        notes.append("put_down_for_is_empty_goal")
        return actions, notes

    if goal_predicate == "is_equippable" and len(goal_args) == 1:
        tool = goal_args[0]
        current_tool = _get_current_tool(initial_state, hand)
        held_part = _get_part_held_by_tool(initial_state, tool)
        actions: list[tuple[str, list[str]]] = []
        notes: list[str] = []
        if current_tool == tool:
            if held_part:
                actions.append(("put_down", [hand, tool, held_part]))
                notes.append("put_down_for_is_equippable_goal")
            actions.append(("unload_tool", [hand, tool]))
            notes.append("unload_for_is_equippable_goal")
            return actions, notes
        if held_part is None:
            return [], ["normalize_is_equippable_goal"]
        return [], [f"is_equippable_goal_tool_not_loaded:{tool}"]

    final_action_name = GOAL_TO_ACTION.get(goal_predicate)
    if final_action_name is None or len(goal_args) != 2:
        return [], [f"unsupported_goal:{goal_predicate}"]

    part, target = goal_args
    required_tool = _infer_tool_for_part(initial_state, part)
    if required_tool is None:
        return [], [f"missing_tool_for_part:{part}"]

    current_tool = _get_current_tool(initial_state, hand)
    current_part = (
        _get_part_held_by_tool(initial_state, current_tool) if current_tool else None
    )
    actions: list[tuple[str, list[str]]] = []
    notes: list[str] = []

    if current_tool and current_tool != required_tool:
        if current_part:
            if not prefer_change_tool and not _hinted("put_down"):
                return [], ["guided_missing_put_down_for_tool_switch"]
            actions.append(("put_down", [hand, current_tool, current_part]))
            notes.append("put_down_current_tool_payload")
        if prefer_change_tool:
            if not _hinted("change_tool"):
                return [], ["guided_missing_change_tool_for_tool_switch"]
            actions.append(("change_tool", [hand, current_tool, required_tool]))
            notes.append("change_wrong_tool")
        else:
            if not _hinted("unload_tool"):
                return [], ["guided_missing_unload_tool_for_tool_switch"]
            actions.append(("unload_tool", [hand, current_tool]))
            notes.append("unload_wrong_tool")
            if not _hinted("load_tool"):
                return [], ["guided_missing_load_tool_for_tool_switch"]
            actions.append(("load_tool", [hand, required_tool]))
            notes.append("load_required_tool")
    elif current_tool is None:
        actions.append(("load_tool", [hand, required_tool]))
        notes.append("load_tool_from_empty_hand")
    elif current_tool == required_tool:
        if current_part and current_part != part:
            if not _hinted("put_down"):
                return [], ["guided_missing_put_down_for_wrong_part"]
            actions.append(("put_down", [hand, current_tool, current_part]))
            notes.append("put_down_wrong_part_from_required_tool")
        notes.append("required_tool_already_loaded")

    if not _find_relation(initial_state, required_tool, "hold", part):
        actions.append(("pick_up", [hand, required_tool, part]))
        notes.append("pick_up_target_part")

    actions.append((final_action_name, [hand, required_tool, part, target]))
    notes.append("apply_final_goal_action")
    return actions, notes


def _resolve_guided_goal_context(
    state: dict[str, Any], target_call: str
) -> dict[str, Any]:
    goal_predicate, goal_args = _parse_call(target_call)
    hand = _infer_hand(state)
    current_tool = _get_current_tool(state, hand)
    current_part = (
        _get_part_held_by_tool(state, current_tool) if current_tool else None
    )
    tool_names = _tool_names(state)
    required_tool = None
    part = None

    if goal_predicate == "hold" and len(goal_args) == 2:
        source, target = goal_args
        if source == hand and target in tool_names:
            required_tool = target
        elif source in tool_names:
            required_tool = source
            part = target
    elif goal_predicate in GOAL_TO_ACTION and len(goal_args) == 2:
        part = goal_args[0]
        required_tool = _infer_tool_for_part(state, part)
    elif goal_predicate in {"is_empty", "is_equippable"} and len(goal_args) == 1:
        required_tool = goal_args[0]

    return {
        "hand": hand,
        "current_tool": current_tool,
        "current_part": current_part,
        "required_tool": required_tool,
        "part": part,
    }


def _guided_subgoal_from_action_hint(
    action_name: str, target_call: str, current_state: dict[str, Any]
) -> tuple[str | None, bool]:
    context = _resolve_guided_goal_context(current_state, target_call)
    hand = context["hand"]
    current_tool = context["current_tool"]
    current_part = context["current_part"]
    required_tool = context["required_tool"]
    part = context["part"]

    if action_name == "put_down":
        if current_tool and current_part:
            return _call("is_empty", [current_tool]), False
        return None, False
    if action_name == "unload_tool":
        if current_tool:
            return _call("is_equippable", [current_tool]), False
        return None, False
    if action_name == "load_tool":
        if required_tool:
            return _call("hold", [hand, required_tool]), False
        return None, False
    if action_name == "change_tool":
        if required_tool:
            return _call("hold", [hand, required_tool]), True
        return None, False
    if action_name == "pick_up":
        if required_tool and part:
            return _call("hold", [required_tool, part]), False
        return None, False
    return None, False


def _apply_action_sequence_to_state(
    state: dict[str, Any], action_sequence: list[tuple[str, list[str]]]
) -> dict[str, Any] | None:
    world = WorldState(copy.deepcopy(state))
    for action_name, action_args in action_sequence:
        if not world.apply_action(action_name, action_args):
            return None
    return world.to_json()


def _synthesize_terminal_closure(
    target_call: str,
    current_state: dict[str, Any],
    prefer_change_tool: bool,
    guided_action_hints: set[str] | None = None,
    enforce_guided_support: bool = False,
    weak_terminal_closure: bool = False,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    state = _normalize_state_for_kios(current_state)
    if _is_call_satisfied(state, target_call):
        return [], ["goal_already_satisfied_after_guidance"]

    goal_predicate, goal_args = _parse_call(target_call)
    if goal_predicate not in GOAL_TO_ACTION or len(goal_args) != 2:
        return _synthesize_action_sequence(
            target_call,
            state,
            prefer_change_tool=prefer_change_tool,
            guided_action_hints=guided_action_hints,
            enforce_guided_support=enforce_guided_support,
        )

    hand = _infer_hand(state)
    part, target = goal_args
    required_tool = _infer_tool_for_part(state, part)
    if required_tool is None:
        return [], [f"missing_tool_for_part:{part}"]

    actions: list[tuple[str, list[str]]] = []
    notes: list[str] = []

    hold_tool_goal = _call("hold", [hand, required_tool])
    if not _is_call_satisfied(state, hold_tool_goal):
        if weak_terminal_closure:
            return actions, notes + ["terminal_closure_skipped:hold_tool"]
        segment, segment_notes = _synthesize_action_sequence(
            hold_tool_goal,
            state,
            prefer_change_tool=prefer_change_tool,
            guided_action_hints=guided_action_hints,
            enforce_guided_support=enforce_guided_support,
        )
        next_state = _apply_action_sequence_to_state(state, segment)
        if next_state is None:
            return actions, notes + ["terminal_closure_failed:hold_tool"]
        actions.extend(segment)
        notes.extend(f"terminal_hold_tool:{note}" for note in segment_notes)
        state = _normalize_state_for_kios(next_state)

    hold_part_goal = _call("hold", [required_tool, part])
    if not _is_call_satisfied(state, hold_part_goal):
        if weak_terminal_closure:
            return actions, notes + ["terminal_closure_skipped:hold_part"]
        segment, segment_notes = _synthesize_action_sequence(
            hold_part_goal,
            state,
            prefer_change_tool=prefer_change_tool,
            guided_action_hints=guided_action_hints,
            enforce_guided_support=enforce_guided_support,
        )
        next_state = _apply_action_sequence_to_state(state, segment)
        if next_state is None:
            return actions, notes + ["terminal_closure_failed:hold_part"]
        actions.extend(segment)
        notes.extend(f"terminal_hold_part:{note}" for note in segment_notes)
        state = _normalize_state_for_kios(next_state)

    final_action_name = GOAL_TO_ACTION[goal_predicate]
    final_action_args = [hand, required_tool, part, target]
    preconditions, _ = ground_action(final_action_name, final_action_args)
    precondition_calls = [_op_to_call(op) for op in preconditions]
    unsatisfied = [call for call in precondition_calls if not _is_call_satisfied(state, call)]
    if unsatisfied:
        return actions, notes + [
            "terminal_closure_unsatisfied_preconditions:" + " | ".join(unsatisfied)
        ]

    actions.append((final_action_name, final_action_args))
    notes.append("terminal_apply_final_goal_action")
    return actions, notes


def _synthesize_action_sequence_with_retrieval_guidance(
    target_call: str,
    initial_state: dict[str, Any],
    retrieved_action_hints: list[str],
    strict_guidance: bool = False,
    weak_terminal_closure: bool = False,
    disable_guided_prefix_closure: bool = False,
) -> tuple[list[tuple[str, list[str]]], list[str], dict[str, Any]]:
    current_state = _normalize_state_for_kios(initial_state)
    if _is_call_satisfied(current_state, target_call):
        return [], ["goal_already_satisfied"], {
            "retrieved_action_hints": retrieved_action_hints,
            "prefer_change_tool": False,
        }

    prefer_change_tool = _prefer_change_tool_from_hints(retrieved_action_hints)
    goal_predicate, _ = _parse_call(target_call)
    terminal_action = GOAL_TO_ACTION.get(goal_predicate)
    prefix_hints = [
        name for name in retrieved_action_hints if not terminal_action or name != terminal_action
    ]

    guided_actions: list[tuple[str, list[str]]] = []
    repair_notes: list[str] = []
    if retrieved_action_hints:
        repair_notes.append(
            "guided_action_hints:" + " -> ".join(retrieved_action_hints)
        )

    if disable_guided_prefix_closure and prefix_hints:
        repair_notes.append("guided_prefix_closure_disabled")

    for action_name in prefix_hints:
        if disable_guided_prefix_closure:
            repair_notes.append(f"guided_hint_skipped_by_config:{action_name}")
            continue
        subgoal_call, local_prefer_change = _guided_subgoal_from_action_hint(
            action_name, target_call, current_state
        )
        if subgoal_call is None:
            repair_notes.append(f"guided_hint_skipped:{action_name}")
            continue
        segment, segment_notes = _synthesize_action_sequence(
            subgoal_call,
            current_state,
            prefer_change_tool=local_prefer_change or prefer_change_tool,
            guided_action_hints=set(retrieved_action_hints) if strict_guidance else None,
            enforce_guided_support=strict_guidance,
        )
        if not segment and not _is_call_satisfied(current_state, subgoal_call):
            repair_notes.append(f"guided_hint_failed:{action_name}")
            continue
        next_state = _apply_action_sequence_to_state(current_state, segment)
        if next_state is None:
            repair_notes.append(f"guided_hint_unapplied:{action_name}")
            continue
        guided_actions.extend(segment)
        current_state = _normalize_state_for_kios(next_state)
        repair_notes.append(f"guided_hint_applied:{action_name}")
        repair_notes.extend(f"guided:{note}" for note in segment_notes)

    if retrieved_action_hints:
        residual_actions, residual_notes = _synthesize_terminal_closure(
            target_call,
            current_state,
            prefer_change_tool=prefer_change_tool,
            guided_action_hints=set(retrieved_action_hints) if strict_guidance else None,
            enforce_guided_support=strict_guidance,
            weak_terminal_closure=weak_terminal_closure,
        )
    else:
        residual_actions, residual_notes = _synthesize_action_sequence(
            target_call,
            current_state,
            prefer_change_tool=prefer_change_tool,
        )
    guided_actions.extend(residual_actions)
    repair_notes.extend(f"guided_final:{note}" for note in residual_notes)
    return guided_actions, repair_notes, {
        "retrieved_action_hints": retrieved_action_hints,
        "prefer_change_tool": prefer_change_tool,
    }


def _build_pyramid_from_actions(
    target_call: str,
    initial_state: dict[str, Any],
    action_sequence: list[tuple[str, list[str]]],
    problem_id: str,
    stable_targets: bool = True,
    repeated_tick_closure_boost: bool = True,
) -> dict[str, Any]:
    goal_predicate, goal_args = _parse_call(target_call)
    goal_item = {
        "id": "g0",
        "type": "goal",
        "predicate": goal_predicate,
        "args": goal_args,
    }

    subgoal_items: list[dict[str, Any]] = []
    method_items: list[dict[str, Any]] = []
    action_items: list[dict[str, Any]] = []
    subgoal_ids_by_call: dict[str, str] = {}
    achieved_calls_by_subgoal_id: dict[str, str] = {}

    def ensure_subgoal(call: str) -> str:
        sg_id = f"sg{len(subgoal_items) + 1}"
        predicate, args = _parse_call(call)
        item = {
            "id": sg_id,
            "type": "subgoal",
            "predicate": predicate,
            "args": args,
            "supports": [],
        }
        subgoal_items.append(item)
        subgoal_ids_by_call[call] = sg_id
        achieved_calls_by_subgoal_id[sg_id] = call
        return sg_id

    achieved_order: list[str] = []
    for step_idx, (action_name, action_args) in enumerate(action_sequence):
        if step_idx == len(action_sequence) - 1:
            achieves = "g0"
        else:
            target_call_for_action = (
                _stable_target_from_action(action_name, action_args)
                if stable_targets
                else _naive_target_from_action(action_name, action_args)
            )
            if target_call_for_action is None:
                continue
            achieves = ensure_subgoal(target_call_for_action)

        preconditions, _ = ground_action(action_name, action_args)
        precondition_calls = [_op_to_call(op) for op in preconditions]
        requires: list[str] = [
            sg_id
            for sg_id in achieved_order
            if achieved_calls_by_subgoal_id[sg_id] in precondition_calls
        ]

        # Repeated-tick BT closure needs stronger ordering than plain precondition
        # intersection for tool-management actions. In particular:
        # - unload_tool should depend on a prior put_down(...)->is_empty(tool)
        # - load_tool should depend on the prior unload_tool(...)->is_equippable(tool)
        # This preserves the repeated-tick semantics that the hand-labeled
        # examples already use.
        if stable_targets and repeated_tick_closure_boost:
            if action_name == "unload_tool":
                empty_tool_call = _call("is_empty", [action_args[1]])
                sg_id = subgoal_ids_by_call.get(empty_tool_call)
                if sg_id and sg_id not in requires:
                    requires.append(sg_id)
            elif action_name == "load_tool":
                if step_idx > 0:
                    prev_name, prev_args = action_sequence[step_idx - 1]
                    if prev_name == "unload_tool":
                        equip_call = _call("is_equippable", [prev_args[1]])
                        sg_id = subgoal_ids_by_call.get(equip_call)
                        if sg_id and sg_id not in requires:
                            requires.append(sg_id)
            elif action_name == "pick_up":
                hold_hand_tool = _call("hold", [action_args[0], action_args[1]])
                sg_id = subgoal_ids_by_call.get(hold_hand_tool)
                if sg_id and sg_id not in requires:
                    requires.append(sg_id)
            elif action_name in {"insert", "place", "screw"}:
                hold_hand_tool = _call("hold", [action_args[0], action_args[1]])
                hold_tool_part = _call("hold", [action_args[1], action_args[2]])
                for call in (hold_hand_tool, hold_tool_part):
                    sg_id = subgoal_ids_by_call.get(call)
                    if sg_id and sg_id not in requires:
                        requires.append(sg_id)

        method_id = f"m{len(method_items) + 1}"
        method_items.append(
            {
                "id": method_id,
                "type": "method",
                "achieves": achieves,
                "action": {"name": action_name, "args": action_args},
                "requires": requires,
            }
        )
        if achieves != "g0":
            for subgoal in subgoal_items:
                if subgoal["id"] == achieves:
                    subgoal["supports"].append(method_id)
                    break
            achieved_order.append(achieves)

        action_items.append(
            {
                "step": step_idx,
                "type": "action",
                "name": action_name,
                "args": action_args,
                "derived_from": method_id,
            }
        )

    return {
        "schema_version": "plan-schema-v1",
        "domain": "assembly_planning",
        "problem_id": problem_id,
        "input": {"target": target_call, "initial_state": initial_state},
        "pyramid": [
            {"level": 0, "name": "goal", "items": [goal_item]},
            {"level": 1, "name": "stable_subgoals", "items": subgoal_items},
            {"level": 2, "name": "methods", "items": method_items},
            {"level": 3, "name": "actions", "items": action_items},
        ],
    }


def _extract_action_sequence_texts(pyramid: dict[str, Any]) -> list[str]:
    actions = pyramid["pyramid"][3]["items"]
    return [_call(action["name"], action["args"]) for action in actions]


def decode_with_slot_retrieval_and_repair(
    predicted_levels: list[torch.Tensor],
    bank: dict[str, Any],
    input_payload: dict[str, Any] | None,
    top_k: int,
    enable_repair: bool = True,
    stable_targets: bool = True,
    repeated_tick_closure_boost: bool = True,
    strict_guidance: bool = False,
    use_target_rerank: bool = True,
    weak_terminal_closure: bool = False,
    disable_guided_prefix_closure: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    template_scores = _retrieve_template_scores(predicted_levels, bank)
    if use_target_rerank and input_payload is not None and input_payload.get("target"):
        template_scores = _rerank_template_scores_for_target(
            template_scores, bank, input_payload["target"]
        )
    best_template = template_scores[0]
    best_entry = _get_bank_entry(bank, best_template["main_id"])
    selected_lengths = list(best_template["template_lengths"])
    target_main_id = input_payload.get("main_id") if input_payload else None
    prototypes = _build_slot_prototypes(bank, target_main_id=target_main_id)
    slot_retrievals = _retrieve_slot_candidates(
        predicted_levels=predicted_levels,
        prototypes=prototypes,
        level_lengths=selected_lengths,
        top_k=top_k,
    )
    template_action_names = _extract_template_action_names(best_entry)
    slot_action_names = _extract_slot_action_names(slot_retrievals)
    if strict_guidance:
        retrieved_action_hints = list(slot_action_names) or list(template_action_names)
    else:
        retrieved_action_hints = _resolve_retrieved_action_hints(
            template_action_names, slot_action_names
        )

    if input_payload is None or input_payload.get("target") is None:
        decoded = {
            "schema_version": "plan-schema-v1",
            "domain": "assembly_planning",
            "problem_id": best_entry.get("problem_id", best_entry["main_id"]),
            "input": {
                "target": best_entry["target"],
                "initial_state": json.loads(best_entry["question"]).get("initial_state"),
            },
            "pyramid": copy.deepcopy(best_entry["pyramid_json"]),
        }
        report = {
            "decoder_version": "plan-slot-retrieval-v2",
            "template_matches": template_scores[:top_k],
            "selected_length_template": selected_lengths,
            "slot_retrievals": slot_retrievals,
            "repair_notes": ["missing_input_payload_fell_back_to_template"],
            "repaired_action_sequence": _extract_action_sequence_texts(decoded),
        }
        return decoded, report

    target_call = input_payload["target"]
    initial_state = _normalize_state_for_kios(input_payload["initial_state"])
    problem_id = input_payload.get("main_id") or best_template["main_id"]

    if not enable_repair:
        decoded = {
            "schema_version": "plan-schema-v1",
            "domain": "assembly_planning",
            "problem_id": problem_id,
            "input": {
                "target": target_call,
                "initial_state": initial_state,
            },
            "pyramid": copy.deepcopy(best_entry["pyramid_json"]),
        }
        repair_notes = ["repair_disabled_template_passthrough"]
        retrieval_guidance = {
            "template_action_names": template_action_names,
            "slot_action_names": slot_action_names,
            "retrieved_action_hints": retrieved_action_hints,
            "prefer_change_tool": None,
        }
    else:
        action_sequence, repair_notes, retrieval_guidance = (
            _synthesize_action_sequence_with_retrieval_guidance(
                target_call=target_call,
                initial_state=initial_state,
                retrieved_action_hints=retrieved_action_hints,
                strict_guidance=strict_guidance,
                weak_terminal_closure=weak_terminal_closure,
                disable_guided_prefix_closure=disable_guided_prefix_closure,
            )
        )
        if not action_sequence and _is_call_satisfied(initial_state, target_call):
            decoded = _build_goal_only_pyramid(target_call, initial_state, problem_id)
        else:
            decoded = _build_pyramid_from_actions(
                target_call=target_call,
                initial_state=initial_state,
                action_sequence=action_sequence,
                problem_id=problem_id,
                stable_targets=stable_targets,
                repeated_tick_closure_boost=repeated_tick_closure_boost,
            )

    report = {
        "decoder_version": "plan-slot-retrieval-v2",
        "template_matches": template_scores[:top_k],
        "selected_length_template": selected_lengths,
        "slot_retrievals": slot_retrievals,
        "repair_notes": repair_notes,
        "repaired_action_sequence": _extract_action_sequence_texts(decoded),
        "target_from_input": target_call,
        "repair_enabled": enable_repair,
        "stable_targets_enabled": stable_targets,
        "repeated_tick_closure_boost_enabled": repeated_tick_closure_boost,
        "strict_guidance_enabled": strict_guidance,
        "weak_terminal_closure_enabled": weak_terminal_closure,
        "guided_prefix_closure_disabled": disable_guided_prefix_closure,
        "template_action_names": template_action_names,
        "slot_action_names": slot_action_names,
        "retrieved_action_hints": retrieved_action_hints,
        "retrieval_guidance": retrieval_guidance,
    }
    return decoded, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode predictor concepts into a discrete plan."
    )
    parser.add_argument(
        "--concepts-file",
        required=True,
        help="Path to concepts.pt emitted by eval_predictor.py.",
    )
    parser.add_argument(
        "--prototype-bank",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_bank_v1.pt"),
        help="Prototype bank path produced by build_plan_bank.py.",
    )
    parser.add_argument(
        "--concept-source",
        choices=("predicted", "gt"),
        default="predicted",
        help="Which tensors from concepts.pt to decode.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional input.json from eval_predictor sample artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many nearest candidates to include in the decode report.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for decoded pyramid, report, and optional BT artifacts.",
    )
    parser.add_argument(
        "--compile-and-evaluate",
        action="store_true",
        help="Compile the decoded pyramid and run sk_sim_run.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    concepts_path = Path(args.concepts_file)
    bank_path = Path(args.prototype_bank)
    input_path = Path(args.input_file) if args.input_file else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_levels, concept_payload = _load_tensor_list(
        concepts_path, args.concept_source
    )
    bank = torch.load(bank_path, map_location="cpu")
    input_payload = _parse_input_payload(input_path)

    decoded, report = decode_with_slot_retrieval_and_repair(
        predicted_levels=predicted_levels,
        bank=bank,
        input_payload=input_payload,
        top_k=args.top_k,
    )

    decoded_path = output_dir / "decoded_pyramid.json"
    report_path = output_dir / "decode_report.json"
    _write_json(decoded_path, decoded)
    _write_json(
        report_path,
        {
            **report,
            "concepts_file": str(concepts_path),
            "concept_source": args.concept_source,
            "prototype_bank": str(bank_path),
            "level_lengths_from_predictor": concept_payload.get("level_lengths"),
        },
    )
    LOGGER.info("Decoded pyramid written to %s", decoded_path)

    if args.compile_and_evaluate:
        behavior_tree = PyramidCompiler(decoded).compile()
        bt_path = output_dir / "compiled_bt.json"
        _write_json(bt_path, behavior_tree)

        evaluation = run_sk_simulation(
            copy.deepcopy(decoded["input"]["initial_state"]),
            copy.deepcopy(behavior_tree),
        )
        eval_path = output_dir / "compile_eval.json"
        _write_json(eval_path, evaluation)
        LOGGER.info("Compiled BT written to %s", bt_path)
        LOGGER.info("sk_sim_run result=%s", evaluation.get("result"))


if __name__ == "__main__":
    main()
