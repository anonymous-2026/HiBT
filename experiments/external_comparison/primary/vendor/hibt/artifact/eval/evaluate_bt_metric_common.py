#!/usr/bin/env python3
"""Common helpers for one-step BT metric evaluation scripts.

These scripts reuse the existing generate -> execution pipeline from
evaluate_bt_batch.py, then compute a specific metric:

- Exec: generated BT can be loaded and executed without evaluation error
- LC: proxy logical coherence for generated BTs
- SR: final sk_sim_run result is success
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from evaluate_bt_batch import (
    DeepSeekBehaviorTreeGenerator,
    evaluate_requests,
    parse_args,
)


def build_generator(args):
    if args.backend == "local":
        from generate_bt_only import BehaviorTreeGenerator

        return BehaviorTreeGenerator(
            model_name=args.model,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            device=args.device,
            prompt_profile=args.prompt_profile,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )

    if args.backend == "deepseek":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"DeepSeek backend requires API key in environment variable {args.api_key_env}"
            )
        return DeepSeekBehaviorTreeGenerator(
            api_key=api_key,
            model_name=args.model,
            prompt_profile=args.prompt_profile,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            base_url=args.base_url,
        )

    from concept_backend import ConceptBehaviorTreeGenerator

    return ConceptBehaviorTreeGenerator(
        predictor_config_path=args.planner_config,
        storage_root=args.planner_storage_root,
        prototype_bank_path=args.plan_bank,
        predictor_checkpoint_path=args.planner_predictor_ckpt or None,
        builder_checkpoint_path=args.planner_builder_ckpt or None,
        device=args.device,
        top_k=args.planner_top_k,
    )


def _normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _extract_action_nodes(node: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    name = node.get("name", "")
    if isinstance(name, str) and name.startswith("action:"):
        actions.append(name.split(":", 1)[1].strip())
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                actions.extend(_extract_action_nodes(child))
    return actions


def _dedup_first(seq: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _final_node_name(record: dict[str, Any]) -> str | None:
    result = record.get("evaluation_result") or {}
    final_node = result.get("final_node") or {}
    return final_node.get("name")


def passes_exec(record: dict[str, Any]) -> tuple[bool, str]:
    if record.get("generation_error"):
        return False, "generation_error"
    if record.get("evaluation_error"):
        return False, "evaluation_error"
    if not record.get("generation_result"):
        return False, "missing_generation_result"
    if not record.get("evaluation_result"):
        return False, "missing_evaluation_result"
    return True, "ok"


def passes_lc(record: dict[str, Any]) -> tuple[bool, str]:
    exec_ok, reason = passes_exec(record)
    if not exec_ok:
        return False, reason

    generation_result = record.get("generation_result") or {}
    action_sequence = generation_result.get("action_sequence")
    behavior_tree = generation_result.get("behavior_tree")
    if not isinstance(action_sequence, list):
        return False, "missing_action_sequence"
    if not isinstance(behavior_tree, dict):
        return False, "missing_behavior_tree"

    tree_actions = _dedup_first(_extract_action_nodes(behavior_tree))
    normalized_seq = [_normalize_action_text(item) for item in action_sequence]
    normalized_tree = [_normalize_action_text(item) for item in tree_actions]
    if normalized_seq != normalized_tree:
        return False, "action_sequence_mismatch"

    final_node_name = _final_node_name(record) or ""
    if final_node_name.startswith("precondition:"):
        return False, "precondition_failure"
    if final_node_name.startswith("target:"):
        return False, "target_failure"

    return True, "ok"


def passes_sr(record: dict[str, Any]) -> tuple[bool, str]:
    exec_ok, reason = passes_exec(record)
    if not exec_ok:
        return False, reason
    result = record.get("evaluation_result") or {}
    if result.get("result") == "success":
        return True, "ok"
    return False, result.get("result", "not_success")


def write_metric_summary(
    summary_output: str | None,
    metric_name: str,
    records: list[dict[str, Any]],
    metric_fn: Callable[[dict[str, Any]], tuple[bool, str]],
) -> dict[str, Any]:
    outcome_counts: dict[str, int] = {}
    passed = 0
    for record in records:
        ok, reason = metric_fn(record)
        record[f"{metric_name}_pass"] = ok
        record[f"{metric_name}_reason"] = reason
        outcome_counts[reason] = outcome_counts.get(reason, 0) + 1
        if ok:
            passed += 1

    summary = {
        "metric": metric_name,
        "total": len(records),
        "passes": passed,
        "pass_rate": passed / len(records) if records else 0.0,
        "reason_counts": outcome_counts,
        "records": records,
    }
    if summary_output:
        path = Path(summary_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def run_metric(metric_name: str, metric_fn):
    args = parse_args()
    generator = build_generator(args)
    records = evaluate_requests(args, generator)
    summary = write_metric_summary(args.summary_output, metric_name, records, metric_fn)
    print(
        json.dumps(
            {
                "metric": metric_name,
                "passes": summary["passes"],
                "total": summary["total"],
                "pass_rate": summary["pass_rate"],
                "reason_counts": summary["reason_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
