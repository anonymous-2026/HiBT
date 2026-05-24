#!/usr/bin/env python3
"""Generate action sequences with a local transformers model, then compile to BT.

This is the action-only planning baseline used to compare against the full
concept-pyramid pipeline. It performs:

    target + initial world state -> task_plan (action sequence)

The resulting action sequence is then deterministically compiled into a
structured plan/pyramid and further into an executable BT skeleton.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import ARTIFACT_DATA_DIR, ARTIFACT_EVAL_DIR, ARTIFACT_PLANNING_DIR

if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))
if str(ARTIFACT_PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_PLANNING_DIR))

from compile_plan_to_bt import PyramidCompiler
from decode_plan_latents import (
    _build_goal_only_pyramid,
    _build_pyramid_from_actions,
    _normalize_state_for_kios,
)
from generate_bt_only import (
    extract_json_object,
    generate_raw_response,
    load_local_model,
)


PROMPT_DIR = ARTIFACT_DATA_DIR / "prompts" / "new" / "seq_plan"


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render_messages(target: str, world_state: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = _read_prompt("system")
    user_sections = [
        _read_prompt("task"),
        _read_prompt("new_domain_nl"),
        _read_prompt("state"),
        _read_prompt("output_format"),
        _read_prompt("new_example"),
        _read_prompt("template").format(
            start_world_state=json.dumps(world_state, ensure_ascii=False),
            target=target,
        ),
        (
            "Return one valid JSON object only. "
            "Do not include markdown fences. "
            "Do not include any text before or after the JSON object."
        ),
    ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]


_ACTION_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*")


def _parse_action_call(text: str) -> tuple[str, list[str]]:
    match = _ACTION_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid action call: {text}")
    action_name = match.group(1)
    args_part = match.group(2).strip()
    args = [] if not args_part else [arg.strip() for arg in args_part.split(",")]
    return action_name, args


def validate_generation_result(result: dict[str, Any]) -> None:
    if "task_plan" not in result:
        raise ValueError('Model output is missing required key "task_plan"')
    if "explanation" not in result:
        raise ValueError('Model output is missing required key "explanation"')
    if not isinstance(result["task_plan"], list):
        raise TypeError('"task_plan" must be a list')
    for item in result["task_plan"]:
        if not isinstance(item, str):
            raise TypeError('"task_plan" items must be strings')


class ActionSequenceBehaviorTreeGenerator:
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        attn_implementation: str | None,
        device: str,
        max_new_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> None:
        load_start = time.perf_counter()
        self.tokenizer, self.model = load_local_model(
            model_name=model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device=device,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.load_duration_sec = time.perf_counter() - load_start

    def generate(
        self, target: str, world_state: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        messages = render_messages(target, world_state)
        start = time.perf_counter()
        raw_output = generate_raw_response(
            tokenizer=self.tokenizer,
            model=self.model,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
        )
        result = extract_json_object(raw_output)
        validate_generation_result(result)

        normalized_state = _normalize_state_for_kios(world_state)
        action_sequence_text = result["task_plan"]
        action_sequence = [_parse_action_call(item) for item in action_sequence_text]
        if action_sequence:
            plan = _build_pyramid_from_actions(
                target_call=target,
                initial_state=normalized_state,
                action_sequence=action_sequence,
                problem_id="<adhoc_actionseq>",
            )
        else:
            plan = _build_goal_only_pyramid(
                target_call=target,
                initial_state=normalized_state,
                problem_id="<adhoc_actionseq>",
            )
        behavior_tree = PyramidCompiler(plan).compile()
        duration = time.perf_counter() - start

        generation = {
            "thought": result["explanation"],
            "action_sequence": action_sequence_text,
            "behavior_tree": behavior_tree,
            "compiled_plan": plan,
            "simulation_world_state": normalized_state,
        }
        return generation, duration
