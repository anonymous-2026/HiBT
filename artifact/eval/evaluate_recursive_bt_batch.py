#!/usr/bin/env python3
"""Batch-evaluate recursive BT generation with a local transformers model.

This reproduces the repository's recursive generation flow at a minimal level:

    target
      -> recursive action planning for a goal
      -> unit-subtree generation for the last action in that plan
      -> recursive expansion of target/precondition nodes
      -> sk_sim_run evaluation

The script deliberately avoids the legacy interactive experiment code and keeps
the robot/scene runtime stubbed so it can run as a pure world-state simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import (
    ARTIFACT_DATA_DIR,
    ARTIFACT_EVAL_DIR,
    bootstrap_runtime,
)

PROMPT_ROOT = ARTIFACT_DATA_DIR / "prompts" / "new"
DEFAULT_MODEL = os.environ.get("BT_MODEL", "Qwen/Qwen3-8B")

bootstrap_runtime()
if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))

from runtime import run_sk_simulation
from generate_bt_only import extract_json_object, generate_raw_response, load_local_model


def _read_prompt(path: Path) -> str:
    return path.read_text()


def _read_world_state(path: str) -> dict[str, Any]:
    world_state_path = Path(path).expanduser().resolve()
    return json.loads(world_state_path.read_text())


def _load_requests(path: str) -> list[dict[str, Any]]:
    requests_path = Path(path).expanduser().resolve()
    requests = json.loads(requests_path.read_text())
    if not isinstance(requests, list):
        raise TypeError("--requests-file must contain a JSON list of request objects")
    return requests


def _resolve_world_state(item: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if "world_state" in item:
        return item["world_state"], "<inline>"
    if "world_state_path" in item:
        raw_path = Path(item["world_state_path"]).expanduser()
        if not raw_path.is_absolute():
            raw_path = (ARTIFACT_DATA_DIR / "examples" / raw_path).resolve()
        world_state_path = str(raw_path.resolve())
        return _read_world_state(world_state_path), world_state_path
    raise ValueError("Each request must include 'world_state' or 'world_state_path'")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class RecursiveBTGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        torch_dtype: str,
        attn_implementation: str | None,
        device: str,
        enable_thinking: bool,
        max_depth: int,
        max_expansions: int,
    ) -> None:
        load_start = time.perf_counter()
        self.tokenizer, self.model = load_local_model(
            model_name=model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device=device,
        )
        self.load_duration_sec = time.perf_counter() - load_start
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.trace: list[dict[str, Any]] = []
        self.expansion_count = 0
        self.prompts = {
            "seq_plan": self._load_prompt_bundle(PROMPT_ROOT / "seq_plan"),
            "state_est": self._load_prompt_bundle(PROMPT_ROOT / "state_est"),
            "ut_gen": self._load_prompt_bundle(PROMPT_ROOT / "ut_gen"),
        }

    @staticmethod
    def _load_prompt_bundle(prompt_dir: Path) -> dict[str, str]:
        bundle: dict[str, str] = {}
        for prompt_file in prompt_dir.glob("*.txt"):
            bundle[prompt_file.stem] = prompt_file.read_text()
        return bundle

    def _invoke_json(
        self,
        messages: list[dict[str, str]],
        stage: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_output = generate_raw_response(
            tokenizer=self.tokenizer,
            model=self.model,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
        )
        parsed = extract_json_object(raw_output)
        self.trace.append(
            {
                "stage": stage,
                "payload": payload,
                "response": parsed,
            }
        )
        return parsed

    def _render_seq_plan_messages(
        self, start_world_state: dict[str, Any], target: str
    ) -> list[dict[str, str]]:
        prompt = self.prompts["seq_plan"]
        system_prompt = prompt["system"].strip()
        user_sections = [
            prompt["task"].strip(),
            prompt["new_domain_nl"].strip(),
            prompt["state"].strip(),
            prompt["output_format"].strip(),
            prompt["new_example"].strip(),
            prompt["template"].format(
                start_world_state=json.dumps(start_world_state, ensure_ascii=False),
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

    def _render_state_est_messages(
        self, start_world_state: dict[str, Any], action_plan: list[str]
    ) -> list[dict[str, str]]:
        prompt = self.prompts["state_est"]
        system_prompt = prompt["system"].strip()
        user_sections = [
            prompt["task"].strip(),
            prompt["new_domain_nl"].strip(),
            prompt["state"].strip(),
            prompt["output_format"].strip(),
            prompt["new_example"].strip(),
            prompt["template"].format(
                start_world_state=json.dumps(start_world_state, ensure_ascii=False),
                action_plan=json.dumps(action_plan, ensure_ascii=False),
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

    def _render_ut_gen_messages(self, action: str) -> list[dict[str, str]]:
        prompt = self.prompts["ut_gen"]
        system_prompt = prompt["system"].strip()
        user_sections = [
            prompt["task"].strip(),
            prompt["new_domain_nl"].strip(),
            prompt["new_behaviortree"].strip(),
            prompt["template"].format(action=action),
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

    def make_plan(self, state: dict[str, Any], goal: str) -> list[str]:
        messages = self._render_seq_plan_messages(state, goal)
        response = self._invoke_json(
            messages=messages,
            stage="seq_plan",
            payload={"goal": goal},
        )
        task_plan = response.get("task_plan")
        if not isinstance(task_plan, list):
            raise TypeError(f"seq_plan output missing list task_plan: {response}")
        for item in task_plan:
            if not isinstance(item, str):
                raise TypeError(f"seq_plan contains non-string action: {item!r}")
        return task_plan

    def estimate_state(
        self, start_world_state: dict[str, Any], action_plan: list[str]
    ) -> dict[str, Any]:
        messages = self._render_state_est_messages(start_world_state, action_plan)
        response = self._invoke_json(
            messages=messages,
            stage="state_est",
            payload={"action_plan": action_plan},
        )
        estimated_world_state = response.get("estimated_world_state")
        if not isinstance(estimated_world_state, dict):
            raise TypeError(
                "state_est output missing dict estimated_world_state: "
                f"{response}"
            )
        return estimated_world_state

    def generate_unit_subtree(self, action: str) -> dict[str, Any]:
        messages = self._render_ut_gen_messages(action)
        response = self._invoke_json(
            messages=messages,
            stage="ut_gen",
            payload={"action": action},
        )
        if not isinstance(response, dict) or "name" not in response:
            raise TypeError(f"ut_gen output is not a subtree object: {response}")
        return response

    @staticmethod
    def match_type(node: dict[str, Any]) -> tuple[str, str]:
        node_name = node["name"]
        match = re.search(
            r"(selector|sequence|action|precondition|condition|target):\s*(.+)",
            node_name,
        )
        if not match:
            raise ValueError(f"Node name does not match any supported type: {node_name}")
        return match.group(1), match.group(2)

    @staticmethod
    def get_node_list_from_tree(unit_subtree: dict[str, Any]) -> list[dict[str, Any]]:
        children = unit_subtree.get("children")
        if not isinstance(children, list) or len(children) < 2:
            raise ValueError(f"Unit subtree missing selector children: {unit_subtree}")
        sequence_node = children[1]
        sequence_children = sequence_node.get("children")
        if not isinstance(sequence_children, list):
            raise ValueError(f"Unit subtree missing sequence children: {unit_subtree}")
        return sequence_children

    def expand_nodes(
        self,
        node_list: list[dict[str, Any]],
        start_state: dict[str, Any],
        overall_tree: list[dict[str, Any]] | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        if depth > self.max_depth:
            raise RecursionError(
                f"Exceeded recursive expansion depth limit ({self.max_depth})"
            )
        if not node_list:
            raise ValueError("expand_nodes received an empty node_list")

        state = deepcopy(start_state)
        for idx in range(len(node_list)):
            node_type, goal_or_action = self.match_type(node_list[idx])
            if node_type == "action":
                continue
            if node_type not in {"precondition", "target"}:
                continue

            if self.expansion_count >= self.max_expansions:
                raise RecursionError(
                    f"Exceeded recursive expansion budget ({self.max_expansions})"
                )

            plan = self.make_plan(state, goal_or_action)
            if len(plan) == 0:
                self.trace.append(
                    {
                        "stage": "skip_goal",
                        "payload": {"goal": goal_or_action},
                        "response": {"reason": "empty_plan"},
                    }
                )
                continue

            last_action = plan[-1]
            unit_subtree = self.generate_unit_subtree(last_action)
            node_list[idx] = unit_subtree
            self.expansion_count += 1

            new_node_list = self.get_node_list_from_tree(unit_subtree)
            self.expand_nodes(
                node_list=new_node_list,
                start_state=state,
                overall_tree=overall_tree,
                depth=depth + 1,
            )
            state = self.estimate_state(state, plan)

        return node_list[0]

    def generate(self, target: str, world_state: dict[str, Any]) -> tuple[dict[str, Any], float]:
        self.trace = []
        self.expansion_count = 0
        start = time.perf_counter()
        root = [
            {
                "summary": f"the target is {target}",
                "name": f"target: {target}",
            }
        ]
        self.expand_nodes(root, deepcopy(world_state), overall_tree=root, depth=0)
        elapsed = time.perf_counter() - start
        return (
            {
                "behavior_tree": root[0],
                "trace": self.trace,
                "expansion_count": self.expansion_count,
            },
            elapsed,
        )


class DeepSeekRecursiveBTGenerator(RecursiveBTGenerator):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        max_depth: int,
        max_expansions: int,
        base_url: str,
        timeout_sec: int = 300,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = False
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.load_duration_sec = 0.0
        self.trace: list[dict[str, Any]] = []
        self.expansion_count = 0
        self.prompts = {
            "seq_plan": self._load_prompt_bundle(PROMPT_ROOT / "seq_plan"),
            "state_est": self._load_prompt_bundle(PROMPT_ROOT / "state_est"),
            "ut_gen": self._load_prompt_bundle(PROMPT_ROOT / "ut_gen"),
        }

    def _invoke_json(
        self,
        messages: list[dict[str, str]],
        stage: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek API HTTP {exc.code}: {error_body}"
            ) from exc

        try:
            raw_output = response_body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected DeepSeek API response shape: {response_body}"
            ) from exc

        parsed = extract_json_object(raw_output)
        self.trace.append(
            {
                "stage": stage,
                "payload": payload,
                "response": parsed,
            }
        )
        return parsed


def evaluate_requests(
    args: argparse.Namespace, generator: RecursiveBTGenerator
) -> list[dict[str, Any]]:
    requests = _load_requests(args.requests_file)
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(requests):
        if not isinstance(item, dict):
            raise TypeError(f"Request #{idx} is not a JSON object")
        target = item.get("target")
        if not target:
            raise ValueError(f"Request #{idx} is missing 'target'")

        world_state, world_state_ref = _resolve_world_state(item)

        generation: dict[str, Any] | None = None
        generation_error: str | None = None
        evaluation: dict[str, Any] | None = None
        evaluation_error: str | None = None
        generation_sec: float | None = None
        evaluation_sec: float | None = None

        try:
            generation, generation_sec = generator.generate(target, world_state)
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"

        if generation is not None:
            try:
                start = time.perf_counter()
                evaluation = run_sk_simulation(world_state, generation["behavior_tree"])
                evaluation_sec = time.perf_counter() - start
            except Exception as exc:
                evaluation_error = f"{type(exc).__name__}: {exc}"

        record = {
            "request_index": idx,
            "target": target,
            "world_state_ref": world_state_ref,
            "generation_sec": generation_sec,
            "evaluation_sec": evaluation_sec,
            "generation_error": generation_error,
            "evaluation_error": evaluation_error,
            "generation_result": generation,
            "evaluation_result": evaluation,
            "success": evaluation is not None and evaluation.get("result") == "success",
        }
        results.append(record)

        print(f"=== request {idx} ===", flush=True)
        if idx == 0:
            print(f"[load_sec] {generator.load_duration_sec:.2f}", flush=True)
        if generation_sec is not None:
            print(f"[generation_sec] {generation_sec:.2f}", flush=True)
        if evaluation_sec is not None:
            print(f"[evaluation_sec] {evaluation_sec:.2f}", flush=True)
        if generation_error:
            print(f"[generation_error] {generation_error}", flush=True)
        if evaluation_error:
            print(f"[evaluation_error] {evaluation_error}", flush=True)
        if evaluation is not None:
            print(
                f"[sk_sim_result] {evaluation.get('result')} - {evaluation.get('summary')}",
                flush=True,
            )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
            _write_json(output_dir / f"request_{idx:03d}.json", record)

        if args.stop_on_error and (generation_error or evaluation_error):
            break

    return results


def write_summary(summary_path: str | None, records: list[dict[str, Any]]) -> None:
    if not summary_path:
        return

    failed_nodes = Counter()
    for item in records:
        result = item.get("evaluation_result") or {}
        final_node = result.get("final_node") or {}
        node_name = final_node.get("name")
        if node_name:
            failed_nodes[node_name] += 1

    summary = {
        "total": len(records),
        "successes": sum(1 for item in records if item["success"]),
        "failures": sum(
            1
            for item in records
            if item["evaluation_result"] is not None
            and item["evaluation_result"].get("result") != "success"
        ),
        "errors": sum(
            1 for item in records if item["generation_error"] or item["evaluation_error"]
        ),
        "avg_generation_sec": (
            sum(item["generation_sec"] for item in records if item["generation_sec"])
            / max(1, sum(1 for item in records if item["generation_sec"]))
        ),
        "avg_evaluation_sec": (
            sum(item["evaluation_sec"] for item in records if item["evaluation_sec"])
            / max(1, sum(1 for item in records if item["evaluation_sec"]))
        ),
        "top_failed_nodes": failed_nodes.most_common(10),
        "records": records,
    }
    _write_json(Path(summary_path).expanduser().resolve(), summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run recursive BT generation followed by execution-time evaluation."
        )
    )
    parser.add_argument(
        "--requests-file",
        required=True,
        help=(
            "Path to a JSON file containing a list of request objects. "
            "Each object must have 'target' and either 'world_state' or "
            "'world_state_path'."
        ),
    )
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "deepseek"],
        help="Generation backend. Use local for transformers or deepseek for API. Default: local.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f'Hugging Face model id/local path for local backend or API model name for '
            f'deepseek backend. Default: "{DEFAULT_MODEL}".'
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help="Maximum number of tokens per recursive sub-call. Default: 1536.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype to use when loading the model. Default: bfloat16.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to transformers. Default: sdpa.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Device to place the model on, e.g. "cuda:0" or "auto". Default: cuda:0.',
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode when supported by the tokenizer template.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable name that stores the DeepSeek API key. Default: DEEPSEEK_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="Base URL for the API backend. Default: https://api.deepseek.com.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum recursive subtree expansion depth. Default: 8.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=32,
        help="Maximum number of condition-node expansions per request. Default: 32.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory to write per-request result JSON files.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional path to write a summary JSON file.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing remaining requests after the first generation/evaluation error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend == "local":
        generator = RecursiveBTGenerator(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            device=args.device,
            enable_thinking=args.enable_thinking,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
        )
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"DeepSeek backend requires API key in environment variable {args.api_key_env}"
            )
        generator = DeepSeekRecursiveBTGenerator(
            api_key=api_key,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            base_url=args.base_url,
        )
    records = evaluate_requests(args, generator)
    write_summary(args.summary_output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
