#!/usr/bin/env python3
"""Batch-generate and evaluate behavior trees.

This script loads the local generation model once, then executes:

    target + world_state -> generate behavior_tree -> sk_sim_run -> success/failure

The evaluation path deliberately avoids robot/scene-side dependencies so the
script can exercise the repository's world-state simulation without requiring a
full execution stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import (
    ARTIFACT_PLANNING_DIR,
    ARTIFACT_DATA_DIR,
    ARTIFACT_EVAL_DIR,
    bootstrap_runtime,
)

DEFAULT_MODEL = os.environ.get("BT_MODEL", "Qwen/Qwen3-8B")

bootstrap_runtime()
if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))
if str(ARTIFACT_PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_PLANNING_DIR))

from runtime import run_sk_simulation

BehaviorTreeGenerator = Any


class DeepSeekBehaviorTreeGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        prompt_profile: str,
        max_new_tokens: int,
        temperature: float,
        base_url: str,
        timeout_sec: int = 300,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_profile = prompt_profile
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.load_duration_sec = 0.0

    def generate(
        self, target: str, world_state: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        from generate_bt_only import (
            extract_json_object,
            render_messages,
            validate_generation_result,
        )

        messages = render_messages(target, world_state, self.prompt_profile)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek API HTTP {exc.code}: {error_body}"
            ) from exc
        duration = time.perf_counter() - start

        try:
            raw_output = response_body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected DeepSeek API response shape: {response_body}"
            ) from exc

        result = extract_json_object(raw_output)
        validate_generation_result(result)
        return result, duration


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


def evaluate_requests(
    args: argparse.Namespace, generator: BehaviorTreeGenerator
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
                sim_world_state = generation.get("simulation_world_state", world_state)
                evaluation = run_sk_simulation(
                    sim_world_state, generation["behavior_tree"]
                )
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
    summary_file = Path(summary_path).expanduser().resolve()
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
        "records": records,
    }
    _write_json(summary_file, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run behavior-tree generation followed by execution "
            "evaluation."
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
        choices=["local", "deepseek", "concept", "actionseq", "concept_raw", "concept_nostable"],
        help=(
            "Generation backend. Use local for transformers, deepseek for API, "
            "concept for the concept predictor + decoder pipeline, or actionseq "
            "for action-sequence generation followed by deterministic compilation. "
            "Use concept_raw to disable decoder repair, or concept_nostable to "
            "disable stable-target construction while keeping repair enabled. "
            "Default: local."
        ),
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
        default=2048,
        help="Maximum number of tokens to generate. Default: 2048.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding. Default: 0.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in apply_chat_template.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model loading: auto, float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention backend, e.g. flash_attention_2 or sdpa.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Model placement target. Use auto or an explicit device like cuda:0.",
    )
    parser.add_argument(
        "--prompt-profile",
        default="full",
        choices=["full", "compact"],
        help="Prompt template profile. Use compact for shorter prompts.",
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
        "--output-dir",
        help="Optional directory to write one detailed result JSON per request.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional path to write the aggregate summary JSON.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first generation or evaluation error.",
    )
    parser.add_argument(
        "--planner-config",
        default=str(
            ARTIFACT_ROOT
            / "configs"
            / "planner"
            / "train_predictor_Qwen3-8B_planlocal_4level_shared_v2_smoke.yml"
        ),
        help="Predictor YAML config for --backend concept.",
    )
    parser.add_argument(
        "--planner-storage-root",
        default=str(ARTIFACT_DATA_DIR / "runtime" / "planner_v2"),
        help="Storage root that contains planner builder/predictor checkpoints.",
    )
    parser.add_argument(
        "--plan-bank",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_bank_v2.pt"),
        help="Prototype bank .pt path for --backend concept.",
    )
    parser.add_argument(
        "--planner-predictor-ckpt",
        default="",
        help="Optional explicit predictor checkpoint for --backend concept.",
    )
    parser.add_argument(
        "--planner-builder-ckpt",
        default="",
        help="Optional explicit builder checkpoint for --backend concept.",
    )
    parser.add_argument(
        "--planner-top-k",
        type=int,
        default=3,
        help="Top-k latent retrieval candidates used by the concept decoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "local":
        from generate_bt_only import BehaviorTreeGenerator as _BehaviorTreeGenerator

        generator = _BehaviorTreeGenerator(
            model_name=args.model,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            device=args.device,
            prompt_profile=args.prompt_profile,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
    elif args.backend == "actionseq":
        from generate_action_sequence_only import (
            ActionSequenceBehaviorTreeGenerator as _ActionSequenceBehaviorTreeGenerator,
        )

        generator = _ActionSequenceBehaviorTreeGenerator(
            model_name=args.model,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
    elif args.backend == "deepseek":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"DeepSeek backend requires API key in environment variable {args.api_key_env}"
            )
        generator = DeepSeekBehaviorTreeGenerator(
            api_key=api_key,
            model_name=args.model,
            prompt_profile=args.prompt_profile,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            base_url=args.base_url,
        )
    elif args.backend in {"concept", "concept_raw", "concept_nostable"}:
        from concept_backend import ConceptBehaviorTreeGenerator

        generator = ConceptBehaviorTreeGenerator(
            predictor_config_path=args.planner_config,
            storage_root=args.planner_storage_root,
            prototype_bank_path=args.plan_bank,
            predictor_checkpoint_path=args.planner_predictor_ckpt or None,
            builder_checkpoint_path=args.planner_builder_ckpt or None,
            device=args.device,
            top_k=args.planner_top_k,
            enable_repair=args.backend != "concept_raw",
            stable_targets=args.backend != "concept_nostable",
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")
    records = evaluate_requests(args, generator)
    write_summary(args.summary_output, records)


if __name__ == "__main__":
    main()
