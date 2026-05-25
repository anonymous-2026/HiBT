#!/usr/bin/env python3
"""Generate a behavior-tree JSON with a local transformers model.

This script reuses the repository's local prompt assets under
``artifact/data/prompts/end_to_end_v3`` and only performs the planning/generation
step:

    target + initial world state -> thought + action_sequence + behavior_tree

It intentionally does not import the robot execution stack, so it avoids
legacy prompt lookups and robot-interface side effects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import ARTIFACT_DATA_DIR

PROMPT_DIR = ARTIFACT_DATA_DIR / "prompts" / "end_to_end_v3"
DEFAULT_MODEL = (
    os.environ.get("BT_LOCAL_MODEL")
    or os.environ.get("BT_MODEL")
    or "Qwen/Qwen3-8B"
)

if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_all_tied_weights_keys(self):
        stored = self.__dict__.get("_all_tied_weights_keys")
        if stored is not None:
            return stored
        return {key: key for key in (getattr(self, "_tied_weights_keys", None) or [])}

    def _set_all_tied_weights_keys(self, value):
        self.__dict__["_all_tied_weights_keys"] = value

    PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys,
        _set_all_tied_weights_keys,
    )


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text()


def render_messages(
    target: str, world_state: dict[str, Any], prompt_profile: str
) -> list[dict[str, str]]:
    system_prompt = _read_prompt("system")
    template_text = _read_prompt("template").format(
        target=target,
        initial_state=json.dumps(world_state, ensure_ascii=False),
    )

    if prompt_profile == "compact":
        compact_bt_rules = """
Generate one executable behavior-tree skeleton in JSON.
Rules:
- The root of behavior_tree must directly be a node object with fields: summary, name, children.
- Use one root selector node.
- The selector's first child is a target node.
- The selector's second child is a sequence node.
- The sequence contains the required precondition nodes and one final action node.
- Every node must have summary and name.
- The behavior_tree must be directly consumable by the repository runtime.
- Do not use generic node names such as "selector", "sequence", "target", "precondition", or "action" alone.
- Every node name must include a grounded predicate or action call.
- Node name grammar:
  - selector node: "selector: <action_or_goal>(arg1, ...)" or "selector: <predicate>(arg1, ...)"
  - sequence node: "sequence: <action_or_goal>(arg1, ...)" or "sequence: <predicate>(arg1, ...)"
  - target node: "target: <predicate>(arg1, ...)"
  - precondition node: "precondition: <predicate>(arg1, ...)"
  - action node: "action: <action>(arg1, ...)"
- Every target/precondition/action node name must be parseable by the runtime.
- First decide a valid action_sequence. Then construct the behavior_tree skeleton strictly from that action_sequence.
- Every action node in behavior_tree must correspond to one action in action_sequence.
- The root selector and root sequence should usually be named after the final action that achieves the target.
- Use only actions and predicates defined in the domain knowledge.
Example skeleton fragment:
{
  "summary": "selector to insert gear3 into shaft2",
  "name": "selector: insert(left_hand, inward_claw, gear3, shaft2)",
  "children": [
    {
      "summary": "the target is to make gear3 inserted into shaft2",
      "name": "target: is_inserted_to(gear3, shaft2)"
    },
    {
      "summary": "sequence to insert gear3 into shaft2",
      "name": "sequence: insert(left_hand, inward_claw, gear3, shaft2)",
      "children": [
        {
          "summary": "check the precondition that left_hand holds inward_claw",
          "name": "precondition: hold(left_hand, inward_claw)"
        },
        {
          "summary": "the action to insert gear3 into shaft2",
          "name": "action: insert(left_hand, inward_claw, gear3, shaft2)"
        }
      ]
    }
  ]
}
"""
        user_sections = [
            _read_prompt("task"),
            _read_prompt("domain"),
            _read_prompt("state"),
            compact_bt_rules.strip(),
            _read_prompt("output_format"),
            template_text,
            (
                "Return one valid JSON object only. "
                "Do not include markdown fences. "
                "Do not include any text before or after the JSON object."
            ),
        ]
    else:
        user_sections = [
            _read_prompt("task"),
            _read_prompt("domain"),
            _read_prompt("state"),
            _read_prompt("behaviortree"),
            _read_prompt("output_format"),
            _read_prompt("example"),
            template_text,
            (
                "Return one valid JSON object only. "
                "Do not include markdown fences. "
                "Do not include any text before or after the JSON object."
            ),
        ]

    user_prompt = "\n\n".join(user_sections)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_local_model(
    model_name: str,
    torch_dtype: str,
    attn_implementation: str | None,
    device: str,
):
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(
            f"Unsupported --torch-dtype value: {torch_dtype}. "
            "Use one of: auto, float16, bfloat16, float32."
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[torch_dtype],
        "low_cpu_mem_usage": False,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        **model_kwargs,
    )
    if hasattr(config, "max_length"):
        model.generation_config.max_length = config.max_length
        try:
            delattr(model.config, "max_length")
        except AttributeError:
            pass
    if device != "auto":
        model = model.to(device)
    return tokenizer, model


def generate_raw_response(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    generated_ids = model.generate(**model_inputs, **generation_kwargs)
    new_token_ids = generated_ids[:, model_inputs.input_ids.shape[1] :]
    output = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
    return output.strip()


def _extract_tagged_json_block(text: str) -> str | None:
    start_tag = "<json>"
    end_tag = "</json>"
    start = text.find(start_tag)
    end = text.rfind(end_tag)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start + len(start_tag) : end].strip()


def _extract_balanced_braces(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()

    tagged = _extract_tagged_json_block(candidate)
    if tagged:
        candidate = tagged

    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        last_fence = candidate.rfind("```")
        if first_newline != -1 and last_fence != -1 and last_fence > first_newline:
            candidate = candidate[first_newline + 1 : last_fence].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        balanced = _extract_balanced_braces(candidate)
        if balanced is None:
            raise
        return json.loads(balanced)


def validate_generation_result(result: dict[str, Any]) -> None:
    required_top_level = ["thought", "action_sequence", "behavior_tree"]
    missing = [key for key in required_top_level if key not in result]
    if missing:
        raise ValueError(f"Model output is missing required keys: {missing}")

    if not isinstance(result["action_sequence"], list):
        raise TypeError('"action_sequence" must be a list')

    if not isinstance(result["behavior_tree"], dict):
        raise TypeError('"behavior_tree" must be a JSON object')

    root = result["behavior_tree"]
    for field in ("summary", "name"):
        if field not in root:
            raise ValueError(f'behavior_tree root is missing "{field}"')


class BehaviorTreeGenerator:
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        attn_implementation: str | None,
        device: str,
        prompt_profile: str,
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
        self.load_duration_sec = time.perf_counter() - load_start
        self.prompt_profile = prompt_profile
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def generate(
        self, target: str, world_state: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        messages = render_messages(target, world_state, self.prompt_profile)
        start = time.perf_counter()
        raw_output = generate_raw_response(
            tokenizer=self.tokenizer,
            model=self.model,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
        )
        duration = time.perf_counter() - start
        result = extract_json_object(raw_output)
        validate_generation_result(result)
        return result, duration


def _write_output(output_path: str | None, result: dict[str, Any]) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if output_path:
        resolved = Path(output_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(rendered + "\n")


def _read_world_state(path: str) -> dict[str, Any]:
    world_state_path = Path(path).expanduser().resolve()
    return json.loads(world_state_path.read_text())


def run_single(args: argparse.Namespace, generator: BehaviorTreeGenerator) -> None:
    world_state = _read_world_state(args.world_state)
    print(f"[load_sec] {generator.load_duration_sec:.2f}", flush=True)
    result, duration = generator.generate(args.target, world_state)
    print(f"[generation_sec] {duration:.2f}", flush=True)
    _write_output(args.output, result)


def run_requests_file(
    args: argparse.Namespace, generator: BehaviorTreeGenerator
) -> None:
    requests_path = Path(args.requests_file).expanduser().resolve()
    requests = json.loads(requests_path.read_text())
    if not isinstance(requests, list):
        raise TypeError("--requests-file must contain a JSON list of request objects")

    for idx, item in enumerate(requests):
        if not isinstance(item, dict):
            raise TypeError(f"Request #{idx} is not a JSON object")
        target = item.get("target")
        if not target:
            raise ValueError(f"Request #{idx} is missing 'target'")
        if "world_state" in item:
            world_state = item["world_state"]
        elif "world_state_path" in item:
            world_state = _read_world_state(item["world_state_path"])
        else:
            raise ValueError(
                f"Request #{idx} must include 'world_state' or 'world_state_path'"
            )

        if idx == 0:
            print(f"[load_sec] {generator.load_duration_sec:.2f}", flush=True)
        result, duration = generator.generate(target, world_state)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print(f"=== request {idx} ===")
        print(f"[generation_sec] {duration:.2f}", flush=True)
        print(rendered)

        output_path = item.get("output")
        if output_path:
            resolved = Path(output_path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(rendered + "\n")


def run_interactive(generator: BehaviorTreeGenerator) -> None:
    print(
        "Interactive mode ready. Enter a target line, then a world state path line. "
        "Type 'quit' to exit.",
        flush=True,
    )
    while True:
        target = input("target> ").strip()
        if target.lower() in {"quit", "exit"}:
            return
        if not target:
            continue

        world_state_path = input("world_state_path> ").strip()
        if world_state_path.lower() in {"quit", "exit"}:
            return
        if not world_state_path:
            continue

        try:
            world_state = _read_world_state(world_state_path)
            result, duration = generator.generate(target, world_state)
            print(f"[generation_sec] {duration:.2f}", flush=True)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a behavior-tree JSON from a target and world state."
    )
    parser.add_argument(
        "--target",
        help='Target predicate or natural-language goal, e.g. "is_inserted_to(gear1, shaft1)".',
    )
    parser.add_argument(
        "--world-state",
        help="Path to the world state JSON file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f'Hugging Face model id or local model path. Default: "{DEFAULT_MODEL}".',
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
        help=(
            "Prompt template profile. Use compact for shorter prompts. "
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the generated JSON. Defaults to stdout only.",
    )
    parser.add_argument(
        "--requests-file",
        help=(
            "Path to a JSON file containing a list of request objects. "
            "Each object must have 'target' and either 'world_state' or "
            "'world_state_path'. Optional 'output' writes one result per request."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Load the model once and accept repeated target/world-state inputs.",
    )
    args = parser.parse_args()
    if args.requests_file or args.interactive:
        return args
    if not args.target or not args.world_state:
        parser.error("--target and --world-state are required unless using --requests-file or --interactive")
    return args


def main() -> None:
    args = parse_args()
    generator = BehaviorTreeGenerator(
        model_name=args.model,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        prompt_profile=args.prompt_profile,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )
    if args.interactive:
        run_interactive(generator)
    elif args.requests_file:
        run_requests_file(args, generator)
    else:
        run_single(args, generator)


if __name__ == "__main__":
    main()
