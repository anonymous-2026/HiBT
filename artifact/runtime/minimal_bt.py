"""Minimal behavior-tree runtime for the repository evaluation path."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
WORLD_DEFINITION_PATH = ARTIFACT_ROOT / "data" / "world_definition.json"

CALL_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\((.*)\)\s*$")


@dataclass(frozen=True)
class PredicateSpec:
    negative: bool
    predicate: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class ObjectProperty:
    object_name: str
    property_name: str
    property_value: str | None
    status: bool


@dataclass(frozen=True)
class ActionSpec:
    name: str
    arg_names: tuple[str, ...]
    preconditions: tuple[PredicateSpec, ...]
    effects: tuple[PredicateSpec, ...]


@dataclass
class EvalResult:
    status: str
    final_node: dict[str, Any] | None = None


def _strip_var(token: str) -> str:
    return token.strip().lstrip("?")


def _parse_template_args(template: str) -> tuple[str, ...]:
    match = CALL_RE.match(template)
    if not match:
        raise ValueError(f"Invalid action template: {template}")
    args_str = match.group(2).strip()
    if not args_str:
        return ()
    return tuple(_strip_var(part) for part in args_str.split(","))


def _parse_predicate(expr: str) -> PredicateSpec:
    expr = expr.strip()
    negative = False
    if expr.startswith("(not"):
        negative = True
        if expr.startswith("(not (") and expr.endswith("))"):
            inner = expr[len("(not (") : -2].strip()
        else:
            raise ValueError(f"Unsupported negated expression: {expr}")
        expr = inner
    parts = expr.split()
    predicate = parts[0]
    variables = tuple(_strip_var(part) for part in parts[1:])
    return PredicateSpec(negative=negative, predicate=predicate, variables=variables)


@lru_cache(maxsize=1)
def load_world_definition() -> dict[str, Any]:
    return json.loads(WORLD_DEFINITION_PATH.read_text())


@lru_cache(maxsize=1)
def action_specs() -> dict[str, ActionSpec]:
    raw = load_world_definition()
    specs: dict[str, ActionSpec] = {}
    for name, payload in raw["actions"].items():
        specs[name] = ActionSpec(
            name=name,
            arg_names=_parse_template_args(payload["template"]),
            preconditions=tuple(_parse_predicate(item) for item in payload["precondition"]),
            effects=tuple(_parse_predicate(item) for item in payload["effect"]),
        )
    return specs


def parse_call(text: str) -> tuple[str, list[str]]:
    match = CALL_RE.match(text.strip())
    if not match:
        raise ValueError(f"Invalid call expression: {text}")
    name = match.group(1)
    args_str = match.group(2).strip()
    if not args_str:
        return name, []
    return name, [part.strip() for part in args_str.split(",")]


def ground_action(
    action_name: str, params: list[str]
) -> tuple[list[ObjectProperty], list[ObjectProperty]]:
    """Ground an action into precondition/effect object-property facts."""
    spec = action_specs().get(action_name)
    if spec is None:
        raise ValueError(f"Action {action_name} is not defined in the world definition")
    if len(params) != len(spec.arg_names):
        raise ValueError(
            f"Action {action_name} expects {len(spec.arg_names)} args, got {len(params)}"
        )
    bindings = dict(zip(spec.arg_names, params))

    def _to_object_properties(predicates: tuple[PredicateSpec, ...]) -> list[ObjectProperty]:
        items: list[ObjectProperty] = []
        for pred in predicates:
            grounded = [bindings[var] for var in pred.variables]
            object_name = grounded[0] if grounded else None
            property_value = grounded[1] if len(grounded) > 1 else None
            if object_name is None:
                raise ValueError(f"Predicate {pred.predicate} has no grounded object")
            items.append(
                ObjectProperty(
                    object_name=object_name,
                    property_name=pred.predicate,
                    property_value=property_value,
                    status=not pred.negative,
                )
            )
        return items

    return _to_object_properties(spec.preconditions), _to_object_properties(spec.effects)


def parse_node_name(name: str) -> tuple[str, str]:
    if ":" not in name:
        raise ValueError(f"Invalid node name: {name}")
    kind, payload = name.split(":", 1)
    return kind.strip(), payload.strip()


class WorldState:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._object_order: list[str] = []
        self._object_properties: dict[str, set[str]] = {}
        for item in payload.get("objects", []):
            name = item["name"]
            self._object_order.append(name)
            self._object_properties[name] = set(item.get("properties", []))

        self._constraints = {
            (item["name"], item["source"], item["target"])
            for item in payload.get("constraints", [])
        }
        self._relations = {
            (item["name"], item["source"], item["target"])
            for item in payload.get("relations", [])
        }

        definition = load_world_definition()
        self._property_preds = set(definition["properties"])
        self._constraint_preds = set(definition["constraints"])
        self._relation_preds = set(definition["relations"])

    def copy(self) -> "WorldState":
        return WorldState(self.to_json())

    def check(self, predicate: str, args: list[str]) -> bool:
        if predicate in self._property_preds:
            if len(args) != 1:
                return False
            return predicate in self._object_properties.get(args[0], set())
        if predicate in self._constraint_preds:
            if len(args) != 2:
                return False
            return (predicate, args[0], args[1]) in self._constraints
        if predicate in self._relation_preds:
            if len(args) != 2:
                return False
            return (predicate, args[0], args[1]) in self._relations
        raise ValueError(f"Unknown predicate: {predicate}")

    def set_fact(self, predicate: str, args: list[str], value: bool) -> None:
        if predicate in self._property_preds:
            if len(args) != 1:
                raise ValueError(f"Property predicate {predicate} needs 1 arg")
            props = self._object_properties.setdefault(args[0], set())
            if value:
                props.add(predicate)
            else:
                props.discard(predicate)
            return
        if predicate in self._constraint_preds:
            key = (predicate, args[0], args[1])
            if value:
                self._constraints.add(key)
            else:
                self._constraints.discard(key)
            return
        if predicate in self._relation_preds:
            key = (predicate, args[0], args[1])
            if value:
                self._relations.add(key)
            else:
                self._relations.discard(key)
            return
        raise ValueError(f"Unknown predicate: {predicate}")

    def apply_action(self, action_name: str, args: list[str]) -> bool:
        spec = action_specs().get(action_name)
        if spec is None:
            raise ValueError(f"Action {action_name} is not defined in the world definition")
        if len(args) != len(spec.arg_names):
            raise ValueError(
                f"Action {action_name} expects {len(spec.arg_names)} args, got {len(args)}"
            )
        bindings = dict(zip(spec.arg_names, args))
        for pred in spec.preconditions:
            grounded_args = [bindings[var] for var in pred.variables]
            if self.check(pred.predicate, grounded_args) == pred.negative:
                return False
        for eff in spec.effects:
            grounded_args = [bindings[var] for var in eff.variables]
            self.set_fact(eff.predicate, grounded_args, not eff.negative)
        return True

    def to_json(self) -> dict[str, Any]:
        objects = [
            {
                "name": name,
                "properties": sorted(self._object_properties.get(name, set())),
            }
            for name in self._object_order
        ]
        constraints = [
            {"name": pred, "source": src, "target": tgt}
            for pred, src, tgt in sorted(self._constraints)
        ]
        relations = [
            {"name": pred, "source": src, "target": tgt}
            for pred, src, tgt in sorted(self._relations)
        ]
        return {
            "objects": objects,
            "constraints": constraints,
            "relations": relations,
        }


class MinimalBehaviorTreeRuntime:
    def __init__(self, world_state: dict[str, Any], behavior_tree: dict[str, Any]) -> None:
        self.world_state = WorldState(copy.deepcopy(world_state))
        self.behavior_tree = copy.deepcopy(behavior_tree)

    def evaluate(self) -> dict[str, Any]:
        result = self._eval_node(self.behavior_tree)
        if result.status == "success":
            return {
                "result": "success",
                "summary": "Behavior tree tick returns success",
                "final_node": None,
                "world_state": self.world_state.to_json(),
            }
        return {
            "result": "failure",
            "summary": "Behavior tree tick returns failure",
            "final_node": result.final_node,
            "world_state": self.world_state.to_json(),
        }

    def _eval_node(self, node: dict[str, Any]) -> EvalResult:
        name = node.get("name")
        if not isinstance(name, str):
            raise ValueError(f"Node is missing string name: {node}")
        kind, payload = parse_node_name(name)
        if kind == "selector":
            last_failure: dict[str, Any] | None = None
            for child in node.get("children", []) or []:
                result = self._eval_node(child)
                if result.status == "success":
                    return EvalResult("success")
                last_failure = result.final_node
            return EvalResult("failure", last_failure or self._final_node(node))
        if kind == "sequence":
            for child in node.get("children", []) or []:
                result = self._eval_node(child)
                if result.status != "success":
                    return result
            return EvalResult("success")
        if kind in {"target", "precondition"}:
            predicate, args = parse_call(payload)
            ok = self.world_state.check(predicate, args)
            if ok:
                return EvalResult("success")
            return EvalResult("failure", self._final_node(node))
        if kind == "action":
            action_name, args = parse_call(payload)
            ok = self.world_state.apply_action(action_name, args)
            if ok:
                return EvalResult("success")
            return EvalResult("failure", self._final_node(node))
        raise ValueError(f"Unsupported BT node kind: {kind}")

    @staticmethod
    def _final_node(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": node.get("summary"),
            "name": node.get("name"),
        }
