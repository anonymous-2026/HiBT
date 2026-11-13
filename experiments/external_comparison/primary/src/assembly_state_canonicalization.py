"""Original goal-independent KIOS state canonicalization at assembly input.

Exact pure upstream functions below. No model, goal, action sequence or evaluator
is consulted. All five methods receive the same canonical state and registry.
"""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
from typing import Any
from assembly_registry import make_registry

STATE_CONTRACT = 'original_kios_state_canonicalization_v1'
UPSTREAM_SOURCE_SHA256 = '717f6fd221e5ad60c432aa8f9ee693fa331bb3b5683113c17709f0ed3a522b08'

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

def canonicalize_request(request):
    result = copy.deepcopy(request)
    result['world_state'] = _normalize_state_for_kios(request['world_state'])
    result['registry'] = make_registry(result)
    return result


def validate_prepared_inputs(raw_path, canonical_path, manifest_path):
    def read(path): return json.loads(Path(path).read_text())
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    raw, canonical, manifest = read(raw_path), read(canonical_path), read(manifest_path)
    if manifest['state_contract'] != STATE_CONTRACT:
        raise ValueError('Unexpected assembly state contract')
    if manifest['raw_requests_sha256'] != sha(raw_path) or manifest['canonical_requests_sha256'] != sha(canonical_path):
        raise ValueError('Raw/canonical assembly inputs changed after preparation')
    if len(raw) != len(canonical) or len(raw) != manifest['n_requests']:
        raise ValueError('Canonicalization changed task membership')
    for original, prepared in zip(raw, canonical):
        if canonicalize_request(original) != prepared:
            raise ValueError('Prepared state is not the common original canonicalization: ' + original['sample_id'])
        if original['sample_id'] != prepared['sample_id'] or original['instruction'] != prepared['instruction'] or original['planning_goal'] != prepared['planning_goal']:
            raise ValueError('Canonicalization altered task language/ID/goal')
    return manifest
