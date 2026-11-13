"""Source-based port of LLM-as-BT-Planner's automated iterative workflow.

The language planner, BT generator and failure-feedback regeneration are distinct
stages. This is a domain/backbone port, not an untouched KIOS/MIOS deployment.
No HiBT decoder/repair or evaluator goal is used.
"""
from __future__ import annotations

from goal_contract import goal_visible_client, goal_user_turn

import copy
import json
import re
from pathlib import Path
from prompt_registry import for_prompt

METHOD = "LLM-as-BT-Planner"
SOURCE_VERSION = "ProNeverFake/kios@e9f16f5bd110ab647242077d55d5cb0a71e4fcd9"
ADAPTER = "source_based_iterative_port_shared_domain_and_backbone"
AUDIT = Path(__file__).resolve().parents[2] / "source_audit" / "kios"


def _json(text):
    text = text.strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            return decoder.raw_decode(text[match.start():])[0]
        except json.JSONDecodeError:
            pass
    raise ValueError("No JSON object/array in model output")


def _source_prompt(name, fallback):
    path = AUDIT / name
    return path.read_text() if path.exists() else fallback


def _arguments(text):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            parts.append(text[start:index].strip()); start = index + 1
    parts.append(text[start:].strip())
    return parts


def _normalize_condition(expression, kind="precondition"):
    expression = expression.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", expression)
    if not match: raise ValueError("Condition must contain an explicit predicate call")
    op, body = match.groups()
    if op.lower() in {"and", "or"}:
        return {"name": ("sequence" if op.lower() == "and" else "selector") + ": explicit Boolean condition",
                "children": [_normalize_condition(x, kind) for x in _arguments(body)]}
    if op.lower() == "not":
        child = _normalize_condition(body, kind)
        if child["name"].startswith(("target:", "precondition:", "condition:")):
            child["negated"] = not child.get("negated", False)
            return child
        return {"name": "inverter: explicit negation", "children": [child]}
    return {"name": kind + ": " + expression}


def _normalize(tree, registry, parent_kind=None, position=None):
    """Lossless schema conversion only: no added actions, goals or ordering.

    Alternative JSON spellings with an explicit type/predicate/action are
    unambiguous syntax. Untyped multi-child roots are rejected, not guessed.
    Raw native artifacts remain unchanged in the result trace.
    """
    if not isinstance(tree, dict): raise ValueError("Native tree is not an object")
    node = copy.deepcopy(tree)
    if "negated" in node and not isinstance(node["negated"], bool):
        raise ValueError("Negation must be an explicit JSON Boolean")
    name, explicit = str(node.get("name", "")), str(node.get("type", "")).lower()
    prefix, sep, payload = name.partition(":")
    control = {"fallback": "selector", "selector": "selector", "sequence": "sequence",
               "and": "sequence", "or": "selector", "inverter": "inverter"}
    kind = prefix.strip().lower() if sep else explicit or name.lower()
    if kind == "root" and len(node.get("children", [])) == 1:
        return _normalize(node["children"][0], registry, parent_kind, position)
    if kind in control:
        mapped = control[kind]
        node["name"] = mapped + ": " + (payload.strip() if sep else name or mapped)
        node["children"] = [_normalize(child, registry, mapped, i)
                            for i, child in enumerate(node.get("children", []))]
        return node
    condition_kind = "target" if parent_kind == "selector" and position == 0 else "precondition"
    if kind in {"target", "precondition", "condition"}:
        expression = payload.strip() if sep else node.get("predicate")
        if expression is None and len(node.get("children", [])) == 1:
            converted = _normalize(node["children"][0], registry, parent_kind, position)
            if node.get("negated"):
                if converted["name"].split(":", 1)[0] in {"target", "precondition", "condition"}:
                    converted["negated"] = not converted.get("negated", False)
                else:
                    converted = {"name": "inverter: explicit wrapper negation", "children": [converted]}
            return converted
        if expression is None: raise ValueError("Condition wrapper lacks an explicit expression")
        if node.get("children"):
            raise ValueError("Condition leaf has nonempty children; conversion cannot discard them")
        converted = _normalize_condition(expression, kind if kind != "condition" else condition_kind)
        if node.get("negated"):
            if converted["name"].split(":", 1)[0] in {"target", "precondition", "condition"}:
                converted["negated"] = not converted.get("negated", False)
            else:
                converted = {"name": "inverter: explicit expression negation", "children": [converted]}
        return converted
    if kind == "action":
        if node.get("children"):
            raise ValueError("Action leaf has nonempty children; conversion cannot discard control flow")
        node["name"] = "action: " + (payload.strip() if sep else node.get("action", ""))
        node.pop("children", None)
        return node
    # A bare call can be classified only when its symbol is in the registry.
    bare = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\([^()]*\)", name)
    if bare:
        symbol = bare[1]
        if symbol in registry.get("actions", {}): return {"name": "action: " + name}
        if symbol in registry.get("predicates", {}): return _normalize_condition(name, condition_kind)
    raise ValueError(f"Ambiguous/unsupported native node spelling: {tree!r}")


def _validate(tree, registry):
    if not isinstance(tree, dict) or not isinstance(tree.get("name"), str):
        raise ValueError("BT node must have a string name")
    kind, separator, payload = tree["name"].partition(":")
    if not separator:
        raise ValueError("Node name must retain KIOS type: payload syntax")
    kind = kind.strip()
    if kind in ("selector", "sequence", "inverter"):
        if not isinstance(tree.get("children"), list) or not tree["children"]:
            raise ValueError("Control node must have nonempty ordered children")
        for child in tree["children"]:
            _validate(child, registry)
        if kind == "inverter" and len(tree["children"]) != 1:
            raise ValueError("Inverter must have exactly one child")
    elif kind in ("target", "precondition", "condition", "action"):
        if not re.fullmatch(r"\s*[\w]+\([^()]*\)\s*", payload):
            raise ValueError("Leaf must be a grounded registry call")
        if tree.get("children"):
            raise ValueError("Leaf has children")
    else:
        raise ValueError(f"Unsupported KIOS node type: {kind}")
    return tree


def _call(llm, messages, budget, trace, stage):
    calls = sum(row.get("event") == "llm" for row in trace)
    if calls >= int(budget.get("max_calls", 12)):
        raise RuntimeError("LLM call budget exhausted")
    try:
        response = llm.complete(messages, max_new_tokens=int(budget.get("max_output_tokens", 2048)),
                                temperature=float(budget.get("temperature", 0.0)),
                                seed=int(budget.get("seed", 0)))
    except Exception as exc:
        trace.append({"event": "llm", "stage": stage, "messages": copy.deepcopy(messages),
                      "response_error": repr(exc)})
        raise
    if isinstance(response, str):
        response = {"text": response}
    trace.append({"event": "llm", "stage": stage, "messages": copy.deepcopy(messages),
                  "response": copy.deepcopy(response)})
    return response["text"]


def _domain(registry):
    # Root registry may include development demonstrations. Never accept an
    # answer supplied as request.target or an evaluator-only private field.
    return {k: copy.deepcopy(v) for k, v in for_prompt(registry).items()
            if k not in {"target", "gold_tree", "reference_plan", "test_solution", "kios_demonstrations"}}


def _planning_prompt(request, registry):
    messages = [
        {"role": "system", "content": (
            "You are the assembly/task planner in LLM-as-BT-Planner. Decompose the human "
            "instruction into ordered product/task subgoals using the full domain and current state. "
            "Return only JSON {\"subgoals\":[\"a natural-language goal\", ...]}. Preserve every "
            "requested goal. Do not output an executable action sequence at this planning stage.")},
    ]
    for example in registry.get("kios_demonstrations", []):
        messages.extend([
            goal_user_turn(json.dumps({"instruction": example["instruction"],
                "world_state": example["world_state"]}), example),
            {"role": "assistant", "content": json.dumps({"subgoals": [example["instruction"]]})}])
    messages.append({"role": "user", "content": json.dumps({"instruction": request["instruction"],
        "world_state": request["world_state"], "domain": _domain(registry)}, ensure_ascii=False)})
    return messages


def _compact_tree(tree):
    return {key: [_compact_tree(x) for x in value] if key == "children" else value
            for key, value in tree.items() if key in {"name", "children", "negated", "control_mode", "memory"}}


def _example_unit(example):
    target = example.get("target")
    expected = "action: " + example["teacher_actions"][-1]["name"] + "(" + ", ".join(example["teacher_actions"][-1]["args"]) + ")"
    def search(node):
        children = node.get("children", [])
        if node.get("name", "").startswith("selector:") and len(children) >= 2:
            body = children[-1]
            if body.get("children") and body["children"][-1].get("name") == expected:
                unit = _compact_tree(node)
                prerequisites = []
                for child in unit["children"][-1]["children"][:-1]:
                    if child["name"].startswith("selector:"):
                        child = copy.deepcopy(child["children"][0])
                    if child["name"].startswith("target:"):
                        child["name"] = "precondition:" + child["name"].split(":", 1)[1]
                    prerequisites.append(child)
                unit["children"][-1]["children"] = prerequisites + [unit["children"][-1]["children"][-1]]
                return unit
        for child in children:
            found = search(child)
            if found: return found
    return search(example["behavior_tree"])


def _generation_prompt(request, registry, subgoal, runtime_state, previous_tree, failed_node):
    system = _source_prompt("system.txt", "You interpret instructions for robotic tasks.")
    task = _source_prompt("task.txt", "Generate a complete behavior tree, revising failures from feedback.")
    bt = _source_prompt("behaviortree.txt", "Use selector/sequence control nodes and target/precondition/action leaves.")
    output = _source_prompt("output_format.txt", "Return JSON containing behavior_tree and action_sequence.")
    # Official output schema, node definitions and iterative inputs retained;
    # the original industrial domain and benchmark-specific examples are not
    # copied into unrelated tests. Root can supply held-out demonstrations.
    payload = {"target": subgoal, "initial_world_state": request["world_state"],
               "runtime_world_state": runtime_state, "last_behavior_tree": previous_tree,
               "last_failed_node": failed_node, "domain": _domain(registry)}
    messages = [{"role": "system", "content": "\n\n".join([system, task, bt, output])}]
    for example in registry.get("kios_demonstrations", []):
        messages.extend([
            goal_user_turn(json.dumps({"target": example["instruction"],
                "initial_world_state": example["world_state"], "runtime_world_state": None,
                "last_behavior_tree": None, "last_failed_node": None}, ensure_ascii=False), example),
            {"role": "assistant", "content": json.dumps({"behavior_tree": _compact_tree(example["behavior_tree"]),
                "action_sequence": [a["name"] + "(" + ", ".join(a["args"]) + ")"
                                    for a in example.get("teacher_actions", [])]}, ensure_ascii=False)}])
    messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    return messages


def plan(request: dict, registry: dict, llm, budget: dict, feedback=None) -> dict:
    """Plan language subgoals, synthesize each BT, and apply native feedback.

    The callback must simulate the supplied full tree from the request's initial
    state and report structural/tick feedback, not reveal an evaluator goal.
    All candidate trees and failed outputs remain in the trace.
    """
    llm = goal_visible_client(llm, request, registry)
    if budget.get("kios_variant") == "recursive":
        return plan_recursive(request, registry, llm, budget, feedback)
    trace, errors, subtrees = [], [], []
    result = {"method": METHOD, "source_version": SOURCE_VERSION,
              "domain_adapter": ADAPTER, "tree": None, "trace": trace,
              "errors": errors, "configuration": "automatic_iterative",
              "native_artifacts": []}
    try:
        if not request.get("instruction"):
            raise ValueError("Missing original natural-language instruction")
        raw = _call(llm, _planning_prompt(request, registry), budget, trace, "assembly_planner")
        parsed = _json(raw)
        subgoals = parsed.get("subgoals") if isinstance(parsed, dict) else parsed
        if not isinstance(subgoals, list) or not subgoals or not all(isinstance(x, str) and x.strip() for x in subgoals):
            raise ValueError("Assembly planner must return nonempty natural-language subgoals")
        result["subgoals"] = subgoals
        runtime_state = copy.deepcopy(request["world_state"])
        for index, subgoal in enumerate(subgoals):
            previous = request.get("_previous_tree") if index == 0 else None
            failed = request.get("_failed_node") if index == 0 else None
            tree = None
            for attempt in range(min(5, int(budget.get("generation_attempts", 5)))):
                messages = _generation_prompt(request, registry, subgoal, runtime_state, previous, failed)
                try:
                    raw = _call(llm, messages, budget, trace, f"bt_generator:{index}:{attempt}")
                    native = _json(raw)
                    result["native_artifacts"].append(copy.deepcopy(native))
                    previous = native.get("behavior_tree", native)
                    tree = _validate(_normalize(previous, registry), registry)
                except (ValueError, TypeError, KeyError) as exc:
                    trace.append({"event": "generation_error", "subgoal": index, "attempt": attempt,
                                  "error": repr(exc), "raw": raw})
                    failed = {"name": "BT serialization", "summary": str(exc)}
                    continue
                combined = {"name": "sequence: inferred subgoals", "children": copy.deepcopy(subtrees) + [tree]}
                result["tree"] = combined
                if feedback is None:
                    break
                fb = feedback(copy.deepcopy(combined))
                trace.append({"event": "simulation_feedback", "subgoal": index, "attempt": attempt,
                              "tree": copy.deepcopy(combined), "feedback": copy.deepcopy(fb)})
                runtime_state = copy.deepcopy(fb.get("world_state", runtime_state))
                if fb.get("result", fb.get("status")) == "success":
                    break
                previous = tree
                failed = fb.get("failed_node", fb.get("final_node")) or {"name": "BT execution", "summary": fb.get("error", fb.get("summary", "failure"))}
            if tree is None:
                raise ValueError(f"No parseable native BT for inferred subgoal {index}")
            subtrees.append(tree)
        result["tree"] = {"name": "sequence: inferred subgoals", "children": subtrees}
    except Exception as exc:
        errors.append({"stage": "planning", "error": repr(exc)})
    result["goal_prompt_events"] = copy.deepcopy(llm.goal_prompt_events)
    result["llm_calls"] = sum(row.get("event") == "llm" for row in trace)
    return result


def plan_recursive(request, registry, llm, budget, feedback=None):
    """Paper Algorithm1: MakePlan, MakeTree, recursive prerequisites, PredictState.

    Both the action sequence and predicted world state are produced by the LLM,
    as in KIOS; the adapter does not replace these with a symbolic oracle.
    """
    llm = goal_visible_client(llm, request, registry)
    trace, errors = [], []
    result = {"method": METHOD, "source_version": SOURCE_VERSION, "domain_adapter": ADAPTER,
              "configuration": "automatic_recursive", "tree": None, "trace": trace,
              "errors": errors, "native_artifacts": []}

    def ask(stage, instruction, payload):
        messages = [{"role": "system", "content": instruction}]
        for example in registry.get("kios_demonstrations", []):
            actions = [a["name"] + "(" + ", ".join(a["args"]) + ")" for a in example.get("teacher_actions", [])]
            pair = None
            if stage == "assembly_planner" and example.get("target"):
                pair = ({"instruction": example["instruction"], "world_state": example["world_state"]},
                        {"goals": [example["target"]]})
            elif stage.startswith("MakePlan") and actions:
                pair = ({"target": example.get("target"), "start_world_state": example["world_state"]},
                        {"task_plan": actions})
            elif stage.startswith("MakeTree") and actions:
                unit = _example_unit(example)
                if unit: pair = ({"action": actions[-1], "target": example.get("target")}, {"behavior_tree": unit})
            elif stage.startswith("PredictState") and example.get("final_world_state"):
                pair = ({"start_world_state": example["world_state"], "action_plan": actions},
                        {"estimated_world_state": example["final_world_state"]})
            if pair:
                messages.extend([goal_user_turn(json.dumps(pair[0], ensure_ascii=False), example),
                                 {"role": "assistant", "content": json.dumps(pair[1], ensure_ascii=False)}])
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        return _json(_call(llm, messages, budget, trace, stage))

    try:
        inferred = ask("assembly_planner", "Interpret the instruction into ordered goal predicates "
                       "from the supplied domain. Return JSON {\"goals\":[\"predicate(args)\", ...]}. "
                       "Use exact known object names. Include every requested goal. Use desired dynamic "
                       "world relations, rather than immutable capability/affordance facts as task goals.",
                       {"instruction": request["instruction"], "world_state": request["world_state"],
                        "domain": _domain(registry)})
        goals = inferred["goals"] if isinstance(inferred, dict) else inferred
        if not isinstance(goals, list) or not goals: raise ValueError("No inferred goals")
        result["subgoals"] = goals
        initial_nodes = [{"name": f"target: {goal}"} for goal in goals]
        tree = {"name": "sequence: recursive inferred subgoals", "children": initial_nodes}
        result["tree"] = tree
        visits = 0

        def expand_nodes(nodes, state, depth):
            nonlocal visits
            if depth > int(budget.get("max_recursion_depth", 24)):
                raise RuntimeError("KIOS recursive expansion depth exhausted")
            state = copy.deepcopy(state)
            for index, node in enumerate(nodes):
                kind, _, goal = node["name"].partition(":")
                if kind.strip() == "action": continue
                if kind.strip() not in {"target", "precondition", "condition"}:
                    raise ValueError("Expected condition while expanding unit subtree")
                visits += 1
                planned = ask(f"MakePlan:{visits}", "You are KIOS MakePlan. Generate an ordered "
                              "action sequence achieving the current condition from the given state. "
                              "Return JSON {\"task_plan\":[\"action(args)\", ...]}. Return [] only "
                              "if the condition is already satisfied. Use the supplied domain.",
                              {"target": goal.strip(), "start_world_state": state, "domain": _domain(registry)})
                actions = planned.get("task_plan") if isinstance(planned, dict) else planned
                if not isinstance(actions, list): raise ValueError("MakePlan did not return action list")
                if not actions: continue
                unit = ask(f"MakeTree:{visits}", "You are KIOS MakeTree. Interpret the given action "
                           "into its unit behavior tree: a selector with its achieved target condition, "
                           "then a sequence of all preconditions followed by that action. Use the "
                           "domain's exact preconditions/effects. Return JSON {\"behavior_tree\": "
                           "{\"name\":\"selector: ...\",\"children\":[...]}}. All leaves use "
                           "target/precondition/action: predicate_or_action(args).", {"action": actions[-1],
                           "target": goal.strip(), "domain": _domain(registry)})
                subtree = _validate(_normalize(unit.get("behavior_tree", unit), registry), registry)
                if not subtree["name"].startswith("selector:") or len(subtree.get("children", [])) < 2:
                    raise ValueError("MakeTree must return guarded unit subtree")
                body = subtree["children"][-1]
                if not body["name"].startswith("sequence:"):
                    raise ValueError("Unit subtree lacks prerequisite/action sequence")
                nodes[index] = subtree
                result["native_artifacts"].append(copy.deepcopy(unit))
                expand_nodes(body["children"], state, depth + 1)
                predicted = ask(f"PredictState:{visits}", "You are KIOS PredictState. Estimate the "
                                "world state after executing the given complete action plan using "
                                "the supplied preconditions and effects. Return only JSON "
                                "{\"estimated_world_state\": {...}} with the same state format.",
                                {"start_world_state": state, "action_plan": actions, "domain": _domain(registry)})
                state = predicted.get("estimated_world_state", predicted)
            return state

        result["predicted_final_state"] = expand_nodes(initial_nodes, request["world_state"], 0)
        if feedback is not None:
            result["simulation_feedback"] = feedback(copy.deepcopy(tree))
    except Exception as exc:
        errors.append({"stage": "recursive_planning", "error": repr(exc)})
    result["goal_prompt_events"] = copy.deepcopy(llm.goal_prompt_events)
    result["llm_calls"] = sum(row.get("event") == "llm" for row in trace)
    return result


def replan(request, registry, llm, budget, feedback=None, tree=None, failed_node=None, runtime_state=None):
    """Native feedback regeneration entry point for actual execution failures."""
    updated = copy.deepcopy(request)
    if runtime_state is not None:
        updated["world_state"] = copy.deepcopy(runtime_state)
    updated["_previous_tree"] = copy.deepcopy(tree)
    updated["_failed_node"] = copy.deepcopy(failed_node)
    return plan(updated, registry, llm, budget, feedback=feedback)


update = replan


def software_smoke():
    """Software-only: verifies generator repair and no evaluator-goal exposure."""
    class MockLLM:
        def __init__(self): self.n = 0
        def complete(self, messages, **kwargs):
            assert "SECRET_EVALUATOR_TARGET" not in json.dumps(messages)
            self.n += 1
            if self.n == 1: return {"text": '{"subgoals":["Move part into receptacle"]}'}
            action = "wrong" if self.n == 2 else "place"
            return {"text": json.dumps({"behavior_tree": {"name": f"action: {action}(part, receptacle)"}})}
    def fb(tree):
        leaf = tree["children"][-1]
        return {"result": "failure" if "wrong" in leaf["name"] else "success", "final_node": leaf,
                "world_state": {"objects": []}}
    result = plan({"instruction": "Put the part into its receptacle", "world_state": {"objects": []},
                   "target": "SECRET_EVALUATOR_TARGET"}, {}, MockLLM(), {"max_calls": 5}, feedback=fb)
    assert not result["errors"] and result["llm_calls"] == 3
    assert "place" in result["tree"]["children"][0]["name"]
    return {"scope": "software_only_mock_no_experimental_result", "passed": True}


if __name__ == "__main__":
    print(json.dumps(software_smoke(), indent=2))
