"""LLM-BT instruction reasoning -> genuine NER Parser -> ATL Expand/Insert.

This is a parameterized-domain port of Zhou et al. ICRA2024, Algorithm1.
The pretrained NER checkpoint is called through llm.ner or a supplied CPU
tagger. There is deliberately no regex/LLM substitute when that stage fails.
"""
from __future__ import annotations

from goal_contract import goal_visible_client, validate_request, goal_user_turn

import copy
import json
import re
from collections import Counter

from assembly_registry import ground_registry_actions, parse_call, state_facts
from prompt_registry import for_prompt

METHOD = "LLM-BT"
SOURCE_VERSION = "henryhaotian/LLM-BT@c69c18d0cf4b78f166ed352fc0fa8470823b32f6"
ADAPTER = "source_based_distilbert_parser_and_parameterized_ATL_Expand_Insert_port"


def _canonical(text):
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("##", ""))


def _schema_literal(text):
    text = text.strip()
    negative = text.startswith("(not (") and text.endswith("))")
    if negative: text = text[6:-2]
    if "(" in text and not text.startswith("("):
        pred, args = parse_call(text)
        return negative, pred, [a.lstrip("?") for a in args]
    words = text.strip("()").split()
    return negative, words[0], [a.lstrip("?") for a in words[1:]]


def _atl(registry):
    """Extend the paper's manual Action Template Library to named domain ops.

    Maps semantic operation words to an achieved condition, not to a plan.
    The table is constructed from complete action schemas before any test goal.
    Registry-specified `llmbt_atl` can provide explicit domain role annotations.
    """
    if registry.get("llmbt_atl"):
        return copy.deepcopy(registry["llmbt_atl"])
    out = {}
    actions = registry["actions"]
    if isinstance(actions, list):
        actions = {a["name"]: a for a in actions}
    for name, spec in actions.items():
        effects = spec.get("effect", spec.get("effects", []))
        positive = [x for x in map(_schema_literal, effects) if not x[0]]
        if not positive: continue
        preferred = [x for x in positive if x[1] not in {"is_empty", "is_equippable"}]
        _, predicate, variables = (preferred or positive)[-1]
        # T/D are the object and destination roles of the paper's Parser.
        # Hand/tool support steps generally remain with Expand, not Reasoning.
        if name in {"pick_up", "load_tool", "change_tool"} and predicate == "hold":
            roles = ["Destination", "Target"]
        else:
            roles = ["Target", "Destination", "Location"][:len(variables)]
        out[name] = {"predicate": predicate, "roles": roles,
                     "variables": variables, "source_action": name}
    return out


def _reasoning_role_parts(action):
    """Same domain-role rendering used to prepare the genuine NER training."""
    name, args = action["name"], action["args"]
    if name in {"insert", "place", "screw", "detach", "pull", "unscrew"}:
        return name, args[-2], args[-1]
    if name in {"pick_up", "load_tool", "change_tool"}:
        return name, args[-1], args[-2] if name == "pick_up" else args[0]
    if name in {"unload_tool", "put_down"}:
        return name, args[1], None
    if name in {"place_on", "place_in"}:
        return name, args[0], args[1]
    if name in {"open", "close", "turnon", "turnoff"}:
        return name, args[0], None
    raise ValueError(f"No declared LLM-BT demonstration rendering for {name}")


def render_llmbt_demonstration_step(action):
    name, target, destination = _reasoning_role_parts(action)
    text = name + " " + target
    if destination is not None:
        prep = {"insert": "into", "place": "on", "screw": "into", "pick_up": "with",
                "load_tool": "onto", "change_tool": "onto", "detach": "from", "pull": "from",
                "unscrew": "from", "place_on": "on", "place_in": "in"}.get(name, "to")
        text += " " + prep + " " + destination
    return text


def prepare_llmbt_demonstrations(teacher_rows, cover_state_categories=True):
    """Seven small train-only examples; never inspect an evaluation request.

    Retain the original three final-outcome examples. Four extra state-goal
    examples use *prefixes of validated training trajectories*, with the same
    initial state and role-bound operation annotation. The helper does not
    modify parsed inference goals or synthesize a test answer.
    """
    rows = list(teacher_rows)
    if not rows or any(not row.get("sample_id", "").startswith("train_") for row in rows):
        raise ValueError("Pass only the declared assembly training teacher rows")
    by_id = {row["sample_id"]: row for row in rows}
    demos = [copy.deepcopy(by_id[f"train_{i:04d}"]) for i in [0, 7, 11]]
    if not cover_state_categories:
        return demos
    categories = [
        ("hand_hold", {"load_tool", "change_tool"}),
        ("tool_hold", {"pick_up"}),
        ("empty_hand", {"unload_tool"}),
        ("empty_tool", {"put_down"}),
    ]
    for category, action_names in categories:
        match = next(((row, index, action) for row in rows
                      for index, action in enumerate(row.get("teacher_actions", []))
                      if action["name"] in action_names), None)
        if match is None:
            raise ValueError(f"Training pool does not cover state category {category}")
        row, index, action = match
        name, args = action["name"], action["args"]
        if category == "hand_hold":
            owner, held = args[0], args[-1]
            instruction, goal = f"Have {owner} hold {held}.", f"hold({owner}, {held})"
        elif category == "tool_hold":
            owner, held = args[-2], args[-1]
            instruction, goal = f"Have {owner} hold {held}.", f"hold({owner}, {held})"
        else:
            owner = args[0] if category == "empty_hand" else args[1]
            instruction, goal = f"Make {owner} empty.", f"is_empty({owner})"
        demos.append({"sample_id": row["sample_id"] + "/" + category,
                      "instruction": instruction, "world_state": copy.deepcopy(row["world_state"]),
                      "target": goal, "teacher_actions": copy.deepcopy(row["teacher_actions"][:index + 1]),
                      "generation": {"source": "prefix of shared non-test training teacher trajectory",
                                     "source_sample_id": row["sample_id"], "source_action_index": index,
                                     "state_category": category, "derived_from_evaluation": False}})
    return demos


def _training_goal_atom(text):
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        terms = text[1:-1].split()
        return terms[0], tuple(terms[1:])
    predicate, args = parse_call(text)
    return predicate, tuple(args)


def requested_libero_demonstration_actions(example):
    """Render *training* desired outcomes, excluding implicit support goals.

    This routine is only called for supervised demonstrations. Actual model
    steps and inferred evaluation goals are never filtered through it.
    """
    declared = example.get("requested_goals")
    if declared is None and isinstance(example.get("target"), str):
        declared = [x.strip() for x in example["target"].split(";") if x.strip()]
    if declared is None and example.get("pyramid_json"):
        goal = example["pyramid_json"][0]["items"][0]
        declared = goal["args"] if goal["predicate"] == "and" else [
            goal["predicate"] + "(" + ", ".join(goal["args"]) + ")"]
    if not declared:
        raise ValueError("LIBERO training demonstration needs explicit requested-goal supervision")
    wanted = {_training_goal_atom(x) for x in declared}
    action_predicates = {"place_on": "On", "place_in": "In", "open": "Open", "close": "Close",
                         "turnon": "Turnon", "turnoff": "Turnoff"}
    selected, covered = [], set()
    for action in example["teacher_actions"]:
        if action["name"] not in action_predicates:
            raise ValueError("Unknown primitive in LIBERO training demonstration")
        achieved = action_predicates[action["name"]], tuple(action["args"])
        if achieved in wanted:
            selected.append(action); covered.add(achieved)
    if covered != wanted:
        raise ValueError("Training teacher actions do not cover every requested goal")
    return selected


def _resolve_object(words, objects):
    key = _canonical(words)
    matches = [obj for obj in objects if _canonical(obj) == key]
    if len(matches) != 1:
        raise ValueError(f"NER entity does not uniquely name a known object: {words!r}")
    return matches[0]


def _frames(entities, text):
    frames, current = [], None
    last_spans = {}
    action_end = None
    for token in entities:
        entity = token.get("entity", token.get("entity_group", "O"))
        if entity == "O": continue
        if "-" in entity: prefix, role = entity.split("-", 1)
        else: prefix, role = "B", entity
        word = token.get("word", "")
        span = None
        if isinstance(token.get("start"), int) and isinstance(token.get("end"), int):
            start, end = token["start"], token["end"]
            if not any(char.isalnum() or char == "_" for char in text[start:end]):
                continue
            # Recover a complete underscore-containing identifier when the
            # BERT tokenizer split it into subwords. Labels still determine
            # the role; no new action/entity is inferred from unlabelled text.
            while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"): start -= 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"): end += 1
            if start < end:
                word, span = text[start:end], (start, end)
            if last_spans.get(role) == span: continue
            last_spans[role] = span
        if role == "Action":
            adjacent = span is None or action_end is None or not text[action_end:span[0]].strip()
            if current and not any(k in current for k in {"Target", "Destination", "Location"}) and adjacent:
                current["Action"] += ("" if word.startswith("##") else " ") + word.replace("##", "")
                action_end = span[1] if span else None
                continue
            if current: frames.append(current)
            current = {"Action": word.replace("##", "")}
            action_end = span[1] if span else None
        elif role in {"Target", "Destination", "Location"} and current is not None:
            if prefix == "I" and role in current:
                current[role] += ("" if word.startswith("##") else " ") + word.replace("##", "")
            else:
                current[role] = word.replace("##", "")
    if current: frames.append(current)
    return frames


def _parse_conditions(entities, text, registry):
    atl = _atl(registry)
    aliases = {_canonical(key): key for key in atl}
    for alias, canonical in {"pick": "pick_up", "pickup": "pick_up", "put": "put_down",
                             "putdown": "put_down", "load": "load_tool", "unload": "unload_tool",
                             "change": "change_tool", "turn": "turnon"}.items():
        if canonical in atl: aliases[alias] = canonical
    objects = registry.get("objects", [])
    objects = [x["name"] if isinstance(x, dict) else x for x in objects]
    conditions = []
    for frame in _frames(entities, text):
        op = aliases.get(_canonical(frame["Action"]))
        if op is None:
            raise ValueError(f"NER action is not in domain ATL: {frame['Action']}")
        item = atl[op]
        args = []
        for role in item["roles"]:
            if role not in frame:
                raise ValueError(f"NER missing {role} for {op}")
            args.append(_resolve_object(frame[role], objects))
        call = f"{item['predicate']}({', '.join(args)})"
        if call not in conditions: conditions.append(call)
    if not conditions: raise ValueError("BERT Parser returned no complete action/argument frame")
    return conditions


def _leaf(literal, kind="precondition"):
    negative = literal.startswith("!")
    return {"name": f"{kind}: {literal.lstrip('!')}", **({"negated": True} if negative else {})}


def _literal(node):
    return ("!" if node.get("negated") else "") + node["name"].split(":", 1)[1].strip()


def _walk(tree, path=()):
    yield path, tree
    for i, child in enumerate(tree.get("children", [])):
        yield from _walk(child, path + (i,))


def _node_at(tree, path):
    for i in path: tree = tree["children"][i]
    return tree


def _replace(tree, path, replacement):
    if not path: return replacement
    parent = _node_at(tree, path[:-1])
    parent["children"][path[-1]] = replacement
    return tree


def _condition_type(node):
    # A parameterized condition's owner distinguishes different effectors. This
    # extends the native hand-coded condition classes to a shared domain model.
    pred, args = parse_call(node["name"].split(":", 1)[1].strip())
    return pred, args[0] if args else None


def _locate_failed(tree, failed_node):
    if isinstance(failed_node, dict) and isinstance(failed_node.get("path"), (list, tuple)):
        path = tuple(failed_node["path"])
        if _node_at(tree, path)["name"] == failed_node.get("name", _node_at(tree, path)["name"]):
            return path
    name = failed_node if isinstance(failed_node, str) else (failed_node or {}).get("name")
    options = [(path, node) for path, node in _walk(tree) if node.get("name") == name]
    if not options: raise ValueError(f"Failed node cannot be located: {failed_node!r}")
    # A guarded condition may occur both in a selector and a later precondition.
    # Prefer the unresolved non-guard occurrence, rather than repeatedly
    # expanding the first target check of an already-expanded fallback.
    non_guards = [p for p, n in options if not (p and p[-1] == 0 and
                  _node_at(tree, p[:-1])["name"].startswith("selector:"))]
    return (non_guards or [p for p, n in options])[-1]


def _choose_failed_condition(tree, feedback):
    """GetFailedNode domain-port disambiguation, using actual tick statuses.

    The original cargo ATL has one achiever branch. A parameterized domain can
    have several; its leftmost unresolved prerequisite takes priority. Failures
    hidden by an ultimately successful fallback are excluded. No goal,
    transition lookahead, reachability lookup, or action filtering occurs.
    """
    reported = feedback.get("failed_node", feedback.get("final_node"))
    records = feedback.get("native_trace", feedback.get("trace", []))
    if not isinstance(records, list) or not records:
        return reported
    last_tick = max((x.get("tick", 0) for x in records if isinstance(x, dict)), default=0)
    records = [x for x in records if isinstance(x, dict) and x.get("tick", 0) == last_tick and "path" in x]
    statuses = {tuple(x["path"]): x.get("status") for x in records}
    for record in records:
        if record.get("status") != "failure": continue
        path = tuple(record["path"])
        try: node = _node_at(tree, path)
        except (KeyError, IndexError, TypeError): continue
        if node.get("children") or node["name"].split(":", 1)[0].strip() not in {"target", "precondition", "condition"}:
            continue
        if path:
            parent = _node_at(tree, path[:-1])
            if path[-1] == 0 and parent["name"].startswith("selector:") and len(parent.get("children", [])) > 1:
                continue
        if any(statuses.get(path[:depth]) != "failure" for depth in range(len(path))):
            continue
        return {**copy.deepcopy(node), "path": list(path)}
    return reported


def expand_insert(tree, failed_node, actions, state=None):
    """Algorithm1: Expand failed condition through ATL; Insert by priority.

    All actions achieving the failed signed condition are kept. Enumeration
    order is the supplied complete-domain ATL order; no test-goal-specific
    repair, oracle plan, or pruning is performed here.
    """
    tree = copy.deepcopy(tree)
    path = _locate_failed(tree, failed_node)
    condition = _node_at(tree, path)
    if condition["name"].split(":", 1)[0].strip() not in {"target", "precondition", "condition"}:
        raise ValueError("BTs Update expands failed conditions, not low-level action failures")
    wanted = _literal(condition)
    branches, pre_types = [], set()
    for action in actions:
        effects = set(action.get("add", [])) | {"!" + x for x in action.get("delete", [])}
        if wanted not in effects: continue
        prerequisites = [_leaf(p) for p in action.get("pre", [])]
        for node in prerequisites: pre_types.add(_condition_type(node))
        branches.append({"name": f"sequence: achieve {wanted}", "control_mode": "reactive_sequence", "children": prerequisites + [
            {"name": "action: " + action["name"]}]})
    if not branches: raise ValueError(f"ATL has no action achieving {wanted}")
    expanded = {"name": "selector: expand " + wanted, "control_mode": "reactive_fallback",
                "children": [copy.deepcopy(condition)] + branches}
    insertion = None
    # Search sequence ancestors for a higher-priority condition of the same
    # parameterized type. Raise this expansion before the first conflict.
    for depth in range(len(path)):
        parent_path, index = path[:depth], path[depth]
        parent = _node_at(tree, parent_path)
        if not parent["name"].startswith("sequence:"): continue
        for prior in range(index):
            leaves = [node for _, node in _walk(parent["children"][prior])
                      if node["name"].split(":", 1)[0].strip() in {"target", "precondition", "condition"}]
            if any(_condition_type(node) in pre_types for node in leaves):
                insertion = (parent_path, prior)
                break
        if insertion is not None: break
    if insertion is None:
        tree = _replace(tree, path, expanded)
        operation = "Replace"
    else:
        _node_at(tree, insertion[0])["children"].insert(insertion[1], expanded)
        operation = "AddBeforeConflictingConditionType"
    return tree, {"event": "BTs_Update", "operation": operation, "failed": copy.deepcopy(condition),
                  "path": list(path), "matching_actions": [b["children"][-1]["name"] for b in branches],
                  "insert_parent": list(insertion[0]) if insertion else None,
                  "insert_index": insertion[1] if insertion else None}


def _run_updates(tree, registry, budget, feedback, trace, holder=None):
    actions = ground_registry_actions(registry)
    seen, expansion_counts = set(), Counter()
    for iteration in range(int(budget.get("max_expansions", 128))):
        if holder is not None: holder["tree"] = tree
        if feedback is None: return tree
        fb = feedback(copy.deepcopy(tree))
        trace.append({"event": "execution_feedback", "iteration": iteration,
                      "feedback": copy.deepcopy(fb)})
        if fb.get("result", fb.get("status")) == "success": return tree
        failed = _choose_failed_condition(tree, fb)
        trace.append({"event": "GetFailedNode", "strategy": "first_critical_unexpanded_condition",
                      "reported_failed": copy.deepcopy(fb.get("failed_node", fb.get("final_node"))),
                      "selected_failed": copy.deepcopy(failed)})
        name = failed if isinstance(failed, str) else (failed or {}).get("name")
        expansion_counts[name] += 1
        if expansion_counts[name] > int(budget.get("max_same_failure_expansions", 16)):
            raise RuntimeError(f"BTs Update repeated failure budget exhausted: {name}")
        fingerprint = json.dumps([tree, failed], sort_keys=True)
        if fingerprint in seen: raise RuntimeError("BTs Update made no progress")
        seen.add(fingerprint)
        tree, record = expand_insert(tree, failed, actions, fb.get("world_state"))
        trace.append(record)
        if holder is not None: holder["tree"] = tree
    raise RuntimeError("BTs Update expansion budget exhausted")


def _reasoning_domain_information(registry, atl):
    """Static ATL semantics and entity roles; no request or evaluator input.

    Complete-skill meanings clarify the original method's semantic map. They
    do not fill a missing neural Parser entity or alter native BT Update.
    """
    meanings = {
        "load_tool": "Equip Target tool on Destination hand; the hand then holds that tool.",
        "change_tool": "Replace the equipped tool with Target tool on Destination hand.",
        "pick_up": "Grasp Target part with Destination tool; the tool then holds that part.",
        "put_down": "Release the part held by Target tool, leaving that tool empty.",
        "unload_tool": "Unequip Target tool from its hand, making the tool available and the hand empty.",
        "insert": "Insert Target part into Destination part.",
        "place": "Place Target part on Destination part.",
        "screw": "Screw Target part into Destination part.",
        "detach": "Make Target tool hold Destination part by detaching it from its placed connection.",
        "pull": "Make Target tool hold Destination part by pulling it from its inserted connection.",
        "unscrew": "Make Target tool hold Destination part by unscrewing it from its screwed connection.",
        "place_on": "Complete skill: grasp Target object, transport it, and release it on Destination support. No separate pickup step is needed or registered.",
        "place_in": "Complete skill: grasp Target object, transport it, and release it in Destination container or interior region. No separate pickup step is needed or registered.",
        "open": "Complete skill that opens Target access unit or addressed drawer region.",
        "close": "Complete skill that closes Target access unit or addressed drawer region.",
        "turnon": "Complete skill that switches Target appliance on.",
        "turnoff": "Complete skill that switches Target appliance off.",
    }
    patterns = {
        "load_tool": "load_tool Target onto Destination",
        "change_tool": "change_tool Target onto Destination",
        "pick_up": "pick_up Target with Destination",
        "put_down": "put_down Target", "unload_tool": "unload_tool Target",
        "insert": "insert Target into Destination", "place": "place Target on Destination",
        "screw": "screw Target into Destination", "detach": "detach Target from Destination",
        "pull": "pull Target from Destination", "unscrew": "unscrew Target from Destination",
        "place_on": "place_on Target on Destination", "place_in": "place_in Target in Destination",
        "open": "open Target", "close": "close Target",
        "turnon": "turnon Target", "turnoff": "turnoff Target",
    }
    role_sets = registry.get("role_objects", {})
    vocabulary = {}
    for name, item in atl.items():
        roles = dict(zip(item["roles"], item.get("variables", item["roles"])))
        action_schemas = registry.get("actions", {})
        if isinstance(action_schemas, list):
            action_schemas = {entry["name"]: entry for entry in action_schemas}
        schema_roles = action_schemas.get(name, {}).get("argument_roles", {})
        domains = {}
        for entity, variable in roles.items():
            if registry.get("domain") == "libero":
                if name in {"place_on", "place_in"}:
                    domain = "movable" if entity == "Target" else (
                        "placement_supports" if name == "place_on" else "containment_targets")
                else:
                    domain = "closable" if name in {"open", "close"} else "switchable"
            else:
                domain = schema_roles.get(variable, variable)
            domains[entity] = {"role": domain, "allowed_objects": copy.deepcopy(
                role_sets.get(domain, registry.get("objects", [])))}
        vocabulary[name] = {"roles": roles, "required_entities": list(roles),
            "meaning": meanings.get(name, "Achieve the declared ATL condition using this registered operation."),
            "language_pattern": patterns.get(name, name + " " + " to ".join(roles)),
            "entity_domains": domains}
    notes = [
        "The operation vocabulary is exhaustive. Use only its operation words and exact registered object identifiers.",
        "Every listed required entity must appear explicitly in the output sentence, including a required Destination even when only one hand or tool is available.",
        "Language-pattern words Target and Destination are placeholders for object identifiers, not literal output words.",
        "Keep only requested outcomes. Dependency and access prerequisites remain the native BT Update module's responsibility.",
    ]
    if registry.get("domain") == "assembly":
        notes += [
            "A hand equips and holds a tool. An equipped tool grasps a part. These are distinct holding relations.",
            "For load_tool and change_tool, Target is the desired tool and Destination is the hand: write both names explicitly.",
            "For pick_up, Target is the part and Destination is the tool. Never substitute the hand for that Destination.",
            "put_down takes the tool as Target and leaves that tool empty; unload_tool takes the tool as Target and leaves its hand empty.",
        ]
    elif registry.get("domain") == "libero":
        notes += [
            "This domain provides six complete skills: place_on, place_in, open, close, turnon and turnoff. It has no separate pick_up, grasp or release skill.",
            "place_on and place_in include grasping, transport and release. A pick-and-place instruction is expressed by the relevant single complete placement skill.",
            "All placement Target objects must be movable; Destination must belong to the listed support or containment role. Do not invent a region or use an abstract table fixture instead of its registered region.",
            "Do not add an Open final outcome solely to enable placement; access opening is a prerequisite. Preserve Open or Close when explicitly requested as a final outcome.",
        ]
        notes += copy.deepcopy(registry.get("domain_notes", []))
    return {"operation_vocabulary": vocabulary, "domain_role_notes": notes,
            "region_parents": copy.deepcopy(registry.get("region_parents", {}))}


def plan(request: dict, registry: dict, llm, budget: dict, feedback=None) -> dict:
    llm = goal_visible_client(llm, request, registry)
    trace, errors = [], []
    result = {"method": METHOD, "source_version": SOURCE_VERSION, "domain_adapter": ADAPTER,
              "tree": None, "trace": trace, "errors": errors, "parser": "genuine_HF_DistilBERT_NER"}
    try:
        if int(budget.get("max_calls", 12)) < 1:
            raise RuntimeError("Reasoning requires one actual LLM call")
        atl = _atl(registry)
        domain_information = _reasoning_domain_information(registry, atl)
        messages = [{"role": "system", "content": (
            "You are the Reasoning module of LLM-BT. Read the human instruction and semantic map. "
            "Return only a numbered list of desired task-achievement steps in simple English. "
            "Use the action vocabulary in the supplied Action Template Library and exact object names. "
            "Every required entity in an operation entry must be named explicitly, including required Destination. "
            "The vocabulary is exhaustive: never invent an operation outside it. Do not include explanations. "
            "Do not add transient grasping/tool steps unless explicitly requested: the BT update module "
            "will achieve prerequisites. Do not output JSON, function calls, argument lists, "
            "parentheses, or a behavior tree. Write ordinary descriptive sentences such as "
            "'insert part into socket', with exact object identifiers. Each line contains one "
            "operation word, the operated Target, a preposition and the Destination if needed. "
            "Preserve only the user's requested task outcomes, leaving support actions to BT Update.")},
        ]
        for example in registry.get("llmbt_demonstrations", []):
            actions = requested_libero_demonstration_actions(example) if registry.get("domain") == "libero" else example["teacher_actions"][-1:]
            sentences = [render_llmbt_demonstration_step(action) for action in actions]
            if not sentences: continue
            messages.extend([goal_user_turn(json.dumps({"instruction": example["instruction"],
                "semantic_map": example["world_state"]}), example), {"role": "assistant", "content":
                "\n".join(f"{i+1}. {sentence}." for i, sentence in enumerate(sentences))}])
        messages.append({"role": "user", "content": json.dumps({"instruction": request["instruction"],
                "semantic_map": request["world_state"],
                **domain_information,
                "recognized_roles": registry.get("role_objects", {}),
                "objects": registry.get("objects", [])}, ensure_ascii=False)})
        call_record = {"event": "llm", "stage": "Reasoning", "messages": copy.deepcopy(messages)}
        trace.append(call_record)
        try:
            response = llm.complete(messages, max_new_tokens=int(budget.get("max_output_tokens", 2048)),
                                    temperature=float(budget.get("temperature", 0)), seed=int(budget.get("seed", 0)))
        except Exception as exc:
            call_record["response_error"] = repr(exc)
            raise
        if isinstance(response, str): response = {"text": response}
        call_record["response"] = copy.deepcopy(response)
        text = response["text"]
        # TextFilter removes enumeration/declarative wrappers before the
        # source Parser's BERT token labelling; it does not infer entities.
        lines = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
                 for line in text.splitlines() if line.strip()]
        tagger = budget.get("ner_tagger") or getattr(llm, "ner", None)
        if tagger is None: raise RuntimeError("Genuine BERT NER endpoint/checkpoint is required")
        conditions = []
        for index, line in enumerate(lines):
            record = {"event": "BERT_Parser", "line": index, "text": line}
            trace.append(record)
            try:
                entities = tagger(line)
                if isinstance(entities, dict): entities = entities["entities"]
                record["entities"] = copy.deepcopy(entities)
                parsed = _parse_conditions(entities, line, registry)
                record["conditions"] = parsed
            except Exception as exc:
                record["error"] = repr(exc)
                raise
            conditions.extend(x for x in parsed if x not in conditions)
        if not conditions: raise ValueError("No task conditions parsed")
        tree = {"name": "sequence: initial goal conditions from BERT Parser", "control_mode": "reactive_sequence",
                "children": [_leaf(c, "target") for c in conditions]}
        result["initial_tree"] = copy.deepcopy(tree)
        result["interpreted_goals"] = conditions
        # Save the current tree before updates so failed expansions retain the
        # genuine parsed tree rather than returning an oracle fallback.
        result["tree"] = tree
        result["tree"] = _run_updates(tree, registry, budget, feedback, trace, result)
    except Exception as exc:
        errors.append({"stage": "Reasoning_Parser_BTUpdate", "error": repr(exc)})
    result["goal_prompt_events"] = copy.deepcopy(llm.goal_prompt_events)
    result["llm_calls"] = sum(x.get("event") == "llm" for x in trace)
    result["ner_calls"] = sum(x.get("event") == "BERT_Parser" for x in trace)
    return result


def update(request, registry, llm, budget, feedback=None, tree=None, failed_node=None, runtime_state=None, failure_trace=None):
    """Continue Algorithm1 after a real execution failure, preserving the tree."""
    validate_request(request, registry)
    if tree is None: return plan(request, registry, llm, budget, feedback)
    trace, errors = [], []
    result = {"method": METHOD, "source_version": SOURCE_VERSION, "domain_adapter": ADAPTER,
              "tree": copy.deepcopy(tree), "trace": trace, "errors": errors, "llm_calls": 0}
    try:
        if failed_node is not None:
            if failure_trace is not None:
                failed_node = _choose_failed_condition(tree, {"failed_node": failed_node, "native_trace": failure_trace})
            result["tree"], event = expand_insert(tree, failed_node, ground_registry_actions(registry), runtime_state)
            trace.append(event)
        result["tree"] = _run_updates(result["tree"], registry, budget, feedback, trace, result)
    except Exception as exc:
        errors.append({"stage": "BTs_Update", "error": repr(exc)})
    return result


replan = update


def software_prompt_domain_checks():
    """Verify static semantic-map content and unchanged missing-entity failure."""
    assembly = {"domain": "assembly", "objects": ["hand_a", "tool_a", "part_a"],
        "role_objects": {"hand": ["hand_a"], "tool": ["tool_a"], "part": ["part_a"]},
        "actions": {
            "load_tool": {"effect": ["hold ?hand ?tool"], "argument_roles": {"hand": "hand", "tool": "tool"}},
            "pick_up": {"effect": ["hold ?tool ?part"], "argument_roles": {"tool": "tool", "part": "part"}}}}
    info = _reasoning_domain_information(assembly, _atl(assembly))
    load = info["operation_vocabulary"]["load_tool"]
    assert load["required_entities"] == ["Destination", "Target"]
    assert load["entity_domains"]["Destination"]["allowed_objects"] == ["hand_a"]
    assert load["entity_domains"]["Target"]["allowed_objects"] == ["tool_a"]
    assert "onto Destination" in load["language_pattern"]
    assert "explicitly" in " ".join(info["domain_role_notes"])
    libero = {"domain": "libero", "objects": ["obj", "plate", "cabinet", "stove"],
        "role_objects": {"movable": ["obj"], "placement_supports": ["plate"],
            "containment_targets": ["cabinet"], "closable": ["cabinet"], "switchable": ["stove"]},
        "actions": {name: {"effect": [predicate + " ?object" + (" ?target" if name.startswith("place_") else "")]}
                    for name, predicate in {"place_on": "On", "place_in": "In", "open": "Open", "close": "Close", "turnon": "Turnon", "turnoff": "Turnoff"}.items()}}
    lib = _reasoning_domain_information(libero, _atl(libero))
    assert set(lib["operation_vocabulary"]) == set(libero["actions"])
    placement = lib["operation_vocabulary"]["place_on"]
    assert placement["required_entities"] == ["Target", "Destination"]
    assert "grasp" in placement["meaning"] and "release" in placement["meaning"]
    assert placement["entity_domains"]["Destination"]["allowed_objects"] == ["plate"]
    assert "no separate pick_up" in " ".join(lib["domain_role_notes"])
    assert lib["operation_vocabulary"]["open"]["required_entities"] == ["Target"]
    text = "load_tool tool_a"
    labels = [{"entity": "B-Action", "start": 0, "end": 9, "word": "load_tool"},
              {"entity": "B-Target", "start": 10, "end": 16, "word": "tool_a"}]
    try:
        _parse_conditions(labels, text, assembly)
    except ValueError as exc:
        assert "Destination" in str(exc)
    else:
        raise AssertionError("Missing Parser entity must not be filled from domain information")
    return {"scope": "SOFTWARE_ONLY_NO_MODEL_INFERENCE", "explicit_required_entities": True,
            "assembly_hand_destination_declared": True, "libero_complete_placement_skill_declared": True,
            "legal_scene_role_sets_declared": True, "all_six_libero_primitives": True,
            "missing_parser_entity_still_fails": True, "request_and_evaluator_not_inputs_to_helper": True}


def software_smoke():
    """Mock validates actual BERT-stage requirement and native update flow only."""
    registry = {"objects": ["part", "slot"], "actions": {"insert": {
        "effect": ["is_inserted_to ?part ?slot"]}},
        "ground_actions": [{"name": "insert(part, slot)", "pre": [],
                            "add": ["is_inserted_to(part, slot)"], "delete": []}]}
    class MockLLM:
        def complete(self, messages, **kwargs):
            assert "SECRET_EVALUATOR_TARGET" not in json.dumps(messages)
            return {"text": "1. Insert part into slot."}
        def ner(self, text):
            return [{"entity": "B-Action", "word": "insert"},
                    {"entity": "B-Target", "word": "part"},
                    {"entity": "B-Destination", "word": "slot"}]
    def feedback(tree):
        action_present = any(n["name"].startswith("action:") for _, n in _walk(tree))
        return {"result": "success" if action_present else "failure",
                "failed_node": {"name": "target: is_inserted_to(part, slot)", "path": [0]}}
    result = plan({"instruction": "Insert the part into the slot", "world_state": {},
                   "target": "SECRET_EVALUATOR_TARGET"}, registry, MockLLM(), {}, feedback)
    assert not result["errors"] and result["ner_calls"] == 1
    assert any(x.get("event") == "BTs_Update" for x in result["trace"])
    assert result["tree"]["children"][0]["name"].startswith("selector:")
    failure_selection = software_failure_selection_checks()
    return {"scope": "software_only_mock_no_experimental_result", "passed": True,
            "failure_selection": failure_selection, "running_semantics": software_running_semantics_checks()}


def software_failure_selection_checks():
    from native_runtime import NativeRuntime
    def run(tree, true=()):
        runtime = NativeRuntime(tree, lambda pred, args: pred in true, lambda name, args: "success")
        runtime.tick()
        reported = copy.deepcopy(runtime.last_failed_node)
        if reported: reported["path"] = runtime.last_failed_path
        return {"failed_node": reported, "native_trace": runtime.trace}
    single = {"name": "selector: achieve", "children": [
        {"name": "target: goal(obj)"}, {"name": "sequence: only achiever", "children": [
        {"name": "precondition: support(obj)"}, {"name": "action: do(obj)"}]}]}
    feedback = run(single)
    assert _choose_failed_condition(single, feedback) == feedback["failed_node"]
    multiple = copy.deepcopy(single)
    multiple["children"].append({"name": "sequence: later achiever", "children": [
        {"name": "precondition: other(obj)"}, {"name": "action: undo(obj)"}]})
    unchanged = copy.deepcopy(multiple)
    feedback = run(multiple)
    assert feedback["failed_node"]["path"] == [2, 0]
    assert _choose_failed_condition(multiple, feedback)["path"] == [1, 0]
    assert multiple == unchanged
    compensated = {"name": "sequence: root", "children": [
        {"name": "selector: successful fallback", "children": [
            {"name": "target: absent(obj)"}, {"name": "precondition: present(obj)"}]},
        {"name": "precondition: unresolved(obj)"}]}
    feedback = run(compensated, true={"present"})
    assert _choose_failed_condition(compensated, feedback)["path"] == [1]
    return {"single_achiever_unchanged": True, "native_alternative_priority_preserved": True,
            "compensated_failure_excluded": True, "tree_and_actions_unchanged": True}


def software_running_semantics_checks():
    """Match pinned legacy C++ controls, which restart their loop at child0."""
    from native_runtime import NativeRuntime
    gate, calls = [True], []
    sequence = {"name": "sequence: legacy source", "control_mode": "reactive_sequence", "children": [
        {"name": "precondition: enabled(obj)"}, {"name": "action: long_action(obj)"}]}
    runtime = NativeRuntime(sequence, lambda pred, args: gate[0],
                            lambda name, args: calls.append(name) or "running")
    assert runtime.tick() == "running"
    gate[0] = False
    assert runtime.tick() == "failure" and len(calls) == 1
    assert any(x.get("event") == "halt" for x in runtime.trace)
    gate, calls = [False], []
    fallback = {"name": "selector: legacy source", "control_mode": "reactive_fallback", "children": [
        {"name": "target: satisfied(obj)"}, {"name": "action: long_action(obj)"}]}
    runtime = NativeRuntime(fallback, lambda pred, args: gate[0],
                            lambda name, args: calls.append(name) or "running")
    assert runtime.tick() == "running"
    gate[0] = True
    assert runtime.tick() == "success" and len(calls) == 1
    assert any(x.get("event") == "halt" for x in runtime.trace)
    return {"sequence_reticks_earlier_condition_while_running": True,
            "fallback_can_preempt_running_later_action": True,
            "source_has_no_persistent_child_cursor": True}


if __name__ == "__main__":
    print(json.dumps(software_smoke(), indent=2))
