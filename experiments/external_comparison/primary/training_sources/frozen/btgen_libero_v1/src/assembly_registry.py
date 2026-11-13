"""Complete assembly domain registry; contains no request-target information.

Static constraint joins and recognized entity roles prune impossible groundings,
never task goals or current dynamic preconditions. All eleven action schemas and
all ten predicates remain. Unknown/empty role domains fall back to all objects.
"""
from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path


def _definition_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'vendor/assembly/artifact/data/world_definition.json'
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Provide the project assembly world_definition.json')


def parse_call(text: str):
    match = re.fullmatch(r'\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(([^()]*)\)\s*', text)
    if not match:
        raise ValueError(f'Invalid grounded predicate/action call: {text!r}')
    args = [x.strip() for x in match[2].split(',') if x.strip()]
    return match[1], args


def call(name, args):
    return f'{name}({", ".join(args)})'


def parse_schema_literal(text: str):
    negative = text.startswith('(not (') and text.endswith('))')
    if negative:
        text = text[6:-2]
    tokens = text.split()
    return negative, tokens[0], [x.lstrip('?') for x in tokens[1:]]


def state_facts(world_state: dict):
    result = {call(prop, [obj['name']]) for obj in world_state.get('objects', [])
              for prop in obj.get('properties', [])}
    for kind in ['constraints', 'relations']:
        result.update(call(row['name'], [row['source'], row['target']])
                      for row in world_state.get(kind, []))
    return result


def make_registry(request: dict, definition_path=None) -> dict:
    definition = json.loads(Path(definition_path or _definition_path()).read_text())
    state = copy.deepcopy(request['world_state'])
    objects = {obj['name'] for obj in state.get('objects', [])}
    for kind in ['constraints', 'relations']:
        for row in state.get(kind, []):
            objects.update([row['source'], row['target']])
    hands = {name for name in objects if 'hand' in name.lower()}
    tools = {row['source'] for row in state.get('constraints', []) if row['name'] == 'can_manipulate'}
    tools.update(obj['name'] for obj in state.get('objects', []) if 'is_equippable' in obj.get('properties', []))
    tools.update(row['target'] for row in state.get('relations', [])
                 if row['name'] == 'hold' and row['source'] in hands)
    tools -= hands
    parts = objects - hands - tools
    role_objects = {'hand': sorted(hands), 'tool': sorted(tools), 'part': sorted(parts)}
    predicates = {}
    meanings = {
        'is_empty': 'The object currently holds nothing.',
        'is_equippable': 'The tool is currently available to be equipped.',
        'can_manipulate': 'Immutable compatibility: the tool can manipulate the part; not a completed manipulation.',
        'can_screw_to': 'Immutable compatibility: the first part can be screwed to the second; not a completed screw connection.',
        'can_insert_to': 'Immutable compatibility: the first part can be inserted into the second; not a completed insertion.',
        'can_place_to': 'Immutable compatibility: the first part can be placed on the second; not a completed placement.',
        'hold': 'The first object currently holds the second object.',
        'is_inserted_to': 'Achieved state: the first part is inserted into the second part.',
        'is_screwed_to': 'Achieved state: the first part is screwed to the second part.',
        'is_placed_to': 'Achieved state: the first part is placed on the second part.',
    }
    for category, arity in [('properties', 1), ('constraints', 2), ('relations', 2)]:
        for name in definition[category]:
            predicates[name] = {'arity': arity, 'arguments': [f'arg{i}' for i in range(arity)], 'kind': category,
                                'static': category == 'constraints', 'description': meanings.get(name, name)}
    actions = {}
    for name, spec in definition['actions'].items():
        _, arguments = parse_call(spec['template'])
        arguments = [arg.lstrip('?') for arg in arguments]
        roles = {arg: 'hand' if arg == 'hand' else 'tool' if re.fullmatch(r'tool\d*', arg)
                 else 'part' if re.fullmatch(r'part\d*', arg) else None for arg in arguments}
        actions[name] = {**copy.deepcopy(spec),
                         'arguments': arguments, 'argument_roles': roles,
                         'cost': 1}
    return {'domain': 'assembly', 'actions': actions, 'predicates': predicates,
            'objects': sorted(objects), 'world_state': state,
            'role_objects': role_objects,
            'facts': sorted(state_facts(state)),
            'grounding_policy': 'All recognized role-compatible object tuples; empty roles fall back to all objects; prune false immutable constraints only; no goal-conditioned pruning'}


def ground_registry_actions(registry: dict):
    """Return complete STRIPS actions with signed literal strings.

    Each literal is its canonical call; negation is prefixed by ``!``. This also
    accepts an explicit `ground_actions` list in a finite other-domain registry.
    The latter must be prepared from the domain, not from the evaluation target.
    """
    if 'ground_actions' in registry:
        return copy.deepcopy(registry['ground_actions'])
    objects = registry['objects']
    static_facts = set(registry.get('facts', state_facts(registry['world_state'])))
    result = []
    for name, spec in registry['actions'].items():
        arguments = spec['arguments']
        def domain(variable):
            role = spec.get('argument_roles', {}).get(variable)
            return registry.get('role_objects', {}).get(role) or objects
        preconditions = [parse_schema_literal(x) for x in spec['precondition']]
        effects = [parse_schema_literal(x) for x in spec['effect']]
        fixed = [{}]
        for negative, predicate, variables in preconditions:
            if not registry['predicates'][predicate].get('static'):
                continue
            next_fixed = []
            for binding in fixed:
                free = list(dict.fromkeys(v for v in variables if v not in binding))
                for values in itertools.product(*(domain(v) for v in free)):
                    candidate = {**binding, **dict(zip(free, values))}
                    holds = call(predicate, [candidate[v] for v in variables]) in static_facts
                    if holds != negative:
                        next_fixed.append(candidate)
            fixed = next_fixed
        seen = set()
        for binding in fixed:
            free = [v for v in arguments if v not in binding]
            for values in itertools.product(*(domain(v) for v in free)):
                assigned = {**binding, **dict(zip(free, values))}
                args = [assigned[v] for v in arguments]
                signature = call(name, args)
                if signature in seen:
                    continue
                seen.add(signature)
                pre = set()
                add, delete = set(), set()
                for negative, predicate, variables in preconditions:
                    literal = call(predicate, [assigned[v] for v in variables])
                    pre.add(('!' if negative else '') + literal)
                for negative, predicate, variables in effects:
                    literal = call(predicate, [assigned[v] for v in variables])
                    (delete if negative else add).add(literal)
                result.append({'name': signature, 'pre': sorted(pre),
                               'add': sorted(add), 'delete': sorted(delete),
                               'cost': spec.get('cost', 1)})
    return result
