"""Goal-specified task interface shared by planning and BTGenBot supervision.

Pure serializers only: no evaluator reads, action lookup, plan construction,
model calls, state modification, or physical-control instruction replacement.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re

VERSION = 'goal_specified_v1'
GOAL_HEADER = 'Explicit final task goal (task input, not an action plan):'
ATOM = re.compile(r'^([A-Za-z_][A-Za-z_0-9]*)\s*\(([^()]*)\)$')


def canonical_planning_goal(value, registry=None):
    """Positive grounded atom/conjunction; sort atoms, preserve goal semantics.

    Predicate case is canonicalized only against the complete supplied registry.
    Object identifiers are exact. No omitted argument or predicate is inferred.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError('planning_goal must be a nonempty final-goal string')
    if any(char in value for char in ['!', '|', ';', '\x00']):
        raise ValueError('This task contract supports positive atoms joined by & only')
    atoms = []
    for raw in value.split('&'):
        match = ATOM.fullmatch(raw.strip())
        if match is None:
            raise ValueError('Expected a grounded final-goal atom: ' + repr(raw))
        name, arguments = match.groups()
        args = [arg.strip() for arg in arguments.split(',')] if arguments.strip() else []
        if any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9-]*', arg) for arg in args):
            raise ValueError('Goal arguments must be explicit object identifiers')
        if registry is not None:
            predicates = registry.get('predicates', {})
            matches = [predicate for predicate in predicates if predicate.lower() == name.lower()]
            if len(matches) != 1:
                raise ValueError('Goal predicate is absent/ambiguous in the complete task domain: ' + name)
            name = matches[0]
            spec = predicates[name]
            arity = spec.get('arity', len(spec.get('arguments', [])))
            if len(args) != arity:
                raise ValueError('Wrong number of explicit goal arguments for ' + name)
            unknown = set(args) - set(registry.get('objects', []))
            if unknown:
                raise ValueError('Goal uses unregistered object identifiers: ' + repr(sorted(unknown)))
        atoms.append(name + '(' + ', '.join(args) + ')')
    return ' & '.join(sorted(set(atoms)))


def serialize_planning_instruction(instruction, planning_goal, registry=None):
    """Exact common planning-only language+goal serialization."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Original task instruction must be nonempty text')
    goal = canonical_planning_goal(planning_goal, registry)
    return instruction + '\n\n' + GOAL_HEADER + '\n' + goal


def validate_request(request, registry=None):
    """Validate the task's goal field; never obtain it from evaluator/labels."""
    if 'planning_goal' not in request:
        raise ValueError('Goal-specified track requires its public planning_goal task input')
    return canonical_planning_goal(request['planning_goal'], registry or request.get('registry'))


def planning_request(request, registry=None):
    """Create a transient planning view; the caller's instruction stays intact.

    Data and rollout requests keep their original instruction/language. The
    returned copy is for a planning prompt only and must not replace the request
    used by the physical rollout controller or its task_prefix.
    """
    goal = validate_request(request, registry)
    original = request.get('_goal_original_instruction', request.get('instruction'))
    text = serialize_planning_instruction(original, goal, registry)
    if request.get('_goal_planning_view') and request.get('instruction') != text:
        raise ValueError('Already serialized planning view has been modified')
    result = copy.deepcopy(request)
    result.update(instruction=text, planning_instruction=text, planning_goal=goal,
                  _goal_original_instruction=original, _goal_planning_view=True,
                  goal_input_contract=VERSION,
                  planning_goal_sha256=hashlib.sha256(goal.encode()).hexdigest())
    return result


def goal_input_digest(request):
    """Training/evaluation identity includes the public task goal and raw NL."""
    goal = validate_request(request)
    original = request.get('_goal_original_instruction', request['instruction'])
    value = {'instruction': original, 'world_state': request.get('world_state', {}),
             'planning_goal': goal, 'goal_input_contract': VERSION}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def append_final_goal(messages, request, registry=None):
    """Add the identical explicit goal block to the final planning user turn.

    BTGenBot training and all four external inference pipelines use this exact
    function. It never changes the caller's messages or physical task request.
    """
    goal = validate_request(request, registry)
    result = copy.deepcopy(messages)
    positions = [index for index, item in enumerate(result) if item.get('role') == 'user']
    if not positions:
        raise ValueError('Planning prompt requires a user turn for its public task goal')
    item = result[positions[-1]]
    if not isinstance(item.get('content'), str):
        raise ValueError('Planning prompt user content must be text')
    suffix = GOAL_HEADER + '\n' + goal
    if not item['content'].endswith(suffix):
        item['content'] += '\n\n' + suffix
    return result


class GoalVisibleClient:
    """Planning-only delegate; genuine models, NER and budgets are untouched."""
    def __init__(self, base, request, registry=None):
        self.base = base
        self.request = copy.deepcopy(request)
        self.registry = registry
        self.planning_goal = validate_request(request, registry)
        self.goal_prompt_events = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def complete(self, messages, *args, **kwargs):
        actual = append_final_goal(messages, self.request, self.registry)
        self.goal_prompt_events.append({'planning_goal': self.planning_goal,
            'actual_messages': copy.deepcopy(actual), 'input_contract': VERSION})
        return self.base.complete(actual, *args, **kwargs)


def goal_visible_client(base, request, registry=None):
    goal = validate_request(request, registry)
    if isinstance(base, GoalVisibleClient):
        if base.planning_goal != goal:
            raise ValueError('A planning client cannot silently change its public task goal')
        return base
    return GoalVisibleClient(base, request, registry)


def goal_user_turn(content, request, registry=None):
    """One common goal-conditioned user turn, also for train-only examples."""
    return append_final_goal([{'role': 'user', 'content': content}], request, registry)[0]
