"""Tick native BT control flow without flattening its actions or conditions.

Shared interpreter only: leaf semantics and method-specific planning/update remain
with the environment and method adapter. Supports the BehaviorTree.CPP control
modes used by the BTGenBot conversion, plus reactive KIOS/OBTEA defaults.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from assembly_registry import call, parse_call


class NativeRuntime:
    def __init__(self, tree, check_predicate, execute_action):
        self.tree = copy.deepcopy(tree)
        self.check_predicate = check_predicate
        self.execute_action = execute_action
        self.trace = []
        self.last_failed_node = None
        self.last_failed_path = None
        self.ticks = 0
        self.action_calls = 0
        self.condition_checks = 0
        self.active_action = None
        self._cursor = {}
        self._retries = {}

    def reset(self):
        self._cursor.clear()
        self._retries.clear()
        self.active_action = None
        self.last_failed_node = None
        self.last_failed_path = None
        self.trace.append({'event': 'reset', 'tick': self.ticks})

    def _forget(self, path):
        for state in [self._cursor, self._retries]:
            for key in list(state):
                if key[:len(path)] == path:
                    del state[key]

    def _record(self, node, path, status, event='tick'):
        self.trace.append({'event': event, 'tick': self.ticks,
                           'path': list(path), 'name': node['name'], 'status': status})
        if status == 'failure' and (self.last_failed_node is None or not node.get('children')):
            self.last_failed_node = copy.deepcopy(node)
            self.last_failed_path = list(path)
        return status

    @staticmethod
    def _status(value):
        if isinstance(value, bool):
            return 'success' if value else 'failure'
        if isinstance(value, dict):
            value = value.get('status', value.get('result'))
        status = str(value).lower()
        if status not in {'success', 'failure', 'running'}:
            raise ValueError(f'Leaf returned invalid BT status: {value!r}')
        return status

    def tick(self):
        self.ticks += 1
        self.last_failed_node = None
        self.last_failed_path = None
        previous_active = self.active_action
        self.active_action = None
        status = self._tick(self.tree, ())
        if previous_active is not None and previous_active != self.active_action:
            self.trace.append({'event': 'halt', 'tick': self.ticks,
                               'path': list(previous_active)})
        return status

    def _tick(self, node, path):
        if not isinstance(node, dict) or not isinstance(node.get('name'), str):
            raise ValueError(f'Invalid native BT node at {path}: {node!r}')
        if ':' not in node['name']:
            raise ValueError(f'Native BT node must include kind prefix: {node["name"]!r}')
        kind, payload = node['name'].split(':', 1)
        kind, payload = kind.strip(), payload.strip()
        children = node.get('children', [])
        if kind in {'sequence', 'selector'}:
            default = 'reactive_sequence' if kind == 'sequence' else 'reactive_fallback'
            mode = node.get('control_mode', 'memory' if node.get('memory') else default)
            if mode not in {'reactive', 'reactive_sequence', 'reactive_fallback',
                            'sequence', 'fallback', 'memory', 'sequence_with_memory'}:
                raise ValueError(f'Unsupported control mode {mode!r}')
            remember_running = mode in {'sequence', 'fallback', 'memory', 'sequence_with_memory'}
            start = self._cursor.get(path, 0) if remember_running else 0
            for index in range(start, len(children)):
                status = self._tick(children[index], path + (index,))
                if status == 'running':
                    if remember_running:
                        self._cursor[path] = index
                    else:
                        self._cursor.pop(path, None)
                    # A reactive control can preempt a different running branch.
                    for later in range(index + 1, len(children)):
                        self._forget(path + (later,))
                    return self._record(node, path, status)
                stop = status == ('failure' if kind == 'sequence' else 'success')
                if stop:
                    if mode == 'sequence_with_memory' and status == 'failure':
                        self._cursor[path] = index
                    else:
                        self._forget(path)
                    return self._record(node, path, status)
            self._forget(path)
            status = 'success' if kind == 'sequence' else 'failure'
            return self._record(node, path, status)
        if kind == 'inverter':
            if len(children) != 1:
                raise ValueError('An inverter requires exactly one child')
            status = self._tick(children[0], path + (0,))
            status = {'success': 'failure', 'failure': 'success'}.get(status, status)
            return self._record(node, path, status)
        if kind == 'retry':
            if len(children) != 1:
                raise ValueError('A retry decorator requires exactly one child')
            maximum = int(node.get('num_attempts', node.get('max_attempts', node.get('attempts', 1))))
            if maximum == 0 or maximum < -1:
                raise ValueError(f'Invalid retry attempt count: {maximum}')
            status = self._tick(children[0], path + (0,))
            if status == 'success':
                self._forget(path)
            elif status == 'failure':
                attempts = self._retries.get(path, 0) + 1
                self._forget(path + (0,))
                self._retries[path] = attempts
                if maximum == -1 or attempts < maximum:
                    status = 'running'
                else:
                    self._forget(path)
            return self._record(node, path, status)
        if children:
            raise ValueError(f'Unexpected child nodes on leaf {node["name"]!r}')
        if kind in {'target', 'precondition', 'condition'}:
            predicate, args = parse_call(payload)
            self.condition_checks += 1
            holds = bool(self.check_predicate(predicate, args))
            if node.get('negated', False):
                holds = not holds
            status = 'success' if holds else 'failure'
            return self._record(node, path, status)
        if kind == 'action':
            action, args = parse_call(payload)
            self.action_calls += 1
            status = self._status(self.execute_action(action, args))
            if status == 'running':
                self.active_action = path
            return self._record(node, path, status)
        raise ValueError(f'Unsupported native BT kind {kind!r}')


def _world_class():
    root = Path(__file__).resolve().parents[1] / 'assembly_artifact'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from runtime.minimal_bt import WorldState
    return WorldState


def evaluate_assembly(tree, world_state, target, max_ticks=100, max_actions=100):
    """Execute repeated native ticks, validating the original evaluator goal.

    A successful control tick is not a successful task: a reactive OBTEA tree
    commonly finishes only one primitive per successful root tick.
    """
    WorldState = _world_class()
    world = WorldState(copy.deepcopy(world_state))
    goal_name, goal_args = parse_call(target)
    actions = []
    reason = None

    def execute(name, args):
        nonlocal reason
        if len(actions) >= max_actions:
            reason = 'max_actions'
            return 'failure'
        before = world.to_json()
        ok = world.apply_action(name, args)
        actions.append({'action': call(name, args), 'success': ok,
                        'before': before, 'after': world.to_json()})
        return 'success' if ok else 'failure'

    runtime = NativeRuntime(tree, world.check, execute)
    last_status = None
    error = None
    success = world.check(goal_name, goal_args)
    try:
        for _ in range(max_ticks):
            if success:
                break
            before = world.to_json()
            last_status = runtime.tick()
            success = world.check(goal_name, goal_args)
            if success:
                break
            if last_status == 'failure':
                reason = reason or 'native_bt_failure'
                break
            if last_status == 'success' and before == world.to_json():
                reason = 'native_bt_success_without_evaluator_goal'
                break
        else:
            reason = 'max_ticks'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        reason = 'execution_error'
    return {'result': 'success' if success else 'failure', 'success': success,
            'world_state': world.to_json(), 'action_trace': actions,
            'actions': [a['action'] for a in actions],
            'failed_node': runtime.last_failed_node, 'failed_path': runtime.last_failed_path,
            'ticks': runtime.ticks, 'action_count': len(actions),
            'condition_checks': runtime.condition_checks, 'native_status': last_status,
            'reason': reason, 'error': error, 'trace': runtime.trace}
