"""Complete-method LIBERO rollouts with native BT ticks and paired pi0.5 noise.

No GPU/model server is started here. MuJoCo rendering defaults to CPU OSMesa;
the root-managed frozen policy is contacted through the supplied loopback port.
Every declared episode, including invalid plans and failed executions, is kept.
"""
from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.error
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import numpy as np

from assembly_registry import call, parse_call
from libero_registry import action_prompt, effect_predicate, make_registry, declared_containment_regions
from native_runtime import NativeRuntime
from goal_contract import VERSION, validate_request

DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULTS = {'suites': ['libero_10'], 'tasks': list(range(10)),
            'init_ids': list(range(10)), 'max_steps': 650, 'chunk_steps': 5,
            'stage_steps': 180, 'max_attempts': 3, 'settle_steps': 10,
            'policy_seed': 0, 'camera_transform': 'rotate180',
            'prompt_mode': 'task_prefix', 'max_no_step_ticks': 20,
            'max_runtime_updates': 128, 'online_expected_model_name': 'Qwen2.5-7B-Instruct',
            'online_adapter_checkpoint': None}


class InfrastructureError(RuntimeError):
    pass


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}.{time.time_ns()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()


def load_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    return value.get('requests', value) if isinstance(value, dict) else value


def canonical_method(name):
    key = re.sub(r'[^a-z0-9]', '', name.lower())
    return {'llmasbtplanner': 'kios', 'llmobtea': 'obtea', 'llmbt': 'llmbt',
            'btgenbot': 'btgenbot', 'hibt': 'hibt', 'concept': 'hibt'}.get(key, key)


def predicate_catalog(registry):
    """All registered scene-feasible bindings, not instruction-target atoms."""
    atoms = set(registry.get('facts', []))
    for action in registry['ground_actions']:
        atoms.update(x.lstrip('!') for field in ['pre', 'add'] for x in action[field])
    roles = registry.get('role_objects', {})
    for pred in ['On', 'In']:
        target_role = 'placement_supports' if pred == 'On' else 'containment_targets'
        for obj, target in itertools.product(roles.get('movable', []), roles.get(target_role, [])):
            if obj != target:
                atoms.add(call(pred, [obj, target]))
    for pred, role in [('Open', 'closable'), ('Close', 'closable'),
                       ('Turnon', 'switchable'), ('Turnoff', 'switchable')]:
        atoms.update(call(pred, [obj]) for obj in roles.get(role, []))
    return sorted(atoms)


def _canonical_predicate(name, registry):
    matches = [p for p in registry['predicates'] if p.lower() == name.lower()]
    if len(matches) != 1:
        return None
    return matches[0]


class UnsupportedPredicateBinding(ValueError):
    """A generated condition has no scene-declared physical interpretation."""


def evaluate_actual_predicate(env, registry, name, args):
    """Bind parent In to the union of its real, declared interior sites.

    LIBERO's In short-circuits contact before ObjectState.check_contain, so a
    scanned parent lacking in_box fails only after contact. The declared
    SiteObjectState has the official geometry API. No task goal is consulted.
    """
    domain = getattr(env, 'env', env)
    if name.lower() != 'in' or len(args) != 2 or not hasattr(domain, 'object_states_dict'):
        return bool(domain._eval_predicate([name.lower()] + list(args)))
    target = args[1]
    state = domain.object_states_dict.get(target)
    if state is None:
        if target in registry.get('objects', []):
            raise RuntimeError('Registered containment target has no physical object state: ' + target)
        raise UnsupportedPredicateBinding('No scene object state for containment target ' + target)
    if getattr(state, 'object_state_type', None) == 'site':
        return bool(domain._eval_predicate(['in'] + list(args)))
    target_object = domain.get_object(target)
    if callable(getattr(target_object, 'in_box', None)):
        return bool(domain._eval_predicate(['in'] + list(args)))
    regions = declared_containment_regions(registry, target)
    if not regions:
        raise UnsupportedPredicateBinding('Containment target has no native in_box or declared interior region: ' + target)
    for region in regions:
        site = domain.object_sites_dict.get(region)
        site_state = domain.object_states_dict.get(region)
        if (site is None or site_state is None or getattr(site, 'parent_name', None) != target
                or getattr(site_state, 'object_state_type', None) != 'site'
                or not callable(getattr(site, 'in_box', None))):
            raise RuntimeError('Declared containment region has no matching physical site API: ' + region)
    return any(bool(domain._eval_predicate(['in', args[0], region])) for region in regions)


def actual_snapshot(env, request, registry, catalog):
    """Expose the complete shared predicate interface, never benchmark goals."""
    state = copy.deepcopy(request['world_state'])
    values, facts, init = [], [], []
    for atom in catalog:
        pred, args = parse_call(atom)
        binding_error = None
        try:
            holds = evaluate_actual_predicate(env, registry, pred, args)
        except UnsupportedPredicateBinding as exc:
            holds, binding_error = False, str(exc)
        except Exception as exc:
            raise InfrastructureError(f'Registered predicate binding failed: {atom}: {exc!r}') from exc
        values.append({'predicate': pred, 'args': args, 'value': holds,
                       **({'binding_error': binding_error} if binding_error else {})})
        if holds:
            facts.append(atom)
            init.append({'predicate': pred, 'args': args, 'raw': '(' + ' '.join([pred] + args) + ')'})
    state['init_predicates'] = init
    state['predicate_values'] = values
    state['facts'] = facts
    return state


def symbolic_feedback(tree, registry, max_ticks=128):
    """Method-native simulation using current facts and complete operator rules.

    No evaluator goal is supplied. Target leaves are the method's own inferred
    condition nodes. Feedback cannot reveal a correct task plan or repair one.
    """
    facts = set(registry.get('facts', []))
    actions = {a['name']: a for a in registry['ground_actions']}
    trace = []

    def check(name, args):
        name = _canonical_predicate(name, registry)
        return name is not None and call(name, args) in facts

    def execute(name, args):
        signature = call(name, args)
        schema = actions.get(signature)
        if schema is None:
            return 'failure'
        for pre in schema['pre']:
            if (pre[1:] in facts) if pre.startswith('!') else (pre not in facts):
                return 'failure'
        facts.difference_update(schema['delete'])
        facts.update(schema['add'])
        trace.append(signature)
        return 'success'

    try:
        runtime = NativeRuntime(tree, check, execute)
        # Structural feedback matches the paper-native simulator. Transient
        # support guards (Open) need not remain true after final Close; checking
        # every nested guard simultaneously would reject a valid reactive BT.
        # Benchmark final-goal success is evaluated separately by the environment.
        status = 'running'
        for _ in range(max_ticks):
            status = runtime.tick()
            if status != 'running':
                break
        if status == 'running':
            status = 'failure'
        current = copy.deepcopy(registry['world_state'])
        current['init_predicates'] = []
        for atom in sorted(facts):
            name, args = parse_call(atom)
            current['init_predicates'].append({'predicate': name, 'args': args})
        current['facts'] = sorted(facts)
        return {'result': status, 'status': status, 'world_state': current,
                'current_state': current, 'failed_node': runtime.last_failed_node,
                'failed_path': runtime.last_failed_path, 'action_trace': trace, 'trace': runtime.trace}
    except Exception as exc:
        return {'result': 'failure', 'status': 'failure', 'world_state': copy.deepcopy(registry['world_state']),
                'failed_node': None, 'error': repr(exc)}


class BudgetLLM:
    """One method call budget with transport failures kept outside its score."""
    def __init__(self, base, initial_calls, maximum, expected_model=None,
                 expected_adapter_checkpoint=None, log_path=None):
        self.base, self.used, self.maximum = base, initial_calls, maximum
        self.expected_model = expected_model
        self.expected_adapter_checkpoint = expected_adapter_checkpoint
        self.log_path = log_path
        self.ner_calls = 0
        self.infra_errors, self.events = [], []

    def raise_if_infrastructure(self):
        if self.infra_errors:
            raise InfrastructureError('Online method transport/provenance failure: ' +
                                      json.dumps(self.infra_errors, ensure_ascii=False))

    def _invoke(self, kind, function, args, kwargs):
        self.raise_if_infrastructure()
        started = time.monotonic()
        event = {'kind': kind, 'request': {'args': args, 'kwargs': kwargs}}
        try:
            response = function(*args, **kwargs)
            event['response'] = response
            if kind == 'llm':
                if not isinstance(response, dict):
                    raise InfrastructureError('Online LLM response lacks provenance object')
                if self.expected_model is not None and response.get('model') != self.expected_model:
                    raise InfrastructureError('Online LLM backbone differs from frozen configuration: ' +
                                              repr(response.get('model')))
                if 'adapter_checkpoint' not in response or response['adapter_checkpoint'] != self.expected_adapter_checkpoint:
                    raise InfrastructureError('Online LLM adapter differs from frozen configuration: ' +
                                              repr(response.get('adapter_checkpoint', '<missing>')))
            return response
        except (OSError, urllib.error.URLError, TimeoutError, InfrastructureError) as exc:
            event['error'] = f'{type(exc).__name__}: {exc}'
            self.infra_errors.append({'kind': kind, 'error': event['error']})
            raise
        finally:
            event['wall_s'] = time.monotonic() - started
            self.events.append(event)
            if self.log_path is not None:
                try:
                    append(self.log_path, event)
                except OSError as exc:
                    self.infra_errors.append({'kind': kind, 'error': f'Call-event persistence failed: {exc!r}'})
                    raise

    def complete(self, *args, **kwargs):
        self.raise_if_infrastructure()
        if self.used >= self.maximum:
            raise RuntimeError('Complete-method shared LLM-call budget exhausted')
        self.used += 1
        return self._invoke('llm', self.base.complete, args, kwargs)

    def ner(self, text):
        self.ner_calls += 1
        return self._invoke('ner', self.base.ner, (text,), {})


def count_initial_calls(row):
    if isinstance(row.get('llm_calls'), int):
        return row['llm_calls']
    return sum(item.get('event') == 'llm' or isinstance(item.get('response'), dict)
               for item in row.get('trace', []))


def input_path(run, value, default):
    path = Path(value) if value is not None else Path(default)
    return path if path.is_absolute() else Path(run) / path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(source_root=None):
    source_root = Path(source_root or Path(__file__).resolve().parent)
    hashes = {str(path.relative_to(source_root)): _sha(path)
              for path in sorted(source_root.rglob('*.py'))}
    lineage = source_root / 'cached_registry_lineage.json'
    if lineage.is_file():
        hashes['cached_registry_lineage.json'] = _sha(lineage)
    # Official prompt assets are runtime inputs of KIOS, not just documentation.
    audit = source_root.parent / 'source_audit' / 'kios'
    for name in ['system.txt', 'task.txt', 'behaviortree.txt', 'state.txt', 'template.txt', 'output_format.txt']:
        path = audit / name
        if path.exists():
            hashes['source_audit/kios/' + name] = _sha(path)
    return hashes


def protocol_fingerprint(run, config, request_path, plan_path, phase, port, source_root=None):
    if not Path(request_path).is_file():
        raise InfrastructureError('Frozen request input is missing: ' + str(request_path))
    if phase != 'preflight' and (plan_path is None or not Path(plan_path).is_file()):
        raise InfrastructureError('Frozen planning input is missing: ' + str(plan_path))
    value = {'version': 1, 'goal_input_contract': VERSION, 'phase': phase, 'config_sha256': _sha(Path(run) / 'config.json'),
             'effective_config': copy.deepcopy(config), 'requests_sha256': _sha(request_path),
             'plans_sha256': _sha(plan_path) if plan_path is not None and Path(plan_path).is_file() else None,
             'source_hashes': source_hashes(source_root), 'policy_port': None if phase == 'preflight' else port}
    value['protocol_id'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def freeze_protocol(run, config, request_path, plan_path, phase, port, source_root=None):
    """Freeze before simulator/service startup; phases have independent dev records."""
    run = Path(run)
    current = protocol_fingerprint(run, config, request_path, plan_path, phase, port, source_root)
    path = run / ('protocol_' + phase + '.json')
    with (run / ('protocol_' + phase + '.lock')).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != current:
                raise InfrastructureError('Immutable LIBERO protocol mismatch; create a new run or restore the exact frozen inputs/source')
        else:
            if phase != 'preflight' and any((run / phase).glob('*/result.json')):
                raise InfrastructureError('Existing episodes have no frozen protocol; preserve this run and create a new version')
            dump(path, current)
    return current


def verify_protocol(run, protocol, config, request_path, plan_path, phase, port, source_root=None):
    current = protocol_fingerprint(run, config, request_path, plan_path, phase, port, source_root)
    if current != protocol:
        raise InfrastructureError('Frozen LIBERO source/config/input changed during execution')


def is_infrastructure_result(result):
    return bool(result.get('infrastructure_error')) or result.get('status') == 'INFRASTRUCTURE_ERROR'


def prepare_episode(output, eid, retry_infrastructure=False, protocol_id=None):
    """Keep completed semantic outcomes; archive a whole infra attempt explicitly."""
    episode = Path(output) / eid
    result_path = episode / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if protocol_id is not None and previous.get('protocol_id') != protocol_id:
            raise InfrastructureError('Existing episode protocol differs from frozen run: ' + eid)
        if not is_infrastructure_result(previous):
            return None
        if not retry_infrastructure:
            raise InfrastructureError('Unresolved infrastructure episode ' + eid +
                                      '; repair the service and pass --retry-infrastructure explicitly')
        attempts = Path(output) / '_infrastructure_attempts'
        attempts.mkdir(exist_ok=True)
        archived = attempts / (eid + '.' + str(time.time_ns()))
        episode.rename(archived)
    elif episode.exists() and any(episode.iterdir()):
        raise InfrastructureError('Incomplete episode has no result record; preserve and inspect before retry: ' + eid)
    episode.mkdir(parents=True, exist_ok=True)
    return episode


def check_unresolved_infrastructure(output, retry_infrastructure=False):
    if retry_infrastructure:
        return
    for path in sorted(Path(output).glob('*/result.json')):
        previous = json.loads(path.read_text())
        if is_infrastructure_result(previous):
            raise InfrastructureError('Unresolved infrastructure episode ' + path.parent.name +
                                      '; pass --retry-infrastructure explicitly after repair')


def initial_state_sha256(value):
    """Hash the selected physical state without changing its dtype or values."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise InfrastructureError('Physical initial state must be a numeric NumPy array')
    metadata = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)},
                          sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(metadata + b'\x00' + array.tobytes(order='C')).hexdigest()
    return digest


def quaternion_axis_angle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    return np.zeros(3) if math.isclose(float(denominator), 0.0) else quat[:3] * 2 * math.acos(float(quat[3])) / denominator


def policy_observation(obs, prompt, seed, image_tools, transform='rotate180'):
    def image(frame):
        if transform == 'rotate180':
            frame = frame[::-1, ::-1]
        elif transform == 'flip_y':
            frame = frame[::-1, :]
        elif transform != 'raw':
            raise InfrastructureError(f'Unsupported camera transform: {transform}')
        frame = np.ascontiguousarray(frame)
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, 224, 224))
    return {'observation/image': image(obs['agentview_image']),
            'observation/wrist_image': image(obs['robot0_eye_in_hand_image']),
            'observation/state': np.concatenate((obs['robot0_eef_pos'], quaternion_axis_angle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])),
            'prompt': prompt, '_hibt_noise_seed': seed}


class NativeEpisode:
    def __init__(self, env, request, registry, plan_row, config, policy, observation_builder, output, llm=None):
        self.env, self.request, self.registry = env, copy.deepcopy(request), copy.deepcopy(registry)
        self.plan_row, self.config, self.policy = copy.deepcopy(plan_row), config, policy
        self.observation_builder, self.output = observation_builder, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = self.output / 'events.jsonl'
        self.catalog = predicate_catalog(registry)
        self.allowed_predicates = set(self.catalog)
        self.action_schemas = {a['name']: a for a in registry['ground_actions']}
        self.steps = self.policy_calls = self.retries = self.updates = 0
        self.latencies, self.semantic_errors = [], []
        self.stage_states = {}
        self.active = None
        self.obs = None
        self.cache = {}
        self.runtime = None
        self.llm = llm
        self.reason = None

    def log(self, event, **kwargs):
        append(self.events, {'event': event, 'step': self.steps, **kwargs})

    def semantic(self, reason, **kwargs):
        row = {'reason': reason, **kwargs}
        if row not in self.semantic_errors:
            self.semantic_errors.append(row)
            self.log('semantic_error', **row)

    def check(self, name, args):
        canonical = _canonical_predicate(name, self.registry)
        signature = call(canonical or name, args)
        if canonical is None or signature not in self.allowed_predicates:
            self.semantic('invalid_generated_predicate_binding', predicate=signature)
            return False
        if signature not in self.cache:
            try:
                self.cache[signature] = evaluate_actual_predicate(self.env, self.registry, canonical, args)
            except UnsupportedPredicateBinding as exc:
                self.semantic('unsupported_generated_predicate_binding', predicate=signature, detail=str(exc))
                self.cache[signature] = False
            except Exception as exc:
                raise InfrastructureError(f'Predicate evaluation failed for {signature}: {exc!r}') from exc
        return self.cache[signature]

    def execute(self, name, args):
        signature = call(name, args)
        schema = self.action_schemas.get(signature)
        if schema is None:
            self.semantic('invalid_generated_action_binding', action=signature)
            return 'failure'
        predicate, pred_args = effect_predicate(name, args)
        state = self.stage_states.setdefault(signature, {'spent': 0, 'attempts': 0, 'completed': False,
                                                         'queue': collections.deque()})
        if self.check(predicate, pred_args):
            state['completed'] = True
            state['queue'].clear()
            return 'success'
        if state['completed']:
            state.update(spent=0, attempts=0, completed=False)
        for pre in schema['pre']:
            negative = pre.startswith('!')
            pred, params = parse_call(pre.lstrip('!'))
            if self.check(pred, params) == negative:
                self.log('action_precondition_failure', action=signature, prerequisite=pre)
                return 'failure'
        if self.steps >= self.config['max_steps']:
            self.reason = 'max_steps'
            return 'failure'
        if state['spent'] >= self.config['stage_steps']:
            state['attempts'] += 1
            state['spent'] = 0
            state['queue'].clear()
            self.retries += 1
            self.log('retry', action=signature, attempt=state['attempts'])
        if state['attempts'] >= self.config['max_attempts']:
            self.log('action_attempts_exhausted', action=signature)
            return 'failure'
        if self.active != signature:
            if self.active in self.stage_states:
                self.stage_states[self.active]['queue'].clear()
            self.active = signature
            state['queue'].clear()
            self.log('select_native_action', action=signature)
        prompt = action_prompt(name, args, self.registry)
        if self.config['prompt_mode'] == 'task_prefix':
            prompt += '; this is part of the task: ' + self.request['instruction']
        if not state['queue']:
            seed = int(np.random.SeedSequence([self.config['policy_seed'], self.request['task_id'],
                                               self.config['init_state_id'], self.policy_calls]).generate_state(1)[0])
            query = self.observation_builder(self.obs, prompt, seed)
            start = time.monotonic()
            try:
                response = self.policy.infer(query)
            except Exception as exc:
                raise InfrastructureError(f'Policy inference failed: {exc!r}') from exc
            elapsed = (time.monotonic() - start) * 1000
            self.latencies.append(elapsed)
            self.policy_calls += 1
            actions = np.asarray(response['actions'])
            if actions.ndim != 2 or actions.shape[0] < self.config['chunk_steps'] or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise InfrastructureError(f'Invalid policy action array: {actions.shape}')
            state['queue'].extend(actions[:self.config['chunk_steps']])
            self.log('query', action=signature, prompt=prompt, noise_seed=seed,
                     latency_ms=elapsed, call_index=self.policy_calls - 1)
        action = np.asarray(state['queue'].popleft()).tolist()
        self.obs, reward, done, info = self.env.step(action)
        self.steps += 1
        state['spent'] += 1
        self.cache.clear()
        self.log('step', action=signature, continuous_action=action, reward=float(reward),
                 official_success=bool(self.env.check_success()))
        # Return RUNNING even if a just-executed control step achieved its effect;
        # the next native tick observes that effect before moving to another leaf.
        return 'running'

    def try_native_update(self, tree):
        method = canonical_method(self.plan_row['method'])
        if method not in {'kios', 'llmbt'}:
            return None
        if self.updates >= self.config['max_runtime_updates']:
            self.reason = 'native_update_budget_exhausted'
            return None
        if method == 'kios' and self.llm is None:
            self.semantic('native_update_llm_not_configured')
            return None
        if self.llm is not None and hasattr(self.llm, 'raise_if_infrastructure'):
            self.llm.raise_if_infrastructure()
        self.updates += 1
        snapshot = actual_snapshot(self.env, self.request, self.registry, self.catalog)
        request = copy.deepcopy(self.request)
        request.pop('target', None)
        request['world_state'] = snapshot
        registry = copy.deepcopy(self.registry)
        registry['world_state'] = snapshot
        registry['facts'] = snapshot['facts']
        failed = copy.deepcopy(self.runtime.last_failed_node)
        failed_path = copy.deepcopy(self.runtime.last_failed_path)
        self.log('native_update_begin', method=method, update=self.updates, failed_node=failed,
                 failed_path=failed_path, current_state=snapshot)
        module = importlib.import_module('methods.' + method)
        budget = copy.deepcopy(self.config.get('method_budget', {}))
        if self.llm is not None:
            budget['max_calls'] = max(0, self.llm.maximum - self.llm.used)
        feedback = lambda candidate: symbolic_feedback(candidate, registry)
        try:
            if method == 'llmbt':
                failed['path'] = copy.deepcopy(failed_path)
            update_kwargs = {'feedback': feedback, 'tree': tree, 'failed_node': failed, 'runtime_state': snapshot}
            if method == 'llmbt':
                update_kwargs['failure_trace'] = copy.deepcopy(self.runtime.trace)
            updated = module.update(request, registry, self.llm, budget, **update_kwargs)
        except Exception as exc:
            if self.llm is not None and hasattr(self.llm, 'raise_if_infrastructure'):
                self.llm.raise_if_infrastructure()
            if isinstance(exc, (OSError, InfrastructureError)):
                raise InfrastructureError(f'Native update infrastructure failure: {exc!r}') from exc
            self.semantic('native_update_failed', detail=repr(exc))
            return None
        dump(self.output / f'native_update_{self.updates:03d}.json', updated)
        if self.llm is not None and hasattr(self.llm, 'raise_if_infrastructure'):
            self.llm.raise_if_infrastructure()
        self.log('native_update_end', method=method, update=self.updates,
                 errors=updated.get('errors', []))
        replacement = updated.get('tree')
        if replacement is None or replacement == tree:
            self.semantic('native_update_no_progress', method=method)
            return None
        return replacement

    def run(self, observation):
        self.obs = observation
        tree = self.plan_row.get('tree')
        dump(self.output / 'initial_plan.json', self.plan_row)
        if tree is None:
            self.reason = 'planning_failure'
            return False
        self.runtime = NativeRuntime(tree, self.check, self.execute)
        no_step_ticks = 0
        while self.steps < self.config['max_steps']:
            if self.env.check_success():
                return True
            before = self.steps
            trace_start = len(self.runtime.trace)
            try:
                status = self.runtime.tick()
            except ValueError as exc:
                self.semantic('invalid_native_tree', detail=str(exc))
                self.reason = 'invalid_native_tree'
                return False
            for event in self.runtime.trace[trace_start:]:
                native = dict(event)
                native['native_event'] = native.pop('event')
                self.log('bt_tick', **native)
            if self.env.check_success():
                return True
            no_step_ticks = no_step_ticks + 1 if self.steps == before else 0
            if status == 'failure':
                updated = self.try_native_update(self.runtime.tree)
                if updated is None:
                    self.reason = self.reason or 'native_bt_failure'
                    return False
                self.runtime = NativeRuntime(updated, self.check, self.execute)
                self.cache.clear()
            if no_step_ticks >= self.config['max_no_step_ticks']:
                self.reason = 'native_tree_stalled_without_official_goal'
                return False
        self.reason = 'max_steps'
        return bool(self.env.check_success())


def make_env(suite, task_id, env_type, bddl_root):
    task = suite.get_task(task_id)
    path = Path(bddl_root) / task.problem_folder / task.bddl_file
    env = env_type(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    return env


def preflight(run, config, requests, suites, env_type, bddl_root, worker_index=0, worker_count=1):
    output = []
    # Software environment/binding coverage uses all supplied requests, normally
    # all40 tasks. It does not choose formal cases based on success outcomes.
    for request_index, request in enumerate(requests):
        if request_index % worker_count != worker_index:
            continue
        name, task = request['suite'], request['task_id']
        env = make_env(suites[name], task, env_type, bddl_root)
        begin = time.monotonic()
        row = {'suite': name, 'task': task, 'passed': False}
        try:
            obs = env.set_init_state(suites[name].get_task_init_states(task)[0])
            for _ in range(2):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            registry = request.get('registry') or make_registry(request, bddl_root)
            catalog = predicate_catalog(registry)
            snapshot = actual_snapshot(env, request, registry, catalog)
            row.update(passed=True, n_registered_actions=len(registry['ground_actions']),
                       n_predicate_bindings=len(catalog),
                       n_true=sum(v['value'] for v in snapshot['predicate_values']),
                       image_shape=list(obs['agentview_image'].shape),
                       camera_transform=config['camera_transform'])
        except Exception as exc:
            row['error'] = repr(exc)
            row['available_object_state_keys'] = sorted(getattr(env.env, 'object_states_dict', {}))
        finally:
            env.close()
        row['wall_s'] = time.monotonic() - begin
        output.append(row)
        print(json.dumps(row), flush=True)
    result = {'scope': 'CPU software/environment preflight only; no policy/model inference',
              'passed': all(x['passed'] for x in output), 'n_tasks': len(output),
              'worker_index': worker_index, 'worker_count': worker_count,
              'gpu_used': False, 'cases': output}
    dump(run / ('preflight.json' if worker_count == 1 else f'preflight_worker{worker_index}.json'), result)
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--retry-infrastructure', action='store_true')
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError('Invalid worker index/count')
    run = args.run_dir.resolve()
    config = {**DEFAULTS, **json.loads((run / 'config.json').read_text())}
    if args.port is None: args.port = int(config['policy_port'])
    if not args.preflight and config.get('protocol_frozen') is not True:
        raise InfrastructureError('Root must freeze this independent goal-specified rollout configuration before policy calls')
    expected_online_model = config.get('expected_model_name', config['online_expected_model_name'])
    if config.get('llm_url') and not isinstance(expected_online_model, str):
        raise InfrastructureError('Frozen online expected backbone must be a nonempty model name')
    if config.get('llm_url') and not expected_online_model:
        raise InfrastructureError('Frozen online expected backbone cannot be empty')
    request_path = input_path(run, config.get('requests_file'), 'requests.json')
    plan_path = input_path(run, config.get('plans_file'), 'planning_outputs.jsonl')
    phase = 'preflight' if args.preflight else ('pilot' if args.pilot else 'formal')
    protocol = freeze_protocol(run, config, request_path, plan_path, phase, args.port)
    output = run / phase
    if not args.preflight:
        output.mkdir(exist_ok=True)
        check_unresolved_infrastructure(output, args.retry_infrastructure)
    requests = load_rows(request_path)
    for request in requests:
        request.setdefault('instruction', request.get('language', ''))
        validate_request(request, request.get('registry'))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites = {name: benchmark.get_benchmark_dict()[name]() for name in {r['suite'] for r in requests}}
    bddl_root = config.get('bddl_root', get_libero_path('bddl_files'))
    if args.preflight:
        preflight(run, config, requests, suites, OffScreenRenderEnv, bddl_root,
                  args.worker_index, args.worker_count)
        return
    if config.get('openpi_client_source'):
        sys.path.insert(0, str(config['openpi_client_source']))
    from openpi_client import image_tools, websocket_client_policy
    policy = websocket_client_policy.WebsocketClientPolicy(config['policy_host'], args.port)
    dump(run / f'server_metadata_worker{args.worker_index}.json', policy.get_server_metadata())
    plan_rows = load_rows(plan_path)
    plans = {(row.get('suite', 'libero_10'), row['task_id'], row['method']): row for row in plan_rows}
    methods = config['methods']
    initials = [0] if args.pilot else config['init_ids']
    case_index = 0
    for initial in initials:
        for request in requests:
            suite_name, task = request['suite'], request['task_id']
            if suite_name not in config['suites'] or task not in config['tasks']:
                continue
            ordering = methods if (task + initial) % 2 == 0 else list(reversed(methods))
            for method in ordering:
                index = case_index
                case_index += 1
                if index % args.worker_count != args.worker_index:
                    continue
                eid = f'{suite_name}_t{task:02d}_i{initial:02d}_{canonical_method(method)}'
                verify_protocol(run, protocol, config, request_path, plan_path, phase, args.port)
                episode = prepare_episode(output, eid, args.retry_infrastructure, protocol['protocol_id'])
                if episode is None:
                    continue
                begin = time.monotonic()
                result = {'episode': eid, 'suite': suite_name, 'task': task,
                          'init': initial, 'method': method, 'phase': phase,
                          'success': False, 'semantic_errors': [], 'infrastructure_error': None,
                          'protocol_id': protocol['protocol_id'],
                          'planning_goal': validate_request(request, request.get('registry')),
                          'goal_input_contract': VERSION}
                env = None
                controller = None
                try:
                    row = plans[(suite_name, task, method)]
                    registry = copy.deepcopy(request.get('registry') or make_registry(request, bddl_root))
                    registry.update(copy.deepcopy(row.get('method_registry_additions', {})))
                    per_episode = {**config, 'init_state_id': initial}
                    llm = None
                    if config.get('llm_url'):
                        from llm_transport import LLMClient
                        base = LLMClient(config['llm_url'], log_path=episode / 'online_llm_calls.jsonl',
                                         ner_url=config.get('ner_url'),
                                         ner_checkpoint=config.get('ner_checkpoint'),
                                         adapter_checkpoint=config['online_adapter_checkpoint'])
                        llm = BudgetLLM(base, count_initial_calls(row),
                                        int(config.get('method_budget', {}).get('max_calls', 12)),
                                        expected_model=expected_online_model,
                                        expected_adapter_checkpoint=config['online_adapter_checkpoint'],
                                        log_path=episode / 'online_transport_events.jsonl')
                    env = make_env(suites[suite_name], task, OffScreenRenderEnv, bddl_root)
                    initial_state = suites[suite_name].get_task_init_states(task)[initial]
                    result['initial_state_sha256'] = initial_state_sha256(initial_state)
                    result['initial_state_dtype'] = np.asarray(initial_state).dtype.str
                    result['initial_state_shape'] = list(np.asarray(initial_state).shape)
                    obs = env.set_init_state(initial_state)
                    for _ in range(config['settle_steps']):
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                    builder = lambda observation, prompt, seed: policy_observation(observation, prompt, seed, image_tools, config['camera_transform'])
                    controller = NativeEpisode(env, request, registry, row, per_episode, policy, builder, episode, llm)
                    result['success'] = controller.run(obs)
                except InfrastructureError as exc:
                    result['infrastructure_error'] = repr(exc)
                except Exception as exc:
                    result['infrastructure_error'] = f'Unexpected software/configuration error: {exc!r}'
                finally:
                    if env is not None:
                        env.close()
                if controller is not None:
                    result.update(steps=controller.steps, policy_calls=controller.policy_calls,
                                  retries=controller.retries, native_updates=controller.updates,
                                  semantic_errors=controller.semantic_errors, termination=controller.reason,
                                  median_policy_ms=float(np.median(controller.latencies)) if controller.latencies else None,
                                  online_llm_calls=(controller.llm.used - count_initial_calls(row)) if controller.llm else 0,
                                  online_infrastructure_errors=controller.llm.infra_errors if controller.llm else [])
                result['wall_s'] = time.monotonic() - begin
                result['status'] = 'INFRASTRUCTURE_ERROR' if result['infrastructure_error'] else 'COMPLETED'
                if result['infrastructure_error']:
                    result['success'] = None
                dump(episode / 'result.json', result)
                append(output / f'episodes_worker{args.worker_index}.jsonl', result)
                print(json.dumps(result), flush=True)
                if result['infrastructure_error']:
                    raise InfrastructureError(result['infrastructure_error'])
    records = [json.loads(p.read_text()) for p in output.glob('*/result.json')]
    summary = {method: {'n': sum(x['method'] == method for x in records),
                        'success': sum(x['method'] == method and bool(x['success']) for x in records),
                        'n_complete': sum(x['method'] == method and not x['infrastructure_error'] for x in records),
                        'infrastructure_errors': sum(x['method'] == method and bool(x['infrastructure_error']) for x in records)}
               for method in methods}
    dump(output / f'summary_worker{args.worker_index}.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
