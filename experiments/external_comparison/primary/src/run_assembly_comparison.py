"""Complete-method assembly comparison with immutable protocol and full traces.

CPU driver only: model services are started/owned by the root coordinator.
"""
from __future__ import annotations
import argparse
import collections
import copy
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error

from assembly_registry import make_registry, parse_call, call, state_facts
from native_runtime import NativeRuntime, _world_class
from goal_contract import VERSION, validate_request
from assembly_state_canonicalization import STATE_CONTRACT, validate_prepared_inputs

METHODS = ['hibt', 'llmbt', 'btgenbot', 'kios', 'obtea']


class ServiceProvenanceError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as out:
        out.write(json.dumps(value, ensure_ascii=False) + '\n');out.flush()


def load(path):
    return json.loads(Path(path).read_text())


def canonical(name):
    key = ''.join(x for x in name.lower() if x.isalnum())
    mapped = {'hibt': 'hibt', 'llmbt': 'llmbt', 'btgenbot': 'btgenbot', 'kios': 'kios',
              'llmasbtplanner': 'kios', 'obtea': 'obtea', 'llmobtea': 'obtea'}.get(key)
    if mapped is None:
        raise ValueError(f'Unknown complete method: {name}')
    return mapped


def source_hashes():
    root = Path(__file__).resolve().parent
    hashes = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py'))}
    lineage = root / 'cached_registry_lineage.json'
    if lineage.is_file():
        hashes['cached_registry_lineage.json'] = sha(lineage)
    external = root.parent / 'source_audit'
    for method in ['obtea', 'kios', 'llmbt', 'btgenbot']:
        official = external / method / 'official'
        if official.exists():
            for p in sorted(official.rglob('*.py')):
                hashes[str(p.relative_to(root.parent))] = sha(p)
    for filename in ['system.txt', 'task.txt', 'behaviortree.txt', 'output_format.txt']:
        path = external / 'kios' / filename
        if not path.is_file():
            raise ValueError('Formal KIOS requires its exact official prompt asset: '+str(path))
        hashes[str(path.relative_to(root.parent))] = sha(path)
    WorldState = _world_class()
    runtime_file = Path(inspect.getfile(WorldState))
    hashes['original_assembly_runtime/minimal_bt.py'] = sha(runtime_file)
    hashes['original_assembly_runtime/world_definition.json'] = sha(runtime_file.parents[1] / 'data/world_definition.json')
    return hashes


def protocol(run, config, request_path, evaluator_path):
    run = Path(run)
    current = {'version': 1, 'goal_input_contract': VERSION, 'config_sha256': sha(run / 'config.json'),
               'requests_sha256': sha(request_path), 'evaluators_sha256': sha(evaluator_path),
               'source_hashes': source_hashes(), 'methods': config['methods'],
               'model_tags': config['model_tags'], 'scope': config.get('phase', 'unspecified')}
    if config.get('raw_requests_file'):
        current['assembly_state_contract'] = STATE_CONTRACT
        for config_key, protocol_key in [('raw_requests_file', 'raw_requests_sha256'),
                                          ('state_canonicalization_manifest', 'state_canonicalization_manifest_sha256')]:
            path = Path(config[config_key])
            current[protocol_key] = sha(path if path.is_absolute() else run / path)
    if config.get('demonstrations_file'):
        path = Path(config['demonstrations_file'])
        if not path.is_absolute():
            path = run / path
        current['demonstrations_sha256'] = sha(path)
    if config.get('training_pool_manifest'):
        path = Path(config['training_pool_manifest'])
        if not path.is_absolute():
            path = run / path
        current['training_pool_manifest_sha256'] = sha(path)
    if config.get('expected_source_hashes') and config['expected_source_hashes'] != current['source_hashes']:
        raise ValueError('Current source differs from root-frozen expected_source_hashes')
    current['protocol_id'] = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    with (run / 'protocol.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = run / 'protocol.json'
        if path.exists():
            if load(path) != current:
                raise ValueError('Immutable protocol/config/input/source mismatch; create a new run')
        else:
            dump(path, current)
    return current


class BudgetClient:
    """Count all planning stages under one per-case LLM allowance."""
    def __init__(self, base, maximum, log_path=None, expected_model=None):
        self.base, self.maximum, self.log_path = base, int(maximum), log_path
        self.expected_model = expected_model
        self.events, self.infra_errors = [], []
        self.llm_calls = self.ner_calls = self.hibt_calls = 0
        self.attempt_id = f'{time.time_ns()}_{os.getpid()}'

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _invoke(self, kind, function, args, kwargs):
        started = time.monotonic()
        event = {'kind': kind, 'attempt_id': self.attempt_id, 'request': {'args': args, 'kwargs': kwargs}}
        try:
            response = function(*args, **kwargs)
            event['response'] = response
            if kind == 'llm' and self.expected_model and response.get('model') != self.expected_model:
                raise ServiceProvenanceError('Actual generation backbone differs from frozen expected_model_name')
            if kind == 'llm' and 'adapter_checkpoint' in response:
                expected_adapter = getattr(self.base, 'metadata', {}).get('adapter_checkpoint')
                if response['adapter_checkpoint'] != expected_adapter:
                    raise ServiceProvenanceError('Actual generation adapter differs from requested frozen checkpoint')
            if kind == 'hibt' and response.get('provenance', {}).get('model_tag'):
                if response['provenance']['model_tag'] != getattr(self.base, 'model_tag', None):
                    raise ServiceProvenanceError('Actual HiBT backbone tag differs from frozen model tag')
            return response
        except Exception as exc:
            event['error'] = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ServiceProvenanceError)):
                self.infra_errors.append(event['error'])
            raise
        finally:
            event['wall_s'] = time.monotonic() - started
            self.events.append(event)
            if self.log_path:
                append(self.log_path, event)

    def complete(self, *args, **kwargs):
        if self.llm_calls >= self.maximum:
            raise RuntimeError('Complete-method global LLM-call budget exhausted')
        self.llm_calls += 1
        return self._invoke('llm', self.base.complete, args, kwargs)

    def ner(self, *args, **kwargs):
        self.ner_calls += 1
        return self._invoke('ner', self.base.ner, args, kwargs)

    def hibt(self, *args, **kwargs):
        self.hibt_calls += 1
        return self._invoke('hibt', self.base.hibt, args, kwargs)

    def metrics(self):
        completions = [x.get('response', {}) for x in self.events if x['kind'] == 'llm']
        return {'llm_calls': self.llm_calls, 'ner_calls': self.ner_calls, 'hibt_calls': self.hibt_calls,
                'prompt_tokens': sum(x.get('prompt_tokens', 0) for x in completions),
                'completion_tokens': sum(x.get('completion_tokens', 0) for x in completions),
                'service_wall_s': sum(x['wall_s'] for x in self.events),
                'reported_model_wall_s': sum(x.get('wall_s', 0.0) for x in completions),
                'infrastructure_errors': self.infra_errors}


def simulation_feedback(tree, state, runtime_config):
    """Pure simulation from a snapshot, without evaluator goal information."""
    WorldState = _world_class()
    world = WorldState(copy.deepcopy(state))
    actions = []
    maximum = int(runtime_config.get('max_feedback_actions', 128))
    def execute(name, args):
        if len(actions) >= maximum:
            return 'failure'
        before = world.to_json()
        okay = world.apply_action(name, args)
        actions.append({'action': call(name, args), 'success': okay,
                        'before': before, 'after': world.to_json()})
        return 'success' if okay else 'failure'
    runtime = NativeRuntime(tree, world.check, execute)
    try:
        status = 'running'
        for _ in range(int(runtime_config.get('max_feedback_ticks', 128))):
            status = runtime.tick()
            if status != 'running':
                break
        if status == 'running':
            status = 'failure'
        return {'result': status, 'status': status, 'world_state': world.to_json(),
                'current_state': world.to_json(), 'failed_node': runtime.last_failed_node,
                'failed_path': runtime.last_failed_path, 'action_trace': actions, 'trace': runtime.trace}
    except Exception as exc:
        return {'result': 'failure', 'status': 'failure', 'world_state': world.to_json(),
                'current_state': world.to_json(), 'failed_node': runtime.last_failed_node,
                'failed_path': runtime.last_failed_path, 'action_trace': actions,
                'error': f'{type(exc).__name__}: {exc}'}


def validate_native_tree(tree, registry):
    """Static registration/structure check, without changing the generated BT."""
    def visit(node):
        if not isinstance(node, dict) or not isinstance(node.get('name'), str) or ':' not in node['name']:
            raise ValueError('Invalid native node structure')
        kind, payload = (x.strip() for x in node['name'].split(':', 1))
        children = node.get('children', [])
        if not isinstance(children, list):
            raise ValueError('Native children must be an ordered list')
        if kind in {'sequence', 'selector'}:
            if node.get('control_mode', 'reactive') not in {'reactive', 'reactive_sequence', 'reactive_fallback', 'sequence', 'fallback', 'memory', 'sequence_with_memory'}:
                raise ValueError('Unknown control semantics')
        elif kind in {'inverter', 'retry'}:
            if len(children) != 1:
                raise ValueError('Decorator requires exactly one child')
        elif kind in {'target', 'precondition', 'condition', 'action'}:
            if children:
                raise ValueError('Leaf cannot contain children')
            name, args = parse_call(payload)
            entries = registry['actions'] if kind == 'action' else registry['predicates']
            if name not in entries:
                raise ValueError(f'Unregistered leaf: {name}')
            spec = entries[name]
            arity = len(spec['arguments']) if 'arguments' in spec else spec['arity']
            if len(args) != arity or set(args) - set(registry['objects']):
                raise ValueError('Unregistered leaf arguments')
        else:
            raise ValueError(f'Unsupported native node kind: {kind}')
        for child in children:
            visit(child)
    visit(tree)


def run_case(request, evaluator, method, module, client, budget, runtime_config):
    """Run actual generation and native execution; no generic repair fallback."""
    forbidden = {'target', 'action_sequence_gold', 'problem_id', 'source_path',
                 'teacher_actions', 'output', 'behavior_tree', 'gold_tree', 'reference_plan', 'pyramid'}
    if forbidden & set(request):
        raise ValueError('Evaluator/gold fields found in model request')
    public_goal = validate_request(request, request['registry'])
    world = _world_class()(copy.deepcopy(request['world_state']))
    goal_name, goal_args = parse_call(evaluator['target'])
    registry = copy.deepcopy(request['registry'])
    row = {'sample_id': request['sample_id'], 'method': method, 'success': False,
           'planning_goal': public_goal, 'goal_input_contract': VERSION,
           'initial_plan': None, 'runtime_updates': [], 'execution_trace': [], 'action_trace': [],
           'condition_checks': 0, 'ticks': 0, 'planning_errors': [], 'execution_error': None,
           'termination': None, 'scope': 'complete instruction-to-nativeBT execution'}
    start = time.monotonic()
    snapshot = copy.deepcopy(request['world_state'])
    feedback = lambda tree: simulation_feedback(tree, snapshot, runtime_config)
    try:
        planned = module.plan(copy.deepcopy(request), registry, client, copy.deepcopy(budget), feedback=feedback)
        row['initial_plan'] = planned
        row['planning_errors'].extend(planned.get('errors', []))
        tree = planned.get('tree')
        has_tree = isinstance(tree, dict) and isinstance(tree.get('name'), str)
        row['valid_generated_tree'] = False
        if has_tree:
            try:
                validate_native_tree(tree, registry)
                row['valid_generated_tree'] = True
            except Exception as exc:
                row['static_validation_error'] = f'{type(exc).__name__}: {exc}'
        if not has_tree:
            row['termination'] = 'planning_failure'
        else:
            max_actions = int(runtime_config.get('max_actions', 128))
            def execute(name, args):
                if len(row['action_trace']) >= max_actions:
                    return 'failure'
                before = world.to_json()
                okay = world.apply_action(name, args)
                row['action_trace'].append({'action': call(name, args), 'success': okay,
                                            'before': before, 'after': world.to_json()})
                return 'success' if okay else 'failure'
            runtime = NativeRuntime(tree, world.check, execute)
            for tick in range(int(runtime_config.get('max_ticks', 128))):
                if world.check(goal_name, goal_args):
                    row['success'] = True;row['termination'] = 'evaluator_goal_reached';break
                before = world.to_json()
                status = runtime.tick()
                row['ticks'] += 1
                if world.check(goal_name, goal_args):
                    row['success'] = True;row['termination'] = 'evaluator_goal_reached';break
                if len(row['action_trace']) >= max_actions:
                    row['termination'] = 'max_actions';break
                if status == 'failure':
                    updater = getattr(module, 'update', getattr(module, 'replan', None)) if method in {'kios', 'llmbt'} else None
                    if updater is None or runtime.last_failed_node is None:
                        row['termination'] = 'native_bt_failure';break
                    if len(row['runtime_updates']) >= int(runtime_config.get('max_runtime_updates', 128)):
                        row['termination'] = 'max_runtime_updates';break
                    current_request = copy.deepcopy(request)
                    current_request['world_state'] = world.to_json()
                    current_registry = make_registry(current_request)
                    for demo_key in ['kios_demonstrations','llmbt_demonstrations']:
                        if registry.get(demo_key):current_registry[demo_key] = copy.deepcopy(registry[demo_key])
                    current_snapshot = copy.deepcopy(current_request['world_state'])
                    actual_feedback = lambda candidate: simulation_feedback(candidate, current_snapshot, runtime_config)
                    failed = copy.deepcopy(runtime.last_failed_node)
                    if method == 'llmbt':
                        failed['path'] = copy.deepcopy(runtime.last_failed_path)
                    update_kwargs = {'feedback': actual_feedback, 'tree': copy.deepcopy(runtime.tree),
                                     'failed_node': failed, 'runtime_state': world.to_json()}
                    if method == 'llmbt':
                        update_kwargs['failure_trace'] = copy.deepcopy(runtime.trace)
                    updated = updater(current_request, current_registry, client, copy.deepcopy(budget), **update_kwargs)
                    row['runtime_updates'].append({'tick': tick + 1, 'actual_failed_node': runtime.last_failed_node,
                                                    'actual_failed_path': runtime.last_failed_path, 'update': updated})
                    row['planning_errors'].extend(updated.get('errors', []))
                    row['execution_trace'].extend(runtime.trace)
                    row['condition_checks'] += runtime.condition_checks
                    runtime.trace = []
                    runtime.condition_checks = 0
                    replacement = updated.get('tree')
                    if not isinstance(replacement, dict):
                        row['termination'] = 'native_update_failure';break
                    if replacement == runtime.tree and before == world.to_json():
                        row['termination'] = 'native_update_no_progress';break
                    runtime = NativeRuntime(replacement, world.check, execute)
                elif status == 'success' and before == world.to_json():
                    row['termination'] = 'native_success_without_evaluator_goal';break
            else:
                row['termination'] = 'max_ticks'
            row['execution_trace'].extend(runtime.trace)
            row['condition_checks'] += runtime.condition_checks
            row['final_tree'] = runtime.tree
            row['final_failed_node'], row['final_failed_path'] = runtime.last_failed_node, runtime.last_failed_path
    except Exception as exc:
        row['execution_error'] = f'{type(exc).__name__}: {exc}'
        row['termination'] = 'method_or_execution_error'
    row['wall_s'] = time.monotonic() - start
    row['final_world_state'] = world.to_json()
    row['success'] = bool(world.check(goal_name, goal_args))
    row['action_count'] = len(row['action_trace'])
    row['model_metrics'] = client.metrics()
    row['attempt_id'] = client.attempt_id
    row['status'] = 'INFRASTRUCTURE_ERROR' if client.infra_errors else 'COMPLETED'
    if client.infra_errors:
        row['success'] = None  # not silently counted as a planner failure
    return row


def create_client(args, config, model_tag, method, log_path):
    from llm_transport import LLMClient
    settings = config.get('model_settings', {}).get(model_tag, {})
    for name in ['llm_url', 'hibt_url', 'ner_url']:
        override = getattr(args, name)
        if override and settings.get(name) and override.rstrip('/') != settings[name].rstrip('/'):
            raise ValueError(f'CLI {name} differs from frozen model_settings; use a new protocol')
    adapter = settings.get('btgenbot_adapter_checkpoint') if method == 'btgenbot' else None
    if method == 'btgenbot' and (not adapter or adapter.startswith('SET_')):
        raise ValueError('Freeze an actual trained BTGenBot adapter checkpoint before formal execution')
    if method == 'llmbt' and (not settings.get('ner_checkpoint') or settings['ner_checkpoint'].startswith('SET_')):
        raise ValueError('Freeze the actual NER checkpoint provenance before formal LLM-BT execution')
    if method == 'hibt' and not (args.hibt_url or settings.get('hibt_url')):
        raise ValueError('HiBT neural backend URL must be explicit in the frozen run')
    kwargs = {'adapter_method': 'BTGenBot' if method == 'btgenbot' else None,
              'adapter_checkpoint': adapter, 'log_path': None,
              'hibt_url': args.hibt_url or settings.get('hibt_url'), 'model_tag': model_tag,
              'ner_checkpoint': settings.get('ner_checkpoint')}
    if 'max_calls' in inspect.signature(LLMClient).parameters:
        kwargs['max_calls'] = config['budget']['max_calls']
    ner_url = args.ner_url or settings.get('ner_url')
    if ner_url:
        if 'ner_url' not in inspect.signature(LLMClient).parameters:
            raise ValueError('Root LLMClient must support ner_url before external NER service use')
        kwargs['ner_url'] = ner_url
    url = args.llm_url or settings.get('llm_url') or (None if model_tag == 'qwen' else None)
    base = LLMClient(url, **kwargs)
    return BudgetClient(base, config['budget']['max_calls'], log_path, settings.get('expected_model_name'))


def aggregate(run, proto, config):
    rows = [load(p) for p in (run / 'results').glob('*.json')]
    if any(x['protocol_id'] != proto['protocol_id'] for x in rows):
        raise ValueError('Mixed protocol results detected')
    summary = {}
    for tag in config['model_tags']:
        for method in config['methods']:
            key = tag + ':' + method
            selected = [x for x in rows if x['model_tag'] == tag and x['method'] == method]
            complete = [x for x in selected if x['status'] == 'COMPLETED']
            summary[key] = {'n_expected': config.get('n_requests', 60), 'n_complete': len(complete),
                            'success': sum(bool(x['success']) for x in complete),
                            'n_infrastructure_pending': sum(x['status'] != 'COMPLETED' for x in selected),
                            'valid_tree': sum(x.get('valid_generated_tree', False) for x in complete),
                            'llm_calls': sum(x['model_metrics']['llm_calls'] for x in complete),
                            'prompt_tokens': sum(x['model_metrics']['prompt_tokens'] for x in complete),
                            'completion_tokens': sum(x['model_metrics']['completion_tokens'] for x in complete),
                            'median_wall_s': statistics.median(x['wall_s'] for x in complete) if complete else None}
    dump(run / 'summary.json', {'protocol_id': proto['protocol_id'], 'groups': summary,
                               'all_expected_complete': all(x['n_complete'] == x['n_expected'] and x['n_infrastructure_pending'] == 0 for x in summary.values())})


def software_smoke(run):
    """A correct tree and a wrong-goal tree distinguish task from tick success."""
    state = {'objects': [{'name': 'dev_hand', 'properties': ['is_empty']},
                         {'name': 'dev_tool', 'properties': ['is_empty', 'is_equippable']},
                         {'name': 'dev_part', 'properties': []}, {'name': 'dev_socket', 'properties': []}],
             'constraints': [{'name': 'can_manipulate', 'source': 'dev_tool', 'target': 'dev_part'},
                             {'name': 'can_insert_to', 'source': 'dev_part', 'target': 'dev_socket'}], 'relations': []}
    request = {'sample_id': 'software_only', 'instruction': 'Insert dev_part into dev_socket.', 'world_state': state, 'suite': 'assembly', 'planning_goal': 'is_inserted_to(dev_part, dev_socket)'}
    request['registry'] = make_registry(request)
    evaluator = {'target': 'is_inserted_to(dev_part, dev_socket)'}
    correct = {'name': 'sequence: correct', 'children': [{'name': 'action: ' + text} for text in
               ['load_tool(dev_hand, dev_tool)', 'pick_up(dev_hand, dev_tool, dev_part)', 'insert(dev_hand, dev_tool, dev_part, dev_socket)']]}
    class Mock:
        metadata = {'software_test_only': True}
        def complete(self, messages, **kwargs):
            assert 'target' not in messages[0]
            return {'text': 'software_only', 'prompt_tokens': 1, 'completion_tokens': 1}
    class MockMethod:
        tree = correct
        @classmethod
        def plan(cls, request, registry, llm, budget, feedback=None):
            llm.complete([{'instruction': request['instruction']}])
            return {'tree': copy.deepcopy(cls.tree), 'errors': [], 'software_test_only': True}
    client = BudgetClient(Mock(), 1)
    passed = run_case(request, evaluator, 'hibt', MockMethod, client, {}, {'max_ticks': 10, 'max_actions': 10})
    assert passed['success'] and passed['action_count'] == 3 and client.llm_calls == 1
    MockMethod.tree = {'name': 'target: is_empty(dev_hand)'}
    failed = run_case(request, evaluator, 'hibt', MockMethod, BudgetClient(Mock(), 1), {}, {'max_ticks': 10})
    assert failed['success'] is False and failed['termination'] == 'native_success_without_evaluator_goal'
    original = copy.deepcopy(state)
    feedback = simulation_feedback(correct, state, {'max_feedback_ticks': 10})
    assert state == original and feedback['result'] == 'success' and 'target' not in feedback
    exhausted = BudgetClient(Mock(), 1)
    exhausted.complete([{}])
    try:
        exhausted.complete([{}]);raise AssertionError('Global budget bypassed')
    except RuntimeError:
        pass
    class UpdateMethod(MockMethod):
        tree = {'name': 'target: hold(dev_hand, dev_tool)'}
        @staticmethod
        def update(request, registry, llm, budget, feedback=None, tree=None, failed_node=None, runtime_state=None, failure_trace=None):
            assert failed_node['name'] == 'target: hold(dev_hand, dev_tool)'
            assert failure_trace and failed_node['path'] == []
            assert 'target' not in request and state_facts(runtime_state) == state_facts(state)
            return {'tree': copy.deepcopy(correct), 'errors': []}
    updated = run_case(request, evaluator, 'llmbt', UpdateMethod, BudgetClient(Mock(), 1), {}, {'max_ticks': 10})
    assert updated['success'] and len(updated['runtime_updates']) == 1
    no_generic = run_case(request, evaluator, 'hibt', UpdateMethod, BudgetClient(Mock(), 1), {}, {'max_ticks': 10})
    assert no_generic['success'] is False and not no_generic['runtime_updates']
    class Unavailable(Mock):
        def complete(self, *args, **kwargs):
            raise urllib.error.URLError('software-only unavailable endpoint')
    unavailable = run_case(request, evaluator, 'hibt', MockMethod, BudgetClient(Unavailable(), 1), {}, {})
    assert unavailable['status'] == 'INFRASTRUCTURE_ERROR' and unavailable['success'] is None
    check_dir = Path(run) / 'protocol_check';check_dir.mkdir(exist_ok=True)
    dump(check_dir / 'config.json', {'version': 1})
    dump(check_dir / 'requests.json', [request]);dump(check_dir / 'evaluators.json', [evaluator])
    cfg = {'methods': ['hibt'], 'model_tags': ['qwen'], 'phase': 'SOFTWARE_ONLY'}
    identity = protocol(check_dir, cfg, check_dir / 'requests.json', check_dir / 'evaluators.json')
    assert protocol(check_dir, cfg, check_dir / 'requests.json', check_dir / 'evaluators.json') == identity
    dump(check_dir / 'config.json', {'version': 2})
    try:
        protocol(check_dir, cfg, check_dir / 'requests.json', check_dir / 'evaluators.json')
        raise AssertionError('Changed immutable config accepted')
    except ValueError:
        pass
    result = {'scope': 'SOFTWARE_ONLY_NO_FORMAL_MODEL_RESULTS', 'checks': {
        'valid_original_transition_execution': True, 'wrong_goal_tick_success_rejected': True,
        'feedback_pure_and_goal_free': True, 'global_budget_enforced': True,
        'only_native_llmbt_update_on_actual_failure': True, 'no_generic_hibt_repair': True,
        'infrastructure_failure_not_planner_score': True, 'resume_requires_identical_immutable_protocol': True}}
    dump(Path(run) / 'software_smoke.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--method', nargs='+')
    p.add_argument('--model-tag', choices=['qwen', 'mistral'], default='qwen')
    p.add_argument('--llm-url');p.add_argument('--hibt-url');p.add_argument('--ner-url')
    p.add_argument('--worker-index', type=int, default=0);p.add_argument('--worker-count', type=int, default=1)
    p.add_argument('--retry-infrastructure', action='store_true')
    p.add_argument('--software-smoke', action='store_true')
    args = p.parse_args();run = args.run_dir.resolve();run.mkdir(parents=True, exist_ok=True)
    if args.software_smoke:
        software_smoke(run);return
    config = load(run / 'config.json')
    if config.get('protocol_frozen') is not True:
        raise ValueError('Root must freeze/review config before any model run')
    config['methods'] = [canonical(x) for x in config['methods']]
    if args.model_tag not in config['model_tags']:
        raise ValueError('Model tag is absent from frozen config')
    requested_methods = [canonical(x) for x in args.method] if args.method else config['methods']
    if set(requested_methods) - set(config['methods']):
        raise ValueError('Requested methods absent from frozen protocol')
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError('Invalid worker partition')
    def resolve(value):
        path = Path(value);return path if path.is_absolute() else run / path
    request_path, evaluator_path = resolve(config['requests_file']), resolve(config['evaluators_file'])
    requests = load(request_path);evaluators = {x['sample_id']: x for x in load(evaluator_path)}
    if config.get('assembly_state_contract') != STATE_CONTRACT:
        raise ValueError('This assembly revision requires the common original state canonicalization')
    validate_prepared_inputs(resolve(config['raw_requests_file']), request_path,
                             resolve(config['state_canonicalization_manifest']))
    if len(requests) != config.get('n_requests', 60) or len({x['sample_id'] for x in requests}) != len(requests):
        raise ValueError('Request count/uniqueness differs from frozen protocol')
    if {x['sample_id'] for x in requests} != set(evaluators):
        raise ValueError('Evaluator/model input IDs do not match')
    proto = protocol(run, config, request_path, evaluator_path)
    demonstrations = load(resolve(config['demonstrations_file'])) if config.get('demonstrations_file') else {}
    if demonstrations and demonstrations.get('source_split') != 'independently_generated_train_only':
        raise ValueError('Unknown demonstration provenance')
    (run / 'results').mkdir(exist_ok=True);(run / 'cases').mkdir(exist_ok=True)
    metadata = {'argv': sys.argv, 'pid': os.getpid(), 'worker_index': args.worker_index,
                'worker_count': args.worker_count, 'protocol_id': proto['protocol_id'], 'started_at': time.time(),
                'gpu_started_by_driver': False}
    dump(run / f'worker_{args.model_tag}_{args.worker_index}_{os.getpid()}.json', metadata)
    for index, request in enumerate(requests):
        for method in requested_methods:
            global_index = index * len(config['methods']) + config['methods'].index(method)
            if global_index % args.worker_count != args.worker_index:
                continue
            if source_hashes() != proto['source_hashes'] or sha(run / 'config.json') != proto['config_sha256']:
                raise ValueError('Frozen source/config changed during run; stop without mixing results')
            key = f'{args.model_tag}__{method}__{index:03d}'
            case = run / 'cases' / key;case.mkdir(exist_ok=True)
            result_path = run / 'results' / (key + '.json')
            with (case / 'case.lock').open('a') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # another worker is actively executing this case
                if result_path.exists():
                    previous = load(result_path)
                    if previous['protocol_id'] != proto['protocol_id']:
                        raise ValueError('Existing case has wrong protocol')
                    if previous['status'] != 'INFRASTRUCTURE_ERROR' or not args.retry_infrastructure:
                        continue
                    history = case / f'infrastructure_attempt_{time.time_ns()}.json'
                    shutil_copy = previous
                    dump(history, shutil_copy)
                budget = copy.deepcopy(config['budget'])
                budget.update(config.get('method_budgets', {}).get(method, {}))
                actual_request = copy.deepcopy(request)
                if demonstrations.get('intent_examples'):
                    budget['intent_examples'] = copy.deepcopy(demonstrations['intent_examples'])
                if method == 'kios' and demonstrations.get('kios_demonstrations'):
                    actual_request['registry']['kios_demonstrations'] = copy.deepcopy(demonstrations['kios_demonstrations'])
                if method == 'llmbt' and demonstrations.get('llmbt_demonstrations'):
                    actual_request['registry']['llmbt_demonstrations'] = copy.deepcopy(demonstrations['llmbt_demonstrations'])
                client = create_client(args, config, args.model_tag, method, case / 'calls.jsonl')
                module = importlib.import_module('methods.' + method)
                record = run_case(actual_request, evaluators[request['sample_id']], method, module, client, budget, config.get('runtime', {}))
                record.update(protocol_id=proto['protocol_id'], model_tag=args.model_tag, request_index=index,
                              model_input_sha256=hashlib.sha256(json.dumps(actual_request, sort_keys=True).encode()).hexdigest(),
                              frozen_budget=budget)
                dump(result_path, record)
                print(json.dumps({key: record.get(key) for key in ['sample_id', 'method', 'success', 'status', 'termination', 'wall_s']}), flush=True)
                if record['status'] == 'INFRASTRUCTURE_ERROR':
                    aggregate(run, proto, config)
                    raise RuntimeError('Infrastructure failure preserved; root must repair and explicitly retry this same case')
    aggregate(run, proto, config)


if __name__ == '__main__':
    main()
