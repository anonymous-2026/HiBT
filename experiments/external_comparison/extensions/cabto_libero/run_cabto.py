"""Independent CABTO assembly/native-LIBERO runner. Never starts a GPU service."""
from __future__ import annotations
import argparse
import base64
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n');temporary.replace(path)


def source_hashes(shared):
    files = {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*.py')
             if 'official_repo' not in p.parts and '__pycache__' not in p.parts}
    files.update({'shared/' + str(p.relative_to(shared)): sha(p) for p in shared.rglob('*.py')})
    for relative in ['source_audit/source_manifest.json', 'source_audit/paper.json', 'source_audit/LICENSE_NOTICE.md']:
        path = HERE / relative
        if path.is_file():
            files[relative] = sha(path)
    import inspect
    from native_runtime import _world_class
    original = Path(inspect.getfile(_world_class()))
    files['original_assembly_runtime/minimal_bt.py'] = sha(original)
    files['original_assembly_runtime/world_definition.json'] = sha(original.parents[1] / 'data/world_definition.json')
    return files


def protocol(run, config, shared):
    identity = {'version': 'cabto_complete_port_v3_complete_skill_execution_boundary', 'mode': config['mode'],
                'execution_boundary': {'name': config.get('execution_boundary'),
                    'scope': 'CABTO-native-only; original per-physical-step execute retained',
                    'generated_guards_effects_and_budgets_unchanged': True,
                    'original_reference': 'pinned stage3_cabto/codegen.py:execute_step; loop.py execute then effect-check'},
                'config_sha256': sha(run / 'config.json'), 'source_hashes': source_hashes(shared),
                'requests_sha256': sha(config['requests_file'])}
    for name in ['evaluators_file', 'input_manifest', 'model_manifest']:
        if config.get(name):
            identity[name + '_sha256'] = sha(config[name])
    identity['protocol_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    with (run / 'protocol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = run / 'protocol.json'
        if path.exists() and read(path) != identity:
            raise ValueError('Frozen source/config/input mismatch; create a separate run')
        if not path.exists():
            dump(path, identity)
    return identity


def observation_images(observation, transform):
    import numpy as np
    from PIL import Image
    images = []
    for key in ['agentview_image', 'robot0_eye_in_hand_image']:
        image = np.asarray(observation[key])
        if transform == 'rotate180':
            image = image[::-1, ::-1]
        elif transform != 'raw':
            raise ValueError('Unsupported visual observation transform')
        image = Image.fromarray(np.ascontiguousarray(image).astype('uint8'))
        buffer = io.BytesIO();image.save(buffer, format='JPEG', quality=85)
        images.append('data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode())
    return images


def assembly_case(request, evaluator, config, client):
    import adapter
    from assembly_registry import make_registry, parse_call, call
    from assembly_state_canonicalization import _normalize_state_for_kios
    from native_runtime import NativeRuntime, _world_class
    from run_assembly_comparison import simulation_feedback, validate_native_tree
    if _normalize_state_for_kios(request['world_state']) != request['world_state']:
        # Normalize equality by fact sets, because original properties may have
        # equivalent list ordering. Do not silently normalize formal inputs.
        from assembly_registry import state_facts
        if state_facts(_normalize_state_for_kios(request['world_state'])) != state_facts(request['world_state']):
            raise ValueError('CABTO assembly requests must use the shared canonical state')
    registry = copy.deepcopy(request['registry'])
    assert make_registry(request) == registry
    world = _world_class()(copy.deepcopy(request['world_state']))
    budget = {**config['budget'], 'max_rounds': config.get('max_rounds', 3)}
    feedback = lambda tree: simulation_feedback(tree, request['world_state'], config.get('runtime', {}))
    plan = adapter.plan(request, registry, client, budget, feedback=feedback)
    row = {'initial_plan': plan, 'runtime_updates': [], 'action_trace': [], 'success': False,
           'termination': 'planning_failure', 'tree': plan.get('tree')}
    tree = plan.get('tree')
    if not isinstance(tree, dict):
        return row
    validate_native_tree(tree, registry)
    goal_name, goal_args = parse_call(evaluator['target'])
    def execute(name, args):
        if len(row['action_trace']) >= config.get('runtime', {}).get('max_actions', 128):
            return 'failure'
        before = world.to_json();okay = world.apply_action(name, args)
        row['action_trace'].append({'action': call(name, args), 'success': okay, 'before': before, 'after': world.to_json()})
        return 'success' if okay else 'failure'
    runtime = NativeRuntime(tree, world.check, execute)
    for tick in range(config.get('runtime', {}).get('max_ticks', 128)):
        if world.check(goal_name, goal_args):
            row['termination'] = 'evaluator_goal_reached';break
        before = world.to_json();status = runtime.tick()
        if world.check(goal_name, goal_args):
            row['termination'] = 'evaluator_goal_reached';break
        if status == 'failure':
            current = copy.deepcopy(request);current['world_state'] = world.to_json();current['registry'] = make_registry(current)
            updated = adapter.update(current, current['registry'], client, budget,
                feedback=lambda candidate: simulation_feedback(candidate, current['world_state'], config.get('runtime', {})),
                tree=runtime.tree, failed_node=runtime.last_failed_node, runtime_state=world.to_json(),
                previous=plan.get('cabto_state'), failure_trace=runtime.trace)
            row['runtime_updates'].append(updated);plan = updated
            if not updated.get('tree'):
                row['termination'] = 'CABTO_refinement_exhausted';break
            runtime = NativeRuntime(updated['tree'], world.check, execute)
        elif status == 'success' and before == world.to_json():
            row['termination'] = 'native_success_without_goal';break
    else:
        row['termination'] = 'max_ticks'
    row.update(success=bool(world.check(goal_name, goal_args)), final_world_state=world.to_json(),
               native_trace=runtime.trace, final_tree=runtime.tree)
    return row


def native_case(request, init_id, config, client, case_dir):
    import adapter
    from execution_boundary import complete_native_skill, OfficialGoalReached
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import run_native_libero as shared
    from native_runtime import NativeRuntime
    if config.get('openpi_client_source'):
        sys.path.insert(0, config['openpi_client_source'])
    from openpi_client import image_tools, websocket_client_policy
    suite = benchmark.get_benchmark_dict()[request['suite']]()
    env = shared.make_env(suite, request['task_id'], OffScreenRenderEnv, config.get('bddl_root', get_libero_path('bddl_files')))
    try:
        initial_state = suite.get_task_init_states(request['task_id'])[init_id]
        task = suite.get_task(request['task_id'])
        actual_bddl = Path(config.get('bddl_root', get_libero_path('bddl_files'))) / task.problem_folder / task.bddl_file
        observation = env.set_init_state(initial_state)
        for _ in range(config.get('settle_steps', 10)):
            observation, _, _, _ = env.step(shared.DUMMY_ACTION)
        registry = copy.deepcopy(request['registry'])
        catalog = shared.predicate_catalog(registry)
        actual = copy.deepcopy(request)
        actual['world_state'] = shared.actual_snapshot(env, request, registry, catalog)
        registry['world_state'] = copy.deepcopy(actual['world_state']);registry['facts'] = actual['world_state']['facts']
        actual['registry'] = registry
        actual['_cabto_observation_images'] = observation_images(observation, config['camera_transform'])
        budget = {**config['budget'], 'max_rounds': config.get('max_rounds', 3)}
        planned = adapter.plan(actual, registry, client, budget,
                               feedback=lambda tree: shared.symbolic_feedback(tree, registry))
        dump(case_dir / 'initial_plan.json', planned)
        class CABTOEpisode(shared.NativeEpisode):
            def execute(self, name, args):
                advance = super().execute
                return complete_native_skill(lambda: advance(name, args), self.env.check_success)

            def try_native_update(self, tree):
                snapshot = shared.actual_snapshot(self.env, self.request, self.registry, self.catalog)
                current = copy.deepcopy(self.request);current['world_state'] = snapshot
                current['_cabto_observation_images'] = observation_images(self.obs, self.config['camera_transform'])
                current_registry = copy.deepcopy(self.registry);current_registry['world_state'] = snapshot;current_registry['facts'] = snapshot['facts']
                updated = adapter.update(current, current_registry, client, budget,
                    feedback=lambda candidate: shared.symbolic_feedback(candidate, current_registry),
                    tree=tree, failed_node=self.runtime.last_failed_node, runtime_state=snapshot,
                    previous=self.plan_row.get('cabto_state'), failure_trace=self.runtime.trace)
                self.updates += 1
                dump(case_dir / ('native_update_' + str(self.updates) + '.json'), updated)
                self.plan_row = updated
                return updated.get('tree')
        policy = websocket_client_policy.WebsocketClientPolicy(config.get('policy_host','localhost'), int(config['policy_port']))
        # Low-level control sees original NL plus primitive prompt; no symbolic
        # goal, generated model, or VLM prompt is sent into the fixed VLA.
        physical_config = {**shared.DEFAULTS, **config, 'init_state_id': init_id}
        builder = lambda obs, prompt, seed: shared.policy_observation(obs, prompt, seed, image_tools, config['camera_transform'])
        episode = CABTOEpisode(env, request, registry, planned, physical_config, policy, builder, case_dir)
        try:
            success = episode.run(observation)
        except OfficialGoalReached:
            # This is the same official per-physical-step success criterion as
            # shared NativeEpisode.run, never a generated/program/model goal.
            success = bool(env.check_success())
            if not success:
                raise RuntimeError('Official evaluator changed without another physical step')
            episode.reason = 'official_evaluator_goal_reached'
        return {'success': success, 'termination': episode.reason,
                'execution_boundary': 'complete_generated_API_skill_before_outer_formal_tick',
                'initial_plan': planned, 'physical_steps': episode.steps, 'policy_calls': episode.policy_calls,
                'primitive_retries': episode.retries, 'runtime_updates': episode.updates,
                'semantic_errors': episode.semantic_errors,
                'initial_state_sha256': shared.initial_state_sha256(initial_state),
                'initial_state_dtype': np.asarray(initial_state).dtype.str,
                'initial_state_shape': list(np.asarray(initial_state).shape),
                'actual_bddl_sha256': sha(actual_bddl),
                'policy_metadata': policy.get_server_metadata()}
    except shared.InfrastructureError as exc:
        from transport import InfrastructureError
        raise InfrastructureError(str(exc)) from exc
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['assembly', 'libero'], required=True)
    parser.add_argument('--worker-count', type=int, default=1);parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--retry-infrastructure', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve();config = read(run / 'config.json')
    if config.get('protocol_frozen') is not True or config['mode'] != args.mode:
        raise ValueError('Root must freeze a matching independent CABTO configuration before model calls')
    if args.mode == 'libero' and config.get('execution_boundary') != 'complete_generated_API_skill_before_outer_formal_tick':
        raise ValueError('Native v3 requires its explicitly reviewed complete-skill execution contract')
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError('Invalid worker partition')
    shared_source = Path(config['shared_source']).resolve();sys.path.insert(0, str(shared_source));sys.path.insert(0, str(HERE))
    from transport import VLMClient, InfrastructureError, BudgetExhausted
    from goal_contract import validate_request
    identity = protocol(run, config, shared_source)
    requests = read(config['requests_file'])
    evaluators = {x['sample_id']: x for x in read(config['evaluators_file'])} if args.mode == 'assembly' else {}
    ids = [request['sample_id'] for request in requests]
    if len(ids) != len(set(ids)) or (args.mode == 'assembly' and set(ids) != set(evaluators)):
        raise ValueError('Membership must be unique and evaluator IDs must match')
    dump(run / ('worker_' + str(os.getpid()) + '.json'), {'pid': os.getpid(), 'argv': sys.argv,
          'protocol_id': identity['protocol_id'], 'inference_service_started_by_client': False})
    matrix = [(i, request, initial) for initial in (config['init_ids'] if args.mode == 'libero' else [None])
              for i, request in enumerate(requests)]
    for ordinal, (index, request, initial) in enumerate(matrix):
        if ordinal % args.worker_count != args.worker_index:
            continue
        key = f'cabto__{index:03d}' if initial is None else f'cabto__{request["suite"]}__t{request["task_id"]:02d}__i{initial:02d}'
        (run / 'locks').mkdir(exist_ok=True)
        with (run / 'locks' / (key + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if protocol(run, config, shared_source) != identity:
                raise ValueError('Immutable CABTO protocol changed')
            result_path = run / 'results' / (key + '.json');case_dir = run / 'cases' / key
            if result_path.exists():
                old = read(result_path)
                if old['protocol_id'] != identity['protocol_id']:
                    raise ValueError('Existing CABTO result uses another protocol')
                if old['status'] == 'COMPLETED':
                    continue
                if not args.retry_infrastructure:
                    raise InfrastructureError('Preserved infrastructure episode requires explicit diagnosed retry: ' + key)
                archive = run / '_infrastructure_attempts' / (key + '.' + str(time.time_ns()))
                archive.parent.mkdir(exist_ok=True)
                if case_dir.exists():
                    shutil.move(str(case_dir), str(archive))
                else:
                    archive.mkdir()
                shutil.move(str(result_path), str(archive / 'result.json'))
            elif case_dir.exists() and any(case_dir.iterdir()):
                raise InfrastructureError('Unfinished partial attempt must be inspected/archived explicitly: ' + key)
            case_dir.mkdir(parents=True, exist_ok=True)
            client = VLMClient(config['vlm_url'], config['expected_model_name'], config['budget']['max_calls'],
                               case_dir / 'model_calls.jsonl', config.get('transport_timeout_s', 900),
                               expected_revision=config.get('expected_revision'))
            row = {'method': 'cabto', 'model_tag': config.get('model_tag', 'qwen35_9b'), 'request_index': index,
                   'sample_id': request['sample_id'], 'suite': request.get('suite', 'assembly'),
                   'task_id': request.get('task_id'), 'init': initial, 'protocol_id': identity['protocol_id'],
                   'planning_goal': validate_request(request, request['registry']),
                   'model_input_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                   'success': False, 'status': 'COMPLETED'}
            start = time.monotonic()
            try:
                result = assembly_case(copy.deepcopy(request), evaluators[request['sample_id']], config, client) if args.mode == 'assembly' else native_case(copy.deepcopy(request), initial, config, client, case_dir)
                row.update(result)
            except (OSError, TimeoutError, ConnectionError) as exc:
                row.update(status='INFRASTRUCTURE_ERROR', success=None, infrastructure_error=repr(exc))
            except BudgetExhausted as exc:
                row.update(termination='CABTO_global_budget_exhausted', semantic_error=str(exc))
            except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                row.update(termination='CABTO_method_error', semantic_error=repr(exc))
            except Exception as exc:
                row.update(status='INFRASTRUCTURE_ERROR', success=None,
                           infrastructure_error='Unexpected software/environment error: ' + repr(exc),
                           infrastructure_traceback=traceback.format_exc())
            row.update(wall_s=time.monotonic() - start, model_metrics=client.metrics())
            if client.infrastructure_errors:
                row.update(status='INFRASTRUCTURE_ERROR', success=None, infrastructure_error=client.infrastructure_errors)
            dump(result_path, row)
            print(json.dumps({key: row.get(key) for key in ['sample_id', 'method', 'success', 'status', 'termination', 'wall_s']}), flush=True)
            if row['status'] != 'COMPLETED':
                raise InfrastructureError('Infrastructure result preserved; stop without scoring it')


if __name__ == '__main__':
    main()
