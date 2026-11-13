"""Prepare/cache complete-method LIBERO plans without launching GPU services.

Formal plans are generated once per task's identical symbolic initial input and
reused across the declared ten physical initial states. The final task goal is
a public input in this independent goal-specified track; evaluator outcomes and
reference action/tree labels remain separate.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from assembly_registry import parse_call
from goal_contract import VERSION, validate_request
from registry_lineage import verify_cached_registry_lineage
from libero_registry import make_registry
from run_assembly_comparison import BudgetClient, canonical, create_client, dump, load, sha, source_hashes, validate_native_tree


def clean_state(value):
    if isinstance(value, dict):
        return {key: clean_state(item) for key, item in value.items() if key != 'objects_of_interest'}
    if isinstance(value, list):
        return [clean_state(item) for item in value]
    return copy.deepcopy(value)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    path = Path(path);temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    temporary.replace(path)




def complete_source_hashes():
    values = source_hashes()
    lineage_path = Path(__file__).with_name('cached_registry_lineage.json')
    values['cached_registry_lineage.json'] = sha(lineage_path)
    audit = Path(__file__).resolve().parents[1] / 'source_audit'
    for method in ['kios', 'llmbt', 'obtea', 'btgenbot']:
        folder = audit / method
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.suffix in {'.py', '.txt', '.json', '.cpp', '.hpp', '.h', '.yaml', '.yml'}:
                    values[str(path.relative_to(audit.parent))] = sha(path)
    return values


def freeze_protocol(run, config, requests_path, manifest_path, demos_path):
    current = {'version': 1, 'goal_input_contract': VERSION, 'phase': config['phase'], 'config_sha256': sha(run / 'config.json'),
               'requests_sha256': sha(requests_path), 'input_manifest_sha256': sha(manifest_path),
               'demonstrations_sha256': sha(demos_path), 'source_hashes': complete_source_hashes(),
               'training_pool_manifest_sha256': sha(config['training_pool_manifest']),
               'methods': config['methods'], 'model_tags': config['model_tags']}
    current['protocol_id'] = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    with (run / 'protocol.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = run / 'protocol.json'
        if path.exists() and load(path) != current:
            raise ValueError('Frozen LIBERO planning protocol/source/input mismatch; use a new run')
        if not path.exists():
            dump(path, current)
    return current


def requested_rows(requests, manifest, phase):
    ids = manifest['development_ids'] if phase == 'development' else manifest['formal_ids']
    selected = [x for x in requests if x['request_id'] in set(ids)]
    if len(selected) != (6 if phase == 'development' else 10):
        raise ValueError('Incorrect complete phase membership')
    if phase == 'development' and any(x['suite'] == 'libero_10' for x in selected):
        raise ValueError('Long task entered development planning')
    if phase == 'formal' and any(x['suite'] != 'libero_10' for x in selected):
        raise ValueError('Non-Long task entered formal planning')
    return selected


def build_plan_row(request, method, module, client, budget, demonstrations, feedback_ticks):
    from run_native_libero import symbolic_feedback
    if {'target', 'teacher_actions', 'output', 'behavior_tree', 'gold_tree', 'reference_plan', 'pyramid'} & set(request):
        raise ValueError('Reference plan/action/BT fields are not task inputs')
    public_goal = validate_request(request, request['registry'])
    actual = copy.deepcopy(request)
    registry = actual['registry']
    additions = {}
    if method == 'kios':
        additions['kios_demonstrations'] = demonstrations['kios_demonstrations']
    if method == 'llmbt':
        additions['llmbt_demonstrations'] = demonstrations['llmbt_demonstrations']
    registry.update(copy.deepcopy(additions))
    effective_budget = copy.deepcopy(budget)
    effective_budget['intent_examples'] = demonstrations['intent_examples']
    feedback = lambda tree: symbolic_feedback(tree, registry, max_ticks=feedback_ticks)
    begin = time.monotonic()
    try:
        result = module.plan(actual, registry, client, effective_budget, feedback=feedback)
    except Exception as exc:
        result = {'tree': None, 'trace': [], 'errors': [f'{type(exc).__name__}: {exc}']}
    tree = result.get('tree')
    valid, validation_error = False, None
    if isinstance(tree, dict):
        try:
            validate_native_tree(tree, registry);valid = True
        except Exception as exc:
            validation_error = f'{type(exc).__name__}: {exc}'
    return {'suite': request['suite'], 'task_id': request['task_id'], 'request_id': request['request_id'],
            'sample_id': request['sample_id'], 'method': method, 'tree': tree,
            'planning_goal': public_goal, 'goal_input_contract': VERSION,
            'trace': result.get('trace', []), 'errors': result.get('errors', []),
            'complete_method_result': result, 'valid_generated_tree': valid,
            'static_validation_error': validation_error,
            'llm_calls': client.llm_calls, 'ner_calls': client.ner_calls, 'hibt_calls': client.hibt_calls,
            'model_metrics': client.metrics(), 'planning_wall_s': time.monotonic() - begin,
            'method_registry_additions': additions, 'frozen_method_budget': effective_budget,
            'status': 'INFRASTRUCTURE_ERROR' if client.infra_errors else 'COMPLETED',
            'planning_outcome': 'PLAN_RECORDED' if tree is not None else 'PLANNING_FAILURE',
            'attempt_id': client.attempt_id,
            'input_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
            'symbolic_input_plan_reuse': 'One cached initial tree for all10declared physical initial states',
            'reference_actions_or_BT_used': False}


def aggregate(run, protocol, expected):
    rows = [load(x) for x in (run / 'results').glob('*.json')]
    if any(x['protocol_id'] != protocol['protocol_id'] for x in rows):
        raise ValueError('Mixed planning protocols')
    rows.sort(key=lambda x: (x['suite'], x['task_id'], x['method'], x['model_tag']))
    write_jsonl(run / 'planning_outputs.jsonl', rows)
    counts = {}
    for tag in protocol['model_tags']:
        for method in protocol['methods']:
            selected = [x for x in rows if x['method'] == method and x['model_tag'] == tag]
            counts[tag + ':' + method] = {'expected': expected, 'recorded': len(selected),
                                         'complete': sum(x['status'] == 'COMPLETED' for x in selected),
                                         'planning_failures': sum(x['status'] == 'COMPLETED' and x['tree'] is None for x in selected),
                                         'infrastructure_pending': sum(x['status'] != 'COMPLETED' for x in selected)}
    dump(run / 'summary.json', {'protocol_id': protocol['protocol_id'], 'phase': protocol['phase'],
                               'groups': counts, 'n_plan_records': len(rows),
                               'all_expected_plans_recorded': all(x['complete'] == expected for x in counts.values()),
                               'robot_rollout_results': 'Not measured by this planning script'})




def main():
    project = Path(__file__).resolve().parents[3]
    e6 = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--phase', choices=['development', 'formal'], default='development')
    p.add_argument('--method', nargs='+')
    p.add_argument('--model-tag', default='qwen', choices=['qwen', 'mistral'])
    p.add_argument('--llm-url');p.add_argument('--hibt-url');p.add_argument('--ner-url')
    p.add_argument('--worker-index', type=int, default=0);p.add_argument('--worker-count', type=int, default=1)
    p.add_argument('--retry-infrastructure', action='store_true')
    args = p.parse_args()
    if args.run_dir is None:
        raise ValueError('--run-dir required; this script never starts model services')
    run = args.run_dir.resolve();run.mkdir(parents=True, exist_ok=True)
    config = load(run / 'config.json')
    if config.get('protocol_frozen') is not True or config.get('phase') != args.phase:
        raise ValueError('Root must freeze a phase-matched config before model calls')
    if args.model_tag not in config['model_tags']:
        raise ValueError('Backbone absent from frozen planning protocol')
    def resolve(value):
        path = Path(value);return path if path.is_absolute() else run / path
    inputs, manifest_path, demos_path = (resolve(config[x]) for x in ['requests_file', 'input_manifest', 'demonstrations_file'])
    config['training_pool_manifest'] = str(resolve(config['training_pool_manifest']))
    requests, manifest, demos = load(inputs), load(manifest_path), load(demos_path)
    if manifest.get('goal_input_contract') != VERSION:
        raise ValueError('This driver requires an independent goal-specified input package')
    registry_lineage = verify_cached_registry_lineage(requests)
    if manifest['requests_sha256'] != sha(inputs) or manifest['demonstrations_sha256'] != sha(demos_path):
        raise ValueError('Clean inputs or training demonstrations changed since preparation')
    config['methods'] = [canonical(x) for x in config['methods']]
    methods = [canonical(x) for x in args.method] if args.method else config['methods']
    if set(methods) - set(config['methods']):
        raise ValueError('Method subset differs from frozen method universe')
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError('Invalid worker partition')
    selected = requested_rows(requests, manifest, args.phase)
    protocol = freeze_protocol(run, config, inputs, manifest_path, demos_path)
    dump(run / 'cached_registry_lineage_audit.json', registry_lineage)
    for folder in ['results', 'cases']:
        (run / folder).mkdir(exist_ok=True)
    dump(run / f'worker_{os.getpid()}.json', {'pid': os.getpid(), 'argv': sys.argv,
                                           'protocol_id': protocol['protocol_id'], 'gpu_started': False})
    for index, request in enumerate(selected):
        for method in methods:
            global_index = index * len(config['methods']) + config['methods'].index(method)
            if global_index % args.worker_count != args.worker_index:
                continue
            if sha(run / 'config.json') != protocol['config_sha256'] or complete_source_hashes() != protocol['source_hashes']:
                raise ValueError('Frozen source/config changed during planning; stop without mixing outcomes')
            key = f'{args.model_tag}__{request["suite"]}__{request["task_id"]:02d}__{method}'
            case = run / 'cases' / key;case.mkdir(exist_ok=True)
            result_path = run / 'results' / (key + '.json')
            with (case / 'case.lock').open('a') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if result_path.exists():
                    previous = load(result_path)
                    if previous['protocol_id'] != protocol['protocol_id']:
                        raise ValueError('Cached plan uses a different immutable protocol')
                    if previous['status'] != 'INFRASTRUCTURE_ERROR' or not args.retry_infrastructure:
                        continue
                    dump(case / f'infrastructure_attempt_{time.time_ns()}.json', previous)
                budget = copy.deepcopy(config['budget'])
                budget.update(config.get('method_budgets', {}).get(method, {}))
                client = create_client(args, config, args.model_tag, method, case / 'model_calls.jsonl')
                if method == 'hibt':
                    client.base.model_tag = config['model_settings'][args.model_tag].get('hibt_model_tag', 'libero_qwen')
                module = importlib.import_module('methods.' + method)
                row = build_plan_row(request, method, module, client, budget, demos, int(config.get('feedback_max_ticks', 128)))
                row.update(protocol_id=protocol['protocol_id'], model_tag=args.model_tag, phase=args.phase)
                dump(result_path, row)
                print(json.dumps({key: row[key] for key in ['request_id', 'method', 'status', 'planning_outcome', 'llm_calls']}), flush=True)
                if row['status'] == 'INFRASTRUCTURE_ERROR':
                    aggregate(run, protocol, len(selected));raise RuntimeError('Infrastructure attempt retained; root must repair before explicit same-case retry')
    aggregate(run, protocol, len(selected))


if __name__ == '__main__':
    main()
