"""Independent, resumable MAESTRO benchmark episodes, no GPU/model launch.

The CLI is a CPU client. The coordinator starts approved model/policy servers.
Use immutable copies of this package, upstream source and input/config files
for a formal run. Every result keeps the full program/tool/feedback transcript.
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
import sys
import time

from .runner import RunConfig, run_episode, UPSTREAM
from .tools import AssemblyBackend, LiberoBackend, InfrastructureError
from .transport import OpenAIChat


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def dump(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=lambda x:x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n');tmp.replace(path)
def append(path, value):
    with Path(path).open('a') as stream: stream.write(json.dumps(value, default=str) + '\n')
def request_id(row): return row.get('request_id', row.get('sample_id'))


def _import_source(folder, name):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(folder).resolve()))
    return importlib.import_module(name)


def protocol(args, config):
    root = Path(__file__).resolve().parent
    value = {'method': 'MAESTRO', 'version': 'benchmark_code_agent_v1',
             'upstream_commit': 'e60d326233a238ddc496bf7a08acb7a278e9e66d',
             'config': config, 'config_sha256': sha(args.config),
             'mode': args.mode, 'phase': args.phase, 'endpoint': args.endpoint, 'model': args.model,
             'inputs': {str(Path(p).resolve()): sha(p) for p in [args.requests_file, args.manifest_file, args.evaluators_file] if p},
             'request_ids': args.request_id, 'init_ids': args.init_ids,
             'source': {str(p.relative_to(root)): sha(p) for p in root.rglob('*.py')},
             'official_source_manifest_sha256': sha(root.parent / 'source_audit/official_manifest.json')}
    # Recheck the actual pinned upstream bytes, not merely a stored manifest.
    official = load(root.parent / 'source_audit/official_manifest.json')
    for name, digest in official['files'].items():
        if sha(UPSTREAM / name) != digest: raise InfrastructureError('Official source changed: ' + name)
    if args.native_source:
        folder = Path(args.native_source);value['shared_native_source'] = {
            str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*.py'))}
        if args.mode == 'assembly':
            import inspect
            native = _import_source(args.native_source, 'native_runtime')
            artifact = Path(inspect.getfile(native._world_class())).parents[1]
            value['shared_original_assembly_runtime'] = {name: sha(artifact / name) for name in
                ['runtime/minimal_bt.py', 'data/world_definition.json']}
    if args.native_config: value['shared_native_config_sha256'] = sha(args.native_config)
    value['policy_port'] = args.policy_port
    value['policy_host'] = args.policy_host
    value['protocol_id'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def select_requests(args):
    rows = load(args.requests_file)
    if not isinstance(rows, list): raise ValueError('requests-file must contain a JSON list')
    ids = [request_id(x) for x in rows]
    if not all(ids) or len(ids) != len(set(ids)): raise ValueError('Missing/duplicate request IDs')
    if args.manifest_file:
        manifest = load(args.manifest_file)
        key = 'formal_ids' if args.phase == 'formal' else 'development_ids'
        allowed = manifest.get(key)
        if allowed is None: raise ValueError('Manifest requires explicit ' + key)
        rows = [x for x in rows if request_id(x) in set(allowed)]
        if len(rows) != len(set(allowed)): raise ValueError('Manifest request membership mismatch')
    elif args.phase == 'development' and args.mode == 'libero' and any(x['suite'] == 'libero_10' for x in rows):
        raise ValueError('LIBERO development requires an explicit non-Long development manifest')
    if args.request_id:
        selected = set(args.request_id)
        if not selected.issubset({request_id(x) for x in rows}): raise ValueError('Requested IDs outside phase manifest')
        rows = [x for x in rows if request_id(x) in selected]
    if not rows: raise ValueError('Empty selected request set')
    for row in rows:
        if args.phase == 'formal' and not row.get('planning_goal'):
            raise ValueError('Primary formal comparison requires the same public planning_goal')
    return rows


def _assembly(args, row, config, evaluators):
    native = _import_source(args.native_source, 'native_runtime')
    WorldState = native._world_class()
    world = WorldState(copy.deepcopy(row['world_state']))
    backend = AssemblyBackend(world, row['registry'], config.get('max_skill_calls', 100))
    target = evaluators[request_id(row)]['target']
    # This evaluator is deliberately outside backend/actor capabilities and chat.
    pred, params = native.parse_call(target)
    return backend, lambda:world.check(pred, params), lambda:None, {}


def _libero(args, row, config, output):
    native = _import_source(args.native_source, 'run_native_libero')
    shared = load(args.native_config)
    # Retain the exact shared physical-control parameters, policy and raw NL.
    required = ['max_steps', 'stage_steps', 'max_attempts', 'chunk_steps',
                'settle_steps', 'policy_seed', 'camera_transform', 'prompt_mode']
    if any(k not in shared for k in required): raise ValueError('Incomplete common physical protocol')
    if shared['openpi_client_source']: sys.path.insert(0, shared['openpi_client_source'])
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools, websocket_client_policy
    suite = benchmark.get_benchmark_dict()[row['suite']]()
    bddl_root = shared.get('bddl_root', get_libero_path('bddl_files'))
    env = native.make_env(suite, row['task_id'], OffScreenRenderEnv, bddl_root)
    try:
        initial = suite.get_task_init_states(row['task_id'])[args.current_init]
        obs = env.set_init_state(initial)
        for _ in range(shared['settle_steps']): obs, _, _, _ = env.step(native.DUMMY_ACTION)
        policy = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
        metadata = policy.get_server_metadata()
        builder = lambda ob,prompt,seed:native.policy_observation(ob,prompt,seed,image_tools,shared['camera_transform'])
        per_episode = {**shared, 'init_state_id': args.current_init}
        # No cached/precomputed plan is supplied: MAESTRO writes/executes programs.
        controller = native.NativeEpisode(env, row, row['registry'], {'method': 'maestro'}, per_episode,
                                          policy, builder, output / 'physical', None)
        controller.obs = obs
        backend = LiberoBackend(controller, native.actual_snapshot, config.get('max_skill_calls', 100))
        identity = {'initial_state_sha256': native.initial_state_sha256(initial),
                    'initial_state_dtype': initial.dtype.str, 'initial_state_shape': list(initial.shape),
                    'policy_metadata': metadata, 'policy_port': args.policy_port,
                    'shared_physical_protocol': {k: shared[k] for k in required}}
        return backend, env.check_success, env.close, identity
    except Exception:
        env.close();raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['assembly', 'libero'], required=True)
    p.add_argument('--phase', choices=['development', 'formal'], required=True)
    p.add_argument('--requests-file', type=Path, required=True)
    p.add_argument('--manifest-file', type=Path)
    p.add_argument('--evaluators-file', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--native-source', type=Path, required=True)
    p.add_argument('--native-config', type=Path)
    p.add_argument('--policy-port', type=int)
    p.add_argument('--policy-host', default='localhost')
    p.add_argument('--request-id', action='append')
    p.add_argument('--init-ids', type=int, nargs='+', default=[0])
    p.add_argument('--worker-count', type=int, default=1)
    p.add_argument('--worker-index', type=int, default=0)
    args = p.parse_args(argv)
    if not 0 <= args.worker_index < args.worker_count: raise ValueError('Invalid worker partition')
    if args.mode == 'assembly' and not args.evaluators_file: raise ValueError('Assembly requires private evaluator file')
    if args.mode == 'libero' and (not args.native_config or not args.policy_port): raise ValueError('LIBERO requires native config and explicit policy port')
    config = load(args.config);rows = select_requests(args);out = args.output_dir;out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'assembly' and args.phase == 'development' and config.get('canonicalize_development', True):
        common = _import_source(args.native_source, 'assembly_state_canonicalization')
        rows = [common.canonicalize_request(row) for row in rows]
    frozen = protocol(args, config)
    with (out / 'protocol.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if (out / 'protocol.json').exists():
            if load(out / 'protocol.json') != frozen: raise InfrastructureError('Immutable run protocol mismatch')
        else: dump(out / 'protocol.json', frozen)
    evaluators = {request_id(x): x for x in load(args.evaluators_file)} if args.evaluators_file else {}
    if args.mode == 'assembly' and any(request_id(x) not in evaluators for x in rows): raise ValueError('Missing private evaluator IDs')
    index = 0
    for init in (args.init_ids if args.mode == 'libero' else [0]):
        for row in rows:
            current = index;index += 1
            if current % args.worker_count != args.worker_index: continue
            key = request_id(row).replace(':', '_').replace('/', '_') + f'__i{init:02d}'
            case = out / 'cases' / key;case.mkdir(parents=True, exist_ok=True)
            with (case / 'case.lock').open('a') as handle:
                try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: continue
                result_file = case / 'result.json'
                if result_file.exists():
                    old = load(result_file)
                    if old['protocol_id'] != frozen['protocol_id']: raise InfrastructureError('Cached result uses another protocol')
                    if old['status'] == 'INFRASTRUCTURE_ERROR': raise InfrastructureError('Preserve/inspect infrastructure record before new attempt')
                    continue
                if (case / 'model_calls.jsonl').exists(): raise InfrastructureError('Incomplete attempt must be preserved before resuming')
                if protocol(args, config) != frozen: raise InfrastructureError('Source/input/config changed before episode')
                args.current_init = init;close = lambda:None;start = time.monotonic()
                try:
                    backend, evaluator, close, identity = (_assembly(args,row,config,evaluators) if args.mode == 'assembly'
                                                          else _libero(args,row,config,case))
                    chat = OpenAIChat(args.endpoint, args.model, seed=config.get('seed', 0),
                                      temperature=config.get('temperature', 0.1),
                                      max_tokens=config.get('max_tokens', 4096),
                                      timeout=config.get('timeout', 600), structured=config.get('structured', False),
                                      log=lambda event:append(case / 'model_calls.jsonl', event))
                    result = run_episode(row, backend, chat, evaluator,
                        RunConfig(config.get('max_attempts',20),config.get('max_planner_calls',21),config.get('interpreter_nodes',20000)),
                        event_sink=lambda event:append(case / 'agent_events.jsonl',event))
                    result.update(identity)
                    if args.mode == 'libero':
                        c = backend.controller;result.update(steps=c.steps,policy_calls=c.policy_calls,retries=c.retries,semantic_errors=c.semantic_errors)
                except Exception as exc:
                    result = {'method':'MAESTRO','status':'INFRASTRUCTURE_ERROR','success':None,
                              'infrastructure_error':type(exc).__name__+': '+str(exc)}
                finally: close()
                result.update(protocol_id=frozen['protocol_id'],request_id=request_id(row),init=init,
                              model=args.model,backbone_adaptation=args.model,
                              wall_s=time.monotonic()-start,input_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest())
                if result['status']=='INFRASTRUCTURE_ERROR':result['success']=None
                dump(result_file,result)
                print(json.dumps({k:result.get(k) for k in ['request_id','init','status','success','planner_calls','skill_calls','infrastructure_error']}),flush=True)
                if result['status']=='INFRASTRUCTURE_ERROR':raise InfrastructureError(result['infrastructure_error'])
    dump(out / f'worker_{args.worker_index}.completed.json', {
        'protocol_id': frozen['protocol_id'], 'worker_index': args.worker_index,
        'worker_count': args.worker_count, 'completed_without_infrastructure_error': True})


if __name__ == '__main__': main()
