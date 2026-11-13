"""Unchanged assembly feedback helpers extracted from the frozen comparison driver."""
import copy
from assembly_registry import call,parse_call
from native_runtime import NativeRuntime,_world_class

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
