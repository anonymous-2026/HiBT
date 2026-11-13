"""Original goal-specified HiBT interface -> actual trained neural backend.

The public final task goal and current state directly condition the original
Predictor/prototype/repair/compiler route. No extra LM goal recovery is added.
"""
from __future__ import annotations
import copy

from goal_contract import VERSION, validate_request

SOURCE_VERSION = 'Original HiBT ConceptBehaviorTreeGenerator; independently trained E6 shared-instance supervision'


def plan(request: dict, registry: dict, llm, budget: dict, feedback=None) -> dict:
    out = {'tree': None, 'trace': [], 'errors': [], 'method': 'HiBT',
           'source_version': SOURCE_VERSION,
           'domain_adapter': 'Public final task goal/state, then trained HiBT Predictor/prototype/repair/BT compilation',
           'execution_contract': 'Execute returned nativeBT without flattening its dependency control flow'}
    try:
        if not callable(getattr(llm, 'hibt', None)):
            raise ValueError('HiBT requires the actual trained neural .hibt backend; no generic-LLM fallback is permitted')
        target = validate_request(request, registry)
        suite = request.get('suite', registry.get('domain', 'assembly'))
        if (suite == 'assembly' or registry.get('domain') == 'assembly') and ' & ' in target:
            raise ValueError('Original assembly HiBT entrypoint requires one positive goal predicate')
        out['planning_goal'] = target
        out['goal_input_contract'] = VERSION
        out['trace'].append({'stage': 'provided_final_task_goal', 'planning_goal': target,
                             'input_contract': VERSION, 'source': 'public task specification'})
        response = llm.hibt(target, copy.deepcopy(request['world_state']), suite=suite,
                            request_id=str(request.get('sample_id', request.get('request_id', '<adhoc>'))))
        out['trace'].append({'stage': 'trained_hibt_synthesis', 'target_from_task_input': target,
                             'neural_response': response})
        if not isinstance(response, dict) or response.get('method') != 'HiBT':
            raise ValueError('Neural service did not identify its result as HiBT')
        if response.get('model_executed') is not True:
            raise ValueError('Neural service did not verify actual trained-model execution')
        if response.get('errors'):
            raise ValueError('HiBT neural backend failed: ' + str(response['errors']))
        tree = response.get('tree', response.get('behavior_tree'))
        if not isinstance(tree, dict) or not isinstance(tree.get('name'), str):
            raise ValueError('HiBT neural backend returned no native behavior tree')
        out['tree'] = tree
        out['neural_provenance'] = response.get('provenance', {})
        for key in ['decoded_plan', 'decode_report', 'planning_ms', 'neural_backend']:
            if key in response:
                out[key] = response[key]
    except Exception as exc:
        out['errors'].append(f'{type(exc).__name__}: {exc}')
    return out
