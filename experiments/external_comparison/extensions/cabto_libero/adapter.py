"""Complete CABTO benchmark port: model proposal, policy sampling, refinement.

The pinned official CABTO planner is one component. Model proposal and generated
robot-API policies always run; no oracle library or hand-written policy fallback.
"""
from __future__ import annotations
import copy
import json
import re
from assembly_registry import state_facts
from goal_contract import validate_request
from prompt_registry import for_prompt
from programs import compile_program, literal, ProgramError
from planner_bridge import plan_models, realize

METHOD = 'CABTO'
SOURCE_VERSION = 'DIDS-EI/CABTO@4cc89d501080015334f7e331471724b7cf6da7bc (last); arXiv:2603.16809v2 Algorithm2'
DOMAIN_ADAPTER = ('Complete proposal/official-planner/generated-policy/effect-feedback port; '
                  'canonical assembly or scene-grounded LIBERO APIs; fixed shared VLA motor carrier')


def planning_context(planning):
    return {key: copy.deepcopy(planning.get(key)) for key in
            ['complete', 'timed_out', 'cost', 'expanded_conditions', 'model_count', 'planner']} | {
                'selected_model_trace': [row['model_id'] for row in planning.get('symbolic_execution', [])]}


def execution_context(feedback):
    """Paper-style diagnostic projection; complete feedback remains in raw trace."""
    return {key: copy.deepcopy(feedback[key]) for key in
            ['result', 'status', 'failed_node', 'failed_path', 'final_node', 'world_state', 'current_state', 'error']
            if key in feedback} | {'executed_actions': [
                {key: row[key] for key in ['action', 'success'] if key in row}
                for row in feedback.get('action_trace', [])]}


def json_object(text, *, required_key=None):
    """Read the first explicit response object with the declared stage key.

    Model prose may contain an argument array before the fenced response object.
    Arrays/scalars are not this API's response and must not reach ``.get``.
    This is syntactic extraction only: no generated field/value is changed,
    completed or repaired. Malformed schemas still follow normal feedback.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            result, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and (required_key is None or required_key in result):
            return result
    suffix = '' if required_key is None else ' with required key ' + repr(required_key)
    raise ValueError('Model did not emit a JSON object' + suffix)


def facts(request, registry):
    if 'facts' in registry:
        return set(registry['facts'])
    if 'facts' in request['world_state']:
        return set(request['world_state']['facts'])
    return set(state_facts(request['world_state']))


def model_library(value, registry):
    if not isinstance(value, dict) or not isinstance(value.get('models'), list):
        raise ValueError('Expected {"models":[...]}')
    result, ids = [], set()
    for entry in value['models']:
        if not isinstance(entry, dict):
            raise ValueError('Each action-model entry must be a JSON object')
        identifier = entry.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z][A-Za-z_0-9]*', identifier) or identifier in ids:
            raise ValueError('Model ids must be unique explicit identifiers')
        ids.add(identifier)
        parsed = {'id': identifier, 'description': str(entry.get('description', ''))}
        for field in ['pre', 'add', 'delete']:
            if not isinstance(entry.get(field), list):
                raise ValueError('Action model needs list field ' + field)
            parsed[field] = sorted({literal(atom, registry, allow_negative=field == 'pre') for atom in entry[field]})
        if set(parsed['add']) & set(parsed['delete']):
            raise ValueError('An action cannot both add and delete the same atom')
        if any(('!' + atom) in parsed['pre'] for atom in parsed['pre'] if not atom.startswith('!')):
            raise ValueError('Contradictory model preconditions')
        parsed['cost'] = float(entry.get('cost', 1))
        if not 0 < parsed['cost'] < float('inf'):
            raise ValueError('Model cost must be finite and positive')
        result.append(parsed)
    return result


def _call(llm, messages, budget, trace, stage, images=None):
    response = llm.complete(messages, max_new_tokens=int(budget.get('max_output_tokens', 4096)),
                            temperature=float(budget.get('temperature', 0)), seed=int(budget.get('seed', 0)),
                            stage=stage, images=images or [])
    trace.append({'stage': stage, 'messages': copy.deepcopy(messages), 'response': copy.deepcopy(response),
                  'n_observation_images': len(images or [])})
    return response['text']


def _pipeline(request, registry, llm, budget, feedback, previous=None, execution_failure=None):
    goal = validate_request(request, registry)
    goal_atoms = goal.split(' & ')
    trace, errors = [], []
    initial_facts = facts(request, registry)
    images = request.get('_cabto_observation_images', [])
    previous = copy.deepcopy(previous or {})
    prior_models = previous.get('models', [])
    start_round = int(previous.get('rounds_used', 0))
    maximum = int(budget.get('max_rounds', 3))
    contexts = []
    if execution_failure:
        trace.append({'stage': 'actual_execution_feedback_full', 'feedback': copy.deepcopy(execution_failure)})
        projected = {key: copy.deepcopy(value) for key, value in execution_failure.items() if key != 'native_failure_trace'}
        native_trace = execution_failure.get('native_failure_trace') or []
        projected.update(native_event_count=len(native_trace), native_event_tail=copy.deepcopy(native_trace[-8:]))
        contexts.append({'type': 'actual_execution_effect_failure', **projected})
    output = {'method': METHOD, 'source_version': SOURCE_VERSION, 'domain_adapter': DOMAIN_ADAPTER,
              'tree': None, 'trace': trace, 'errors': errors, 'planning_goal': goal,
              'stages_required': ['model_proposal', 'BT_planning_context', 'policy_code_sampling',
                                  'effect_consistency', 'cross_level_refinement_when_needed'],
              'grounding_observation_mode': 'actual_scene_images_and_predicates' if images else 'symbolic_state_api',
              'reference_actions_or_BT_used': False}
    for round_index in range(start_round, maximum):
        proposal_payload = {'instruction': request['instruction'], 'public_final_goal': goal,
                            'initial_state': request['world_state'], 'complete_domain_and_robot_API': for_prompt(registry),
                            'previous_models': prior_models, 'planning_and_execution_context': contexts}
        system = ('You implement CABTO high-level model proposal and cross-level refinement. '
                  'Propose a complete action-model library whose formal BT can solve the given task. '
                  'Every model has id, description, pre, add, delete, cost. pre may include !predicate(...); '
                  'add/delete contain positive grounded predicates. Use only supplied scene predicates/objects. '
                  'These are candidate intended transitions, not a preselected action sequence. '
                  'Use planning failure/effect inconsistency context to revise the models. '
                  'Return only JSON {"models":[{"id":"a0","description":"...","pre":[...],"add":[...],"delete":[...],"cost":1}]}.')
        try:
            raw = _call(llm, [{'role': 'system', 'content': system},
                              {'role': 'user', 'content': json.dumps(proposal_payload, ensure_ascii=False)}],
                        budget, trace, 'model_proposal' if not contexts else 'cross_level_model_refinement', images)
            models = model_library(json_object(raw, required_key='models'), registry)
            if len(models) > int(budget.get('max_models', 16)):
                raise ValueError('Library exceeds the declared model-count allowance')
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            contexts.append({'type': 'model_proposal_validation_failure', 'error': str(exc)})
            trace.append(contexts[-1]);continue
        prior_models = models
        try:
            abstract_tree, planning = plan_models(models, initial_facts, goal_atoms,
                                                  timeout_s=float(budget.get('planning_timeout_s', 30)))
        except (ValueError, KeyError, TypeError) as exc:
            contexts.append({'type': 'formal_planning_error', 'error': str(exc)})
            trace.append(contexts[-1]);continue
        trace.append({'stage': 'BT_planning_context', 'round': round_index,
                      'abstract_tree': abstract_tree, 'planning': planning, 'models': models})
        if not planning['complete']:
            contexts.append({'type': 'incomplete_model_library', 'planning': planning_context(planning), 'abstract_tree': abstract_tree})
            continue
        programs, sampling_failures, policy_prompts, samples_used = {}, [], {}, {}
        for model in models:
            policy_context = []
            for sample_index in range(int(budget.get('policy_samples', 3))):
                prompt = {'action_model': model, 'task_language': request['instruction'],
                          'public_final_goal': goal, 'current_observed_state': request['world_state'],
                          'complete_robot_API': for_prompt(registry), 'planning_context': planning_context(planning),
                          'previous_policy_execution_context': contexts + policy_context}
                code_system = ('You are CABTO low-level policy sampler. Ground the proposed action model into '
                               'a Python policy composed of the available robot control APIs. Scene images, '
                               'when supplied, are actual current observations; use them for object binding. '
                               'The registered skills call a fixed robot motor policy and include all their '
                               'declared physical behavior. In LIBERO place_on/place_in are complete '
                               'pick-transfer-place skills, not release-only commands. '
                               'Return JSON {"code":"def policy(api):\\n    ..."}. '
                               'Allowed syntax: literal api.skill("registered_name",["object",...]); '
                               'api.require("predicate(objects)"); if api.check("predicate(objects)"): ... '
                               'else: ...; if not api.check(...): ...; pass. '
                               'Preserve intended ordered calls. No imports, assignments, external I/O, '
                               'arbitrary Python, teleportation or direct state writes. Do not invent APIs. '
                               'Use execution/compilation feedback to sample an improved program.')
                policy_prompts[model['id']] = (code_system, copy.deepcopy(prompt))
                samples_used[model['id']] = sample_index + 1
                try:
                    response = _call(llm, [{'role': 'system', 'content': code_system},
                                           {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)}],
                                     budget, trace, 'policy_code_sampling:' + model['id'] + ':' + str(sample_index), images)
                    parsed = json_object(response, required_key='code')
                    code = parsed.get('code')
                    if not isinstance(code, str):
                        raise ProgramError('Policy response needs a code string')
                    programs[model['id']] = compile_program(code, registry)
                    trace.append({'stage': 'policy_grounded', 'model_id': model['id'],
                                  'sample_index': sample_index, **copy.deepcopy(programs[model['id']])})
                    break
                except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                    policy_context.append({'type': 'generated_policy_validation_failure', 'error': str(exc),
                                           'model_id': model['id'], 'sample_index': sample_index})
                    trace.append(policy_context[-1])
            if model['id'] not in programs:
                sampling_failures.extend(policy_context)
        if sampling_failures:
            contexts.append({'type': 'no_consistent_policy_sample', 'failures': sampling_failures})
            continue
        tree = realize(abstract_tree, models, programs)
        output['tree'] = tree
        state = {'models': models, 'programs': programs, 'rounds_used': round_index + 1,
                 'abstract_tree': abstract_tree, 'planning': planning}
        output['cabto_state'] = state
        if feedback is not None:
            actual = feedback(copy.deepcopy(tree))
            trace.append({'stage': 'effect_consistency', 'source': 'shared transition/scene feedback, no evaluator oracle',
                          'feedback': copy.deepcopy(actual), 'round': round_index})
            status = actual.get('result', actual.get('status'))
            # Sample another policy for the *same* model before cross-level
            # model refinement, using actual failed-effect context (Alg2:16–25).
            while status not in {'success', True}:
                failed = actual.get('failed_node') or actual.get('final_node') or {}
                model_id = failed.get('cabto_model_id')
                if model_id not in programs or samples_used[model_id] >= int(budget.get('policy_samples', 3)):
                    break
                code_system, retry_prompt = copy.deepcopy(policy_prompts[model_id])
                retry_prompt['previous_policy_execution_context'].append({
                    'failed_effect_feedback': execution_context(actual), 'previous_code': programs[model_id]['source']})
                sample_index = samples_used[model_id]
                samples_used[model_id] += 1
                try:
                    response = _call(llm, [{'role': 'system', 'content': code_system},
                                           {'role': 'user', 'content': json.dumps(retry_prompt, ensure_ascii=False)}],
                                     budget, trace, 'policy_code_resampling:' + model_id + ':' + str(sample_index), images)
                    programs[model_id] = compile_program(json_object(response, required_key='code')['code'], registry)
                    tree = realize(abstract_tree, models, programs)
                    output['tree'] = tree
                    state['programs'] = programs
                    actual = feedback(copy.deepcopy(tree))
                    trace.append({'stage': 'effect_consistency_after_policy_resampling', 'model_id': model_id,
                                  'sample_index': sample_index, 'feedback': copy.deepcopy(actual),
                                  'generated_policy': copy.deepcopy(programs[model_id])})
                    status = actual.get('result', actual.get('status'))
                except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                    trace.append({'stage': 'policy_resampling_validation_failure', 'model_id': model_id,
                                  'sample_index': sample_index, 'error': str(exc)})
            if status not in {'success', True}:
                contexts.append({'type': 'policy_execution_or_model_effect_inconsistency',
                                 'feedback': execution_context(actual),
                                 'generated_programs': {key: value['source'] for key, value in programs.items()},
                                 'planning': planning_context(planning)})
                continue
        output['errors'] = []
        return output
    errors.append('CABTO exhausted its declared grounding/refinement rounds without a complete consistent grounded library')
    output['tree'] = None
    output.setdefault('cabto_state', {'models': prior_models, 'rounds_used': maximum})
    return output


def plan(request, registry, llm, budget, feedback=None):
    return _pipeline(request, registry, llm, budget, feedback)


def update(request, registry, llm, budget, feedback=None, tree=None, failed_node=None,
           runtime_state=None, previous=None, failure_trace=None, **kwargs):
    context = {'failed_node': copy.deepcopy(failed_node), 'previous_tree': copy.deepcopy(tree),
               'actual_current_state': copy.deepcopy(runtime_state), 'native_failure_trace': copy.deepcopy(failure_trace)}
    return _pipeline(request, registry, llm, budget, feedback, previous=previous,
                     execution_failure=context)
