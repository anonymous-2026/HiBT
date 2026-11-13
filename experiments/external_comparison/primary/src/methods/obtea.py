"""LLM goal interpretation plus the unchanged official IJCAI2024 OBTEA core.

Domain adaptation: shared action schemas, finite object grounding, closed-world
negative literal compilation, and lossless BT-node conversion. The learned
intent stage always runs, sees no evaluator goal, and applies grammar/semantic
reflective feedback as in the paper. No HiBT repair or preselected plan is used.
"""
from __future__ import annotations

from goal_contract import goal_visible_client, goal_user_turn

import contextlib
import copy
import importlib.util
import json
import re
import signal
import sys
import threading
import time
from pathlib import Path

from assembly_registry import call, ground_registry_actions, parse_call, state_facts

SOURCE_VERSION = 'DIDS-EI/LLM-OBTEA@52c52267b74b63db8e4003d0228b0819884f1e9f'
_OFFICIAL = Path(__file__).resolve().parents[2] / 'source_audit/obtea/official/OBTEA'


def _load_official():
    """Import vendored untouched files without importing any robot/API entrypoint."""
    tree_name = '_hibt_external_official_obtea_behavior_tree'
    core_name = '_hibt_external_official_obtea_core'
    if core_name in sys.modules:
        return sys.modules[core_name]
    spec = importlib.util.spec_from_file_location(tree_name, _OFFICIAL / 'BehaviorTree.py')
    tree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tree)
    sys.modules[tree_name] = tree
    old = sys.modules.get('BehaviorTree')
    sys.modules['BehaviorTree'] = tree
    try:
        spec = importlib.util.spec_from_file_location(core_name, _OFFICIAL / 'OptimalBTExpansionAlgorithm.py')
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        sys.modules[core_name] = core
    finally:
        if old is None:
            sys.modules.pop('BehaviorTree', None)
        else:
            sys.modules['BehaviorTree'] = old
    return core


class FormulaError(ValueError):
    pass


def _tokenize(text):
    text = text.replace('∧', '&').replace('∨', '|').replace('¬', '!')
    tokens = []
    while text.strip():
        text = text.lstrip()
        atom = re.match(r'([A-Za-z_][A-Za-z_0-9]*)\s*\(([^()]*)\)', text)
        if atom and atom[1].lower() not in {'and', 'or', 'not'}:
            name, args = parse_call(atom[0])
            tokens.append(('literal', call(name, args)))
            text = text[atom.end():]
            continue
        boolean = re.match(r'(and|or|not)\b', text, re.I)
        if boolean:
            tokens.append((boolean[1].lower(), boolean[1]))
            text = text[boolean.end():]
            continue
        if text[0] in '&|!~()':
            token = {'&': 'and', '|': 'or', '!': 'not', '~': 'not'}.get(text[0], text[0])
            tokens.append((token, text[0]))
            text = text[1:]
            continue
        raise FormulaError(f'Invalid formula text near {text[:60]!r}')
    return tokens


def formula_to_dnf(text: str, registry: dict, max_terms=256):
    """Safe WFF parser and DNF normalization; no eval or model-answer repair."""
    tokens = _tokenize(text)
    index = 0

    def unary():
        nonlocal index
        if index >= len(tokens):
            raise FormulaError('Unexpected end of formula')
        kind, value = tokens[index]
        index += 1
        if kind == 'not':
            return ('not', unary())
        if kind == '(':
            item = disjunction()
            if index >= len(tokens) or tokens[index][0] != ')':
                raise FormulaError('Missing closing parenthesis')
            index += 1
            return item
        if kind != 'literal':
            raise FormulaError(f'Expected predicate, got {value!r}')
        predicate, args = parse_call(value)
        if predicate not in registry['predicates']:
            raise FormulaError(f'Unknown predicate {predicate!r}; use the predicate whitelist')
        spec = registry['predicates'][predicate]
        arity = spec.get('arity', len(spec.get('arguments', [])))
        if len(args) != arity:
            raise FormulaError(f'{predicate} requires {arity} arguments separated by commas, received {len(args)}. '
                               'Use predicate(object_A, object_B) for a binary predicate; do not separate arguments only by spaces.')
        unknown = sorted(set(args) - set(registry['objects']))
        if unknown:
            raise FormulaError(f'Unknown objects {unknown}; use only the object whitelist')
        return ('literal', value)

    def conjunction():
        nonlocal index
        item = unary()
        while index < len(tokens) and tokens[index][0] == 'and':
            index += 1
            item = ('and', item, unary())
        return item

    def disjunction():
        nonlocal index
        item = conjunction()
        while index < len(tokens) and tokens[index][0] == 'or':
            index += 1
            item = ('or', item, conjunction())
        return item

    if not tokens:
        raise FormulaError('Empty goal formula')
    ast = disjunction()
    if index != len(tokens):
        raise FormulaError(f'Unexpected token {tokens[index][1]!r}')

    def normalize(node, negative=False):
        kind = node[0]
        if kind == 'literal':
            return [frozenset([('!' if negative else '') + node[1]])]
        if kind == 'not':
            return normalize(node[1], not negative)
        op = ('or' if kind == 'and' else 'and') if negative else kind
        left, right = normalize(node[1], negative), normalize(node[2], negative)
        if op == 'or':
            terms = left + right
        else:
            if len(left) * len(right) > max_terms:
                raise FormulaError(f'DNF exceeds {max_terms} terms')
            terms = [a | b for a in left for b in right]
        if len(terms) > max_terms:
            raise FormulaError(f'DNF exceeds {max_terms} terms')
        return terms

    terms = normalize(ast)
    unique = []
    for term in terms:
        if any(('!' + literal if not literal.startswith('!') else literal[1:]) in term for literal in term):
            continue
        if term not in unique:
            unique.append(term)
    if not unique:
        raise FormulaError('Goal formula is contradictory')
    return [set(term) for term in unique]


def _extract_formula(raw):
    text = raw.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3:
            raise FormulaError('Incomplete code fence')
        text = '\n'.join(lines[1:-1]).strip()
    if text.startswith('{'):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormulaError(f'Invalid JSON goal: {exc}') from exc
        if not isinstance(data.get('goal'), str):
            raise FormulaError('Expected JSON with string field "goal"')
        return data['goal']
    return text


def _assembly_goal_contract(request, registry):
    """Static domain API, independent of the requested/evaluator goal value."""
    return registry.get('domain') == 'assembly' or request.get('suite') == 'assembly'


def _validate_goal_contract(formula, dnf, request, registry):
    if not _assembly_goal_contract(request, registry):
        return
    tokens = _tokenize(formula)
    literals = [value for kind, value in tokens if kind == 'literal']
    if (len(literals) != 1 or any(kind not in {'literal', '(', ')'} for kind, _ in tokens)
            or len(dnf) != 1 or len(dnf[0]) != 1):
        raise FormulaError('Assembly goal interface requires exactly one positive grounded dynamic '
                           'condition explicitly requested by the instruction. Return one predicate '
                           'call only, without &, |, !, implicit consequences, initial-state facts, '
                           'or action preconditions. Re-interpret the same instruction; do not list '
                           'additional conditions.')
    name, _ = parse_call(literals[0])
    if registry['predicates'][name].get('static', False):
        raise FormulaError('Assembly goal interface requires a dynamic achieved-state condition, '
                           'not an immutable capability or compatibility predicate. Return exactly '
                           'one positive predicate call explicitly requested by the instruction.')


def _intent_messages(request, registry, budget):
    meanings = {'On': 'The first object is physically on the second object or region.',
                'In': 'The first object is physically inside the second container or region.',
                'Open': 'The container is open.', 'Close': 'The container is closed.',
                'Turnon': 'The appliance is on.', 'Turnoff': 'The appliance is off.'}
    predicates = [{'name': name, 'arity': spec.get('arity', len(spec.get('arguments', []))),
                   'arguments': spec.get('arguments', []), 'kind': spec.get('kind', 'condition'),
                   'static': spec.get('static', False),
                   'description': spec.get('description', meanings.get(name, name))}
                  for name, spec in registry['predicates'].items()]
    system = ('You interpret a robot user instruction as a goal condition, not an action plan. '
              'Use only the provided condition predicates and objects. Produce a well-formed '
              'formula using predicate(object, ...) calls, & for and, | for or, ! for not, '
              'and parentheses. Preserve all requested goals, alternatives and negation. '
              'Separate predicate arguments with commas: predicate(object_A, object_B). '
              'Distinguish immutable capability/compatibility conditions from achieved states. '
              'A request to perform an operation asks for its achieved state, not merely the '
              'capability to perform it. Include only end conditions actually requested by '
              'the user, rather than intermediate holding/tool conditions needed to get there. '
              'Return only JSON {"goal":"formula"}. Do not list actions, infer a sequence, '
              'or invent objects or predicates. State facts describe the initial state, not '
              'additional requested goals.')
    if _assembly_goal_contract(request, registry):
        system += (' This assembly domain has a single-goal interface: every instruction explicitly '
                   'requests exactly one positive grounded dynamic condition. Return that single '
                   'predicate call as the goal. Do not use &, |, ! or negation. Do not add implied '
                   'consequences (such as a holding state implying non-emptiness), extra conditions '
                   'to make an empty state explicit, initial-state facts, or prerequisites. The '
                   'planner handles prerequisites; the goal records only the explicitly requested '
                   'end condition.')
    messages = [{'role': 'system', 'content': system}]
    examples = budget.get('intent_examples', registry.get('intent_examples', []))
    for item in examples:
        messages.append(goal_user_turn(item['instruction'], item))
        messages.append({'role': 'assistant', 'content': json.dumps({'goal': item['goal']})})
    messages.append({'role': 'user', 'content': json.dumps({
        'instruction': request['instruction'], 'available_objects': registry['objects'],
        'condition_predicates': predicates,
        'initial_state': request['world_state']}, ensure_ascii=False)})
    return messages


@contextlib.contextmanager
def _deadline(seconds):
    if not seconds or threading.current_thread() is not threading.main_thread():
        yield
        return
    def expired(*_):
        raise TimeoutError(f'OBTEA exceeded {seconds}s planning deadline')
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _literal_tree(literal):
    if literal.startswith('!'):
        return {'name': 'inverter: not', 'children': [{'name': 'precondition: ' + literal[1:]}]}
    return {'name': 'precondition: ' + literal}


def _convert(node):
    if hasattr(node, 'content'):
        if node.type == 'act':
            if node.content is None:
                raise ValueError('Unexpected empty official action leaf')
            return {'name': 'action: ' + node.content.name}
        if node.type == 'cond':
            return {'name': 'sequence: conjunctive condition',
                    'children': [_literal_tree(x) for x in sorted(node.content)]}
        raise ValueError(f'Unsupported official leaf {node.type}')
    children = [_convert(c) for c in node.children]
    if node.type in {'act', 'cond'}:
        if len(children) != 1:
            raise ValueError('Unexpected official wrapper arity')
        return children[0]
    if node.type not in {'?', '>'}:
        raise ValueError(f'Unsupported official control {node.type}')
    return {'name': ('selector: ' if node.type == '?' else 'sequence: ') + 'OBTEA',
            'children': children}


def compile_goal(dnf, registry, budget):
    """Invoke the official core independently per disjunct, then cost-sort BTs."""
    started = time.perf_counter()
    core = _load_official()
    ground = ground_registry_actions(registry)
    positive_start = set(registry.get('facts', state_facts(registry['world_state'])))
    universe = set(positive_start)
    for term in dnf:
        universe.update(x.lstrip('!') for x in term)
    for item in ground:
        universe.update(x.lstrip('!') for field in ['pre', 'add', 'delete'] for x in item[field])
    start = positive_start | {'!' + atom for atom in universe - positive_start}
    actions = []
    for item in ground:
        # Explicit negative literals make closed-world STRIPS semantics available
        # to the unmodified set-based core, including negative goal conditions.
        add = set(item['add']) | {'!' + x for x in item['delete']}
        delete = set(item['delete']) | {'!' + x for x in item['add']}
        actions.append(core.Action(name=item['name'], pre=set(item['pre']),
                                   add=add, del_set=delete, cost=item.get('cost', 1)))
    subtrees = []
    details = []
    with _deadline(float(budget.get('planning_timeout_s', 30))):
        for term in dnf:
            algo = core.OptBTExpAlgorithm(verbose=False)
            algo.clear()
            official_tree, cost = algo.run_algorithm_selTree(start, set(term), actions,
                                                            merge_time=int(budget.get('merge_depth', 3)))
            algo.bt = official_tree
            subtrees.append((cost, _convert(official_tree)))
            details.append({'goal': sorted(term), 'cost': cost if cost != float('inf') else None,
                            'expanded_conditions': algo.expand_conds,
                            'traversed_states': algo.traversed_state_num,
                            'native_ptml': algo.get_ptml() if algo.bt is not None else None})
    subtrees.sort(key=lambda row: row[0])
    tree = {'name': 'selector: DNF alternatives by OBTEA cost',
            'children': [row[1] for row in subtrees]}
    return tree, {'ground_action_count': len(actions), 'literal_count': len(universe),
                  'planning_ms': (time.perf_counter() - started) * 1000,
                  'subgoals': details, 'compaction_depth': int(budget.get('merge_depth', 3))}


def interpret_goal(request: dict, registry: dict, llm, budget: dict) -> dict:
    """Shared genuine language interface only; never invokes symbolic planning."""
    llm = goal_visible_client(llm, request, registry)
    result = {'tree': None, 'trace': [], 'errors': [], 'method': 'LLM-goal-interpretation',
              'source_version': SOURCE_VERSION,
              'domain_adapter': 'Paper-based LLM WFF interpretation and grammar/semantic reflection only'}
    messages = _intent_messages(request, registry, budget)
    max_feedbacks = int(budget.get('intent_max_feedbacks', 5))
    max_calls = min(int(budget.get('max_calls', 12)), 1 + max_feedbacks)
    result['intent_config'] = {'few_shot_count': len(budget.get('intent_examples', registry.get('intent_examples', []))),
                               'max_feedbacks': max_feedbacks, 'max_calls': max_calls,
                               'goal_contract': 'single_positive_dynamic_atom' if _assembly_goal_contract(request, registry)
                                                else 'general_boolean_formula'}
    blacklisted = []
    dnf = None
    for index in range(max_calls):
        entry = {'stage': 'intent_understanding' if index == 0 else 'reflective_feedback',
                 'messages': copy.deepcopy(messages)}
        try:
            response = llm.complete(messages, max_new_tokens=int(budget.get('max_output_tokens', 2048)),
                                    temperature=float(budget.get('temperature', 0.0)),
                                    seed=int(budget.get('seed', 0)) + index)
            entry['response'] = response
            formula = _extract_formula(response['text'])
            candidate_dnf = formula_to_dnf(formula, registry)
            _validate_goal_contract(formula, candidate_dnf, request, registry)
            dnf = candidate_dnf
            entry['formula'] = formula
            entry['dnf'] = [sorted(term) for term in dnf]
            result['trace'].append(entry)
            result['interpreted_goal'] = formula
            result['interpreted_goal_dnf'] = entry['dnf']
            break
        except (FormulaError, ValueError, KeyError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            entry['validation_error'] = error
            result['trace'].append(entry)
            blacklisted.append(error)
            if 'response' not in entry:
                result['errors'].append(error)
                result['goal_prompt_events'] = copy.deepcopy(llm.goal_prompt_events)
                return result
            messages.append({'role': 'assistant', 'content': entry['response'].get('text', '')})
            messages.append({'role': 'user', 'content': 'Automatic grammar/semantic checker rejected the goal. '
                             + error + ' Correct the goal for the same instruction. Prior rejected issues: '
                             + json.dumps(blacklisted) + '. Return only valid JSON with field goal.'})
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            result['trace'].append(entry)
            result['errors'].append(entry['error'])
            result['goal_prompt_events'] = copy.deepcopy(llm.goal_prompt_events)
            return result
    if dnf is None:
        result['errors'].append('Intent interpretation exhausted the shared LLM-call budget')
    result['goal_prompt_events'] = copy.deepcopy(llm.goal_prompt_events)
    return result


def plan(request: dict, registry: dict, llm, budget: dict, feedback=None) -> dict:
    result = interpret_goal(request, registry, llm, budget)
    result.update(method='LLM-OBTEA',
                  domain_adapter='Paper-based LLM WFF interpretation/reflection; unmodified official OBTEA; complete-domain grounding; native BT conversion',
                  execution_contract='Re-tick native BT after each primitive until interpreted/evaluator goal or shared budget termination')
    if result['errors']:
        return result
    dnf = [set(term) for term in result['interpreted_goal_dnf']]
    try:
        tree, details = compile_goal(dnf, registry, budget)
        result['tree'] = tree
        result['trace'].append({'stage': 'official_OBTEA', **details})
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result
