"""Invoke the pinned CABTO OBTEA core on generated action models only."""
from __future__ import annotations
import copy
import importlib
import random
import sys
import types
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parent / 'source_audit/pinned'
_CACHE = None


def official_core():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    import numpy as np
    prefix = 'btgym'
    existing = {k: v for k, v in sys.modules.items() if k == prefix or k.startswith(prefix + '.')}
    random_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        for name in list(existing):
            del sys.modules[name]
        for name, relative in [('btgym', 'btgym'), ('btgym.algos', 'btgym/algos'),
                               ('btgym.algos.bt_planning', 'btgym/algos/bt_planning')]:
            module = types.ModuleType(name)
            module.__path__ = [str(OFFICIAL / relative)]
            sys.modules[name] = module
        module = importlib.import_module('btgym.algos.bt_planning.OBTEA')
        action = importlib.import_module('btgym.algos.bt_planning.Action')
        _CACHE = module.OBTEA, action.Action
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                del sys.modules[name]
        sys.modules.update(existing)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    return _CACHE


def plan_models(models, initial_facts, goal_atoms, *, timeout_s=30, max_actions=128):
    """No registered transition rules are used to manufacture candidate models.

    Predicate strings are bijectively renamed to opaque propositional symbols;
    this prevents the original household-name conflict heuristics from imposing
    a different domain's ontology. Signed complements retain closed-world facts.
    """
    Algorithm, Action = official_core()
    atoms = {x.lstrip('!') for x in initial_facts | set(goal_atoms)}
    for model in models:
        atoms.update(x.lstrip('!') for field in ['pre', 'add', 'delete'] for x in model[field])
    symbol = {atom: 'cabto_p' + str(i) for i, atom in enumerate(sorted(atoms))}
    reverse = {}
    def encode(literal):
        value = ('cabto_not_' if literal.startswith('!') else '') + symbol[literal.lstrip('!')]
        reverse[value] = literal
        return value
    start = {encode(atom if atom in initial_facts else '!' + atom) for atom in sorted(atoms)}
    goal = {encode(atom) for atom in goal_atoms}
    actions = []
    for model in models:
        positive_add = set(model['add'])
        positive_delete = set(model['delete'])
        # Model lists describe positive add/delete effects. Explicit negative
        # preconditions are represented by complementary proposition symbols.
        add = {encode(x) for x in positive_add} | {encode('!' + x) for x in positive_delete}
        delete = {encode(x) for x in positive_delete} | {encode('!' + x) for x in positive_add}
        actions.append(Action(name=model['id'], pre={encode(x) for x in model['pre']},
                              add=add, del_set=delete, cost=model['cost']))
    algorithm = Algorithm(verbose=False, time_limit=float(timeout_s), output_just_best=True, bt_merge=True)
    algorithm.clear()
    tree, cost, timed_out = algorithm.run_algorithm_selTree(start, goal, actions, merge_time=3)
    state = set(start)
    trace = []
    complete = goal <= state
    for _ in range(max_actions):
        if complete:
            break
        status, selected = tree.tick(state)
        if status != 'running' or not hasattr(selected, 'pre') or not selected.pre <= state:
            break
        before = set(state)
        state = (state | selected.add) - selected.del_set
        trace.append({'model_id': selected.name, 'before': sorted(reverse.get(x, x) for x in before),
                      'after': sorted(reverse.get(x, x) for x in state)})
        complete = goal <= state
        if state == before:
            break
    def node(value):
        if value.type == 'cond' and not getattr(value, 'children', None):
            return {'kind': 'conditions', 'literals': sorted(reverse[x] for x in value.content)}
        if value.type == 'act' and not getattr(value, 'children', None):
            return {'kind': 'model', 'model_id': value.content.name}
        kind = {'?': 'selector', '>': 'sequence', 'cond': 'sequence', 'act': 'sequence'}[value.type]
        return {'kind': kind, 'children': [node(child) for child in value.children]}
    converted = node(tree)
    return converted, {'complete': complete, 'timed_out': bool(timed_out),
                       'cost': cost if cost != float('inf') else None,
                       'expanded_conditions': len(algorithm.expanded),
                       'expanded_condition_sets': [sorted(reverse.get(x, x) for x in condition)
                                                   for condition in algorithm.expanded if isinstance(condition, set)],
                       'symbolic_execution': trace, 'model_count': len(models),
                       'planner': 'Pinned CABTO last-branch official OBTEA',
                       'literal_renaming': 'bijective signed propositional encoding; no domain-rule insertion'}


def realize(tree, models, programs):
    """Replace each abstract action with its generated policy and effect checks."""
    by_id = {model['id']: model for model in models}
    def condition(literal, model_id=None):
        result = {'name': 'precondition: ' + literal.lstrip('!')}
        if literal.startswith('!'):
            result['negated'] = True
        if model_id:
            result['cabto_model_id'] = model_id
        return result
    def annotate(node, model_id):
        out = copy.deepcopy(node)
        out['cabto_model_id'] = model_id
        out['children'] = [annotate(x, model_id) for x in out['children']] if 'children' in out else None
        if out.get('children') is None:
            out.pop('children', None)
        return out
    if tree['kind'] == 'conditions':
        return {'name': 'sequence: CABTO planned conditions', 'children': [condition(x) for x in tree['literals']]}
    if tree['kind'] in {'selector', 'sequence'}:
        return {'name': tree['kind'] + ': CABTO formal plan', 'control_mode': 'reactive',
                'children': [realize(child, models, programs) for child in tree['children']]}
    model_id = tree['model_id']
    model = by_id[model_id]
    required_after = (set(model['pre']) | set(model['add'])) - set(model['delete'])
    required_after -= {'!' + literal for literal in model['add']}
    required_after |= {'!' + literal for literal in model['delete']}
    return {'name': 'sequence: CABTO grounded model ' + model_id, 'control_mode': 'memory',
            'cabto_model_id': model_id,
            'children': [*[condition(x, model_id) for x in model['pre']],
                         annotate(programs[model_id]['tree'], model_id),
                         {'name': 'sequence: CABTO effect consistency ' + model_id,
                          'cabto_model_id': model_id,
                          'children': [condition(x, model_id) for x in sorted(required_after)]}]}
