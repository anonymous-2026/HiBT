"""Meaningful CPU-only closed-loop checks; mocks never count as experiment SR."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
SHARED = HERE.parent / 'common'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SHARED));sys.path.insert(0, str(HERE))
from assembly_registry import make_registry
from run_assembly_comparison import simulation_feedback
from programs import compile_program, ProgramError
import adapter
import run_cabto
from transport import VLMClient, BudgetExhausted


def main():
    state = {'objects': [{'name': 'dev_hand', 'properties': []}, {'name': 'dev_tool', 'properties': []},
                          {'name': 'dev_part', 'properties': []}],
             'constraints': [{'source': 'dev_tool', 'name': 'can_manipulate', 'target': 'dev_part'}],
             'relations': [{'source': 'dev_hand', 'name': 'hold', 'target': 'dev_tool'},
                           {'source': 'dev_tool', 'name': 'hold', 'target': 'dev_part'}]}
    request = {'sample_id': 'SOFTWARE_ONLY_CABTO', 'suite': 'assembly', 'instruction': 'Leave the tool empty.',
               'planning_goal': 'is_empty(dev_tool)', 'world_state': state}
    registry = make_registry(request);request['registry'] = registry
    model = {'id': 'put', 'pre': ['hold(dev_hand, dev_tool)', 'hold(dev_tool, dev_part)'],
             'add': ['is_empty(dev_tool)'], 'delete': ['hold(dev_tool, dev_part)'], 'cost': 1}
    code = 'def policy(api):\n    api.skill("put_down", ["dev_hand", "dev_tool", "dev_part"])'
    budget = {'max_calls': 36, 'max_output_tokens': 4096, 'seed': 0, 'max_rounds': 3, 'policy_samples': 3}
    class Mock:
        software_test_only = True
        def __init__(self, kind='normal'):
            self.kind, self.calls, self.model_calls, self.policy_calls = kind, [], 0, 0
        def complete(self, messages, **kwargs):
            self.calls.append({'messages': copy.deepcopy(messages), 'kwargs': copy.deepcopy(kwargs)})
            stage = kwargs['stage']
            if 'model' in stage:
                self.model_calls += 1
                chosen = copy.deepcopy(model)
                if self.kind == 'missing_library' and self.model_calls == 1:
                    value = {'models': []}
                else:
                    if self.kind == 'bad_effect_model' and self.model_calls == 1:
                        chosen['delete'] = []
                    value = {'models': [chosen]}
            else:
                self.policy_calls += 1
                generated = code
                if self.kind == 'bad_program' and self.policy_calls == 1:
                    generated = 'def policy(api):\n    api.skill("put_down", ["unknown", "dev_tool", "dev_part"])'
                if self.kind == 'bad_execution' and self.policy_calls == 1:
                    generated = 'def policy(api):\n    api.require("is_empty(dev_tool)")'
                value = {'code': generated}
            return {'text': json.dumps(value), 'model': 'SOFTWARE_MOCK', 'software_test_only': True}
    results = []
    for kind in ['normal', 'missing_library', 'bad_program', 'bad_execution', 'bad_effect_model']:
        client = Mock(kind)
        output = adapter.plan(copy.deepcopy(request), copy.deepcopy(registry), client, budget,
                              feedback=lambda tree: simulation_feedback(tree, state, {}))
        assert output['tree'] is not None, (kind, output['errors'], output['trace'])
        actual = simulation_feedback(output['tree'], state, {})
        assert actual['result'] == 'success', (kind, actual)
        assert any(event['stage'] == 'BT_planning_context' for event in output['trace'] if 'stage' in event)
        assert client.model_calls >= 1 and client.policy_calls >= 1
        if kind == 'missing_library':
            assert client.model_calls == 2
        if kind == 'bad_execution':
            assert client.model_calls == 1 and client.policy_calls == 2
        if kind == 'bad_effect_model':
            assert client.model_calls == 2 and client.policy_calls == 4
        results.append({'fixture': kind, 'software_only': True, 'proposal_calls': client.model_calls,
                        'code_sampling_calls': client.policy_calls, 'feedback_executes_generated_program': True})
    bad = ['import os\ndef policy(api):\n    pass',
           'def policy(api):\n    open("secret")',
           'def policy(api):\n    api.teleport("dev_part")',
           'def policy(api):\n    api.skill("put_down", ["unknown", "dev_tool", "dev_part"])']
    for source in bad:
        try:
            compile_program(source, registry)
        except (ProgramError, SyntaxError):
            pass
        else:
            raise AssertionError('Unsupported/illegal program accepted')
    conditional = 'def policy(api):\n    if not api.check("is_empty(dev_tool)"):\n        api.skill("put_down", ["dev_hand", "dev_tool", "dev_part"])\n    else:\n        pass'
    compiled = compile_program(conditional, registry)
    assert simulation_feedback(compiled['tree'], state, {})['result'] == 'success'
    dispatched = run_cabto.assembly_case(copy.deepcopy(request), {'target': request['planning_goal']},
        {'budget': budget, 'runtime': {'max_ticks': 128, 'max_actions': 128}, 'max_rounds': 3}, Mock())
    assert dispatched['success'] is True and dispatched['action_trace'][0]['action'] == 'put_down(dev_hand, dev_tool, dev_part)'
    with tempfile.TemporaryDirectory() as temporary:
        client = VLMClient('http://127.0.0.1:1', 'SOFTWARE_ONLY', 0, Path(temporary) / 'calls.jsonl')
        try:
            client.complete([])
        except BudgetExhausted:
            pass
        else:
            raise AssertionError('Global call ceiling bypassed')
    report = {'scope': 'CPU_SOFTWARE_MOCKS_ONLY_NO_REAL_MODEL_OR_ROBOT_RESULTS', 'all_pass': True,
              'checks': results, 'unsupported_code_no_exec': True, 'conditional_order_preserved': True,
              'assembly_driver_end_to_end_generated_policy_on_original_transition_model': True,
              'global_call_budget': True, 'real_model_calls': 0, 'GPU_processes_started': 0}
    output = HERE / 'tests/cpu_smoke.json';output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
