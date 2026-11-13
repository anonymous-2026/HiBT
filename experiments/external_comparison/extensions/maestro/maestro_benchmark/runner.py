"""Port of MAESTRO's initial-plan / execute / visual-replan orchestration.

Source: penn-pal-lab/maestro@e60d326233a238ddc496bf7a08acb7a278e9e66d,
maestro/runner.py (FirstStepPlan, Replan, run_trial, VLMCodeExtractor).
The benchmark tool API and bounded interpreter replace hardware tools and exec;
planner judgments and the independent final evaluator are recorded separately.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from .sandbox import Interpreter, SandboxViolation
from .tools import InfrastructureError, PlannerBudgetExhausted


UPSTREAM = Path(__file__).resolve().parents[1] / 'upstream'
SCHEMAS = {
    'FirstStepPlan': {'describe_scene': str, 'reason_and_plan': str, 'actor_code': str},
    'Replan': {'reason_and_plan': str, 'success': bool, 'task_complete': bool, 'actor_code': str},
}


@dataclass(frozen=True)
class RunConfig:
    max_attempts: int = 20
    max_planner_calls: int = 21
    interpreter_nodes: int = 20000


def schema_json(name):
    return {'type': 'object', 'properties': {k: {'type': 'boolean' if t is bool else 'string'}
                                           for k, t in SCHEMAS[name].items()},
            'required': list(SCHEMAS[name]), 'additionalProperties': False}


def parse_response(value, name):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('```'):
            lines = text.splitlines();text = '\n'.join(lines[1:-1])
        value = json.loads(text)
    # The original Gemini chat asks for list[schema] and returns parsed[0].
    if isinstance(value, list) and len(value) == 1: value = value[0]
    if not isinstance(value, dict): raise ValueError('Expected structured ' + name)
    for key, kind in SCHEMAS[name].items():
        if key not in value or type(value[key]) is not kind:
            raise ValueError(f'{name}.{key} must be {kind.__name__}')
    return {key: value[key] for key in SCHEMAS[name]}


def public_request(request):
    """Allow only the task inputs, never current-task teacher/evaluator artifacts."""
    keys = ['request_id', 'instruction', 'language', 'planning_goal', 'suite', 'task_id']
    return {k: copy.deepcopy(request[k]) for k in keys if k in request}


def build_prompt(request, backend):
    original = (UPSTREAM / 'maestro/prompts/maestro_prompt.txt').read_text()
    # Retain the original receive/describe/plan/execute/verify procedure, replacing
    # the hardware-specific coordinate and tool-library sections explicitly.
    procedure = original.split('### **Coordinate System and Constraints**')[0]
    procedure = procedure.replace('Franka Panda robot arm', 'benchmark robot')
    api = '''
### Benchmark robot tools (the hardware-tool adapter for this evaluation)
Write real Python actor_code. Compose tools with variables, functions, lists,
dictionaries, for/while, conditions, assertions, and print. Variables persist
between programs. Only `import math` is allowed. Do not use open, exec, eval,
network, OS operations, arbitrary attributes, numpy, or undocumented tools.

robot.get_robot_state() / robot.observe(): the CURRENT observed state.
robot.list_skills(): the complete shared skill schemas, including prerequisites.
robot.check_predicate(name, [exact_object_names]): current boolean condition.
robot.execute_skill(name, [exact_object_names]): actually execute one registered
skill; returns status='success'/'failure' and execution feedback. This is not a
plan submission. An unavailable prerequisite is reported, not repaired for you.
Use all arguments in the documented order. A failed skill leaves actual feedback
for replanning. Print relevant results; check actual state before/after action.
No action sequence or task-specific template is supplied. Determine how to
compose the registered skills yourself. After a failed step, diagnose the
observed error, reach a suitable state using legal skills, and revise your code.
For multi-stage instructions, preserve every requested stage. Current images,
when supplied, are the real external and wrist camera views. No synthetic image
is substituted for a symbolic assembly state. The benchmark-specific tool set
replaces the released Franka/ZED geometry, grasp, and collision-planning APIs.
image_editor.draw_point([x,y], image_index=0, label=''): draw a pixel point on
the next real camera feedback image (0 external, 1 wrist); clear() removes marks.
This benchmark annotation uses image pixels rather than Franka 3D projection.
Keep code and printed feedback concise; retain all prior text in the dialogue.
'''
    return [procedure + api,
            'Shared domain interface: ' + json.dumps({
                **{key: backend.registry[key] for key in ['objects', 'predicates', 'region_parents', 'role_objects', 'domain_notes'] if key in backend.registry},
                'skills': backend.list_skills()}, sort_keys=True),
            'Goal Instruction: ' + request.get('instruction', request.get('language', '')),
            *(['Explicit final task goal (task input, not an action plan):\n' + request['planning_goal']]
              if request.get('planning_goal') else []),
            'History Code and failure analysis:\n']


def run_episode(request, backend, chat, evaluator, config=None, event_sink=None):
    """Complete generated-code execution; evaluator is not exposed to tools/chat.

    `chat.send(parts, schema_name)` returns (parsed-dict-or-JSON, raw-response).
    No fixed/saved actor program is accepted as a replacement for model output.
    The evaluator only measures the final actual environment; planner-reported
    completion controls stopping exactly as the released harness does.
    """
    config = config or RunConfig();request = public_request(request)
    interpreter = Interpreter(backend.capabilities(), max_nodes=config.interpreter_nodes)
    events, planner_calls, attempts, reported_complete = [], 0, 0, False
    reason, infrastructure = 'attempt_budget_exhausted', None
    def log(event, **fields):
        row = {'event': event, 'attempt': attempts, **fields};events.append(row)
        if event_sink: event_sink(row)
    def ask(parts, schema):
        nonlocal planner_calls
        while True:
            if planner_calls >= config.max_planner_calls: raise PlannerBudgetExhausted('Planner-call budget exhausted')
            planner_calls += 1
            value, raw = chat.send(parts, schema)
            log('planner_response', schema=schema, raw=raw)
            try: return parse_response(value, schema)
            except (ValueError, TypeError) as exc:
                # The released providers use structured output. An unconstrained
                # replacement VLM receives format-only feedback, with every
                # malformed generation retained and charged to the same budget.
                log('semantic_format_error', schema=schema, error=str(exc))
                parts = ['Your previous response did not match the required ' + schema +
                         ' JSON format: ' + str(exc) + '. Return the same intended response as one valid JSON object. '
                         'Do not execute anything or invent new state. Required schema: ' + json.dumps(schema_json(schema))]
    try:
        parts = build_prompt(request, backend)
        parts += ['State: ' + json.dumps(backend.snapshot(), sort_keys=True), *backend.images(),
                  'Finish Receive Instruction, Describe the Scene, and Steps Planning. Return the code for the first step.']
        plan = ask(parts, 'FirstStepPlan')
        for _ in range(config.max_attempts):
            attempts += 1
            code = plan['actor_code'];log('actor_code', code=code)
            try:
                output = interpreter.execute(code)
                execution_error = None
            except InfrastructureError: raise
            except Exception as exc:
                output = ''.join(interpreter.output)
                execution_error = type(exc).__name__ + ': ' + str(exc)
                output += '\nError executing code: ' + execution_error
            log('execution_feedback', stdout=output, error=execution_error,
                tool_calls=backend.skill_calls, state=backend.snapshot())
            parts = ['Goal Instruction: ' + request.get('instruction', request.get('language', '')),
                     f'Step {attempts} execution results: Code Output (judge the observed state and image, not printed claims): {output}',
                     'Final observed state: ' + json.dumps(backend.snapshot(), sort_keys=True),
                     'Final Image:', *backend.images(),
                     'Describe the execution result and compare it with the requested goal/substep. '
                     'If the whole goal is achieved, set task_complete=true; the next actor_code will not run. '
                     'Otherwise, if this step succeeded, write the next step. If it failed, diagnose the code/tool '
                     'failure, return to a feasible state and write better code. Do not skip stages.']
            if request.get('planning_goal'): parts += ['Explicit final task goal (task input, not an action plan):\n' + request['planning_goal']]
            plan = ask(parts, 'Replan')
            log('replan_judgment', **plan)
            if plan['task_complete']:
                reported_complete, reason = True, 'planner_reported_complete';break
            if backend.skill_calls >= backend.max_skill_calls:
                reason = 'skill_call_budget_exhausted';break
            c = getattr(backend, 'controller', None)
            if c is not None and c.steps >= c.config['max_steps']:
                reason = 'physical_step_budget_exhausted';break
    except PlannerBudgetExhausted as exc:
        log('semantic_resource_budget_exhausted', error=str(exc));reason = 'planner_resource_budget_exhausted'
    except InfrastructureError as exc:
        infrastructure, reason = str(exc), 'infrastructure_error'
    except Exception as exc:
        # Invalid structured model output is a method failure, not a service outage.
        log('semantic_error', error=type(exc).__name__ + ': ' + str(exc));reason = 'invalid_planner_response'
    try: success = bool(evaluator())
    except Exception as exc:
        success = False;infrastructure = 'Final evaluator infrastructure failure: ' + str(exc);reason = 'infrastructure_error'
    return {'method': 'MAESTRO', 'adaptation': 'shared-benchmark-tool-library',
            'upstream_commit': 'e60d326233a238ddc496bf7a08acb7a278e9e66d',
            'success': success, 'planner_reported_complete': reported_complete,
            'status': 'INFRASTRUCTURE_ERROR' if infrastructure else 'COMPLETED',
            'infrastructure_error': infrastructure, 'termination_reason': reason,
            'attempts': attempts, 'planner_calls': planner_calls,
            'skill_calls': backend.skill_calls, 'tool_trace': backend.trace,
            'final_state': backend.snapshot() if not infrastructure else None, 'events': events}
