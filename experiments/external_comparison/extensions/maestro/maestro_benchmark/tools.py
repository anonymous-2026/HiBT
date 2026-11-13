"""Benchmark-domain robot tools; no planner, closure, teacher or repair is used."""
from __future__ import annotations

import copy
import hashlib
import json
import re


class InfrastructureError(RuntimeError):
    """Unavailable model/environment service; never a semantic method score."""


def call(name, args): return f'{name}({", ".join(args)})'


class ToolError(ValueError): pass


class PlannerBudgetExhausted(RuntimeError):
    """Declared model input/output resource limit, retained as a semantic outcome."""


class BenchmarkBackend:
    def __init__(self, registry, max_skill_calls=100):
        self.registry = copy.deepcopy(registry)
        self.max_skill_calls, self.skill_calls = max_skill_calls, 0
        self.trace, self.annotations = [], []

    def _validate(self, name, args, kind):
        if not isinstance(name, str) or not isinstance(args, (list, tuple)) or any(not isinstance(x, str) for x in args):
            raise ToolError('Use a registered name and a list of exact object-name strings')
        schemas = self.registry[kind]
        if isinstance(schemas, list): schemas = {x['name']: x for x in schemas}
        canonical = next((x for x in schemas if x.lower() == name.lower()), None)
        if canonical is None: raise ToolError('Unknown registered ' + kind + ': ' + name)
        schema = schemas[canonical]
        arity = schema.get('arity', len(schema.get('arguments', schema.get('parameters', []))))
        if len(args) != arity: raise ToolError(f'{canonical} expects {arity} arguments')
        objects = self.registry['objects']
        if any(arg not in objects for arg in args): raise ToolError('Unknown object binding: ' + repr(args))
        return canonical, list(args)

    def check_predicate(self, name, args): raise NotImplementedError
    def snapshot(self): raise NotImplementedError
    def execute_skill(self, name, args): raise NotImplementedError
    def images(self): return []

    def list_skills(self):
        schemas = self.registry['actions']
        if isinstance(schemas, list): schemas = {x['name']: x for x in schemas}
        return [{'name': name, 'arguments': spec.get('arguments', spec.get('parameters', [])),
                 'preconditions': spec.get('precondition', spec.get('pre', [])),
                 'conditional_precondition': spec.get('conditional_precondition'),
                 'effects': spec.get('effect', spec.get('add', []))}
                for name, spec in schemas.items()]

    def capabilities(self):
        return {'robot': {'get_robot_state': self.snapshot, 'observe': self.snapshot,
                          'list_skills': self.list_skills,
                          'check_predicate': self.check_predicate,
                          'execute_skill': self.execute_skill},
                'image_editor': {'draw_point': self.draw_point, 'clear': self.annotations.clear}}

    def draw_point(self, position, image_index=0, label=''):
        if not self.images(): raise ToolError('This symbolic benchmark has no actual camera image to annotate')
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ToolError('Benchmark image point is [x,y] in actual image pixels, not a 3D world position')
        if type(image_index) is not int or not 0 <= image_index < 2 or len(str(label)) > 100:
            raise ToolError('Invalid image index/label')
        point = [float(x) for x in position]
        if any(not 0 <= x <= 10000 for x in point): raise ToolError('Invalid pixel point')
        if len(self.annotations) >= 100: raise ToolError('Image annotation budget exhausted')
        self.annotations.append({'point':point,'image_index':image_index,'label':str(label)})
        return True

    def _count(self, name, args):
        if self.skill_calls >= self.max_skill_calls: raise ToolError('Episode skill-call budget exhausted')
        self.skill_calls += 1


class AssemblyBackend(BenchmarkBackend):
    """Uses the caller's original WorldState after common canonicalization.

    The adapter does not canonicalize/repair a state, fill preconditions, choose
    actions, or read the evaluator goal. Invalid/disabled actions produce real
    feedback and leave the transition model unchanged.
    """
    def __init__(self, world, registry, max_skill_calls=100):
        super().__init__(registry, max_skill_calls)
        self.world = world

    def snapshot(self): return copy.deepcopy(self.world.to_json())

    def check_predicate(self, name, args):
        name, args = self._validate(name, args, 'predicates')
        return bool(self.world.check(name, args))

    def execute_skill(self, name, args):
        name, args = self._validate(name, args, 'actions');self._count(name, args)
        # WorldState itself evaluates every signed original-domain prerequisite.
        ok = bool(self.world.apply_action(name, args))
        result = {'status': 'success' if ok else 'failure', 'action': call(name, args),
                  'feedback': 'Applied original transition' if ok else 'Original action preconditions are not satisfied'}
        self.trace.append({**result, 'state': self.snapshot()})
        return result


class LiberoBackend(BenchmarkBackend):
    """Synchronous code tool over the common real, chunked VLA skill executor.

    NativeEpisode.execute is reused solely as a shared physical skill. MAESTRO
    does not call its BT run/update loop, inherited plan, or symbolic simulator.
    Each RUNNING result advances the real simulator; the program receives the
    terminal status, actual state and subsequent real RGB feedback.
    """
    def __init__(self, controller, actual_snapshot, max_skill_calls=100):
        super().__init__(controller.registry, max_skill_calls)
        self.controller, self.actual_snapshot = controller, actual_snapshot
        self._last_snapshot_sha = None

    def snapshot(self):
        c = self.controller
        try: state = self.actual_snapshot(c.env, c.request, c.registry, c.catalog)
        except Exception as exc: raise InfrastructureError('Actual state snapshot failed: ' + str(exc)) from exc
        digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        if digest != self._last_snapshot_sha:
            if hasattr(c, 'log'): c.log('maestro_actual_state', state=state, state_sha256=digest)
            self._last_snapshot_sha = digest
        # Also expose actual proprioception, like the original robot-state feedback.
        obs = c.obs or {}
        proprioception = {k: v.tolist() if hasattr(v, 'tolist') else v
                          for k, v in obs.items()
                          if k in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']}
        # Closed-world encoding of the same complete predicate truth assignment:
        # avoid repeating a large false/true table plus duplicate init_predicates
        # in every model turn. Every true atom and every binding error survives.
        return {'facts': state.get('facts', []), 'objects': self.registry['objects'],
                'robot_state': proprioception,
                'binding_errors': [x for x in state.get('predicate_values', []) if x.get('binding_error')],
                'state_encoding': 'Complete true facts; absent registered predicates are false, except reported unsupported bindings.'}

    def check_predicate(self, name, args):
        name, args = self._validate(name, args, 'predicates')
        self.controller.cache.clear()
        try: return bool(self.controller.check(name, args))
        except Exception as exc: raise InfrastructureError('Actual predicate evaluation failed: ' + str(exc)) from exc

    def images(self):
        from PIL import Image, ImageDraw
        c = self.controller;obs = c.obs or {};result = []
        for name in ['agentview_image', 'robot0_eye_in_hand_image']:
            if name in obs:
                frame = obs[name]
                transform = c.config.get('camera_transform', 'rotate180')
                if transform == 'rotate180': frame = frame[::-1, ::-1]
                elif transform == 'flip_y': frame = frame[::-1, :]
                elif transform != 'raw': raise InfrastructureError('Unknown camera transform ' + transform)
                result.append(Image.fromarray(frame.copy()).convert('RGB'))
        if not result: raise InfrastructureError('No real LIBERO camera observation available')
        for annotation in self.annotations:
            if annotation['image_index'] >= len(result): continue
            image = result[annotation['image_index']];x,y = annotation['point']
            if x >= image.width or y >= image.height: continue
            draw = ImageDraw.Draw(image);draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2)
            if annotation['label']: draw.text((x+4,y+4),annotation['label'],fill='red')
        return result

    def execute_skill(self, name, args):
        name, args = self._validate(name, args, 'actions');self._count(name, args)
        c = self.controller
        started = c.steps
        ticks = 0
        while True:
            ticks += 1
            if ticks > c.config['max_steps'] + 100:
                raise InfrastructureError('Shared skill returned RUNNING without bounded physical progress')
            c.cache.clear()
            try: status = c.execute(name, args)
            except Exception as exc: raise InfrastructureError('Physical skill infrastructure failure: ' + str(exc)) from exc
            if status in ['success', 'failure']: break
            if status != 'running': raise InfrastructureError('Unknown shared skill status: ' + str(status))
            if c.steps >= c.config['max_steps']: status = 'failure';break
        result = {'status': status, 'action': call(name, args), 'physical_steps': c.steps - started,
                  'total_physical_steps': c.steps, 'reason': c.reason,
                  'semantic_errors': copy.deepcopy(c.semantic_errors)}
        self.trace.append(result)
        return result
