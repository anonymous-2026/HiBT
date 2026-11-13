"""A bounded Python AST interpreter: generated programs never use Python exec.

Only published tool capabilities can cross the interpreter boundary. There are
no Python object attributes, filesystem, network, imports other than math,
reflection, exception traceback objects, or arbitrary native callables.
"""
from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass


class SandboxViolation(RuntimeError):
    pass


class _Return(Exception):
    def __init__(self, value): self.value = value


class _Break(Exception): pass
class _Continue(Exception): pass


@dataclass
class Capability:
    function: object


@dataclass
class Namespace:
    members: dict


@dataclass
class UserFunction:
    node: ast.FunctionDef
    closure: dict


class Interpreter:
    """Persistent program variables with fresh per-program execution fuel."""
    def __init__(self, tools, *, max_nodes=20000, max_items=1000, max_output=12000):
        self.max_nodes, self.max_items, self.max_output = max_nodes, max_items, max_output
        self.output, self.depth, self.remaining = [], 0, max_nodes
        self.namespace = {name: Namespace({k: Capability(v) for k, v in funcs.items()})
                          for name, funcs in tools.items()}
        builtins = {'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                    'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
                    'all': all, 'any': any, 'sorted': sorted, 'list': list,
                    'dict': dict, 'tuple': tuple, 'set': set, 'enumerate': lambda x: list(enumerate(x)),
                    'zip': lambda *x: list(zip(*x)), 'range': self._range,
                    'print': self._print}
        self.namespace.update({k: Capability(v) for k, v in builtins.items()})
        self.math = Namespace({name: Capability(getattr(math, name)) for name in
                               ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
                                'sqrt', 'hypot', 'degrees', 'radians', 'floor', 'ceil', 'isclose']})
        self.math.members.update(pi=math.pi, e=math.e)
        self.protected = set(self.namespace) | {'math'}

    def _print(self, *args, sep=' ', end='\n'):
        text = str(sep).join(map(str, args)) + str(end)
        if sum(map(len, self.output)) + len(text) > self.max_output:
            raise SandboxViolation('Printed output budget exhausted')
        self.output.append(text)

    def _range(self, *args):
        result = range(*args)
        if len(result) > self.max_items: raise SandboxViolation('Range exceeds item limit')
        return list(result)

    def _bounded(self, value, depth=0):
        if depth > 20: raise SandboxViolation('Container nesting limit')
        if isinstance(value, (str, bytes)) and len(value) > 32000:
            raise SandboxViolation('String exceeds size limit')
        if isinstance(value, int) and value.bit_length() > 4096:
            raise SandboxViolation('Integer exceeds size limit')
        if isinstance(value, (list, tuple, set, dict)):
            if len(value) > self.max_items: raise SandboxViolation('Container exceeds item limit')
            for item in (list(value.keys()) + list(value.values()) if isinstance(value, dict) else value):
                self._bounded(item, depth + 1)
        return value

    def _fuel(self):
        self.remaining -= 1
        if self.remaining < 0: raise SandboxViolation('Program instruction budget exhausted')

    def execute(self, source):
        self.output, self.remaining = [], self.max_nodes
        if not isinstance(source, str) or len(source) > 32000:
            raise SandboxViolation('Actor code must be text of at most 32000 characters')
        tree = ast.parse(source)
        # Validate the entire program before any real tool can have a side effect.
        forbidden = (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom,
                     ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFor,
                     ast.Delete, ast.ImportFrom, ast.Raise, ast.Try, ast.Lambda)
        for node in ast.walk(tree):
            if isinstance(node, forbidden): raise SandboxViolation('Unsupported syntax: ' + type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise SandboxViolation('Private names are unavailable')
            if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                raise SandboxViolation('Private attributes are unavailable')
            if isinstance(node, ast.Import) and any(n.name != 'math' for n in node.names):
                raise SandboxViolation('Only the bounded math namespace can be imported')
        self._block(tree.body, self.namespace)
        return ''.join(self.output)

    def _block(self, nodes, env):
        for node in nodes: self._statement(node, env)

    def _assign(self, target, value, env):
        self._bounded(value)
        if isinstance(target, ast.Name):
            if target.id in self.protected: raise SandboxViolation('Cannot replace API/builtin name')
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value): raise ValueError('Unpacking arity mismatch')
            for key, item in zip(target.elts, value): self._assign(key, item, env)
        elif isinstance(target, ast.Subscript):
            obj, key = self._expr(target.value, env), self._expr(target.slice, env)
            if not isinstance(obj, (dict, list)): raise SandboxViolation('Mutable assignment requires a dict/list')
            obj[key] = value; self._bounded(obj)
        else: raise SandboxViolation('Assignment target unavailable')

    def _statement(self, node, env):
        self._fuel()
        if isinstance(node, ast.Expr): self._expr(node.value, env)
        elif isinstance(node, ast.Assign):
            value = self._expr(node.value, env)
            for target in node.targets: self._assign(target, value, env)
        elif isinstance(node, ast.AnnAssign): self._assign(node.target, self._expr(node.value, env), env)
        elif isinstance(node, ast.AugAssign):
            value = self._binary(node.op, self._expr(node.target, env), self._expr(node.value, env))
            self._assign(node.target, value, env)
        elif isinstance(node, ast.If): self._block(node.body if self._expr(node.test, env) else node.orelse, env)
        elif isinstance(node, (ast.For, ast.While)):
            values = iter(self._expr(node.iter, env)) if isinstance(node, ast.For) else None
            broke = False
            while True:
                self._fuel()
                if values is None:
                    if not self._expr(node.test, env): break
                else:
                    try: value = next(values)
                    except StopIteration: break
                    self._assign(node.target, value, env)
                try: self._block(node.body, env)
                except _Break: broke = True; break
                except _Continue: continue
            if not broke: self._block(node.orelse, env)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.args.vararg or node.args.kwarg:
                raise SandboxViolation('Function decorators/variadic arguments unavailable')
            self._assign(ast.Name(id=node.name), UserFunction(node, env), env)
        elif isinstance(node, ast.Return): raise _Return(self._expr(node.value, env) if node.value else None)
        elif isinstance(node, ast.Break): raise _Break()
        elif isinstance(node, ast.Continue): raise _Continue()
        elif isinstance(node, ast.Pass): pass
        elif isinstance(node, ast.Assert):
            if not self._expr(node.test, env): raise AssertionError(self._expr(node.msg, env) if node.msg else 'Program assertion failed')
        elif isinstance(node, ast.Import):
            for name in node.names: env[name.asname or name.name] = self.math
        else: raise SandboxViolation('Unsupported statement: ' + type(node).__name__)

    def _binary(self, op, left, right):
        if isinstance(op, ast.Pow) and (not isinstance(right, (int, float)) or abs(right) > 100):
            raise SandboxViolation('Exponent exceeds limit')
        if isinstance(op, ast.Mult):
            for seq, count in [(left, right), (right, left)]:
                if isinstance(seq, (str, list, tuple)) and isinstance(count, int) and len(seq) * max(0, count) > self.max_items:
                    raise SandboxViolation('Sequence repetition exceeds limit')
        functions = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                     ast.Pow: operator.pow}
        if type(op) not in functions: raise SandboxViolation('Operator unavailable')
        return self._bounded(functions[type(op)](left, right))

    def _comprehension(self, node, env, index=0):
        if index == len(node.generators):
            return [(self._expr(node.key, env), self._expr(node.value, env))] if isinstance(node, ast.DictComp) else [self._expr(node.elt, env)]
        gen = node.generators[index]
        if gen.is_async: raise SandboxViolation('Async comprehension unavailable')
        result = []
        for value in self._expr(gen.iter, env):
            self._fuel(); self._assign(gen.target, value, env)
            if all(self._expr(c, env) for c in gen.ifs): result.extend(self._comprehension(node, env, index + 1))
            self._bounded(result)
        return result

    def _expr(self, node, env):
        self._fuel()
        if isinstance(node, ast.Constant): return self._bounded(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env: raise NameError('Unavailable name: ' + node.id)
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self._expr(n, env) for n in node.elts]
            return self._bounded(tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values)
        if isinstance(node, ast.Dict): return self._bounded({self._expr(k, env): self._expr(v, env) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.Subscript): return self._expr(node.value, env)[self._expr(node.slice, env)]
        if isinstance(node, ast.Slice): return slice(*(self._expr(x, env) if x else None for x in [node.lower, node.upper, node.step]))
        if isinstance(node, ast.BinOp): return self._binary(node.op, self._expr(node.left, env), self._expr(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self._expr(node.operand, env)
            return not value if isinstance(node.op, ast.Not) else -value if isinstance(node.op, ast.USub) else +value
        if isinstance(node, ast.BoolOp):
            value = self._expr(node.values[0], env)
            for n in node.values[1:]:
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value): break
                value = self._expr(n, env)
            return value
        if isinstance(node, ast.Compare):
            current = self._expr(node.left, env)
            funcs = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                     ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                     ast.In: lambda a,b:a in b, ast.NotIn: lambda a,b:a not in b,
                     ast.Is: operator.is_, ast.IsNot: operator.is_not}
            for op, other in zip(node.ops, node.comparators):
                value = self._expr(other, env)
                if type(op) not in funcs: raise SandboxViolation('Comparison unavailable')
                if not funcs[type(op)](current, value): return False
                current = value
            return True
        if isinstance(node, ast.IfExp): return self._expr(node.body if self._expr(node.test, env) else node.orelse, env)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            values = self._comprehension(node, dict(env))
            return dict(values) if isinstance(node, ast.DictComp) else set(values) if isinstance(node, ast.SetComp) else values
        if isinstance(node, ast.Attribute):
            value = self._expr(node.value, env)
            if isinstance(value, Namespace) and node.attr in value.members: return value.members[node.attr]
            allowed = {dict: {'get', 'keys', 'values', 'items', 'update', 'pop'},
                       list: {'append', 'extend', 'pop', 'index', 'count'},
                       str: {'lower', 'upper', 'strip', 'split', 'startswith', 'endswith', 'join', 'replace'}}
            if type(value) in allowed and node.attr in allowed[type(value)]: return Capability(getattr(value, node.attr))
            raise SandboxViolation('Attribute unavailable: ' + node.attr)
        if isinstance(node, ast.JoinedStr):
            return self._bounded(''.join(str(self._expr(n.value, env)) if isinstance(n, ast.FormattedValue) else n.value for n in node.values))
        if isinstance(node, ast.Call):
            function = self._expr(node.func, env)
            args = [self._expr(n, env) for n in node.args]
            kwargs = {n.arg: self._expr(n.value, env) for n in node.keywords}
            if None in kwargs: raise SandboxViolation('Keyword expansion unavailable')
            if isinstance(function, Capability):
                value = function.function(*args, **kwargs)
                owner = getattr(function.function, '__self__', None)
                if isinstance(owner, (list, dict, set)): self._bounded(owner)
                return self._bounded(value)
            if isinstance(function, UserFunction):
                self.depth += 1
                try:
                    if self.depth > 20: raise SandboxViolation('Function depth exceeds limit')
                    names = [x.arg for x in function.node.args.args]; local = dict(function.closure)
                    if len(args) > len(names) or any(k not in names for k in kwargs): raise TypeError('Function argument mismatch')
                    defaults = function.node.args.defaults
                    for name, value in zip(names[-len(defaults):], defaults): local[name] = self._expr(value, function.closure)
                    local.update(zip(names, args));local.update(kwargs)
                    if any(n not in local for n in names): raise TypeError('Missing function argument')
                    try: self._block(function.node.body, local)
                    except _Return as result: return result.value
                    return None
                finally: self.depth -= 1
            raise SandboxViolation('Only published capabilities and program functions can be called')
        raise SandboxViolation('Unsupported expression: ' + type(node).__name__)
