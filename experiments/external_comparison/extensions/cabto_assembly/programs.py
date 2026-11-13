"""Lossless compiler for the declared generated robot-API Python subset.

The model emits executable policy code; unsupported constructs trigger feedback.
No arbitrary exec, primitive addition, semantic repair or missing-step fallback.
"""
from __future__ import annotations
import ast
import copy
from assembly_registry import call, parse_call, ground_registry_actions


class ProgramError(ValueError):
    pass


def literal(text, registry, *, allow_negative=True):
    if not isinstance(text, str):
        raise ProgramError('Predicate must be an explicit grounded string')
    negative = text.startswith('!')
    if negative and not allow_negative:
        raise ProgramError('Add/delete lists contain positive atoms, not signed expressions')
    name, args = parse_call(text.lstrip('!'))
    candidates = [x for x in registry['predicates'] if x.lower() == name.lower()]
    if len(candidates) != 1:
        raise ProgramError('Unknown predicate: ' + name)
    name = candidates[0]
    spec = registry['predicates'][name]
    if len(args) != spec.get('arity', len(spec.get('arguments', []))) or set(args) - set(registry['objects']):
        raise ProgramError('Illegal predicate arity/object binding: ' + text)
    return ('!' if negative else '') + call(name, args)


def compile_program(source, registry):
    parsed = ast.parse(source)
    functions = [x for x in parsed.body if isinstance(x, ast.FunctionDef)]
    if len(parsed.body) != 1 or len(functions) != 1 or functions[0].name != 'policy':
        raise ProgramError('Return exactly def policy(api): with no imports/top-level code')
    function = functions[0]
    if [x.arg for x in function.args.args] != ['api'] or function.decorator_list:
        raise ProgramError('Policy signature must be policy(api) without decorators')
    legal_actions = {entry['name'] for entry in ground_registry_actions(registry)}
    operations = []
    def value(node):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError) as exc:
            raise ProgramError('Use explicit literal API names/object arguments') from exc
    def api_call(node, expected=None):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'api':
            raise ProgramError('Only declared api.skill/api.check/api.require calls are allowed')
        name = node.func.attr
        if expected and name != expected:
            raise ProgramError('Expected api.' + expected)
        if node.keywords:
            raise ProgramError('Use positional arguments in the declared policy API')
        return name, [value(arg) for arg in node.args]
    def predicate(test):
        negated = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
        call_node = test.operand if negated else test
        _, args = api_call(call_node, 'check')
        if len(args) != 1:
            raise ProgramError('api.check requires one grounded predicate')
        atom = literal(args[0], registry)
        if negated:
            atom = atom[1:] if atom.startswith('!') else '!' + atom
        return atom
    def condition(atom):
        result = {'name': 'precondition: ' + atom.lstrip('!')}
        if atom.startswith('!'):
            result['negated'] = True
        return result
    def sequence(children):
        return {'name': 'sequence: generated CABTO program', 'control_mode': 'memory', 'children': children}
    def statements(body):
        children = []
        for statement in body:
            if isinstance(statement, ast.Expr):
                if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                    continue  # Python docstring has no robot side effect.
                name, args = api_call(statement.value)
                if name == 'skill' and len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], list):
                    signature = call(args[0], args[1])
                    if signature not in legal_actions:
                        raise ProgramError('Unregistered robot skill binding: ' + signature)
                    children.append({'name': 'action: ' + signature})
                    operations.append({'api': 'skill', 'name': args[0], 'args': args[1]})
                elif name == 'require' and len(args) == 1:
                    atom = literal(args[0], registry)
                    children.append(condition(atom));operations.append({'api': 'require', 'predicate': atom})
                else:
                    raise ProgramError('Expression must call api.skill(name,args) or api.require(predicate)')
            elif isinstance(statement, ast.If):
                atom = predicate(statement.test)
                opposite = atom[1:] if atom.startswith('!') else '!' + atom
                children.append({'name': 'selector: generated if', 'control_mode': 'memory', 'children': [
                    sequence([condition(atom), statements(statement.body)]),
                    sequence([condition(opposite), statements(statement.orelse)])]})
            elif isinstance(statement, ast.Pass):
                continue
            else:
                raise ProgramError('Unsupported Python construct: ' + type(statement).__name__)
        return sequence(children)
    tree = statements(function.body)
    if not operations:
        raise ProgramError('Policy must contain an actual registered skill or condition check')
    return {'source': source, 'tree': tree, 'operations': copy.deepcopy(operations),
            'compiler': 'Declared robot API Python subset; imperative order/if preserved as memory BT controls'}
