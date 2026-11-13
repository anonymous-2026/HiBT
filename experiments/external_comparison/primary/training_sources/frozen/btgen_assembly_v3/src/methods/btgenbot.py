"""BTGenBot (IROS2024), domain-bound XML generation and execution conversion.

This is an explicitly labeled domain adapter, not the original ROS2 validator.
The model must have BTGenBot LoRA/SFT provenance. XML control flow is preserved;
the converter never generates missing actions or repairs a plan.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

METHOD = 'BTGenBot'
SOURCE_VERSION = 'AIRLab-POLIMI/BTGenBot@fb6016f4f243bc71685a1df7d8272ff542e9863a'
SYSTEM = (
    'You will be provided a summary of a task performed by a robot, its current '
    'state, and the complete registered actions and conditions. Express the '
    'task as a BehaviorTree.CPP behavior tree in XML format. Return only one '
    '<root BTCPP_format="4" main_tree_to_execute="MainTree"> containing a '
    '<BehaviorTree ID="MainTree">. Use the registered leaf names and named '
    'argument ports exactly. Sequence and Fallback preserve their ordered '
    'children; conditions inspect the world and actions execute robot skills. '
    'Use only registered leaves and literal object names. Do not use blackboard '
    'variables, scripting, Parallel, or invented actions. Preserve necessary '
    'condition checks and control flow, and achieve the instruction.'
)


class InvalidTree(ValueError):
    pass


def _arguments(spec: Any) -> list[str]:
    if isinstance(spec, int):
        return [f'arg{i}' for i in range(spec)]
    if isinstance(spec, (list, tuple)):
        return [str(x.get('name', x.get('parameter', ''))) if isinstance(x, dict)
                else str(x).lstrip('?') for x in spec]
    if not isinstance(spec, dict):
        raise InvalidTree(f'Invalid registered schema: {spec!r}')
    for key in ['parameters', 'arg_names', 'args', 'arguments']:
        if key in spec:
            value = spec[key]
            if isinstance(value, dict):
                return [str(x).lstrip('?') for x in value]
            return _arguments(value)
    if 'template' in spec:
        match = re.fullmatch(r'[^()]+\((.*)\)', spec['template'])
        if not match:
            raise InvalidTree(f'Invalid schema template: {spec["template"]}')
        return [x.strip().lstrip('?') for x in match[1].split(',') if x.strip()]
    if 'arity' in spec:
        return _arguments(int(spec['arity']))
    raise InvalidTree(f'No argument specification: {spec!r}')


def registry_leaves(registry: dict) -> dict[str, dict]:
    """Resolve exact node names/ports without inventing task-specific options."""
    leaves: dict[str, dict] = {}
    for category in ['actions', 'predicates']:
        entries = registry.get(category, {})
        if isinstance(entries, list):
            entries = {x['name']: x for x in entries}
        if not isinstance(entries, dict):
            raise InvalidTree(f'{category} must be a name→schema mapping or list')
        for name, spec in entries.items():
            xml_name = spec.get('xml_name', name) if isinstance(spec, dict) else name
            if xml_name in leaves:
                raise InvalidTree(f'Duplicate XML leaf name: {xml_name}')
            leaves[xml_name] = {'name': name, 'kind': 'action' if category == 'actions' else 'precondition',
                                'ports': _arguments(spec), 'schema': spec}
    if not leaves:
        raise InvalidTree('Empty action/predicate registry')
    return leaves


def format_input(request: dict, registry: dict) -> str:
    """Same structured input for domain SFT and inference; evaluator goal omitted."""
    try:
        from prompt_registry import for_prompt
    except ModuleNotFoundError as exc:
        if exc.name != 'prompt_registry':
            raise
        for_prompt = None
    # The registry may carry cached state/facts for symbolic backends. State is
    # already present below; omit only these redundant copies, not schemas.
    visible = for_prompt(registry) if for_prompt is not None else registry
    domain = {key: value for key, value in visible.items()
              if key not in {'world_state', 'facts', 'ground_actions'} and not key.startswith('_')}
    return json.dumps({'instruction': request['instruction'],
                       'initial_state': request.get('world_state', {}),
                       'registered_domain': domain}, ensure_ascii=False, sort_keys=True)


def format_messages(request: dict, registry: dict, example: dict | None = None) -> list[dict]:
    messages = [{'role': 'system', 'content': SYSTEM}]
    if example is not None:
        if example.get('split') != 'train':
            raise InvalidTree('One-shot example must be explicitly from training split')
        messages.extend([{'role': 'user', 'content': format_input(example, example['registry'])},
                         {'role': 'assistant', 'content': example['output']}])
    messages.append({'role': 'user', 'content': format_input(request, registry)})
    return messages


def extract_xml(text: str) -> str:
    """Official notebook takes the last complete root in generated output."""
    matches = re.findall(r'<root\b[^>]*>[\s\S]*?</root\s*>', text)
    if not matches:
        raise InvalidTree('Generation contains no complete <root> behavior tree')
    xml = matches[-1]
    if '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper():
        raise InvalidTree('DTD/entity declarations are unsupported')
    return xml


def xml_to_tree(xml: str, registry: dict) -> dict:
    leaves = registry_leaves(registry)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise InvalidTree(f'Malformed XML: {exc}') from exc
    if root.tag != 'root':
        raise InvalidTree('Expected root XML element')
    if root.attrib.get('BTCPP_format', '4') not in {'3', '4'}:
        raise InvalidTree('Unsupported BTCPP format')
    definitions = {x.attrib.get('ID', ''): x for x in root if x.tag == 'BehaviorTree'}
    if not definitions or len(definitions) != sum(x.tag == 'BehaviorTree' for x in root):
        raise InvalidTree('Missing or duplicate BehaviorTree IDs')
    main = root.attrib.get('main_tree_to_execute')
    if main is None and len(definitions) == 1:
        main = next(iter(definitions))
    if main not in definitions:
        raise InvalidTree('Unknown main tree ID')

    def convert(node: ET.Element, ancestors: tuple[str, ...]) -> dict:
        tag, attrs, children = node.tag, dict(node.attrib), list(node)
        if tag == 'SubTree':
            if children:
                raise InvalidTree('SubTree cannot contain inline children')
            target = attrs.get('ID')
            if target not in definitions or target in ancestors:
                raise InvalidTree('Missing or cyclic SubTree')
            if set(attrs) - {'ID', 'name'}:
                raise InvalidTree('SubTree blackboard remapping unsupported')
            result = tree_body(definitions[target], ancestors + (target,))
            result = dict(result, source_subtree=target)
            return result
        controls = {'Sequence': ('sequence', 'sequence'),
                    'ReactiveSequence': ('sequence', 'reactive_sequence'),
                    'SequenceWithMemory': ('sequence', 'sequence_with_memory'),
                    'Fallback': ('selector', 'fallback'),
                    'ReactiveFallback': ('selector', 'reactive_fallback')}
        if tag in controls:
            if not children:
                raise InvalidTree(f'Empty {tag}')
            if set(attrs) - {'name'}:
                raise InvalidTree(f'Unsupported {tag} ports: {sorted(attrs)}')
            kind, mode = controls[tag]
            return {'name': f'{kind}: {attrs.get("name", tag)}', 'control_mode': mode,
                    'children': [convert(c, ancestors) for c in children], 'native_xml_tag': tag}
        if tag in {'Inverter', 'RetryUntilSuccessful'}:
            if len(children) != 1:
                raise InvalidTree(f'{tag} must have one child')
            allowed = {'name'} if tag == 'Inverter' else {'name', 'num_attempts'}
            if set(attrs) - allowed:
                raise InvalidTree(f'Unsupported {tag} ports: {sorted(attrs)}')
            out = {'name': f'{"inverter" if tag == "Inverter" else "retry"}: {attrs.get("name", tag)}',
                   'children': [convert(children[0], ancestors)], 'native_xml_tag': tag}
            if tag == 'RetryUntilSuccessful':
                try:
                    attempts = int(attrs['num_attempts'])
                except (KeyError, ValueError) as exc:
                    raise InvalidTree('Retry requires literal num_attempts') from exc
                if not 1 <= attempts <= 100:
                    raise InvalidTree('Retry attempts outside bounded supported range1–100')
                out['num_attempts'] = attempts
            return out
        if children:
            raise InvalidTree(f'Unsupported control node: {tag}')
        if tag in {'Action', 'Condition'}:
            leaf_id = attrs.pop('ID', '')
            expected = 'action' if tag == 'Action' else 'precondition'
        else:
            leaf_id, expected = tag, None
        if leaf_id not in leaves:
            raise InvalidTree(f'Unregistered leaf: {leaf_id}')
        spec = leaves[leaf_id]
        if expected is not None and expected != spec['kind']:
            raise InvalidTree(f'Leaf type mismatch: {leaf_id}')
        attrs.pop('name', None)
        if set(attrs) != set(spec['ports']):
            raise InvalidTree(f'{leaf_id} expects ports{spec["ports"]}, got{sorted(attrs)}')
        values = [attrs[key] for key in spec['ports']]
        if any(not value or any(char in value for char in '{}(),') for value in values):
            raise InvalidTree('Leaf ports require literal object names')
        return {'name': f'{spec["kind"]}: {spec["name"]}({", ".join(values)})',
                'native_xml_tag': tag, 'native_xml_ports': dict(attrs)}

    def tree_body(node: ET.Element, ancestors: tuple[str, ...]) -> dict:
        if len(node) != 1:
            raise InvalidTree('BehaviorTree must contain exactly one root node')
        return convert(node[0], ancestors)

    return tree_body(definitions[main], (main,))


def tree_to_xml(tree: dict, registry: dict) -> str:
    """Serialize an existing supervision tree, preserving control and leaf order."""
    leaves = registry_leaves(registry)
    by_native = {(x['kind'], x['name']): (tag, x) for tag, x in leaves.items()}

    def convert(node: dict) -> ET.Element:
        kind, _, payload = node['name'].partition(':')
        kind, payload = kind.strip(), payload.strip()
        if kind in {'sequence', 'selector'}:
            mode = node.get('control_mode', 'sequence' if kind == 'sequence' else 'fallback')
            tags = {'sequence': 'Sequence', 'reactive_sequence': 'ReactiveSequence',
                    'sequence_with_memory': 'SequenceWithMemory', 'fallback': 'Fallback',
                    'reactive_fallback': 'ReactiveFallback'}
            if mode not in tags:
                raise InvalidTree(f'Unknown control mode: {mode}')
            el = ET.Element(tags[mode], {'name': payload})
            for child in node.get('children', []):
                el.append(convert(child))
            return el
        if kind in {'inverter', 'retry'}:
            attrs = {'name': payload}
            if kind == 'retry':
                attrs['num_attempts'] = str(node['num_attempts'])
            el = ET.Element('Inverter' if kind == 'inverter' else 'RetryUntilSuccessful', attrs)
            for child in node.get('children', []):
                el.append(convert(child))
            return el
        match = re.fullmatch(r'([\w]+)\((.*)\)', payload)
        if not match:
            raise InvalidTree(f'Cannot serialize leaf: {node["name"]}')
        native_kind = 'precondition' if kind in {'target', 'precondition', 'condition'} else kind
        key = native_kind, match[1]
        if key not in by_native:
            raise InvalidTree(f'Unregistered native leaf: {key}')
        tag, spec = by_native[key]
        args = [x.strip() for x in match[2].split(',') if x.strip()]
        if len(args) != len(spec['ports']):
            raise InvalidTree('Native leaf arity mismatch')
        return ET.Element(tag, dict(zip(spec['ports'], args)))

    root = ET.Element('root', {'BTCPP_format': '4', 'main_tree_to_execute': 'MainTree'})
    body = ET.SubElement(root, 'BehaviorTree', {'ID': 'MainTree'})
    body.append(convert(tree))
    xml = ET.tostring(root, encoding='unicode')
    xml_to_tree(xml, registry)  # reject unsupported or malformed supervision
    return xml


def _provenance(llm, budget: dict) -> dict:
    metadata = getattr(llm, 'metadata', {})
    if callable(metadata):
        metadata = metadata()
    provenance = dict(metadata or {})
    provenance.update(budget.get('btgenbot_adapter_provenance', {}))
    if budget.get('software_test_only'):
        provenance['software_test_only'] = True
        return provenance
    if provenance.get('adapter_method') != METHOD or not provenance.get('adapter_checkpoint'):
        raise InvalidTree('BTGenBot requires verified SFT/LoRA checkpoint provenance; vanilla LLM is not this baseline')
    return provenance


def plan(request: dict, registry: dict, llm, budget: dict, feedback=None) -> dict:
    out = {'method': METHOD, 'source_version': SOURCE_VERSION, 'tree': None,
           'native_xml': None, 'trace': [], 'errors': [],
           'domain_adapter': 'Registered-domain XML leaves; native BT control flow retained; no plan repair',
           'validator': 'Domain-port XML/registration validation followed by shared native-BT execution; original task-specific ROS2 validator not run'}
    try:
        out['adapter_provenance'] = _provenance(llm, budget)
        if int(budget.get('max_calls', 1)) < 1:
            raise InvalidTree('No generation calls allowed')
        messages = format_messages(request, registry, budget.get('one_shot_example'))
        completion = llm.complete(messages, max_new_tokens=int(budget.get('max_output_tokens', 2048)),
                                  temperature=float(budget.get('temperature', 0.0)), seed=int(budget.get('seed', 0)))
        out['trace'].append({'stage': 'sft_xml_generation', 'messages': messages, 'completion': completion})
        xml = extract_xml(completion['text'])
        out['native_xml'] = xml
        out['tree'] = xml_to_tree(xml, registry)
    except Exception as exc:
        out['errors'].append(f'{type(exc).__name__}: {exc}')
    return out
