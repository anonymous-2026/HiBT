#!/usr/bin/env python3
"""Summarize your own completed runs; saved-output analysis, not a new experiment."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def method_name(value):
    key = re.sub(r'[^a-z0-9]', '', str(value).lower())
    return {'llmbt': 'llmbt', 'btgenbot': 'btgenbot', 'llmasbtplanner': 'kios',
            'llmobtea': 'obtea', 'hibt': 'hibt', 'maestro': 'maestro',
            'cabto': 'cabto'}.get(key, key)


def read_rows(path, benchmark):
    """Read one JSON/JSONL or the canonical retained results directory only."""
    if path.is_file():
        files = [path]
    else:
        require(path.is_dir(), 'Input path does not exist: ' + str(path))
        nested = path / (benchmark + '_formal')
        if nested.is_dir():
            path = nested
        candidates = (['formal/*/result.json', 'cases/*/result.json', 'results/*.json', '*/result.json']
                      if benchmark == 'libero' else
                      ['results/*.json', 'cases/*/result.json', 'formal/*/result.json', '*/result.json'])
        files = []
        for pattern in candidates:
            files = sorted(p for p in path.glob(pattern)
                           if not any(part.startswith('_') for part in p.relative_to(path).parts))
            if files:
                break
        require(files, 'No canonical result files found: ' + str(path))
    rows = []
    for file in files:
        text = file.read_text()
        if file.suffix == '.jsonl':
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            values = value if isinstance(value, list) else value.get(benchmark, [value])
        require(isinstance(values, list) and all(isinstance(v, dict) for v in values),
                'Input must contain result objects')
        rows.extend(values)
    require(rows, 'No input records')
    return rows


def task_id(row):
    value = row.get('task', row.get('task_id'))
    if value is None:
        identifier = str(row.get('request_id', row.get('sample_id', '')))
        value = identifier.rsplit(':', 1)[-1]
    require(isinstance(value, (int, str)) and str(value).isdigit(), 'Missing physical task ID')
    return int(value)


def cost(row, metric):
    value = row.get(metric, row.get('physical_steps') if metric == 'steps' else None)
    require(type(value) is int and value >= 0, 'Missing/noninteger execution counter: ' + metric)
    return value


def fingerprint(row):
    keys = ('initial_state_sha256', 'initial_state_dtype', 'initial_state_shape')
    if not all(key in row for key in keys):
        return None
    require(isinstance(row[keys[0]], str) and re.fullmatch('[0-9a-f]{64}', row[keys[0]]),
            'Invalid physical initial-state hash')
    return row[keys[0]], row[keys[1]], tuple(row[keys[2]])


def summarize(inputs, allow_partial=False, reference='hibt', reference_model=None):
    groups = {}
    for cohort, benchmark, path in inputs:
        for row in read_rows(path, benchmark):
            require(row.get('status') == 'COMPLETED' and type(row.get('success')) is bool,
                    'Only COMPLETED Boolean outcomes are supported; resolve infrastructure errors first')
            require(not row.get('infrastructure_error'), 'Infrastructure errors are not semantic failures')
            method = method_name(row.get('method', cohort))
            if cohort != 'primary':
                require(method == cohort, 'Method differs from the requested additional system')
            require(method, 'Missing method')
            model = str(row.get('model_tag', row.get('model', 'unspecified')))
            key = cohort, benchmark, method, model
            group = groups.setdefault(key, {})
            if benchmark == 'assembly':
                case = row.get('sample_id', row.get('request_id'))
                require(isinstance(case, str) and case, 'Missing assembly sample ID')
            else:
                initial = row.get('init')
                require(type(initial) is int and 0 <= initial < 10, 'Physical init must be in 0..9')
                task = task_id(row)
                require(0 <= task < 10, 'Physical task must be in 0..9')
                case = str(row.get('suite', 'libero_10')), task, initial
            require(case not in group, 'Duplicate case within method/model group')
            group[case] = row
    summary = []
    for key, rows in sorted(groups.items()):
        cohort, benchmark, method, model = key
        expected = 60 if benchmark == 'assembly' else 100
        if not allow_partial:
            require(len(rows) == expected, 'Incomplete ' + '/'.join(key) + ': expected ' + str(expected))
            if benchmark == 'libero':
                require({(case[1], case[2]) for case in rows} ==
                        {(task, initial) for task in range(10) for initial in range(10)},
                        'Incomplete ten-task by ten-state physical membership')
        summary.append({'cohort': cohort, 'benchmark': benchmark, 'method': method,
                        'model': model, 'n': len(rows),
                        'successes': sum(row['success'] for row in rows.values()),
                        'failures': sum(not row['success'] for row in rows.values())})
    references = [(key, rows) for key, rows in groups.items()
                  if key[1] == 'libero' and key[2] == method_name(reference)
                  and (reference_model is None or key[3] == reference_model)]
    require(len(references) <= 1, 'Multiple physical reference groups; specify --reference-model')
    paired = []
    if references:
        refkey, ours = references[0]
        for key, theirs in sorted(groups.items()):
            if key[1] != 'libero' or key == refkey:
                continue
            shared = set(ours) & set(theirs)
            if not allow_partial:
                require(set(ours) == set(theirs), 'Physical task/state membership differs')
            verified = 0
            for case in shared:
                left, right = fingerprint(ours[case]), fingerprint(theirs[case])
                if left is not None and right is not None:
                    require(left == right, 'Different physical initial state for paired case')
                    verified += 1
            both = [(ours[c], theirs[c]) for c in sorted(shared)
                    if ours[c]['success'] and theirs[c]['success']]
            metrics = {}
            for metric in ('steps', 'policy_calls'):
                differences = [cost(a, metric) - cost(b, metric) for a, b in both]
                metrics[metric] = {
                    'reference_mean': statistics.mean(cost(a, metric) for a, b in both) if both else None,
                    'comparator_mean': statistics.mean(cost(b, metric) for a, b in both) if both else None,
                    'mean_reference_minus_comparator': statistics.mean(differences) if both else None,
                    'reference_lower': sum(d < 0 for d in differences),
                    'ties': sum(d == 0 for d in differences),
                    'reference_higher': sum(d > 0 for d in differences)}
            paired.append({'reference': {'method': refkey[2], 'model': refkey[3]},
                           'comparator': {'cohort': key[0], 'method': key[2], 'model': key[3]},
                           'shared_starts': len(shared), 'fingerprint_verified_starts': verified,
                           'both_success_n': len(both), 'costs': metrics})
    return {'analysis': 'User-supplied completed outputs only; no embedded measured results',
            'partial_inputs_allowed': allow_partial, 'success_counts': summary,
            'both_success_execution_cost': paired,
            'notes': ['All provided semantic failures remain in success denominators.',
                      'Cost differences use only shared starts where both methods succeed; failure early stopping is not an efficiency gain.',
                      'Reference-minus-comparator is negative for lower reference cost.',
                      'A pair is fingerprint-verified only when both records provide identical hash, dtype and shape; otherwise pairing uses supplied IDs only.',
                      'Model/backbone labels remain separate. Control steps and policy queries are not a controlled wall-clock speed benchmark.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in ('primary', 'maestro', 'cabto'):
        for benchmark in ('assembly', 'libero'):
            parser.add_argument('--' + cohort + '-' + benchmark, type=Path,
                                help='Your new output directory or JSON/JSONL result file')
    parser.add_argument('--allow-partial', action='store_true', help='Allow incomplete development outputs; default checks 60 assembly or 100 physical cases per group')
    parser.add_argument('--reference', default='hibt', help='Reference method for both-success physical costs (default: hibt)')
    parser.add_argument('--reference-model', help='Disambiguate multiple reference model tags')
    parser.add_argument('--output', type=Path, help='Write JSON here instead of stdout; must be outside input directories')
    args = parser.parse_args()
    inputs = [(cohort, benchmark, getattr(args, cohort + '_' + benchmark).resolve())
              for cohort in ('primary', 'maestro', 'cabto') for benchmark in ('assembly', 'libero')
              if getattr(args, cohort + '_' + benchmark) is not None]
    try:
        require(inputs, 'Provide at least one output path; see --help')
        if args.output:
            output = args.output.resolve()
            for _, _, path in inputs:
                require(output != path and (not path.is_dir() or path not in output.parents),
                        '--output must not alter input files/directories')
        result = summarize(inputs, args.allow_partial, args.reference, args.reference_model)
        text = json.dumps(result, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end='')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('ERROR:', str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
