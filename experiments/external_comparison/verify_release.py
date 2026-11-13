#!/usr/bin/env python3
"""Verify the published experiment package against its content manifest.

This checks released source and input files. Run the evaluation and analysis
scripts to generate and summarize your own measurements.
"""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parent


def main():
    manifest_path = ROOT / 'RELEASE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    errors = []
    expected = {}
    for item in manifest['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in expected:
            errors.append('Invalid or repeated manifest path: ' + str(rel))
            continue
        expected[str(rel)] = item
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            errors.append('Missing file or symbolic link: ' + str(rel))
            continue
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            errors.append('Content mismatch: ' + str(rel))
    actual = {
        str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts
        and p.suffix != '.pyc' and p != manifest_path
        and 'outputs' not in p.relative_to(ROOT).parts
    }
    report = {
        'status': 'PASS' if not errors else 'FAIL',
        'files_checked': len(expected),
        'scope': 'Released source/input file integrity; no saved experimental outcomes.',
        'additional_files_not_part_of_release': sorted(actual - set(expected)),
        'errors': errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
