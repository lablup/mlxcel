#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Archive explicitly verified benchmark runs; never infer verification.

Usage: archive_batched_decode.py --source-root RUNS --destination RAW --mapping runs.json
Mapping example (all paths in config/corpus are local input files):
{"runs": [{"label": "baseline-gpt-oss", "destination": "gpt-oss/baseline",
 "verified": true, "source_revision": "a29a2692", "expected_trace_rows": [1],
 "config": "/model/config.json",
 "corpus": "/corpora/architecture.md", "instrumentation_note": "none"}]}

A label may be nested, allowing an existing staged archive as source. Source
revisions are supplied facts, not inferred from the current checkout. Hashes
reflect the supplied input files at archive time; compare them to run-time
metadata before setting verified=true. Existing run destinations are refused.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def child(root, relative):
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Path must be below its root: {relative}')
    return path


def plan_run(source_root, destination, run):
    if run.get('verified') is not True or not run.get('source_revision'):
        raise ValueError('Each run requires verified=true and source_revision')
    source = child(source_root, run['label'])
    target = child(destination, run['destination'])
    if target.exists():
        raise ValueError(f'Refusing existing run destination: {target}')
    config, corpus = Path(run['config']).resolve(), Path(run['corpus']).resolve()
    if not config.is_file() or not corpus.is_file():
        raise ValueError(f'Missing config/corpus for {run["label"]}')
    expected_rows = run.get('expected_trace_rows')
    if (not isinstance(expected_rows, list) or not expected_rows
            or any(type(row) is not int or row < 1 for row in expected_rows)
            or len(set(expected_rows)) != len(expected_rows)):
        raise ValueError('Each run requires explicit distinct positive expected_trace_rows')
    phases = ['greedy', 'ladder', *[f'trace-b{row}' for row in expected_rows]]
    files = []
    for phase in phases:
        output, status = source / (phase + '.out'), source / (phase + '.status.json')
        stderr = source / (phase + '.err')
        if not all(path.is_file() for path in (output, status, stderr)):
            raise ValueError(f'Missing output/status/stderr for {source}/{phase}')
        state = json.loads(status.read_text())
        if state.get('returncode') != 0 or state.get('error'):
            raise ValueError(f'Unsuccessful phase cannot be archived: {status}')
        if phase == 'ladder':
            validate_ladder(output.read_text())
            if 'request error:' in stderr.read_text().lower():
                raise ValueError(f'Serving ladder stderr contains a request error: {stderr}')
        files.extend([output, status, stderr])
    for name in ('b1-repeat.out', 'b1-v-b4.txt', 'baseline-v-b1.txt'):
        if (source / name).is_file():
            files.append(source / name)
    # A staged archive already separates original runtime metadata from its wrapper.
    original_manifest = source / 'run-metadata.json'
    if not original_manifest.is_file():
        original_manifest = source / 'manifest.json'
    if original_manifest.is_file():
        files.append(original_manifest)
    route = source / 'server-route.log'
    if route.is_file():
        route_text = route.read_text()
    elif (source / 'server.log').is_file():
        route_text = ''.join(line for line in (source / 'server.log').read_text().splitlines(True)
                             if re.search(r'clamp|decode path:|native batched forward', line, re.I))
    else:
        raise ValueError(f'Missing server route evidence: {source}')
    if not route_text.strip():
        raise ValueError(f'Empty server route evidence: {source}')
    metadata = {key: run[key] for key in
                ('label', 'destination', 'source_revision', 'instrumentation_note', 'expected_trace_rows') if key in run}
    metadata.update({
        'verified': True,
        'archived_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'config': {'path_at_archive': str(config), 'sha256': digest(config)},
            'corpus': {'path_at_archive': str(corpus), 'sha256': digest(corpus)},
        },
        'hash_note': 'Input hashes computed at archive time; caller verified the files match the run.',
    })
    return target, files, route_text, metadata


def validate_ladder(text):
    """Reject partial or failed ladders even when the benchmark exits zero."""
    if 'request error:' in text.lower():
        raise ValueError('Serving ladder contains a request error')
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 7 or not all(value.isdigit() for value in fields[:3]):
            continue
        concurrency, succeeded, failed = map(int, fields[:3])
        if concurrency not in (1, 2, 4):
            continue
        if concurrency in rows:
            raise ValueError(f'Duplicate concurrency row: {concurrency}')
        if failed != 0 or succeeded != concurrency:
            raise ValueError(f'Incomplete or failed concurrency row: {concurrency}')
        metrics = [float(value) for value in fields[3:]]
        if not all(math.isfinite(value) and value >= 0 for value in metrics):
            raise ValueError(f'Invalid throughput/timing metrics: {concurrency}')
        rows[concurrency] = fields
    if set(rows) != {1, 2, 4}:
        raise ValueError('Serving ladder requires successful concurrency rows 1, 2, 4')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--mapping', required=True, type=Path)
    args = parser.parse_args()
    root, destination = args.source_root.resolve(), args.destination.resolve()
    runs = json.loads(args.mapping.read_text())['runs']
    if not runs:
        parser.error('No runs selected')
    plans = [plan_run(root, destination, run) for run in runs]
    targets = [plan[0] for plan in plans]
    if len(set(targets)) != len(targets) or any(a in b.parents for a in targets for b in targets if a != b):
        parser.error('Run destinations must be distinct and non-overlapping')
    # Validate every selection before copying. Failed verification leaves no archive edits.
    for target, files, route_text, metadata in plans:
        target.mkdir(parents=True)
        for source in files:
            # Keep the measurement manifest distinct from this archive's manifest.
            name = 'run-metadata.json' if source.name == 'manifest.json' else source.name
            shutil.copyfile(source, target / name)
        (target / 'server-route.log').write_text(route_text)
        metadata['artifacts'] = {
            path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(target.iterdir()) if path.is_file()
        }
        (target / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Refresh after all copies, including existing runs and auxiliary input metadata.
    artifacts = {
        str(path.relative_to(destination)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
        for path in sorted(destination.rglob('*'))
        if path.is_file() and path != destination / 'manifest.json'
    }
    (destination / 'manifest.json').write_text(json.dumps({'artifacts': artifacts}, indent=2) + '\n')
    print(f'Archived {len(plans)} runs; manifest covers {len(artifacts)} files')


if __name__ == '__main__':
    main()
