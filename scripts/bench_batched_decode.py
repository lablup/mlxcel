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

"""Capture one benchmark arm using existing binaries and a local checkpoint.

No model downloads, builds, uploads, or implicit source checkouts. Run arms
sequentially. --scalar-reference requires a logit_trace binary that supports
MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE; record instrumentation changes separately.
Declare the actual backend. Metal runs require the exact loaded library path,
not a search directory or a collection of possible library candidates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


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


def validate_backend(backend, loaded_metallib):
    """Require the caller to identify the actual backend and loaded library."""
    if backend not in ('metal', 'cpu', 'cuda', 'rocm'):
        raise ValueError('An explicit backend is required: metal, cpu, cuda, or rocm')
    if backend == 'metal':
        if loaded_metallib is None:
            raise ValueError('--backend metal requires --loaded-metallib for the actual loaded library')
        library = Path(loaded_metallib).resolve()
        if not library.is_file():
            raise ValueError('The actual loaded metallib file is missing')
        return {'backend': backend,
                'loaded_metallib': {'path': str(library), 'sha256': digest(library)}}
    if loaded_metallib is not None:
        raise ValueError('--loaded-metallib is only valid with --backend metal')
    return {'backend': backend, 'loaded_metallib': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--backend', choices=('metal', 'cpu', 'cuda', 'rocm'), required=True,
                        help='Actual backend used by the measured binaries; not a runtime selector')
    parser.add_argument('--loaded-metallib', type=Path,
                        help='Exact library loaded by all Metal binaries; required with --backend metal')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True,
                        help='Revision of the measured binaries, not necessarily --repo HEAD')
    parser.add_argument('--instrumentation-note', default='none')
    parser.add_argument('--rows', default='1,4')
    parser.add_argument('--port', type=int, default=18574)
    parser.add_argument('--scalar-reference', action='store_true')
    args = parser.parse_args()
    try:
        rows = [int(value) for value in args.rows.split(',')]
    except ValueError:
        parser.error('--rows must contain integers')
    controlled = {'MLXCEL_TRACE_BATCH_ROWS', 'MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE'}
    unexpected = sorted(key for key in os.environ
                        if key.startswith(('MLXCEL_TRACE_', 'MLXCEL_LOGIT_TRACE_'))
                        and key not in controlled)
    if unexpected:
        parser.error('Unset unexpected trace flags: ' + ', '.join(unexpected))
    # Capture model/kernel switches without exposing credential-valued keys.
    inherited = {key: value for key, value in os.environ.items()
                 if key.startswith('MLXCEL_') and key not in controlled
                 and not any(part in key.upper() for part in
                             ('TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'API_KEY', 'AUTH'))}
    if not rows or any(row < 1 for row in rows) or len(set(rows)) != len(rows):
        parser.error('--rows must contain distinct positive integers')
    if args.scalar_reference and rows != [1]:
        parser.error('--scalar-reference requires --rows 1')
    for name in ('repo', 'binary_dir', 'model', 'corpus', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    binaries = {name: args.binary_dir / name for name in
                ('mlxcel', 'mlxcel-server', 'examples/logit_trace')}
    required = [*binaries.values(), args.corpus,
                args.repo / 'scripts/bench_serving_concurrency.py']
    required.append(args.model / 'config.json')
    benchmark_script = args.repo / 'scripts/bench_serving_concurrency.py'
    if not args.model.is_dir() or not all(path.is_file() for path in required):
        parser.error('Checkpoint, corpus, benchmark script, or binaries are missing')
    try:
        backend_metadata = validate_backend(args.backend, args.loaded_metallib)
    except ValueError as error:
        parser.error(str(error))
    # Refuse an occupied port, rather than benchmarking an unrelated server.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'source_revision': args.source_revision,
        'instrumentation_note': args.instrumentation_note,
        'prompt_tokens_nominal': 512,
        'prompt_tokens_note': 'Requested approximate text length; actual tokenization varies.',
        'trace_max_history_tokens': 512,
        'trace_history_note': 'Maximum budget; history grows from the corpus start.',
        'model': str(args.model), 'corpus': str(args.corpus),
        'corpus_sha256': digest(args.corpus),
        'binary_sha256': {name: digest(path) for name, path in binaries.items()},
        **backend_metadata,
        'config_sha256': digest(args.model / 'config.json'),
        'benchmark_script_sha256': digest(benchmark_script),
        'inherited_mlxcel_environment': inherited,
        'phases': [],
    }

    def save():
        metadata['artifacts'] = {
            path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in args.output.iterdir()
            if path.is_file() and path.name != 'manifest.json'
        }
        (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def run(name, command, overrides=None, timeout=1200):
        record = {'name': name, 'command': list(map(str, command)),
                  'environment_overrides': overrides or {}}
        start = time.monotonic()
        try:
            with (args.output / (name + '.out')).open('w') as out, \
                    (args.output / (name + '.err')).open('w') as err:
                result = subprocess.run(record['command'], cwd=args.repo,
                                        env={**os.environ, **(overrides or {})},
                                        stdout=out, stderr=err, timeout=timeout)
                record['returncode'] = result.returncode
                if result.returncode:
                    raise RuntimeError(f'{name} exited {result.returncode}')
            if name == 'ladder':
                validate_ladder((args.output / 'ladder.out').read_text())
                if 'request error:' in (args.output / 'ladder.err').read_text().lower():
                    raise ValueError('Serving ladder stderr contains a request error')
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            record['seconds'] = time.monotonic() - start
            metadata['phases'].append(record)
            (args.output / (name + '.status.json')).write_text(json.dumps(record, indent=2) + '\n')
            save()

    server = None
    try:
        run('greedy', [binaries['mlxcel'], 'generate', '-m', args.model, '-p',
                       'Hello', '-n', '128', '--temp', '0', '--show-reasoning'])
        for row in rows:
            env = {'MLXCEL_TRACE_BATCH_ROWS': str(row),
                   'MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE': '1' if args.scalar_reference else '0'}
            run(f'trace-b{row}', [binaries['examples/logit_trace'], args.model,
                                args.corpus, '64', '4', '8', '512'], env)
        command = [str(binaries['mlxcel-server']), '-m', str(args.model),
                   '--parallel', '4', '--port', str(args.port), '--offline', '--metrics']
        metadata['server_command'] = command
        with (args.output / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode} before readiness')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Server readiness exceeded 600 seconds')
                time.sleep(1)
            run('ladder', [sys.executable, benchmark_script, '--port', str(args.port),
                           '--concurrency', '1,2,4', '--prompt-tokens', '512',
                           '--max-tokens', '128', '--metrics'], timeout=1800)
    except BaseException as error:
        metadata['error'] = str(error)
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            metadata['server_returncode'] = server.returncode
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(args.output)


if __name__ == '__main__':
    main()
