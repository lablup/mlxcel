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

"""Compare solo and concurrent VLM messages against an existing local server binary.

Starts one offline server on localhost; performs no downloads or external HTTP.
Preserves two text prompts and one image prompt, temperature 0, max_tokens 48.
Requests are saved without base64; the copied image plus its SHA-256 reconstructs
exact wire payloads. --constrained is a separately labeled diagnostic workload.
The backend records provenance, not a runtime device selection. Metal requires
one explicit --loaded-metallib identifying the library all requests actually use.
"""
import argparse
import base64
import concurrent.futures
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

IMAGE_MARKER = '@artifact:input-image.png'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_backend(backend, loaded_metallib):
    if backend not in ('metal', 'cpu', 'cuda', 'rocm'):
        raise ValueError('An explicit backend is required: metal, cpu, cuda, or rocm')
    if backend == 'metal':
        if loaded_metallib is None:
            raise ValueError('--backend metal requires --loaded-metallib')
        path = Path(loaded_metallib).resolve()
        if not path.is_file():
            raise ValueError('The actual loaded metallib file is missing')
        return {'backend': backend, 'loaded_metallib': {'path': str(path), 'sha256': digest(path)}}
    if loaded_metallib is not None:
        raise ValueError('--loaded-metallib is only valid with --backend metal')
    return {'backend': backend, 'loaded_metallib': None}


def request_manifest(image_path, constrained=False):
    """Store complete requests with one explicitly documented image placeholder."""
    prompts = ['Write a short description of a mountain.', 'Count from one to ten in words.']
    if constrained:
        prompts = ['Reply with exactly this word: mountain',
                   'What is two plus two? Reply with only the numeral.']
    prompts.append([{'type': 'image_url', 'image_url': {'url': IMAGE_MARKER}},
                    {'type': 'text', 'text': 'Describe this image briefly.'}])
    return {
        'workload': 'constrained_diagnostic' if constrained else 'original_freeform',
        'constrained_prompts': constrained,
        'image': {'artifact': 'input-image.png', 'original_path': str(Path(image_path).resolve()),
                  'mime_type': 'image/png', 'sha256': digest(image_path)},
        'image_expansion': 'Replace @artifact:input-image.png with data:image/png;base64, plus base64 of the hash-verified image artifact.',
        'payloads': [{'model': 'epic-test', 'messages': [{'role': 'user', 'content': prompt}],
                      'temperature': 0, 'max_tokens': 48, 'stream': False} for prompt in prompts],
    }


def expand_requests(manifest, image_path):
    """Rebuild exact API payloads from the durable input record."""
    if digest(image_path) != manifest['image']['sha256']:
        raise ValueError('Image artifact hash does not match request manifest')
    payloads = copy.deepcopy(manifest['payloads'])
    image_url = payloads[2]['messages'][0]['content'][0]['image_url']
    if image_url['url'] != IMAGE_MARKER:
        raise ValueError('Unexpected image marker in request manifest')
    image_url['url'] = 'data:image/png;base64,' + base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    return payloads


def response_message(response):
    if not isinstance(response, dict) or 'error' in response:
        raise ValueError('Completion response is not successful')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('Completion response must contain exactly one choice')
    if not isinstance(choices[0], dict) or not isinstance(choices[0].get('message'), dict):
        raise ValueError('Completion response is missing its full message')
    message = choices[0]['message']
    if message.get('role') != 'assistant':
        raise ValueError('Completion message must have assistant role')
    if not any(isinstance(message.get(key), str) and message[key]
               for key in ('content', 'reasoning', 'reasoning_content')) and not message.get('tool_calls'):
        raise ValueError('Completion message has no text, reasoning or tool calls')
    return message


def compare_messages(solo, concurrent):
    """Compare every message field, including reasoning; reject partial results."""
    if len(solo) != 3 or len(concurrent) != 3:
        raise ValueError('Expected all three solo and concurrent results')
    def ordered(results):
        indexed = {result['index']: result for result in results}
        if set(indexed) != {0, 1, 2}:
            raise ValueError('Results must contain distinct request indices 0, 1, 2')
        return [response_message(indexed[index]['response']) for index in range(3)]
    left, right = ordered(solo), ordered(concurrent)
    matches = [a == b for a, b in zip(left, right)]
    return {'exact_message_matches': matches, 'all_match': all(matches)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def ensure_no_listener(port):
    """Reject a live listener without confusing closed TCP TIME_WAIT sockets.

    A live loopback listener accepts at once. Windows answers a closed port by
    retrying the SYN for about two seconds instead of refusing it, so a
    timeout also means nothing is listening.
    """
    try:
        connection = socket.create_connection(('127.0.0.1', port), timeout=0.5)
    except (ConnectionRefusedError, TimeoutError):
        return
    connection.close()
    raise ValueError(f'Port {port} already has a listening service')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'binary', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--image', type=Path, help='PNG fixture; defaults to REPO/tests/fixtures/test_image.png')
    parser.add_argument('--backend', choices=('metal', 'cpu', 'cuda', 'rocm'), required=True)
    parser.add_argument('--loaded-metallib', type=Path, help='Actual loaded library, required for Metal')
    parser.add_argument('--constrained', action='store_true', help='Use separately labeled constrained diagnostic text prompts')
    parser.add_argument('--readiness-timeout', type=float, default=600)
    parser.add_argument('--request-timeout', type=float, default=600)
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535
            or not all(math.isfinite(value) and value > 0
                       for value in (args.readiness_timeout, args.request_timeout))):
        parser.error('Port must be 1..65535 and timeouts must be positive')
    for name in ('repo', 'binary', 'model', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    image_path = (args.image or args.repo / 'tests/fixtures/test_image.png').resolve()
    if not args.repo.is_dir() or not args.binary.is_file() or not (args.model / 'config.json').is_file() or not image_path.is_file():
        parser.error('Repository, server binary, model config or image fixture is missing')
    try:
        backend = validate_backend(args.backend, args.loaded_metallib)
    except ValueError as error:
        parser.error(str(error))
    # Check before creating artifacts or sending requests to prevent accidental reuse.
    if args.output.exists():
        parser.error('Output already exists; refusing to overwrite')
    ensure_no_listener(args.port)
    args.output.mkdir(parents=True, exist_ok=False)
    input_record = request_manifest(image_path, args.constrained)
    shutil.copyfile(image_path, args.output / 'input-image.png')
    write_json(args.output / 'requests.json', input_record)
    payloads = expand_requests(input_record, args.output / 'input-image.png')
    server_command = [str(args.binary), '-m', str(args.model), '--parallel', '4', '--port',
                      str(args.port), '--offline', '--metrics']
    manifest = {'source_revision': args.source_revision, 'started_utc': utc_now(),
                'command': [sys.executable, *sys.argv], 'server_command': server_command,
                'binary': {'path': str(args.binary), 'sha256': digest(args.binary)},
                'config': {'path': str(args.model / 'config.json'), 'sha256': digest(args.model / 'config.json')},
                'driver_sha256': digest(Path(__file__).resolve()),
                'workload': input_record['workload'], 'constrained_prompts': args.constrained,
                'readiness_timeout_seconds': args.readiness_timeout,
                'request_timeout_seconds': args.request_timeout, 'phases': [], **backend}
    server = None
    base_url = f'http://127.0.0.1:{args.port}'
    # Do not let proxy settings redirect localhost requests to an external proxy.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    def opener():
        return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def request(index, barrier=None):
        if barrier:
            barrier.wait(timeout=15)
        started = utc_now()
        begin = time.monotonic()
        req = urllib.request.Request(base_url + '/v1/chat/completions',
                                     data=json.dumps(payloads[index]).encode(),
                                     headers={'Content-Type': 'application/json'})
        with opener().open(req, timeout=args.request_timeout) as response:
            data = json.load(response)
        response_message(data)
        return {'index': index, 'started_utc': started, 'completed_utc': utc_now(),
                'seconds': time.monotonic() - begin, 'response': data}
    def phase(name, action):
        record = {'name': name, 'started_utc': utc_now()}
        begin = time.monotonic()
        try:
            value = action()
            write_json(args.output / (name + '.json'), value)
            record['returncode'] = 0
            return value
        except BaseException as error:
            record.update({'returncode': 1, 'error': str(error), 'error_type': type(error).__name__})
            raise
        finally:
            record.update({'completed_utc': utc_now(), 'seconds': time.monotonic() - begin})
            manifest['phases'].append(record)
            write_json(args.output / (name + '.status.json'), record)
    try:
        with (args.output / 'server.log').open('w') as log:
            server = subprocess.Popen(server_command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
            def wait_ready():
                deadline = time.monotonic() + args.readiness_timeout
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'Server exited {server.returncode} before readiness')
                    try:
                        with opener().open(base_url + '/health', timeout=min(2, args.readiness_timeout)) as response:
                            if response.status == 200:
                                return {'ready': True}
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Server readiness deadline exceeded')
                    time.sleep(0.2)
            phase('readiness', wait_ready)
            solo = phase('solo', lambda: [request(index) for index in range(3)])
            def concurrent_requests():
                barrier = threading.Barrier(3)
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    return list(pool.map(lambda index: request(index, barrier), range(3)))
            concurrent_results = phase('concurrent', concurrent_requests)
            comparison = {**compare_messages(solo, concurrent_results), 'model': str(args.model),
                          'constrained_prompts': args.constrained, 'workload': input_record['workload']}
            write_json(args.output / 'comparison.json', comparison)
            if not comparison['all_match']:
                raise RuntimeError('Solo/concurrent full messages differ; see comparison.json')
            manifest['returncode'] = 0
    except BaseException as error:
        manifest.update({'returncode': 1, 'error': str(error), 'error_type': type(error).__name__})
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            manifest['server_returncode'] = server.returncode
        manifest['completed_utc'] = utc_now()
        manifest['artifacts'] = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                                 for path in sorted(args.output.iterdir())
                                 if path.is_file() and path.name != 'manifest.json'}
        write_json(args.output / 'manifest.json', manifest)
    print(args.output)


if __name__ == '__main__':
    main()
