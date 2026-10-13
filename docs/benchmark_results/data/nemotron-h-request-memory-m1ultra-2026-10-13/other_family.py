#!/usr/bin/env python3
"""Unprotected Qwen family smoke under an intentionally tiny target-only budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
from harness import request, require

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
base = f'http://127.0.0.1:{port}'
cmd = [a.binary, '--model', 'models/mlx/qwen2.5-7b-instruct-4bit', '--offline', '--host', '127.0.0.1', '--port', str(port), '--parallel', '1', '--ctx-size', '4096', '--alias', 'repro']
logpath = Path(a.output + '.log')
result = {'command': cmd, 'budget': '1', 'binary_sha256': hashlib.sha256(Path(a.binary).read_bytes()).hexdigest(), 'load_before': os.getloadavg(), 'requests': []}
with logpath.open('w') as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'MLXCEL_REQUEST_MEMORY_BUDGET': '1', 'RUST_LOG': 'info'})
    try:
        deadline = time.monotonic() + 180
        while True:
            require(proc.poll() is None, 'server exited')
            try:
                if request(base, '/v1/models')['status'] == 200:
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            require(time.monotonic() < deadline, 'readiness timed out')
            time.sleep(0.25)
        for stream in [False, True]:
            r = request(base, '/v1/completions', {'model': 'repro', 'prompt': 'The capital of France is', 'max_tokens': 8, 'temperature': 0.0, 'stream': stream})
            require(r['status'] == 200, f'Qwen incorrectly guarded: {r}')
            if stream:
                require('[DONE]' in r['body'], 'incomplete stream')
            result['requests'].append(r)
        result['health'] = request(base, '/health')
        require(result['health']['status'] == 200, 'health failed')
        require('request_memory_admission' not in logpath.read_text(), 'unexpected target guard')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps({'requests': len(result['requests']), 'health': result['health']['status']}))
