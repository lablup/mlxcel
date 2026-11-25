#!/usr/bin/env python3
"""Strict Falcon-H1 displayed CLI/concurrent HTTP parity; run under the GPU guard.

Requires tokenizers. Select --device explicitly and use the pinned IREE environment.
The CLI parser removes only its documented preamble and two framing newlines;
model-generated trailing whitespace is preserved and compared byte for byte.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--oracle', type=Path, default=ROOT / 'spike/openxla/falcon_h1_reference/hf-f32-oracle.json')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', choices=['local-task', 'cuda', 'hip'], required=True)
parser.add_argument('--bin-dir', type=Path, default=ROOT / 'target/test-fast')
parser.add_argument('--binary-source-head', required=True)
args = parser.parse_args()
OUT, MODEL, ORACLE = args.out.resolve(), args.model.resolve(), args.oracle.resolve()
PROMPTS = [
    'The capital of France is',
    'The capital of Japan is',
    'The capital of Germany is',
    'The capital of Italy is',
]
ENV = os.environ.copy()
ENV.update(MLXCEL_BACKEND='xla', MLXCEL_XLA_DEVICE=args.device, MLXCEL_XLA_PRECISION='f32',
           MLXCEL_XLA_CONTEXT_CAPACITY='256', MLXCEL_CACHE_DIR=str(OUT / 'cache'),
           MLXCEL_PRINT_TOKEN_IDS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stop_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def token_ids(text: str, label: str) -> list[int]:
    matches = re.findall(r'\[' + re.escape(label) + r' \((\d+)\): ([0-9 ]*)\]', text)
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one {label} line, got {len(matches)}')
    count, body = matches[0]
    ids = [int(x) for x in body.split()]
    if len(ids) != int(count):
        raise RuntimeError(f'{label}: count mismatch')
    return ids


def cli(name: str, prompt: str, budget: int, tokenizer: Tokenizer) -> dict[str, object]:
    command = [str(args.bin_dir / 'mlxcel'), 'generate', '-m', str(MODEL),
               '-p', prompt, '-n', str(budget), '--temp', '0', '--no-chat-template']
    started = time.time()
    print(f'{name}: starting CLI', flush=True)
    with (OUT / f'{name}.stdout').open('wb') as stdout, (OUT / f'{name}.stderr').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=600)
        finally:
            stop_owned(process)
    stderr_text = (OUT / f'{name}.stderr').read_text()
    record: dict[str, object] = {'command': command, 'exit_code': code,
        'start_time': started, 'end_time': time.time(), 'prompt': prompt, 'budget': budget}
    save(f'{name}.json', record)
    if code != 0:
        raise RuntimeError(f'{name}: CLI failed with exit {code}')
    generated = token_ids(stderr_text, 'token ids')
    stdout_text = (OUT / f'{name}.stdout').read_bytes().decode('utf-8')
    prefix = 'Generating...\n' + prompt
    if stdout_text.count(prefix) != 1:
        raise RuntimeError(f'{name}: unexpected CLI preamble')
    displayed = stdout_text.split(prefix, 1)[1]
    framing = re.search(r'\n\n\[Generated \d+ tokens in [^\n]*\]\n\Z', displayed)
    if framing is None:
        raise RuntimeError(f'{name}: unexpected CLI timing suffix')
    displayed = displayed[:framing.start()]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    record.update(prompt_ids=token_ids(stderr_text, 'prompt ids'), token_ids=generated,
                  text=displayed, unfiltered_token_decode=decoded, display_matches_token_decode=displayed == decoded)
    if displayed != decoded:
        save(f'{name}.json', record)
        raise RuntimeError(f'{name}: displayed CLI text differs from unfiltered token decode')
    save(f'{name}.json', record)
    print(f'{name}: exit0, {len(generated)} tokens, {time.time()-started:.2f}s', flush=True)
    return record


def request_json(url: str, payload: dict[str, object] | None = None, timeout: int = 10) -> object:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    provenance: dict[str, object] = {
        'head': head, 'binary_source_head': args.binary_source_head, 'server_ctx_size': 2048, 'checkpoint_revision': 'e6389502a0b12cd8da894b395ba5bf7436873b16',
        'model': str(MODEL), 'precision': 'f32', 'device': args.device,
        'context_capacity_override': 256, 'max_batch_size': 4,
        'scope': 'CLI 40-token oracle and four concurrent completions versus separate 40-token displayed CLI references',
        'gpu_guard': 'caller must hold gpu-lock on GB10 or rocm_gpu_guard on ROCm',
        'binary_sha256': {name: hashlib.sha256((args.bin_dir / name).read_bytes()).hexdigest()
                          for name in ['mlxcel', 'mlxcel-server']},
        'environment': {k: ENV[k] for k in ['MLXCEL_BACKEND', 'MLXCEL_XLA_DEVICE',
            'MLXCEL_XLA_PRECISION', 'MLXCEL_XLA_CONTEXT_CAPACITY', 'MLXCEL_CACHE_DIR']},
    }
    save('provenance.json', provenance)
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    oracle = json.loads(ORACLE.read_text())
    oracle_cli = cli('cli-oracle40', oracle['prompt_text'], 40, tokenizer)
    if oracle_cli['prompt_ids'] != oracle['prompt_ids'] or oracle_cli['token_ids'] != oracle['ref_token_ids']:
        raise RuntimeError('CLI 40-token output differs from independent HF fp32 oracle')
    print('CLI/HF ORACLE: 40/40 TOKEN-EXACT PASS', flush=True)
    references = [cli(f'cli-ref-{i}', prompt, 40, tokenizer) for i, prompt in enumerate(PROMPTS)]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(args.bin_dir / 'mlxcel-server'), '-m', str(MODEL), '--host', '127.0.0.1',
               '--port', str(port), '--max-batch-size', '4', '--ctx-size', '2048']
    save('server-command.json', {'command': command, 'base_url': base})
    with (OUT/'server.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        save('server-process.json', {'pid': process.pid, 'start_time': time.time()})
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited before readiness: {process.returncode}')
                try:
                    health = request_json(base+'/health', timeout=2)
                    models = request_json(base+'/v1/models', timeout=2)
                    model_id = models['data'][0]['id']
                    save('server-ready.json', {'health': health, 'models': models})
                    break
                except (urllib.error.URLError, TimeoutError, OSError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Server readiness timeout')
                    time.sleep(1)
            print(f'Server ready at {base}; issuing four requests concurrently', flush=True)
            barrier = threading.Barrier(4)

            def complete(i: int) -> dict[str, object]:
                payload = {'model': model_id, 'prompt': PROMPTS[i], 'temperature': 0,
                           'max_tokens': 40, 'stream': False}
                save(f'request-{i}.json', payload)
                barrier.wait(timeout=10)
                started = time.time()
                body = request_json(base+'/v1/completions', payload, timeout=300)
                result = {'request': i, 'started': started, 'ended': time.time(), 'body': body}
                save(f'response-{i}.json', result)
                return result

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(complete, range(4)))
            comparisons = []
            for i, (result, reference) in enumerate(zip(results, references, strict=True)):
                body = result['body']
                actual = body['choices'][0]['text']
                usage = body['usage']
                match = actual == reference['text']
                counts = (usage['prompt_tokens'] == len(reference['prompt_ids'])
                          and usage['completion_tokens'] == len(reference['token_ids']))
                comparisons.append({'request': i, 'text_exact': match, 'counts_exact': counts,
                    'expected_text': reference['text'], 'actual_text': actual, 'usage': usage,
                    'cli_token_ids': reference['token_ids'], 'finish_reason': body['choices'][0]['finish_reason']})
                print(f'request{i}: text_exact={match} counts_exact={counts}', flush=True)
            overlap = max(r['started'] for r in results) < min(r['ended'] for r in results)
            passed = overlap and all(c['text_exact'] and c['counts_exact'] for c in comparisons)
            save('result.json', {'head': head, 'passed': passed, 'cli_oracle_40_token_exact': True,
                'four_requests_overlapped': overlap, 'server_cli_comparisons': comparisons,
                'evidence_boundary': f'{args.device} HTTP versus displayed CLI text and token-count parity; server API does not expose token IDs.'})
            if not passed:
                raise RuntimeError('Concurrent server/CLI parity failed; inspect result.json')
            print('SERVER/CLI: 4/4 EXACT-TEXT AND TOKEN-COUNT PASS', flush=True)
        finally:
            stop_owned(process)
            save('server-exit.json', {'pid': process.pid, 'returncode': process.returncode,
                                     'reaped': process.poll() is not None, 'end_time': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save('failure.json', {'error': str(error), 'type': type(error).__name__})
        raise
