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

"""Record sequential, deterministic native completion measurements."""
import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any
from pathlib import Path


def request(
    port: int, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--port', type=int, default=19833)
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--scenario', choices=['science', 'software', 'history'])
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            request(args.port, '/health')
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    else:
        raise RuntimeError('server did not become ready')
    scenarios = [
        ('science', 'Explain why the seasons change, and describe the role of axial tilt and sunlight. '),
        ('software', 'Explain a reliable method for testing concurrent software and investigating race conditions. '),
        ('history', 'Explain how libraries preserve knowledge and make information accessible across generations. '),
    ]
    if args.long:
        scenarios = [(
            "long",
            "A careful engineering report records assumptions, observations, "
            "measurements, and the limitations of its conclusions. ",
        )]
    if args.scenario and not args.long:
        scenarios = [(name, topic) for name, topic in scenarios if name == args.scenario]
    for name, topic in scenarios:
        instruction = "Write a detailed, coherent answer with several examples."
        if args.long:
            instruction = (
                "Write a detailed engineering report of at least 2000 words "
                "with many examples and concrete measurements."
            )
        prompt = (
            "<bos><start_of_turn>user\n"
            + topic * (400 if args.long else 28)
            + f"\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        )
        if args.long:
            tokens = request(
                args.port, "/tokenize", {"content": prompt, "add_special": False}
            )["tokens"]
            selected = tokens[:6349] + tokens[-100:]
            if len(selected) != 6449:
                raise RuntimeError(f'long prompt has {len(selected)} tokens')
            prompt = request(args.port, '/detokenize', {'tokens': selected})['content']
        payload = {
            "prompt": prompt,
            "n_predict": 513 if args.long else 256,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
            "stream": False,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
        }
        (args.out/f'{name}.request.json').write_text(json.dumps(payload, indent=2))
        start = time.monotonic()
        result = request(args.port, '/completion', payload)
        wall = time.monotonic()-start
        (args.out/f'{name}.json').write_text(json.dumps(result, indent=2))
        (args.out/f'{name}.txt').write_text(result.get('content', ''))
        print(json.dumps({
            "name": name,
            "wall_s": wall,
            "tokens_evaluated": result.get("tokens_evaluated"),
            "tokens_predicted": result.get("tokens_predicted"),
            "timings": result.get("timings"),
        }), flush=True)


if __name__ == '__main__':
    main()
