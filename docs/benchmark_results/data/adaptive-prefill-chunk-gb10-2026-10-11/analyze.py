#!/usr/bin/env python3
"""Check issue 2228 acceptance without rounding measurements."""
import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
result = {'admission': [], 'ttft': []}
for round_no in range(1, 4):
    path = root / f'admit-policy-r{round_no}.txt'
    match = re.search(r'admission window .*?p95 ([0-9.]+) ms', path.read_text())
    if not match:
        raise SystemExit(f'missing admission p95: {path}')
    p95 = float(match[1])
    result['admission'].append({'round': round_no, 'p95_ms': p95, 'pass': p95 <= 62.0})
for model in ['qwen3-1.7b-4bit', 'llama-3.2-1b-instruct-4bit']:
    rows = [json.loads(line) for line in (root / f'ttft-{model}.jsonl').read_text().splitlines()]
    arms = {name: {} for name in ['c2048', 'policy', 'c2048-null']}
    for row in rows:
        if row.get('kind') == 'header':
            continue
        if row['path'] != 'server' or row['prompt_target_len'] != 8192:
            raise SystemExit(f'unexpected benchmark cell: {row}')
        arm = arms[row['arm']]
        if row['round'] in arm:
            raise SystemExit('duplicate round; do not append a rerun to original measurements')
        value = row['ttft_ms']
        if not math.isfinite(value) or value <= 0:
            raise SystemExit('invalid TTFT')
        arm[row['round']] = value
    if any(set(arm) != set(range(5)) for arm in arms.values()):
        raise SystemExit(f'incomplete rounds: {model}')
    delta = lambda name: [(arms[name][r] / arms['c2048'][r] - 1) * 100 for r in range(5)]
    policy, null = delta('policy'), delta('c2048-null')
    median = statistics.median(policy)
    result['ttft'].append({'model': model, 'policy_median_delta_pct': median,
        'policy_range_pct': [min(policy), max(policy)], 'null_range_pct': [min(null), max(null)],
        'null_median_delta_pct': statistics.median(null),
        'pass': min(null) <= median <= max(null)})
result['pass'] = all(row['pass'] for group in ['admission', 'ttft'] for row in result[group])
(root / 'acceptance-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
sys.exit(0 if result['pass'] else 1)
