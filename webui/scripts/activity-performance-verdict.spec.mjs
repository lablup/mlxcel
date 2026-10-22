// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The Activity decode-overhead verdict (#1925), checked against the fixture that
// scripts/webui/activity_performance_verdict_tests.py evaluates too.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { verdictPolicy, validatePolicy, signTestRank, medianInterval, lookVerdict, modeSummary, pairsFromSamples, performanceMode, performanceCompletion } from './activity-performance-config.mjs';

const fixture = JSON.parse(readFileSync(new URL('./activity-performance-fixtures.json', import.meta.url), 'utf8'));
const pick = (value, expected) => Object.fromEntries(Object.keys(expected).map((key) => [key, value[key]]));

test('fixture was generated under the policy in force', () => {
  assert.deepEqual(fixture.policy, { ...verdictPolicy });
});
for (const item of fixture.cases) {
  test(`fixture: ${item.name}`, () => {
    const evaluate = () => {
      const byMode = pairsFromSamples(item.samples);
      if (item.check === 'pairs') return {};
      return Object.fromEntries([...byMode].map(([mode, pairs]) => [mode, item.check === 'look' ? lookVerdict(pairs) : modeSummary(mode, pairs)]));
    };
    if (item.expected.error) {
      assert.throws(evaluate, (error) => error.message.includes(item.expected.error));
      return;
    }
    const result = evaluate();
    assert.deepEqual(Object.keys(result).sort(), Object.keys(item.expected).sort());
    for (const [mode, expected] of Object.entries(item.expected)) assert.deepEqual(pick(result[mode], expected), expected, `${item.name}: ${mode}`);
  });
}
test('none of the recorded runs that failed on noise is a slowdown', () => {
  const recorded = fixture.cases.filter((item) => item.name.startsWith('recorded '));
  assert.ok(recorded.length >= 10);
  for (const item of recorded) for (const verdict of Object.values(item.expected)) assert.notEqual(verdict.status, 'investigate');
});
test('sign-test rank matches the exact binomial tail', () => {
  // P(Bin(12, 1/2) <= 1) = 13/4096 <= 0.01 < P(<= 2) = 79/4096, so the 98% interval is [x(2), x(11)].
  assert.equal(signTestRank(12, 0.01), 2);
  assert.equal(signTestRank(6, 0.01), 0);
  assert.equal(signTestRank(7, 0.01), 1);
  assert.equal(signTestRank(40, 0.01), 13);
  assert.deepEqual(medianInterval([5, 1, 4, 2, 3, 6, 7, 8, 9, 10, 11, 12]), { median: 6.5, interval: [2, 11] });
  assert.equal(medianInterval([1, 2, 3]).interval, null);
});
test('looks happen only at min_pairs plus whole steps', () => {
  const pairs = (n, on) => Array.from({ length: n }, () => ({ off: 100, on }));
  assert.equal(lookVerdict(pairs(11, 90)).status, 'continue');
  assert.equal(lookVerdict(pairs(13, 90)).status, 'continue');
  assert.equal(lookVerdict(pairs(12, 90)).status, 'investigate');
  assert.equal(lookVerdict(pairs(12, 101)).status, 'within-target');
});
test('policy validation rejects unsafe or incomplete policies', () => {
  assert.doesNotThrow(() => validatePolicy({ ...verdictPolicy }));
  for (const change of [{ tail_alpha: 0 }, { tail_alpha: 0.5 }, { min_pairs: 6 }, { pair_step: 1 }, { max_pairs: 52 }, { max_pairs: 41 }, { precision_ceiling_percent: 1 }, { budget_percent: Number.NaN }, { extra: 1 }]) {
    assert.throws(() => validatePolicy({ ...verdictPolicy, ...change }), /policy/);
  }
  const { max_pairs: _ignored, ...missing } = verdictPolicy;
  assert.throws(() => validatePolicy(missing), /policy/);
});
test('full mode reports an undecided mode as inconclusive, never as a slowdown', () => {
  const full = performanceMode();
  assert.equal(performanceCompletion(full, [{ status: 'within-target' }, { status: 'inconclusive' }]).status, 'inconclusive');
  assert.equal(performanceCompletion(full, [{ status: 'inconclusive' }, { status: 'investigate' }]).status, 'investigate');
  assert.equal(performanceCompletion(performanceMode('visible-only-headed'), [{ status: 'inconclusive' }]).status, 'incomplete');
});
