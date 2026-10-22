// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global URL */
// Pure acceptance policy: diagnostic evidence must never become a full pass.
import { readFileSync } from 'node:fs';

export function performanceMode(value = 'full') {
  if (value === 'full') return { name: value, headless: false, modes: ['one-visible', 'two-visible', 'hidden'] };
  if (value === 'visible-only-headless') return { name: value, headless: true, modes: ['one-visible', 'two-visible'] };
  // Headed visible-mode overhead without the hidden acceptance. The hidden mode needs the browser
  // to report a real window-visibility change through document.hidden, which the automation
  // browser does not do; this mode keeps the observation-overhead measurement on the real headed
  // path and still declares the hidden acceptance not run.
  if (value === 'visible-only-headed') return { name: value, headless: false, modes: ['one-visible', 'two-visible'] };
  throw new Error('WEBUI_PERF_MODE must be full, visible-only-headed, or visible-only-headless.');
}
export function assertGeometry(value) {
  if (![value.innerWidth, value.innerHeight, value.visualWidth, value.visualHeight].every((n) => Number.isFinite(n) && n > 0)) throw new Error('Browser has no usable layout/visual viewport; no inference measurement may begin.');
}
export function performanceCompletion(mode, summaries) {
  // Completeness follows whether the hidden acceptance actually ran, not whether the browser was
  // headed: a headed run that skips the hidden mode is still an incomplete acceptance. A measured
  // slowdown outranks an undecided measurement, so `investigate` wins over `inconclusive`.
  const hiddenRan = mode.modes.includes('hidden');
  const has = (status) => summaries.some((x) => x.status === status);
  return {
    status: !hiddenRan ? 'incomplete' : has('investigate') ? 'investigate' : has('inconclusive') ? 'inconclusive' : 'within-target',
    scope: mode.name,
    hidden_native: hiddenRan ? 'measured' : 'not-run',
    limitation: hiddenRan ? null : mode.headless
      ? 'Visible headless diagnostic only; native hidden-tab and interactive host acceptance remain outstanding.'
      : 'Headed visible-mode overhead only; native hidden acceptance is not run because the automation browser reports document.hidden as false for a window it reports as minimized.',
  };
}

// Decode-overhead verdict (#1925). Every constant lives in activity-performance-policy.json, which
// scripts/webui/activity_performance_verdict.py reads too, so the harness and both validators judge
// one rule. Both implementations are checked against activity-performance-fixtures.json.
//
// Each pair measures the same request with the Activity clients off and on, in alternating order,
// and scores (off - on) / off * 100: positive is a slowdown. Per-pair noise on the GB10 runner is
// heavy-tailed (kurtosis above 10 across 636 recorded runs), so the point estimate is the median
// and its interval is the distribution-free sign-test interval [x(k), x(n+1-k)], which needs no
// variance estimate and no bootstrap randomness. Sampling starts at min_pairs and adds pair_step
// pairs per look until one of:
//   investigate   - the interval lies wholly above budget_percent: evidence of a slowdown.
//   within-target - the interval does not lie above the budget and its upper bound is at most
//                   precision_ceiling_percent, so the run was precise enough to have caught a
//                   slowdown well past the budget.
//   inconclusive  - max_pairs reached with neither: measurement noise, not a slowdown.
// Baseline CV and the paired range are diagnostics only; neither decides anything.
const policyKeys = ['budget_percent', 'precision_ceiling_percent', 'tail_alpha', 'min_pairs', 'pair_step', 'max_pairs'];
export function validatePolicy(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Activity verdict policy must be an object.');
  const keys = Object.keys(value).sort();
  if (keys.join() !== [...policyKeys].sort().join()) throw new Error(`Activity verdict policy must have exactly: ${policyKeys.join(', ')}.`);
  const { budget_percent: budget, precision_ceiling_percent: ceiling, tail_alpha: alpha, min_pairs: min, pair_step: step, max_pairs: max } = value;
  if (![budget, ceiling, alpha].every(Number.isFinite) || !(ceiling >= budget) || !(alpha > 0 && alpha < 0.5)) throw new Error('Activity verdict policy thresholds are invalid.');
  // Binomial sums stay exact in a double up to n = 50, which keeps JavaScript and Python identical.
  if (![min, step, max].every(Number.isInteger) || step < 2 || step % 2 !== 0 || min % 2 !== 0 || max > 50 || max < min || (max - min) % step !== 0) throw new Error('Activity verdict policy pair counts are invalid.');
  if (signTestRank(min, alpha) < 1) throw new Error('Activity verdict policy min_pairs is too small for its tail_alpha.');
  return Object.freeze({ ...value });
}
export const verdictPolicy = validatePolicy(JSON.parse(readFileSync(new URL('./activity-performance-policy.json', import.meta.url), 'utf8')));

// Largest k with P(Binomial(n, 1/2) <= k - 1) <= alpha; 0 when no order statistic qualifies.
export function signTestRank(n, alpha) {
  let choose = 1; let cumulative = 0; let k = 0;
  for (let j = 0; j <= n; j++) {
    cumulative += choose;
    if (cumulative > alpha * 2 ** n) break;
    k = j + 1;
    choose = choose * (n - j) / (j + 1);
  }
  return k;
}
export function pairDegradation(pair) {
  if (!(Number.isFinite(pair.off) && pair.off > 0 && Number.isFinite(pair.on) && pair.on > 0)) throw new Error('Activity pair rates must be finite and positive.');
  return (pair.off - pair.on) / pair.off * 100;
}
export function medianInterval(values, policy = verdictPolicy) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) throw new Error('Activity verdict needs at least one pair.');
  const median = n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const k = signTestRank(n, policy.tail_alpha);
  return { median, interval: k < 1 ? null : [sorted[k - 1], sorted[n - k]] };
}
// The verdict for exactly these pairs at one look. `continue` means: not a look, or undecided below max_pairs.
export function lookVerdict(pairs, policy = verdictPolicy) {
  const n = pairs.length;
  const { median, interval } = medianInterval(pairs.map(pairDegradation), policy);
  const base = { paired_runs: n, median_decode_degradation_percent: median, interval_percent: interval };
  if (n < policy.min_pairs || n > policy.max_pairs || (n - policy.min_pairs) % policy.pair_step !== 0) return { ...base, status: 'continue' };
  if (interval[0] > policy.budget_percent) return { ...base, status: 'investigate' };
  if (interval[1] <= policy.precision_ceiling_percent) return { ...base, status: 'within-target' };
  return { ...base, status: n >= policy.max_pairs ? 'inconclusive' : 'continue' };
}
// Replays every look over a recorded sequence; the harness must have stopped at the first decision.
export function sequentialVerdict(pairs, policy = verdictPolicy) {
  for (let n = policy.min_pairs; n <= Math.min(pairs.length, policy.max_pairs); n += policy.pair_step) {
    const verdict = lookVerdict(pairs.slice(0, n), policy);
    if (verdict.status !== 'continue') {
      if (n !== pairs.length) throw new Error(`Activity sampling continued after the verdict at ${n} pairs.`);
      return verdict;
    }
  }
  throw new Error(`Activity sampling stopped at ${pairs.length} pairs without a verdict.`);
}
export function verdictRule(summary, policy = verdictPolicy) {
  const [low, high] = summary.interval_percent;
  const confidence = Math.round((1 - 2 * policy.tail_alpha) * 100);
  const head = `median ${summary.median_decode_degradation_percent.toFixed(2)}%, ${confidence}% interval [${low.toFixed(2)}%, ${high.toFixed(2)}%] over ${summary.paired_runs} pairs`;
  if (summary.status === 'investigate') return `${head}; slowdown: the interval lies wholly above the ${policy.budget_percent}% budget`;
  if (summary.status === 'within-target') return `${head}; within target: the interval reaches no further than ${policy.precision_ceiling_percent}% and does not lie above the ${policy.budget_percent}% budget`;
  return `${head}; measurement noise, not a slowdown: after the ${policy.max_pairs}-pair cap the interval still straddles the ${policy.budget_percent}% budget and reaches past ${policy.precision_ceiling_percent}%`;
}
export function modeSummary(mode, pairs, policy = verdictPolicy) {
  const verdict = sequentialVerdict(pairs, policy);
  const degradations = pairs.map(pairDegradation);
  const baseline = pairs.map((pair) => pair.off);
  const mean = baseline.reduce((a, b) => a + b, 0) / baseline.length;
  const cv = Math.sqrt(baseline.reduce((sum, value) => sum + (value - mean) ** 2, 0) / baseline.length) / mean * 100;
  const summary = { mode, status: verdict.status, paired_runs: verdict.paired_runs, median_decode_degradation_percent: verdict.median_decode_degradation_percent, interval_percent: verdict.interval_percent, baseline_cv_percent: cv, paired_range_percent: [Math.min(...degradations), Math.max(...degradations)] };
  return { ...summary, rule: verdictRule(summary, policy) };
}
// Pairs a recorded sample list (warmup first, then each mode's block) by adjacency: every pair is
// one `off` and one mode sample with the same pair index, `off` first on even pairs.
export function pairsFromSamples(samples) {
  const measured = samples.filter((sample) => sample.mode !== 'warmup');
  if (measured.length % 2 !== 0) throw new Error('Activity samples do not form whole pairs.');
  const byMode = new Map();
  for (let i = 0; i < measured.length; i += 2) {
    const [first, second] = [measured[i], measured[i + 1]];
    const off = first.mode === 'off' ? first : second;
    const on = first.mode === 'off' ? second : first;
    if (off.mode !== 'off' || on.mode === 'off' || first.pair !== second.pair) throw new Error(`Activity samples ${i + 1} and ${i + 2} are not one off/on pair.`);
    const pairs = byMode.get(on.mode) ?? [];
    if (on.pair !== pairs.length || (on.pair % 2 === 0) !== (first.mode === 'off')) throw new Error(`Activity ${on.mode} pair ${on.pair} is out of order.`);
    pairs.push({ off: off.tokens_per_second, on: on.tokens_per_second });
    byMode.set(on.mode, pairs);
  }
  return byMode;
}
