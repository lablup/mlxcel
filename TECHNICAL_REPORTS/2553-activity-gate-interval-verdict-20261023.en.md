# PR #2553: Judge Activity decode overhead by an interval, not 5 pairs

**Date**: 2026-10-23
**Status**: PR open; Closes #1925

## Executive Summary

The `WebUI installed artifact` job's Activity performance gate measured five alternating off/on pairs per mode and failed the mode when the median paired degradation exceeded the 2% budget or the baseline CV exceeded 5%. On the self-hosted GB10 runner `lablup-dgxspark21` the per-pair noise (SD 3.6 points, kurtosis 12 to 25) makes both conditions fire on unchanged code: #1925 and its comments record ten failed attempts on PRs that could not affect decode, including one that failed on CV while the Activity arm measured 6.76% faster. PR #2553 replaces the rule with a sequential, distribution-free test: the point estimate is the median of the paired degradations, the interval is the 98% sign-test interval `[x(k), x(n+1-k)]`, and each mode samples from 12 pairs in steps of 2 up to 40 until the interval decides `investigate` (wholly above 2%), `within-target` (upper bound at most 4%), or, at the cap, `inconclusive`, which fails with a message naming measurement noise. All constants move to one policy file read by the JavaScript harness and a Python port; the verifier re-derives every mode's verdict from the raw samples, and both languages are tested against one shared fixture of recorded sample vectors. A resampling estimate over 636 recorded runs gives 0.21% false failure per mode on a true 0% change and at least 95% detection at 5% and above. On the runner, five consecutive attempts of the unchanged tree passed, and an injected slowdown on a throwaway PR failed with `investigate` in both modes. The issue's suggested injections (a CPU burst, a delay keyed on Activity polls) did not reach decode on GB10 at all. 15 files, +3995/-120, of which 3278 lines are the fixture.

## 1. Problem Statement

The gate's verdict lived in `webui/scripts/activity-performance.mjs`: 5 pairs per mode, `status: degradation > 2 || cv > 5 ? 'investigate' : 'within-target'`. The same rule was enforced twice more with independent copies of the thresholds and counts: `scripts/webui/verify_activity_performance.py` required `paired_runs == 5`, re-applied `degradation > 2 or cv > 5`, and hard-coded the per-mode sample and preflight counts; `scripts/webui/summarize_activity_evidence.py` repeated the status, `paired_runs == 5`, `<= 2` and `<= 5` checks.

Two defects followed from that rule:

- **Five pairs cannot resolve 2%.** The per-pair spread in the failed runs was 6 to 21 points. Flat changes crossed the median threshold: PR #1920 (display-name strings only) failed one-visible at +3.38%, and PR #2279 (YAML parser only) failed one-visible at +4.65% on its second attempt.
- **CV measured drift, not overhead.** Baseline CV mostly records slow drift across the block, which the alternating pairs exist to cancel. In run 35425050365 attempt 1 the `off` rate fell from 171.4 to 141.7 tok/s over the two-visible block and the mode failed on CV 7.45 although the Activity arm was 6.76% faster. PR #2279 attempt 1 failed on CV 5.17 with every pair faster.

Each rerun on the single GB10 runner costs 10 to 40 minutes. The issue rejected raising the 2% or 5% thresholds, since that hides noise and real regressions alike. A maintainer comment on the issue added that host-level contention (a concurrent compile on the same machine, invisible to the GPU-process check) also moved scores, for example PR #1944 and #1946; that half was split into #1949 (host-quiet precheck), and this PR covers only the verdict.

## 2. Change Summary

| Area | Change |
|---|---|
| `webui/scripts/activity-performance-policy.json` (new) | The only copy of the constants: `budget_percent` 2, `precision_ceiling_percent` 4, `tail_alpha` 0.01, `min_pairs` 12, `pair_step` 2, `max_pairs` 40 |
| `webui/scripts/activity-performance-config.mjs` | Pure verdict functions: `validatePolicy`, `verdictPolicy` (loaded and validated at import), `signTestRank`, `pairDegradation`, `medianInterval`, `lookVerdict`, `sequentialVerdict`, `verdictRule`, `modeSummary`, `pairsFromSamples`; `performanceCompletion` gains `inconclusive` |
| `webui/scripts/activity-performance.mjs` | Sampling loop runs until `lookVerdict` leaves `continue`; prints `mode: status: rule` per mode; writes `verdict_policy` into the output; the old `median` helper (which took the upper middle element for even n) is gone |
| `scripts/webui/activity_performance_verdict.py` (new) | Python port of the same functions, operation for operation, plus `check_summary_consistency`, `failure_message`, `expected_counts` |
| `scripts/webui/verify_activity_performance.py` | Re-derives each mode from `samples`; exact match on `status`, `paired_runs`, median and interval; 1e-9 relative tolerance on CV and paired range; sample and preflight counts derived from `paired_runs`, replacing the two hard-coded count tables |
| `scripts/webui/summarize_activity_evidence.py` | No samples available, so it checks structure and internal consistency under the policy; requires `verdict_policy` to equal the repository's; emits `interval_percent` and `verdict_policy`; CV is no longer bounded above |
| `webui/scripts/activity-performance-fixtures.json` (new) | 22 cases: the ten recorded failing attempts, two concatenated null sequences, synthetic slowdown, speedup, drift, inconclusive and boundary cases, four rejection cases |
| `webui/scripts/activity-performance-verdict.spec.mjs` (new), `scripts/webui/activity_performance_verdict_tests.py` (new) | Both evaluate every fixture case and must reproduce `expected` exactly; JS also tests the sign-test rank, look schedule, policy validation and `performanceCompletion` |
| `webui/scripts/activity-stream.spec.mjs` | Imports the verdict spec (`package.json`, which lists unit files, is part of the WebUI bundle's source digest); visible-only tests cover `inconclusive` |
| `scripts/webui/*_tests.py` (verifier, summarizer) | Rewritten for variable `paired_runs` and the policy |
| `.github/workflows/ci.yml`, `Makefile` | The new Python test added to the helper-test lists; no timeout changed |
| `docs/webui-integration-matrix.md` | Performance row describes the new verdict |

Commits: `744c11f4` (the change), `a7cc03fa` (`/* global URL */` for lint, test script made executable, one blank line removed).

## 3. Technical Decisions

### 3.1 Median with a sign-test interval

Per-pair noise on this runner is heavy-tailed (kurtosis above 10 across 636 runs) and skewed (-0.7). The sign-test interval needs no variance estimate and no random bootstrap; its only input is `k`, the largest rank with `P(Binomial(n, 1/2) <= k - 1) <= alpha`. At `alpha` 0.01 per tail, `k` is 2 at n = 12 (98% interval `[x(2), x(11)]`, since `P(<= 1) = 13/4096`) and 13 at n = 40; at n = 6 no order statistic qualifies, which is why `validatePolicy` rejects a `min_pairs` too small for its `tail_alpha`. Because both languages sort the same doubles and the binomial sums are exact in a double up to n = 50 (hence `max_pairs <= 50` in validation), JavaScript and Python agree bit for bit, so the verifier can compare medians and bounds with `!=` instead of a tolerance. Rejected: mean with a t-interval (hurt by the heavy tails) and Hodges-Lehmann (assumes a symmetry the data lacks); both had similar power.

### 3.2 Three outcomes, and what the 4% ceiling means

- `investigate`: the interval lies wholly above the 2% budget. An interval touching 2.0 is not above it (fixture case `[2.0, 2.0]` is `within-target`).
- `within-target`: not `investigate`, and the upper bound is at most 4%.
- `continue` below the cap, `inconclusive` at 40 pairs.

The 2% budget is unchanged. The 4% ceiling is a precision requirement, not a second budget: a pass must have been precise enough to see a slowdown well past 2%. Under the resampling model, 4% gives 0.15% null `inconclusive` per mode while 3% gives 2.4%, which breaks the 1% false-failure target. Passing only when the interval lies wholly below 2% was rejected because it needs far more pairs, a tradeoff the issue had already recorded.

The cost of this asymmetry is in the PR's own table: a true 2% slowdown passes 81.6% of the time and a true 4% slowdown passes 12.9% of the time (78.6% `investigate`). The gate is built to catch effects of 5% and above (96.0%, 99.67% at 6%); between 2% and 5% it is a partial detector. The issue asked for at least 95% detection of the injected slowdown, which is why the acceptance injection was sized at about 6%.

### 3.3 Resampling estimate

Noise model: all 636 complete diagnostics artifacts on this runner (2026-09-15 to 2026-10-22, 6,360 pairs) centred on a true 0% change, iid pairs drawn from that pool plus a per-run shift N(0, 0.83^2). 0.83 is the between-run SD estimated from runs after #2119; the one-visible and two-visible means in one run are uncorrelated (r = -0.07). 20,000 simulated runs per row, sequential looks included:

| True degradation | investigate | inconclusive | within-target | mean pairs |
|---|---|---|---|---|
| 0% | 0.06% | 0.15% | 99.79% | 13.5 |
| 2% | 11.4% | 7.0% | 81.6% | 20.4 |
| 4% | 78.6% | 8.5% | 12.9% | 21.6 |
| 5% | 96.0% | 2.3% | 1.6% | 17.1 |
| 6% | 99.67% | 0.27% | 0.07% | 14.1 |
| 8% | 100% | 0% | 0% | 12.4 |

False failure on a true 0% change is 0.21% per mode, about 0.4% per CI job (two modes) and about 0.6% in `full` mode; with iid pairs and no run shift it is below 0.01%. The run shift dominates: at a shift SD of 1.5 false `investigate` rises to 2.4%.

### 3.4 Sequential looks, replayed by the validators

Looks happen only at `min_pairs + k * pair_step` (12, 14, ..., 40); `lookVerdict` returns `continue` at any other count. The harness stops at the first decision. `sequentialVerdict` replays every look over a recorded sequence and throws if the samples continued past a decision ("continued after the verdict") or stopped without one ("without a verdict"), so a recorded run that kept sampling after its decision, or stopped before one, is rejected. The resampling estimate includes the sequential looks, so the 0.21% already pays for the repeated testing.

### 3.5 One policy, three enforcers

The harness, the verifier and the summarizer used to carry their own thresholds. Now the policy file is the single copy, validated identically in both languages (exact key set, `ceiling >= budget`, `0 < alpha < 0.5`, even `min_pairs` and `pair_step`, `(max - min) % step == 0`). The harness writes the policy into its output, and both Python validators reject evidence produced under a different policy. The verifier, which has the samples, re-derives each mode with the Python port; the summarizer, which only sees summaries, checks that the status follows from the recorded interval, that the median lies inside it, and that `inconclusive` occurs only at the cap. Sample and preflight counts come from `expected_counts(paired_runs)`: one warmup, one `off` and one mode sample per pair, and preflight records equal to the mode's client count (1 for one-visible and hidden, 2 for two-visible) times `1 + paired_runs`.

### 3.6 Reporting and completion status

Every failing mode is reported as `mode: status (slowdown | measurement noise): median, 98% interval, pair count, rule`, both on the harness's stdout and in the verifier's assertion, so the job log separates `investigate` from `inconclusive` without downloading the artifact. In `full` mode `performanceCompletion` ranks `investigate` above `inconclusive` above `within-target`; under `visible-only-headed`, which CI runs, the top-level status stays `incomplete` and the per-mode verdict is what fails the job.

### 3.7 Warmup and wall time

The data show no first-pair bias (pair 0 mean -0.50 against -0.48 overall, no order effect), so no leading pairs are discarded. The existing warmup request runs 1.6% slower than the first baseline and was already excluded. Per pair, one-visible takes 7.2 s median (7.7 s max) and two-visible 8.4 s (9.2 s max), plus about 34 s before the first pair. In CI the worst case at 40 pairs per mode is about 12 minutes and the expected null run about 4 minutes (about 2 before); `full` mode's worst case is about 20 minutes, inside the 1800 s activity timeout. The issue set a 15-minute worst-case target.

## 4. Verification

Local checks from the PR test plan: `pnpm --dir webui run unit` and `lint`, `make verify-webui-helper-tests` (including `activity_performance_verdict_tests.py`), the verifier and summarizer test scripts, and `make verify-versions verify-env-var-docs verify-python-tooling`.

Fixture: the per-sample logs of every failing attempt named in #1925 and its comments (35423171315 a1; 35425297407 a1, a2, a3; 35425050365 a1; 35421166203 a1; 35567777700 a1 for #1944; 35588958676 a1 for #1946; 38265665905 a1, a2 for #2279) were copied before the artifacts expired. Concatenated in run order, their pairs decide `within-target` at 12 pairs in both modes (one-visible median -1.31%, interval [-4.50%, 1.83%]; two-visible -0.15%, [-4.99%, 2.45%]). With the Activity arm 6% slower the same pairs decide `investigate` at 14 pairs, and a drift case with baseline CV 12.19% decides `within-target`.

On `lablup-dgxspark21`:

| Run | Result |
|---|---|
| 39853154267 (test plan) | one-visible `within-target` (median -0.15%, [-1.83%, 1.25%]), two-visible `within-target` (-0.76%, [-2.64%, 0.47%]), 12 pairs each; verifier and summarizer accepted the evidence; verifier took 3 min 42 s |
| 39853154267, attempts 1 to 5 (jobs 120084449655, 120097378392, 120110916929, 120116442394, 120121248222) | Five consecutive green `WebUI installed artifact` runs at head `a7cc03fa` |
| 39870659579 on draft PR #2554 (closed, never merged) | Injected slowdown: one-visible `investigate` (median 4.58%, [3.89%, 5.68%]), two-visible `investigate` (8.05%, [7.11%, 9.62%]), 12 pairs each |

The injection sleeps 400 us per decode tick for each open `/ui-api/v1/events` stream, which the Activity page holds open exactly while visible.

Injections that did not work on GB10:

- A per-poll CPU burst of 80 spinning threads (run 39862888632) raised runnable tasks to 46 to 87 in the visible arms, but decode stayed at 152 tok/s in both arms.
- A delay keyed on Activity polls (runs 39866915190 and 39868894929) measured 0%. The instrumented run showed that no operations poll landed inside a decode window, because each poll was answered the moment a request finished.

## 5. Unverified and Open Items

- **True effect of the injection not pre-measured.** The issue's criterion asked for an injection "whose true effect is measured beforehand at 4 to 8 percent". A local build on the shared CI host was not permitted, so the effect is characterised only by the gate's own output: one-visible (4.58%, [3.89%, 5.68%]) is inside 4 to 8%; two-visible's median (8.05%) and upper bound sit just above it.
- **The recorded fixture cases cannot return `investigate` by construction.** At their recorded five pairs, n is below `min_pairs` and the sign-test rank at n = 5 is 0, so every one returns `continue` with a null interval. The acceptance item "none of the six runs returns `investigate`" is therefore satisfied structurally; the substantive null evidence is the concatenated 12-pair sequences and the resampling table.
- **Guarantee depends on run-level shifts.** The 0.21% false-failure figure holds while between-run shifts stay near the measured 0.83 points; at 1.5 it is 2.4%. Keeping host contention out is the job of the #1949 host-quiet precheck, not of this verdict.
- **2% to 5% slowdowns are often passed** (81.6% at a true 2%, 12.9% at 4%), as described in 3.2.
- **`full` mode not exercised.** The native hidden acceptance is still user-deferred; its 20-minute worst case is an estimate from per-pair timings, not a measured run.
- **CI wall time roughly doubles on a null run** (about 4 minutes expected, about 2 before); the measured acceptance run took 3 min 42 s in the verifier. `ci.yml` timeouts are unchanged.
- **Rule text formatting.** Python's `f"{value:.2f}"` stands in for JavaScript's `toFixed(2)` "for the magnitudes seen here"; the verifier does not compare the `rule` string, so a divergence would be cosmetic only.

## 6. Learning Points

- A fixed-count gate against a small budget fails on noise by design; size the sample from the measured noise and let the interval decide when to stop, and make "could not decide" a distinct, named failure so nobody reads noise as a regression.
- Pick a statistic both implementations can compute identically. Order statistics with an exact binomial rank need no randomness and no tolerance, which lets a Python verifier re-derive a JavaScript verdict with exact equality.
- When a rule is enforced in several places, move its constants into one file, have the producer stamp the policy into its output, and have consumers reject evidence produced under another policy.
- An injected slowdown has to share a resource with the thing being measured at the moment it is measured. On GB10, CPU pressure did not slow decode, and Activity polls were answered between requests, never during a decode window, so a poll-keyed delay measured 0%. A delay tied to the event stream the page keeps open during decode did reach it.
- Copy evidence that a fixture depends on into the repository before CI artifacts expire.
