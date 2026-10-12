# Server prompt-lookup break-even governor on GB10

**Date:** 2026-10-12
**Hardware:** NVIDIA GB10
**Model:** Qwen3-1.7B affine 4-bit
**Release source:** `b274abf05d79c5bf2616c537bcbb387693f10323`
**Result:** PASS

## Purpose

This measurement validates the server-only prompt-lookup break-even policy from issue #2264. The policy must remove the ordinary-chat regression without giving up the established copy-heavy gains.

The acceptance matrix requires:

- median paired chat delta of at least −1%;
- every short-copy request to retain at least 90% of its historical gain;
- paged 1024/2048/4096 and dense 8192 to retain at least 90% of their historical speedup above 1×;
- concurrency-two on/off ranges to show no disjoint regression;
- paged 8192 to remain closed to prompt lookup.

The CLI policy was not performance-tested here. Its configuration defaults remain unchanged; this report makes no CLI performance claim.

## Policy Under Test

The CUDA server opts into:

- emitted-token floor: 4 tokens per verify round;
- shadow confirmation: 7 consecutive tokens;
- proposal-free pipeline delay: 0;
- paged prompt-lookup context cap: 4608 tokens.

A round emits the accepted proposals plus the target model's bonus token. Once emitted-token EMA is below the floor, a partial low-yield round pauses lookup. A fully accepted short proposal remains active because it exhausted its available copy rather than rejecting proposed tokens.

One abrupt complete miss receives one additional probe only when the accepted-token EMA before that miss is at least four. Consecutive misses still pause. Full acceptance, a floor-meeting round, and shadow-confirmed resume reset the relevant low-yield history.

This zero-versus-partial distinction is a trace-derived field-boundary heuristic. In the captured copy workloads, complete misses followed accepted-token EMAs of 4.25, 4.981, and 6.982. Chat complete misses occurred below four except for q3's copy-identical `2/2 → 7/7 → 0/7` prefix. That prefix correctly received grace and added 7 proposed tokens, 0 accepted tokens, and 1 verify round; the other observed high-EMA chat lows were partial matches. These observations justify this workload boundary; they do not establish that zero acceptance is generally more profitable than partial acceptance.

## Method

All flag-on and flag-off arms used the same release binary and Qwen3-1.7B model. The chat test used eight chat-template questions, five alternating paired rounds, and 256 generated tokens. Short-copy used four requests over five paired rounds. Concurrency two used three rounds per prompt style. Long-copy used three rounds per storage/context cell.

The analyzer computes:

- chat from aggregate decode tokens divided by aggregate decode seconds per paired round;
- short-copy from each request's median paired gain;
- concurrency from median and full round ranges;
- long-copy from the ratio of median flag-on and flag-off decode rates.

No observations were filtered. All flag-on and flag-off rates below are decode tokens per second (tok/s).

## Final Results

### Ordinary chat, B=1

| Round | Flag on (tok/s) | Flag off (tok/s) | Delta |
|---:|---:|---:|---:|
| 1 | 152.867 | 156.110 | −2.077% |
| 2 | 155.904 | 154.055 | +1.200% |
| 3 | 152.473 | 152.894 | −0.275% |
| 4 | 157.006 | 154.012 | +1.944% |
| 5 | 152.221 | 155.604 | −2.174% |

Median paired delta: **−0.275%**, with range **−2.174% to +1.944%**. The −1% criterion passes.

### Short copy, B=1

| Request | Measured gain | Historical gain | Required floor |
|---:|---:|---:|---:|
| q0 | +159.283% | +61.736% | +55.563% |
| q1 | +157.037% | +53.320% | +47.988% |
| q2 | +151.289% | +65.449% | +58.904% |
| q3 | +148.602% | +66.599% | +59.939% |

All four requests pass their unchanged preservation floors.

### Chat-template concurrency two

| Style | Flag on median tok/s (range) | Flag off median tok/s (range) |
|---|---:|---:|
| Plain | 137.900 (135.900–141.700) | 136.900 (135.300–137.300) |
| Copy | 220.000 (216.600–225.100) | 217.400 (216.900–226.800) |

Neither style shows a disjoint regression. The concurrency criterion passes.

### Long copy

| Storage/context | Flag on (tok/s) | Flag off (tok/s) | Ratio | Required floor |
|---|---:|---:|---:|---:|
| Paged 1024 | 314.430 | 114.970 | 2.7349× | 2.521× |
| Paged 2048 | 255.000 | 90.460 | 2.8189× | 2.611× |
| Paged 4096 | 188.050 | 123.550 | 1.5221× | 1.378× |
| Dense 8192 | 246.140 | 104.850 | 2.3475× | 1.990× |

All preservation cells pass. The paged-8192 control reported zero prompt-lookup proposals and a 0.9841× ratio, confirming that the 4608-token gate stayed closed.

## Diagnostic Basis

Recording actual proposal length fixed a real distinction between a fully accepted `2/2` proposal and partial `2/7` acceptance, but attempt 2 showed that this correction alone did not restore copy gains.

A broad two-low-round grace restored the exact copy trajectory from 137 proposed / 112 accepted / 21 verify rounds to 239 / 214 / 35. Attempt 3 then passed every copy, concurrency, and gate criterion but failed chat at −1.419%. Compared with the earlier chat behavior, broad grace added 60 proposals, 8 accepted tokens, and 12 verify rounds.

The selective policy kept copy at **239/214/35** while reducing the exact eight-question chat workload from **226/83/43** under broad grace to **173/75/32**. Focused validation passed 54 governor/drafter tests and 20 scheduler tests. Temporary runtime tracing was removed before the final release build.

## Validation and Provenance

- Release build: PASS, 6m16s.
- Test build: PASS.
- Full CUDA tests: 8,936 root and 2,093 core tests; zero integration failures.
- Clippy and formatting: PASS.
- Contract checks: 3 PASS.
- Binary SHA-256: `2388e9f8ef914e11ca0e621801a0cad7bf4101fc359a572a49d6a352afae9485`.
- Native MLX pin: `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`.
- Host gate: zero wait; no CI job running.

The final analyzer exited zero with `"passed": true` and no failures.

## Evidence Archive

All attempts remain preserved under `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/`:

- parent directory and `attempt-1.md`: complete failed attempt 1;
- `attempt-2/`: intentionally stopped failed partial attempt 2;
- `attempt-3/`: complete failed attempt 3;
- `attempt-4/`: final passing matrix;
- trajectory diagnostic directories: causal and selective-policy counter evidence.

Each attempt has its own summary or checkpoint note and checksum manifest. Failed observations were retained rather than overwritten.
