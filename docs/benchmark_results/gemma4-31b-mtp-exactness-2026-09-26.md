# Gemma 4 31B MTP exactness on M5 Max

## Scope and provenance

Measured on 2026-09-26, Apple M5 Max / 128 GiB, Metal, using the **test-fast** profile (`opt-level=3`, no release LTO). These are not release throughput measurements. Baseline: main `d7279c52`; corrected implementation: the PR closing #1983. A [machine-readable evidence summary](gemma4-31b-mtp-exactness-2026-09-26.json) records timings, output hashes, checkpoint config hashes and teacher-trace reports. Both binaries were rebuilt from their respective source trees; the old `target/release` binaries were not used. GPU jobs ran serially and throughput runs began after compiler processes exited.

Checkpoints came from the repository's `models/` store: `gemma-4-31b-it-qat-4bit`, `gemma-4-31b-it-4bit`, and the same `gemma-4-31b-it-assistant-bf16` drafter. The GB10 host was unavailable; **CUDA remains unmeasured**.

## Localization and correction

The failure-only diagnostic reruns the existing startup probe with per-layer attention-output, MLP-output, layer-output and final norm/head captures. Layer output additionally covers residual, per-layer input and scaling operations between the named sub-operations. Captures materialize tensors and can affect fusion, so the ordinary failing verdict remains authoritative if the rerun cannot localize it. Request forwards do not enter the capture scope or access its thread-local storage.

At `MLXCEL_QMV_WIDE=0`, the original short probe first differs in full-attention output: QAT layer 41, position 0; non-QAT layer 5, position 2. Thus the failure is not specific to QAT's 8-bit MLPs. Identical non-QAT failures at widths 2, 3 and 4 (227,042/524,288 bytes in this baseline build) reproduce the T=1 versus T>1 distinction. With the default wide kernel, the corrected binary still reports the historical 245,722-byte projection-path mismatch at all three widths, then automatically retries narrow and passes; no inexact override is required. QAT width 4 also passes the default automatic retry.

The startup probe now also covers a 1,056-token prefill and the actual buffered rotating cache. With only the initial full-attention correction, both targets diverged first at layer 0 sliding-attention output. Chronological buffered keys have the same visible set as classic decode's physical ring, but a different reduction order. Matching the physical ring moved QAT's first divergence to layer 5 full attention; matching classic's maskless single-query dispatch then made all startup draws pass. An all-zero mask can select different arithmetic despite hiding no keys.

The correction is limited to the measured 31B geometry (32 query heads, 16 sliding KV heads at dimension 256, 4 global KV heads at dimension 512), B=1 linear MTP verification. Projections remain batched; attention reproduces each M=1 query's physical key order and width. Classic forwards retain their arithmetic. Batched and nonlinear-tree MTP decline to the supported path. The Unified 12B configuration is not assigned this correction merely because it also has 512-wide full heads.

A further 6,449-token server control identified a separate prefix-state mismatch: default classic serving prefills in 512-token chunks, while the MTP adapter previously forwarded the whole prompt. Corrected verify matched a classic whole-prompt control byte-for-byte for all 513 generated tokens, but differed from default chunked classic. The 31B adapter now uses the scheduler's configured chunk size, evaluates each chunk and projects only its final row through the head, retaining the last chunk's assistant seed. Offline adapters use the same `MLXCEL_PREFILL_CHUNK` setting as classic CLI generation (default 2,048; zero means one pass). Other Gemma geometries keep their existing prefill path.

The shared rotating-cache implementation is used by Gemma 3, AFMoE, Gemma 4 and Muse Glimmer; the attention correction itself is restricted to the 31B Gemma 4 geometry above. A cache-owned reference ring anchor survives speculative rollback, buffer compaction, scheduler tick reconstruction, snapshot restore and detach/adopt. Old buffered snapshots without the anchor are declined for a cold prefill; ordinary unbuffered snapshots remain compatible. Malformed origins and K/V shapes are rejected before mutating state. Completed 31B MTP requests do not donate buffered rotating caches to the automatic prompt cache: ordinary decode and suffix prefill cannot reuse that chronological layout exactly. Unbuffered prompt snapshots remain available; in-flight slice parking retains live per-sequence state, and low-level detach/restore retains the metadata.

The short-mask warning is emitted once per process, then at DEBUG. Its effective causal/sliding visibility is covered by an identity-value attention test. This visibility test does not claim floating-point equality between chronological and physical-ring reductions.

The strengthened QAT startup probe took about 5.51 s from scheduler startup to its passing verdict in one run, versus about 2.26 s for the old short probe. This is startup work, not request latency. The additional long buffered draw is restricted to the measured 31B geometry; other Gemma variants retain their original three short draws. Failure-only localization can still add startup work on a failing other-Gemma probe.

## Teacher-forced trace

All arms used `MLXCEL_QMV_WIDE=0`, the same tokenized `docs/architecture.md`, one 1,536-token prefill and 256 contiguous forced positions at verify width 4. The block arms prefill through the real MTP target adapter, including the rotating buffer. Full acceptance advances the same cache across 64 blocks; free generation below covers rejection/rollback. “Original block” disables the correction in the trace harness; it reproduces the old attention arithmetic without changing classic decode.

| Target | Arm versus M=1 chain | Top-1 disagreements | Decided disagreements (reference gap >=2) | Largest gap at disagreement | Reference-choice logit delta p50/p90/p99/max |
|---|---|---:|---:|---:|---|
| QAT | Original block | 6/256 (2.344%) | 0/79 | 0.25 | 0 / 0.125 / 0.375 / 0.625 |
| QAT | Corrected verify | 0/256 | 0/79 | None | 0/0/0/0 |
| Non-QAT | Original block | 15/256 (5.859%) | 0/161 | 1.75 | 0.125 / 0.5 / 1.8125 / 3.625 |
| Non-QAT | Corrected verify | 0/256 | 0/161 | None | 0/0/0/0 |

Reference and corrected perplexities also match: QAT 1073.9542, non-QAT 20436.3508. These absolute corpus values are diagnostic, not a model-quality comparison. The startup probe compares full logit bytes; this trace reports top-k and reference-choice numerical metrics, so its zero deltas should not be described as a full-vocabulary byte comparison.

Reproduce the trace with the `MLXCEL_TRACE_GEMMA4` recipe in [benchmarks.md](../benchmarks.md#judging-a-change-that-moves-the-numbers), once per target.

## Free-running parity and throughput

Each target ran science, software and history prompts of 486, 402 and 374 input tokens, respectively, with 256 generated tokens per prompt. All **six** baseline-classic / corrected-classic / corrected-MTP triples produced byte-identical text. There was one measured request per arm and scenario (no statistical confidence interval). For each target the order was science, software, history; each scenario ran baseline classic, corrected classic, then corrected MTP in fresh server processes, with no cooldown. QAT ran before non-QAT.

| Target | Scenario | Classic before tok/s | Classic after tok/s | MTP tok/s | MTP / classic after | Accepted / proposed |
|---|---|---:|---:|---:|---:|---:|
| QAT | Science | 17.44 | 17.28 | 19.76 | 1.14x | 146/329 |
| QAT | Software | 15.96 | 13.82 | 11.71 | 0.85x | 133/367 |
| QAT | History | 9.54 | 8.97 | 10.87 | 1.21x | 139/349 |
| Non-QAT | Science | 12.86 | 12.89 | 17.80 | 1.38x | 167/264 |
| Non-QAT | Software | 12.75 | 12.20 | 15.99 | 1.31x | 157/293 |
| Non-QAT | History | 13.35 | 13.77 | 16.25 | 1.18x | 154/303 |

Classic before/after changes ranged from -13.38% to -0.94% for QAT and -4.32% to +3.18% for non-QAT. These samples do **not** establish unchanged classic throughput. A prior sweep also showed large drift (including non-QAT classic near 25 tok/s), so it is excluded from this table. A single host snapshot during the paired sweep reported 100% GPU utilization, 690 MHz, 68.6 C and thermal pressure `Fair`; it is evidence of operating conditions, not proof of the timing cause. Classic tensor arithmetic is unchanged and inactive tracing does no TLS access, but a stable performance comparison remains necessary before claiming zero overhead. MTP was slower on one QAT scenario; the static B=1 preference remains subordinate to exactness and adaptive profitability.

A single reverse-order software control ran corrected classic first (17.7342 tok/s), then baseline classic (17.7614 tok/s), a -0.153% difference with byte-identical output. It did not reproduce the earlier -13.38% gap. This supports the need to account for run conditions; one reversed pair still cannot establish a confidence interval.

### Other model checks

Classic before/after regression smoke used narrow QMV, 512-token prefill chunks, temperature 0, seed 42, disabled penalties and 64 generated tokens. `gemma-3-4b-it-4bit` used a 1,101-token prefix, beyond its 1,024-token sliding window, and produced identical 230-byte text; `qwen2.5-7b-instruct-4bit` used 1,100 input tokens and produced identical 352-byte text. Single-sample decode rates were 152.17/148.24 and 103.79/101.78 tok/s respectively (before/after); these short smoke timings are not performance evidence. These are classic-path checks, not Qwen MTP checks.

A development diagnostic temporarily applied the long buffered draw to the available Unified 12B target (`gemma-4-12b-it-4bit` with `gemma-4-12b-it-assistant-4bit`). It exposed a layer-0 sliding-attention difference of 275,832/524,288 bytes even after the narrow retry. Startup reached health in 13.08 s, with 6.08 s from cache setup to verdict. This unresolved long-context mismatch is tracked in [#1986](https://github.com/lablup/mlxcel/issues/1986) and needs separate baseline localization; it is not proof of a newly introduced arithmetic regression. The final change scopes the added long draw and buffered probe conversion to 31B, retaining 12B's existing short gate. No claim of 12B long-context exactness is made.

A separate actual MTP regression check used `qwen3.8-27b-4bit` and `qwen3.8-27b-mtp-4bit` at width 4: 498 input tokens and 256 generated tokens. Baseline MTP, corrected MTP and corrected classic produced byte-identical text. Both MTP arms proposed 274 tokens, accepted 164 and used 92 rounds. Their single-sample decode rates were 41.34/39.86 tok/s (before/after), versus 31.03 for corrected classic; timings remain diagnostic only. Qwen 3.6 had no compatible local drafter (the available target's hidden size was 2,048 versus 5,120 for the 3.8 drafter), so no actual Qwen 3.6 MTP validation is claimed.

## Long-prompt acceptance

The original issue's exact 6,449-token prompt was unavailable. The workload here is recreated by `scripts/bench_gemma4_mtp_exactness.py --long`, preserving both the beginning and final instruction, with 6,449 input tokens and a 513-token output cap. It must not be presented as the original issue's workload.

Both arms used the same request, effective narrow QMV, block size 4, singleton serving, 512-token scheduler prefill chunks, adaptive sizing disabled and slice grant limit disabled. The baseline required `MLXCEL_MTP_ALLOW_INEXACT=1` because its exactness gate failed; the corrected arm used the normal automatic narrow retry without that override. The adapter's actual prefill changes from whole-prompt to scheduler-sized chunks as part of this fix.

| Metric | Baseline, forced inexact | Corrected exact |
|---|---:|---:|
| Input / generated tokens | 6,449 / 513 | 6,449 / 513 |
| Accepted / proposed draft tokens | 265 / 739 | 280 / 698 |
| Acceptance | 0.3586 | 0.4011 |
| Verify rounds | 247 | 233 |
| Emitted tokens per verify | 2.0729 | 2.1974 |
| Zero / partial / full acceptance rounds | 104 / 99 / 44 | 92 / 87 / 54 |
| Draft time | 2,316.6 ms | 1,936.8 ms |
| Verify-forward time | 37,252.5 ms | 30,401.1 ms |
| Prefill / seed time | 22,230.6 ms | 11,471.0 ms |
| Decode throughput | 12.93 tok/s | 15.83 tok/s |

These are single samples taken at different times under the drifting host conditions above, so the timing difference is not a controlled speedup estimate. The acceptance and round counts describe each observed trajectory; the baseline is inexact and does not establish a greedy reference. The corrected 513-token output is byte-identical to corrected classic with the default 512-token prefill chunks (2,253 output bytes).

## Reproduction

Build with `cargo build --profile test-fast --features metal,accelerate --bins --example logit_trace`. For each classic arm, start `target/test-fast/mlxcel-server --model MODEL --parallel 1 --batch-size 512 --ctx-size 16384 --host 127.0.0.1 --port 19833`. For MTP add `--draft-kind mtp --model-draft models/gemma-4-31b-it-assistant-bf16 --draft-block-size 4`. Both use `MLXCEL_QMV_WIDE=0`; MTP measurements also use `MLXCEL_ENABLE_MTP_B1=1 MLXCEL_MTP_ADAPTIVE=0 MLXCEL_MTP_SLICE_GRANT_ROUNDS=0`, with `MLXCEL_MTP_ALLOW_INEXACT` unset.

Run `python3 scripts/bench_gemma4_mtp_exactness.py OUTPUT_DIRECTORY` and optionally repeat with `--long`. The script saves the exact requests, responses, generated text and timing summaries. It explicitly disables repetition/frequency/presence/DRY penalties and uses temperature 0, seed 42, `cache_prompt=false`; that flag bypasses both prompt-cache lookup and insertion. Compare the saved text files byte-for-byte.
