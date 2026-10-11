# Technical Report: PR #2271 - B=1 terminal decode lookahead

**Date**: 2026-10-11
**Status**: Completed and verified on CUDA (GB10), with the Qwen prompt-cache limitation below
**Languages**: Rust, Python, Bash, Markdown
**Risk Level**: Low to medium

## Executive summary

The scheduler's B=1 lookahead pipeline submitted token `n+1` before reading pending token `n`. When `n` exhausted `max_tokens`, it unwound two speculative appends and synchronously replayed an already sampled token. PR #2271 checks the known length budget first, skips `n+1`, and commits the pending token through the existing fused finish path. This removes two complete forwards once per fused-lookahead-eligible capped B=1 request while preserving B>1 behavior.

Five-round GB10 measurements pass the required -1.0% median no-regression threshold in all four model/prompt cells. Every candidate range overlaps its repeated-baseline null range, so the evidence supports the threshold and does not establish a speedup.

## 1. Problem statement

Temporary host timers measured the pre-change Qwen B=1 server tick at 5,318.203 microseconds. Row lookup and build together were 0.0053%, metrics publication 0.0462%, provider channel send 0.0698%, and lookahead collection 0.0836%. Issue #2227 required candidates below 0.1% to remain unchanged, so this PR does not alter those paths. The 0.1445% fused finish call remains because it owns the shared sampling and finish transition.

Nsight's aggregate CUDA traces showed 126 more graph launches and 125 more graph-exec updates on the server than CLI. The traces include warm-up and measured requests and contain multiple graph calls per model forward, so they are corroborating evidence only. Scheduler forward-count tests establish the exact request-level cause.

## 2. Technical decisions and implementation

- The shortcut applies only when one sequence is active and its pending token is guaranteed to spend the length budget.
- The pending token still passes through shared fused token application, finish-state evaluation, event emission, and completion finalization.
- EOS wins when it coincides with the length boundary.
- A failed or malformed pending-token read uses the existing unwind and synchronous fallback.
- Seeded stochastic sampling commits the existing device draw without submitting a future draw or redrawing the terminal token.
- B>1 retains the previous next-submit, teardown, and synchronous replay behavior.

The live dense external-KV cycle-Llama fixture counts real scheduler submit and step calls. At `max_tokens=3`, optimized B=1 stays at two forwards while B=2 stays at four. Replacing the budget guard with `false` makes the test fail with actual four versus expected two, proving that the branch removes the measured work. Separate Muse tests cover generated IDs, finish reason, stream text, pool offset, and model-owned offsets without treating the non-rewindable Muse model as lookahead-eligible.

The change adds no public option, dependency, or storage-format change. Test-only counters are compiled under `cfg(test)` and have no release-path cost.

## 3. Performance results

The acceptance run used five interleaved quiet-host rounds against pre-epic baseline `4c44e317`, with a repeated-baseline null arm.

| Model | Prompt | Server dense vs baseline CLI | Null arm |
|---|---:|---:|---:|
| Qwen3-1.7B 4-bit | 256 | +0.57% (-0.25%..+1.54%) | +0.04% (-0.41%..+0.76%) |
| Qwen3-1.7B 4-bit | 8192 | +0.32% (+0.04%..+0.63%) | +0.22% (-0.13%..+0.66%) |
| Llama-3.2-1B 4-bit | 256 | +0.48% (+0.19%..+1.41%) | +0.07% (-12.38%..+0.54%) |
| Llama-3.2-1B 4-bit | 8192 | -0.16% (-1.15%..+0.05%) | -0.17% (-0.29%..+0.14%) |

All four medians pass the -1.0% threshold. Every candidate range overlaps its null range. The Llama 256 null arm contains -1.01% in round 2 and an unexplained -12.38% in round 3; both remain in the raw data and analysis without exclusion or rerun.

## 4. Correctness and validation

- Real `mlxcel run` generated text is byte-identical before and after the production change, with generated-text SHA-256 `ecbb6c3137fa73714f6fe6df1dd6414b14c6af75796efe19662a3aaab50eaea6` in both captures.
- Full Llama parity reports 13 identical pairs, zero divergent pairs, and one inapplicable pair.
- Qwen parity with prompt-cache cases excluded reports nine identical pairs, zero divergent pairs, and one inapplicable pair.
- Full Qwen parity still fails on prompt-cache miss versus hit at greedy token 58 (`374` versus `594`) and seeded token 61 (`576` versus `362`). Forced synchronous decode reproduces the same positions and token IDs while the new guard is unreachable. This is an existing prompt-cache-path limitation outside #2227, not a result of the terminal lookahead change.
- Formatting, CUDA workspace clippy, repository contracts, scheduler tests, and post-mutation binary restoration passed.
- The restored full CUDA suite reports 8,930 root tests and 2,072 core tests passed, with zero integration failures; 165 root and two core tests were ignored.
- The archived acceptance analyzer replays to `PASS (4/4 cells >= -1.00%)`, and every tracked evidence file matches the committed SHA-256 manifest.

## 5. Provenance and limitations

The release binary used for correctness and performance was built from `7c8830cd5aa01198e6ff93b661f90c048ccaeb1d`. Later checkout differences through `93e2f9ab` were confined to two test modules; the production-file diff is empty. During the after-run capture the source checkout contained the controlled guard mutation, but the release binary was never rebuilt. Its hash and archived source patch independently identify the executed code.

Repository policy excludes generated binary profiling captures. The original CLI `.nsys-rep` was 3,077,212 bytes with SHA-256 `646508acb7a4193870b5bbf1d93b090e6abc6667cda9da1783239ae63b176719`; the server capture was 3,147,242 bytes with SHA-256 `19efc62eaf08bcc9b25f3c79c1be9ccb43220f4d84ef1b0091bb5b04e38884ab`. Both remain in the run host's `/tmp/mlxcel-auto-plan/2227-profile/` scratch directory outside git. The committed Nsight statistics and logs, plain logs, instrumentation patch, profile analysis, acceptance JSONL, and correctness records preserve the evidence used by this report.

## 6. Change summary

| Category | Result |
|---|---|
| Performance | Removes two terminal decode forwards per eligible capped B=1 request; all four acceptance medians pass the no-regression threshold |
| Correctness | Preserves token output, finish semantics, cache offsets, stochastic draw order, EOS precedence, and B>1 behavior in targeted tests |
| Compatibility | No public API, dependency, model format, or B>1 behavior change |
| Known limitation | Full Qwen prompt-cache miss/hit parity remains divergent and is reproducible when the new branch cannot execute |

## References

- Issue #2227
- PR #2271
- `docs/benchmark_results/b1-decode-overhead-gb10-2026-10-11.md`
- `docs/benchmark_results/data/b1-decode-overhead-gb10-2026-10-11/`
