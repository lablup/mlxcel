# Gemma 4 12B long-buffered MTP parity on M1 Ultra

## Scope and baseline

The [machine-readable evidence](gemma4-12b-mtp-exactness-m1-ultra-2026-10-15.json) retains commands, flags, host load, output hashes, timings, draft counters and artifact hashes.

Investigation of [#1986](https://github.com/lablup/mlxcel/issues/1986), measured on 2026-10-15 KST on Apple M1 Ultra / 128 GiB, Metal. This is **not the original M5 Max host**. M5 and CUDA are unmeasured here; the earlier CUDA-wide failure is not evidence for this Metal issue.

The diagnostic baseline is main `a269883588c3a3b40ef2bcc56096ee7f6435647b`, pinned MLX `243bdf09`, built with `cargo build --profile test-fast --features metal,accelerate --bin mlxcel-server`. A diagnostic-only patch adds the issue's opt-in long buffered draw, draw-list override and cache metadata logging without changing attention or default startup policy. The diagnostic executable SHA-256 is `b60e1cba62156c6dfa04c9875ce3f3601502370c60c8524bb6fe42af8bc9edc9`; its patch SHA-256 is `7ff9fb734998a6b8a8823a2c24e17024475813adda54a5505775bf1a86a10551`.

The immutable release before-arm was built from `c0ae26b2` (SHA-256 `4fdb66333c5862fbf7d6b14129c11fde6f017d4a27aae8090035a0346e7c27ba`). Four unrelated Kimi Linear, RWKV7, Bailing MoE Linear and RecurrentGemma prefill fixes landed before branch creation. Their diff changes no Gemma 4, Qwen, cache-core, or MLX code; nevertheless, the two source revisions are not identical and must not be described as such. Test-fast timing is diagnostic, not release throughput.

Checkpoints are `gemma-4-12b-it-4bit` and `gemma-4-12b-it-assistant-4bit`. Their `config.json` SHA-256 hashes are respectively `a17d0966c7e4be270f5ce096e286935ad4386d8262c9c6d6982e61ed4d50f286` and `09eec70b34616c5f48abca37b396a75a6f29776d548424228e6d0593fe49a844`. The target is Unified 12B: 48 layers, 16 query heads, 8 sliding KV heads of dimension 256, 1 full KV head of dimension 512, sliding window 1,024. Model aliases in this checkout were stale; measurements used the existing external model store directly without changing the aliases or downloading weights.

## Reproduction and localization

All startup runs use `--draft-kind mtp --draft-block-size 4 --parallel 1`, `MLXCEL_QMV_WIDE=0`, `MLXCEL_ENABLE_MTP_B1=1` and no inexact override. The diagnostic patch sets `MLXCEL_DIAG_GEMMA4_LONG_PROBE=1` and overrides draws with `MLXCEL_DIAG_GEMMA4_PROBE_LENGTHS`.

| Draw sequence | Ordinary verdict | Failure-only localization | Traced verdict |
|---|---|---|---|
| 8, 8, 8, 1056 | Three short passes; long draw: position 0, 209,372 / 524,288 logit bytes differ | Layer 8 sliding-attention output, position 2 | Same byte count |
| 1056 alone | Position 0, 232,915 / 524,288 bytes differ | Layer 5 full-attention output, position 0 | Same byte count |
| 1056 alone, repeat | Position 0, 232,915 / 524,288 bytes differ | Layer 5 full-attention output, position 0 | Same byte count |

Draw index changes the synthetic token salt, so the combined and isolated draws intentionally use different values. Final-logit comparison scans positions first; failure localization scans operations/layers first and only then positions. Thus the earlier layer-8 difference at position 2 does not imply that position 2 caused the final position-0 difference. The ordinary and traced full-logit mismatch counts agree for every run; the isolated repeat also reproduces exactly. Capturing intermediate tensors may change fusion, so localization remains a failure-only diagnostic rather than a proof of every uncaptured operation's arithmetic.

Both failing runs decline to classic decode, demonstrating fail-closed behavior. The first process reached health in 42.44 s (including cold model access and localization); isolated repeats took 16.36 and 16.38 s. These are not clean startup guard costs. Host load averages before/after the first run were 7.57/7.66/5.88 and 6.01/7.21/5.80; the isolated runs began at 5.86/7.14/5.79 and 6.36/7.14/5.88.

### Actual rotating-cache layout

The first six layers' cache snapshots establish the layout, rather than inferring it from logical context length. For layer 0 after the 1,056-token prefill, classic has offset/index 1,056 and K/V shape `[1,8,1056,256]`. Its next four single-token updates reduce storage to `[1,8,1024,256]` and write indices 1, 2, 3, 4 at offsets 1,057 through 1,060.

Enabling the speculative buffer records ring anchor `(1056,0)`, window 1,024, slack 32, chronological start position 32 and write index 1,024. Its allocation is `[1,8,1280,256]`; allocation capacity is **not** the visible reduction width. After the four-token verify, start remains 32, index becomes 1,028, offset 1,060 and the reference ring cursor becomes 4. Full-attention layer 5 uses `[1,1,1280,512]` storage with logical offset 1,056 before verify.

For query row 0, chronological visibility is logical tokens 33 through 1,056, whereas classic's physical order is `[1056,33,...,1055]`. At row 3 it is `[1056,1057,1058,1059,36,...,1055]`. These have equal mathematical visible sets but different floating-point reduction order. A candidate must also reproduce maskless single-query dispatch, not merely mask out future draft keys in a wider reduction.

## Isolated attention experiments

The diagnostic candidate adds only an opt-in `Attention::attend` selector for 12B sliding rows (16/8/256), then full rows (16/1/512). It does not widen the config predicate or change serving/prefill policies.

| Correction | 8,8,8,1056 | 1056 alone |
|---|---|---|
| Sliding only | Short draws pass; long first divergence moves to layer 29 full attention, position 0; 209,372 bytes differ | First divergence remains layer 5 full attention, position 0; 209,181 bytes differ |
| Sliding and full | All four draws byte-identical | Byte-identical |

This establishes the need for both attention corrections on this host and checkpoint. It does not independently establish the need for changing prefill chunking, cache donation, or scheduling policy; those require the integration checks below.

## Serving prefix-state control

Three deterministic 256-token completions use science, software and history prompts of 1,638, 1,338 and 1,238 input tokens, respectively. The same diagnostic executable serves every arm, with greedy sampling, seed 42, penalties disabled and `cache_prompt=false`. MTP uses width 4, adaptive sizing disabled and the slice grant limit disabled. Default classic serving uses 512-token prefill chunks.

All three attention-only MTP outputs differ from classic. A fresh-process classic repeat reproduces **all three classic output hashes exactly**, excluding baseline instability for these cases. A classic whole-prompt control (`--batch-size 16384`) instead reproduces **all three attention-only MTP outputs exactly**. MTP counters are nonzero: proposed/accepted/rounds are 287/160/96, 294/157/98 and 316/149/106. This is not a comparison of classic against a silently declined MTP arm.

Thus attention correction alone is insufficient: 12B also requires the target adapter to use classic's configured prefill chunking. The ordinary short startup test and the isolated long attention test both start from equal whole-prompt prefixes, so neither catches this serving-only prefix mismatch.

The product candidate reuses the existing exact-geometry predicate rather than adding an attention-only gate. This also keeps the correction inside its linear, unpadded query-layout contract, declines unmeasured Metal batched/nonlinear-tree execution, and prevents chronological buffered caches from being donated to classic automatic prompt-cache reuse. Those restrictions are necessary because row-wise attention intentionally drops structural masks and reproduces physical-ring order; a passing singleton startup draw does not certify other layouts. CUDA's per-row RoPE correction stays CUDA-only. Classic forward arithmetic and the 31B geometry are unchanged.

## Teacher-forced and integrated free-generation parity

The final test-fast candidate traces `docs/architecture.md` with 1,536 prefix tokens and 256 contiguous forced positions at width 4. Both block arms prefill through the actual MTP target adapter and its rotating buffer; the chain uses M=1 decode. The trace deliberately gives all arms the same whole-prompt prefix, separating attention parity from the serving prefix control above.

| Arm versus chain | Top-1 disagreements | Decided disagreements (reference gap >=2) | Largest disagreement gap | Reference-choice logit delta p50/p90/p99/max |
|---|---:|---:|---:|---|
| Ordinary block | 18/256 (7.031%) | 2/155 (1.290%) | 5.344 | 0.125 / 1 / 5.6875 / 8.4531 |
| Corrected verify | 0/256 | 0/155 | None | 0 / 0 / 0 / 0 |

Reference/corrected perplexity is 246.8612; ordinary block is 249.2134. These corpus values are diagnostics, not quality evaluations. The top-8 trace does not compare every vocabulary logit byte, unlike the startup probe.

With the integrated prefill policy, all three real-serving 256-token outputs now match the two stable classic reference runs byte-for-byte at the default 512-token prefill setting. Actual proposed/accepted/round counts are 317/150/106, 295/157/99 and 322/148/108. The strengthened startup guard passes with normal product settings, no diagnostic variables and no inexact override.

## Other target regressions and tests

Real MTP before/after runs use the same test-fast profile and source baseline, width 4, no inexact override and three 256-token completions per target. The non-QAT 31B target uses its bf16 assistant and the same long Gemma prompts (1,638/1,338/1,238 tokens). Qwen3.8 27B uses its matching 4-bit MTP checkpoint and Qwen chat delimiters (468/384/356 input tokens). All six before/after output pairs match byte-for-byte, with identical proposed/accepted/round counts:

| Pair | Science | Software | History |
|---|---|---|---|
| 31B + bf16 assistant | 269 / 166 / 90 | 314 / 151 / 105 | 287 / 160 / 96 |
| Qwen 27B + 4-bit MTP | 265 / 167 / 89 | 273 / 164 / 92 | 279 / 161 / 94 |

These are actual MTP regressions, not classic-only smoke tests. Their single-run timings overlap compilation and are not performance evidence. 31B QAT and Qwen bf16-drafter variants are not measured in this run.

Narrow tests pass: `gemma4::` 82 tests (one unrelated ffmpeg test ignored); `gemma4_tests::` 12 ordinary tests plus all 13 serial-only tests explicitly selected with `--ignored --test-threads=1`. This includes the exact-geometry/long-probe gate, 12B 16/8/256 attention at a realistic 1,024-token window, rollover and partial acceptance, and existing snapshot/cache tests. Formatting and cross-repository reference checks pass.

## Pending validation

Startup cost and repeated release throughput are not yet recorded. No completion or performance claim follows from the correctness runs, which may overlap compilation. M5, CUDA, ROCm and CPU remain unmeasured.
