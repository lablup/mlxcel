# PR #2976: Split the ragged load timing and restore decode-only benchmark throughput

**Date**: 2026-11-11
**Status**: Completed (draft PR; closes #2961)

## Executive Summary

The reported f16 `ragged-context-and-upload` regression is not a regression, and the `0.7 s` reference does not reproduce. A pre-#2933 control built from `92c07eb8^` (`fd2a65ea`) with an identical `Cargo.lock` and the same IREE ROCm distribution loads the same span in `4.89 s` at B=4 and `4.93 s` at B=8; HEAD is `5.62 s` and `5.03 s`, both inside the run-to-run copy-time spread. The cost is the f16 host expansion itself: dequantize `1.8` to `1.9 s`, pack to f16 `1.9 s`, and device copy `0.9` to `1.9 s` for `2,997,100,544` bytes across 146 tensors. It is flat in capacity.

The `20.6` to `22.5 tok/s` reading that motivated the issue has a different cause: #2936 moved engine construction inside the timed window of `xla_batch_bench`, so the one-time weight upload sits in the denominator. Decode-only the same runs measure `64.0 tok/s`. The benchmark now loads outside its timed region and prints the load times separately.

With that corrected, the #2933 absolute targets are mostly met: native B=4 `61.7` (target 60.0 to 62.2, pass), native B=8 `75.8` (target 74.5 to 76.7, pass), f16 B=8 `60.4` (target 59.8, pass), f16 B=4 `64.0` against `65.6`, which is 2.4 percent short and is reported as such.

## 1. Problem Statement

Issue #2961 was opened because the f16 ragged load appeared to take about `4.8 s` where a pre-#2933 binary took about `0.7 s`, and because the f16 arm measured `20.6` to `22.5 tok/s` at B=4 where #2933 recorded `65.6`. Two load-side functions (`upload_weights_with_sink`, `create_ctx_with_diagnostics`) are byte-identical to `92c07eb8^`, so the issue asked for a measurement rather than a guess: split the phase, test shape scaling, count uploads, reproduce the control honestly.

## 2. Measurement

### 2.1 Phase split

`report_load_timing` now reports `language-module-load` (instance, device, session, vmfb) apart from `language-weight-upload`, and `language-weight-upload-parts` appends `tensors`, `uploaded_bytes`, `dequantize_ms`, `pack_ms` and `copy_ms`. A capacity bucket prints the parts line with zero counters because it shares the base context's weights, which is also asserted by a device test.

Medians of three runs, `gfx1151`, capacity 256 unless stated:

| Arm | Cap | B | Load s | Module s | Upload s | Dequant s | Pack s | Copy s | Tensors | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HEAD f16 | 256 | 4 | 5.62 | 0.20 | 5.37 | 1.88 | 1.91 | 1.56 | 146 | 2,997,100,544 |
| HEAD f16 | 256 | 8 | 5.03 | 0.20 | 4.83 | 1.88 | 1.94 | 0.99 | 146 | 2,997,100,544 |
| HEAD f16 | 1024 | 4 | 4.92 | 0.20 | 4.71 | 1.80 | 1.91 | 0.98 | 146 | 2,997,100,544 |
| HEAD f16 | 1024 | 8 | 4.94 | 0.20 | 4.74 | 1.83 | 1.91 | 0.97 | 146 | 2,997,100,544 |
| HEAD native | 256 | 4 | 1.17 | 0.21 | 0.95 | 0.45 | 0 | 0.46 | 370 | 1,598,300,160 |
| HEAD native | 256 | 8 | 1.15 | 0.21 | 0.93 | 0.41 | 0 | 0.43 | 370 | 1,598,300,160 |
| Control f16 | 256 | 4 | 4.89 | - | - | - | - | - | - | - |
| Control f16 | 256 | 8 | 4.93 | - | - | - | - | - | - | - |

The control cannot print the new labels; its inner `language-vmfb-and-device-upload` covers the same span. The harness loads the ragged engine twice, and the phases are identified: phase 1 is the single-sequence reference engine (147 tensors, `3,522,437,120` bytes, about `12.6 s` upload), phase 2 is the ragged batch engine and is the only load with `ragged-context-and-upload`.

### 2.2 What the numbers say

- The module load is `0.20 s` in every row, so the vmfb instantiation is not the cost.
- The upload is flat in capacity and in bucket, so it is not a shape-dependent re-upload.
- Upload count is one per declared spec; a bucket context reports zero calls and zero bytes.
- The f16 upload is `3.0 GB` at 2 bytes per weight against `1.6 GB` for native packed words, and the host work (dequantize plus pack) is `3.8 s` of the `5.4 s` upload span. This is the cost, not a change from #2933.

## 3. The throughput reading

`xla_batch_bench` between #2936 and this PR created the batch engine inside `let t_run = Instant::now();`. With the load in the window, f16 B=4 reproduces the issue's number: median `20.8 tok/s` (`19.5`, `20.8`, `22.4`). Decode-only the same runs measure `64.0`. The benchmark loads first and prints `phase 2 load: ... <n> s` separately, so `tok/s` and `BENCH` lines now describe decode.

## 4. #2933 target status

Decode-only, three runs, medians:

| Arm | B | #2933 target | Runs | Median | Status |
|---|---:|---|---:|---:|---|
| f16 | 4 | 65.6 | 66.6, 61.2, 64.0 | 64.0 | 2.4 percent short, not a clean pass |
| f16 | 8 | 59.8 | 59.9, 60.4, 65.0 | 60.4 | pass |
| native | 4 | 60.0 to 62.2 | 59.8, 61.7, 70.7 | 61.7 | pass |
| native | 8 | 74.5 to 76.7 | 69.6, 75.8, 80.9 | 75.8 | pass |

Control reference points at the same protocol: f16 B=4 median `63.4` (`60.0`, `63.4`, `63.7`), B=8 median `63.5` (`56.3`, `63.5`, `71.2`). The host spread is 5 to 15 percent with a mild upward drift, so all runs are reported rather than a single median.

Every native run and every f16 B=8 run prints `REFERENCE-EXACT PASS`. The f16 B=4 arm diverges at request 2, token 3 with the synthetic random-id prompt artifact, in all three HEAD runs and in all three control runs; a natural-text artifact of the same shape prints `REFERENCE-EXACT PASS` at `63.1 tok/s`. The divergence is therefore a prompt-specific near-tie, not a path regression, and it is recorded rather than hidden.

## 5. Decision

The f16 ragged engine keeps f16-resident weights. Removing the `3.8 s` of host dequantize and pack would mean uploading packed U32 and dequantizing on the device, which is a different ragged graph with its own correctness surface and is outside this issue's scope. The measurement shows the load is not the reason the decode target was missed; the benchmark window was.

## 6. Verification

- `make verify-fmt` pass.
- `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings` pass.
- `cargo test -p mlxcel-xla --lib` 407 passed, 19 ignored.
- `cargo test --release -p mlxcel-xla --lib --features iree load_timing_accounting_tests` 3 passed.
- `cargo test --test llama_compat_manifest` 4 passed.
- `make verify-env-var-docs` 15 checks pass.
- `cargo test --release --features xla-iree,test-utils --test xla_ragged_hardening base_load_uploads_each_weight_once_and_a_bucket_uploads_none -- --ignored` 1 passed, asserting one upload per declared spec and zero calls and bytes for a bucket.
- Configurations: every run at `MLXCEL_XLA_CONTEXT_CAPACITY` 256 and 1024, B=4 and B=8, both arms, all under `scripts/rocm_gpu_guard.sh --hold`.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 7 |
| Lines added | +352 |
| Lines deleted | -44 |

Related: #2961 (this issue), #2933 (the targets), #2936 (the benchmark window change), #2955 (the report that found the discrepancy).
