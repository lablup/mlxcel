# Technical Report: PR #1514 - KV Binary Byte Units

**Date**: 2026-08-30
**Status**: Open PR, ready for central merge
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #1514 corrects byte-size labels in the KV-transfer benchmark so values divided by powers of 1024 are printed with binary unit names. The change is limited to display strings and exact formatter assertions; benchmark measurement arithmetic, transfer strategy behavior, throughput units, and model execution paths are unchanged.

## 1. Problem Statement

The KV-transfer benchmark rendered byte counts as `GB`, `MB`, and `KB` even though the formatter divided by `1024` at each tier. Those labels describe decimal units, while the calculation was binary. The mismatch affected benchmark report strings and could mislead readers comparing KV cache size and transfer summaries.

The issue thread had no comments, so the issue body was the complete specification. The requested fix was to use `GiB`, `MiB`, and `KiB`, then update the formatter tests.

## 2. Technical Decisions

### 2.1 Fix display labels, not measurements

Only the labels changed. The byte counts, benchmark inputs, compression ratios, timing, and throughput calculations remain untouched. This preserves measurement behavior while making the rendered units match the existing arithmetic.

### 2.2 Cover both byte-size formatters in the benchmark module

The private `format_bytes` helper was the direct issue target. `TransferBenchConfig::total_size_str` used the same 1024-based formatting pattern in the same benchmark module, so it was corrected as the same report-string defect. Throughput remains labelled `MB/s` because it is not a byte-size formatter and was outside this issue's scope.

### 2.3 Replace substring checks with exact assertions

The old tests accepted any string containing `KB`, `MB`, or `GB`, which allowed the wrong labels to pass. The tests now assert exact outputs such as `2.0KiB`, `5.0MiB`, `2.0GiB`, and `256.0 MiB`.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 2 implementation/test files, plus reports |
| Source/test delta | 11 insertions, 11 deletions |
| Public API changes | None |
| Measurement logic changes | None |
| Hardware/model requirements | None |

- Changed `TransferBenchConfig::total_size_str` labels from `GB`/`MB`/`KB` to `GiB`/`MiB`/`KiB`.
- Changed the private KV-transfer `format_bytes` helper labels from `GB`/`MB`/`KB` to `GiB`/`MiB`/`KiB`.
- Updated `bench_config_total_size_str` to assert the exact `256.0 MiB` output.
- Updated `format_bytes_ranges` to assert exact byte, KiB, MiB, and GiB outputs.

## 4. Validation

- `cargo fmt --check`: passed.
- `git diff --check -- src/distributed/kv_cache_transfer/benchmark.rs src/distributed/kv_cache_transfer/benchmark_tests.rs`: passed.
- `cargo test --profile test-fast -p mlxcel distributed::kv_cache_transfer::benchmark::tests::format_bytes_ranges -- --exact --nocapture`: passed, 1 matching test passed, remaining tests filtered out.
- `cargo test --profile test-fast -p mlxcel distributed::kv_cache_transfer::benchmark::tests::bench_config_total_size_str -- --exact --nocapture`: passed, 1 matching test passed, remaining tests filtered out.
- `cargo clippy --profile test-fast -p mlxcel --lib -- -D warnings`: passed.

Broad workspace tests were not run because this wave-runner unit forbids broad cargo and workspace commands. Hardware/model validation is not applicable because this PR changes benchmark display strings and formatter assertions only.

## 5. Review Findings

| Finding | Severity | Resolution |
|---------|----------|------------|
| 1024-based byte-size values were labelled with decimal unit names | Low | Use binary labels `KiB`, `MiB`, and `GiB` |
| Formatter tests used substring checks that could miss label drift | Low | Use exact assertions for the relevant formatter outputs |

No security issue was introduced. The change adds no input parsing, I/O, unsafe code, public API surface, concurrency, caching, or resource-management behavior.

## 6. Remaining Limits

- Hosted CI should be read from GitHub before central merge.
- The PR does not attempt to standardize unrelated throughput labels such as `MB/s`; that unit is outside the byte-size formatter scope.
