# Technical Report: PR #3082 - Run the fused-MoE Dff harness test on Windows

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Low

## Executive Summary

The Windows build job's `tests/` Python suite failed in `test_bench_fused_moe_dff` for two reasons. `loadavg()` read `/proc/loadavg`, which exists only on Linux, and `main()` returned without closing `harness.log`, so Windows could not remove the test's temporary directory. `loadavg()` now falls back to `os.getloadavg()` and then to `"na"`, and `main()` opens its files through a `contextlib.ExitStack`, so every return path closes them. The job is path-filtered and rarely runs, which is how this reached main; #3069 triggered it.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files | `scripts/bench_fused_moe_dff.py`, `tests/test_bench_fused_moe_dff.py` |
| Tests | 2 new `loadavg()` fallback tests; the `tests/` suites pass (240) |
