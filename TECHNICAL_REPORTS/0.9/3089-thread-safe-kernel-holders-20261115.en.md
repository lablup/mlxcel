# Technical Report: PR #3089 - Initialize the lazy custom-kernel holders once across threads

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Medium

## Executive Summary

The 22 kernel holders in `mlx_cxx_kernels.cpp` built their `CustomKernelFunction` on first use behind a plain `if (!initialized)` flag, so two threads entering the same holder raced. In parallel `granitemoehybrid` test runs, the fused MoE launch threw `std::bad_function_call` or `__next_prime overflow`, and the process sometimes died with SIGSEGV. On an M5 Max, 7 of 10 runs failed at #3069's merge, against 0 of 10 just before it: the race predates #3069, which changed thread timing enough to expose it. A server that dispatches the same kernel from two threads on first use would hit it too. Each holder now initializes through `std::call_once`.

## 1. Problem Statement

The race was located by bisecting between the two commits and instrumenting the `.ok()` that `switch_layers::forward_fused_kernel` used to swallow the launch's exception.

## 7. Change Summary

| Item | Value |
|------|-------|
| Granite module, parallel | 12 of 12 runs pass (was 3 of 10 failing on main) |
| mlxcel-core lib suite, parallel | 3 of 3 runs pass |
