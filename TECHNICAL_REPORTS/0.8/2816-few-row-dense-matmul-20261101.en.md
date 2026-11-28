# PR #2816: Fix batched decode losses found by the M1 Ultra acceptance

**Date**: 2026-11-01
**Status**: Completed
**Languages**: C++, Rust
**Risk Level**: Medium (arithmetic routes on M1 and M2)

## Executive Summary

The #2587 acceptance run found reproducible B=2 and B=4 losses on M1 Ultra. Dense products of 2 to 6 rows now stay on GEMV on GPU generations 13 and 14, and `SwitchGLU::forward` takes the gate/up kernel at one token or on M3 and later.

## 1. Problem Statement

Generations 13 and 14 are excluded from `gemv_wide`, so dense products with 2 or more rows fell to the steel GEMM tile at several times a one-row GEMV. pythia-1b dropped to 165 tok/s at B=2, below its B=1. Separately, the #2514 gate/up kernel, measured only on M3 Ultra, lost about 10% against `gather_qmm` on M1 Ultra.

## 7. Change Summary

| Area | Summary |
|---|---|
| Metal | New matmul overlay, `MLXCEL_DENSE_GEMV_ROWS` |
| MoE | `switch_gate_up_preferred_in_forward` |
| Evidence | `docs/benchmark_results/versions/v0.8.0.md#few-row-dense-matmul-m1ultra-2026-11-01` |

## 8. Follow-up Actions

Residual gaps, base M1 and M2 measurement are tracked in #2817.
