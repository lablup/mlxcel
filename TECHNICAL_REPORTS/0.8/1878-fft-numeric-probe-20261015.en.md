# PR #1878: Reliable FFT numeric and plan-cache probes

**Date**: 2026-10-15
**Status**: Implemented and validated on CUDA
**Risk**: Low; standalone diagnostic example only

## Summary

Promote the scratch FFT programs from #1825 and #1876 into a reusable mlxcel-core example. The default mode compares real and complex transforms with the CPU stream, while `--plans N` exercises distinct forward/inverse transform shapes under the backend's plan-cache limit. Integrate main `cb73c7d6` and MLX `243bdf09` through the mlxcelverse layout.

## Problem

The submitted stress mode printed numerical failures but always returned success, allowing a script to accept incorrect results. Missing or malformed plan counts silently became 40, zero performed no work, and unchecked counts could overflow the i32 transform or batch dimensions. The previous coverage also omitted odd real-transform lengths.

## Decisions and changes

Both numerical modes share comparison accounting: non-finite, negative, or out-of-tolerance errors remain failures even after later successful comparisons, and produce exit status 1. A strict parser rejects invalid, zero, extra, and arithmetic-overflow arguments with status 2 before device work. The upper bound comes from the i32 batch-times-length geometry, not a benchmark budget. Help returns success.

Add lengths 3, 7, and 401, including odd inverse transforms and a complex transform, for 35 comparisons. Preserve the CPU stream reference, evaluation within each device guard, deterministic inputs, and the 1e-5 tolerance. Clarify that shape count is not retained-plan count: forward/inverse plans may be evicted at the configured limit. Document backend features and build before applying a runtime timeout, so cold compilation cannot be mistaken for the historical ROCm hang. CPU-only builds remain degenerate smoke tests.

## Validation

On NVIDIA GB10, all 35 CUDA/CPU comparisons pass, with maximum reported relative error 9.538e-7. Forty distinct stress shapes complete with maximum round-trip error 8.946e-7 and no numerical failures. Both modes return 0. Help returns 0; missing, zero, malformed, overflowing, and extra arguments each return 2 in actual subprocess tests. Four unit tests cover parsing boundaries, geometry, and failure persistence including NaN/infinities and the tolerance boundary. Full CUDA workspace/all-targets Clippy with warnings denied, formatting, license-header, and diff checks pass.

Validated source tree: `590b55dd21d412bd7d071ce6d3ffbd66abd6f673`. Probe SHA-256: `8377fca87228e0d66d87c823735f5a9c1f0d7740745754952062d0ef4f845913`. Reports added afterward do not alter that production source.

The original PR recorded gfx1151 comparisons below 5e-7 and a 40-shape run at cache size 8, versus a hang near shape 17 at size 32. Those are historical measurements, not fresh results on this pin. AMD hardware is unavailable here; this PR supplies the reproducer and does not claim to fix hipFFT allocation or the cache-size hang.

## Change summary

One executable example with embedded unit tests and bilingual reports; no runtime backend, dependency, or cache policy changes. No blocking correctness, security, or performance issue remains in the diagnostic tool.

Refs #1825, #1876.
