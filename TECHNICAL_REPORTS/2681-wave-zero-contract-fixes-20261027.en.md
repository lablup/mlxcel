# Technical Report: PR #2681 - Wave Zero Contract Fixes

**Date**: 2026-10-27  
**Status**: Completed  
**Languages**: Rust  
**Risk Level**: Low

## Executive Summary

PR #2681 repairs two integration checks exposed after the parallel wave-zero changes for epic #2574 landed. It classifies the new distributed authentication setting without exposing secret-derived data and makes the model-owned state atomicity assertion tolerate only normal floating-point variation between separate model instances.

## 1. Problem Statement

The full workspace suite reported two failures. `ServerConfig` grew from 113 to 114 fields when `cluster_auth` was added, but the runtime-settings roster and schema had not classified the field. The DeepSeek-V4 atomicity test compared continuations produced by separate model instances with an absolute tolerance of zero; parallel execution exposed differences up to approximately `1.5e-7` even though rejected calls had not advanced state.

## 2. Technical Decisions

`cluster_auth` is a read-only distributed setting. Its schema value is only `{ "configured": true|false }`; the authentication handle and derived secret material never enter the settings response. A regression assertion constructs a real authentication handle and verifies that representation.

The strict B=1 routing check remains at zero tolerance. Only the post-rejection state-invariance comparison between separate model instances uses `1e-5`, the same threshold already used by the six-step model-owned parity tests. Malformed row counts and missing or duplicate IDs must still fail before mutation.

## 3. Change Summary

| Area | Change |
|---|---|
| Runtime settings | Classify all 114 fields in source order; expose only the authentication configured state |
| Batched decode tests | Apply the existing f32 tolerance to the separate-model atomicity comparison |
| Security | Prevent secret-derived authentication state from being serialized |

## Appendix: Validation

- DeepSeek-V4 model-owned multistep: 1 passed.
- Runtime-settings module: 8 passed.
- `cargo fmt --check` and `git diff --check`: passed.
- Full workspace rerun: pending in the epic integration gate; this report does not claim it has completed.

Related: #2574.
