# Technical Report: PR #1851 — WebUI contract gate

**Date**: 2026-09-12
**Status**: Pre-merge contract validation complete; runtime implementation follows in dependent issues
**Languages**: OpenAPI 3.1 / JSON Schema, TypeScript, Python, English/Korean documentation
**Risk Level**: Medium

## Executive Summary

PR #1851 establishes the shared contract for the bundled WebUI epic #1834, implementing its blocking child #1835. It adds schemas, generated TypeScript declarations, 30 JSON fixtures, state/UX specifications, and a CI drift/validation gate without adding server routes, model workers, or browser pages.

## 1. Problem Statement

Independent backend and frontend implementation requires more than endpoint names. Without pinned identity, resource ownership, event ordering, nullability, and action semantics, individually plausible implementations can disagree about whether a model is ready, whether memory has been released, or whether a missing measurement means zero.

The contract also separates the local model catalog from the existing downloaded-model listing and inference model identifiers. The UI must project the current runtime rather than introduce a competing registry or lifecycle controller.

## 2. Technical Decisions

- OpenAPI 3.1 is stored as JSON-compatible YAML and consumed by a pinned JSON Schema Draft 2020-12 verifier. Strict TypeScript declarations are generated deterministically; the gate rejects drift and unknown schema keywords.
- Stable model IDs hash canonical source/entry identity. Content fingerprints and revisions remain separate so rescans and content changes do not silently identify a different model. Canonical vectors and colliding-source fixtures constrain downstream implementations.
- `draining` and `unloading` retain resource ownership until worker exit is observed. A drain timeout is not permission to start a second worker. Request leases must eventually cover streaming response bodies, not only handler dispatch.
- Separately fetched resource snapshots fence SSE replay from their minimum sequence, with duplicate suppression against each resource's own snapshot. A maximum-sequence cursor would lose events between the older and newer snapshots. Gaps and process restarts force resnapshot, not automatic action replay.
- Next-load profiles expose only `ctx_size`, `n_parallel`, and `kv_cache_mode`. Request sampling, live partial-success settings, explicit CLI overrides, and startup-only settings remain distinct scopes.

## 3. Review and Regression Findings

Operation targets/results were changed from permissive bags to discriminated DTOs, and SSE envelopes require event identity and timestamps. Security review bounded identifiers, input fields, response collections, and same-origin API bases, while adding authentication-error and seeded-redaction fixtures.

Finalization found that a combined-invalid download test masked a remaining acceptance bug: `o/n` followed by a newline passed the repository-ID pattern because `$` can match before the final newline. Patterns now use the ECMAScript-compatible strict end assertion `(?![\s\S])`; independent mutations test each field. All 20 patterns were compiled with JavaScript as well as exercised through Python validation. Rust producers must translate lookaround into equivalent structural/full-string validation rather than copy it into the Rust `regex` crate.

The isolated `jsonschema` environment does not enable date-time validation automatically. An explicit checker now rejects malformed timestamps, impossible dates, missing offsets, and invalid offset ranges; the gate also fails when a schema requests an unavailable format checker. This is a validation dependency concern, not proof that a future runtime emits valid timestamps.

The hand-written verifier was split into CLI, reusable checks, and negative tests to keep each source module below 500 lines. The generated declaration file remains a single machine-produced artifact.

## 4. Validation and Limits

- Passed: 30 contract fixtures, generated DTO drift checks, identity vectors, allowed transitions/continuity, independent negative cases, and strict TypeScript 5.9.3 `--noEmit` compilation.
- Passed: JavaScript pattern compilation/trailing-newline rejection, Python byte compilation, license headers, `cargo fmt --check`, compatibility-manifest cases, workspace version consistency, kernel dtype-key checks, and diff whitespace checks.
- Cross-repository reference checking used its documented local fallback; added bare references were reviewed as issues in this repository.
- The orchestrator separately reported 27 CLI-help and 4 compatibility integration tests passing at the preflight baseline; those results are not live WebUI tests.
- No WebUI route, Rust serialization, model inference, GPU resource-release, CUDA, Safari, accessibility, or screenshot behavior is implemented or validated by this contract-only PR. Those remain gates for dependent issues, especially #1848. No runtime performance improvement is claimed.

## 5. Change Summary and Follow-up

The change surface is the shared architecture/API/state/UX documentation, generated declarations, fixture corpus, Python verifier dependencies, Makefile target, and CI job. Contract changes must update producer, consumer, fixtures, and requirement mapping together before dependent work proceeds.

Backend owners must validate actual Rust responses against these fixtures and enforce the lifecycle/security rules at runtime. Frontend owners must consume the generated DTOs and shared string/test-ID contracts. The integration owner must prove the complete system, including model-free startup and real checkpoint/browser tests, rather than treating this specification gate as product completion.

Related: [PR #1851](https://github.com/lablup/mlxcel/pull/1851), [contract issue #1835](https://github.com/lablup/mlxcel/issues/1835), [epic #1834](https://github.com/lablup/mlxcel/issues/1834).
