# Technical Report: PR #1857 — Reproducible WebUI bundle

**Date**: 2026-09-12
**Status**: Pre-merge; local bundle/static-router validation complete, CI and downstream production integration pending
**Languages**: Rust, TypeScript/React, Python, English/Korean documentation
**Risk Level**: Medium

## Executive Summary

[PR #1857](https://github.com/lablup/mlxcel/pull/1857) implements the offline bundle foundation in #1836, following the contracts established by #1835 within epic #1834. The checked-in React assets embed in development and release Rust artifacts without runtime Node or a source checkout. This is not the production `--webui` startup implementation; #1838 owns that integration.

## 1. Problem Statement

A browser shell is not a self-contained distribution if development binaries read files from a checkout, Cargo silently runs a network-dependent frontend build, or generated assets drift from their source. Parallel page implementation also needs one regeneration pipeline and explicit payload budgets rather than independently maintained build recipes.

## 2. Change Summary and Decisions

- Strict TypeScript/React and Vite use a locked pnpm dependency graph and exact Node 26.5.1/pnpm 11.18.0 toolchain. The Python build entry point adds a source/file manifest after Vite; running raw `pnpm build` alone is deliberately not the complete shipping pipeline.
- Verification builds twice in independent temporary directories, compares bytes with each other and committed assets, and rejects stale inputs, missing assets, invalid shell structure, and budget overruns. No timestamp or absolute build path belongs in the manifest.
- The default `webui` feature enables optional embedding/MIME dependencies. `rust-embed` uses `debug-embed`, avoiding its development-time filesystem dependency. Disabling default features excludes the static module and embedded assets; backend features remain independently selectable.
- The reusable router owns only `/webui` and its assets, supports nesting under a validated API prefix, preserves health/API routes, and uses hash navigation instead of a broad SPA fallback. Static assets remain public; administrative API authentication and browser-origin enforcement belong to #1837.
- English/Korean contributor documentation explains feature combinations, canonical regeneration, router mounting, test scope, and relocated-artifact requirements.

## 3. Review and Security Findings

Review hardened weak/list ETag handling and ensured unsupported methods use the same security-header policy as ordinary, redirect, missing-path, and conditional responses. Encoded traversal, NUL, and backslash inputs are rejected; an unbootable index is not returned as a successful blank page. HTML and manifests revalidate while content-hashed assets can be immutable.

Security fixes retained the React, React DOM, and scheduler MIT notices in both `NOTICE` and an embedded license asset. The verifier now enforces exact toolchain versions, validates generated hashes/sizes and source digests, separates initial from total JavaScript budgets, and checks for debug artifacts and active external-origin references. These static checks complement, rather than replace, browser tests and later API security controls.

## 4. Validation Evidence

Local implementation/review runs reported these passing checks, independently repeated for the bundle and shared contract by the unit coordinator:

- Frontend frozen-lockfile installation, typecheck, lint, unit test, and Playwright Chromium scaffold test.
- `make verify-webui-bundle`: two clean builds matched checked-in bytes; bundle digest `1b50b860b289ce10a4fac31da4f9ecddafc7b5480ab25e942c54d40d13064cd3`. Stale-source, empty/blank-bundle, and missing-JavaScript negative cases failed as intended.
- Twelve focused Rust static-router tests (`cargo test --profile test-fast assets_tests --features webui`), feature-on/off checks including `cargo check --profile test-fast --no-default-features --lib`, and 30 shared contract fixtures.
- Formatting, license-header scope, documentation links, and whitespace checks. Local cross-repository-reference validation used its manual fallback; apparent unrelated numeric references were CSS colors.

The root validation run separately built and relocated **development and shipping-profile release** static harness artifacts from runtime/assets revision `dd918147`; subsequent `3d2259cc` changes were documentation only. Both recorded `result: pass`, two fetched JS/CSS assets, and a 453-byte HTML document. Sandbox negative controls confirmed source-asset reads and outbound network access were denied, with system tools only and no Node on PATH. The run exercised health/API sentinels, shell/manifest/assets, HEAD, 304, 404, and 405. Release compilation took 7 minutes 22 seconds.

The harness uses the production static router but stub health/model routes. This proves asset relocation independence, not production CLI startup or inference correctness. MLX dynamic-library and Metal-resource requirements remain separate from browser assets.

| Measured bundle | Bytes | Limit |
|---|---:|---:|
| Initial JavaScript gzip | 68,364 | 204,800 |
| Total JavaScript gzip | 68,364 | 716,800 |
| All embedded assets | 227,718 | 5,242,880 |

At report preparation, PR CI was still running and CUDA compilation was queued. CUDA dependency-graph isolation was inspected locally, but this report claims neither CUDA compilation nor inference success. No Safari, real-model, production-server relocation, or administrative authentication acceptance is inferred from the Chromium/static tests.

## 5. Learning Points and Remaining Integration

A reproducible frontend build, an embedded development artifact, and an offline relocated process are separate properties: byte comparison cannot substitute for denying source-file access at runtime. Likewise a static router harness isolates bundling well but must not be described as the completed WebUI server.

The startup owner must mount the router once under the validated prefix and rerun production relocation checks. Security/client/page owners must preserve same-origin policy and regenerate assets through the canonical pipeline. Later chat and visualization code must retain the budgets through lazy loading when needed; final macOS/Safari and real-checkpoint acceptance remains in #1848.
