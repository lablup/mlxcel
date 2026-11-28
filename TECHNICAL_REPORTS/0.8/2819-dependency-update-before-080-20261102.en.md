# PR #2819: Bring Rust, web and pnpm dependencies to latest

**Date**: 2026-11-02 (generated before merge)
**Status**: In review
**Languages**: Rust, TypeScript, Python
**Risk Level**: Medium (TLS roots, template semantics and HTTP routing move with the new majors)

## Executive Summary

Every declared dependency moves to latest ahead of v0.8.0, and pnpm to 12.16.1, except hf-hub (0.5) and TypeScript (6.0.3). `make verify` passes, and 12 real checkpoints produce identical output, templates and tokens against the previous build.

## 1. Problem Statement

Security reports were arriving and many declared requirements were behind by a major version, so the maintainer asked to update everything before tagging v0.8.0.

## 3. Technical Decisions

- **ring stays the rustls provider.** reqwest 0.13's `rustls` feature selects aws-lc-rs, a C build the repository avoids. `rustls-no-provider` plus `src/http_tls.rs` keeps ring. Every process and test binary that builds a client must go through that helper, or client construction panics.
- **hf-hub is held at 0.5.** Version 1.0 forces aws-lc-rs through hf-xet and removes the sync API. reqwest 0.12 and native-tls therefore stay in the graph for hf-hub alone.
- **TypeScript is held at 6.0.3.** typescript-eslint 8.72 requires below 6.1, and TypeScript 7 removes the compiler API that `tests/models-fixtures.ts` uses.
- **A failed route chunk stays failed.** The old retry returned a new promise per render, so `use()` suspended forever. React 19.3.1 made this reliably visible. Browsers keep a failed dynamic import for the document's lifetime, so retrying without a reload never helped.

## 5. Learning Points

- pnpm 12 lockfiles open with a separate environment document that has its own `snapshots:` key, so text parsers must take the last one.
- reqwest 0.13 verifies against the OS trust store. A host with a custom corporate CA can now succeed, and a host with a damaged store can now fail, where bundled roots behaved the opposite way.
- minijinja 3 moves toward Python semantics (`//` floors, half-even `round`, Python list quoting). Identical output on 12 checkpoints does not cover every chat template.

## 7. Change Summary

| Area | Summary |
|---|---|
| Rust | axum 0.8, tower 0.5, tower-http 0.7, reqwest 0.13, sha2 0.11, symphonia 0.6, minijinja 3, tokenizers 0.23, safetensors 0.8, serde-saphyr 2, toml 1, and others |
| Web | pnpm 12.16.1, React 19.3.1, Vite 8.3.5, axe 4.14, lucide-react 1.66, Next 16.4 on the site |
| Fixes | Failed route chunk, two WCAG 2.5.3 names, site brand icons |

## 8. Follow-up Actions

- Move hf-hub when 1.x can run without aws-lc-rs, and TypeScript when typescript-eslint accepts 7.
- `minimatch` in `webpage/site` has 1 moderate dev-only advisory, waiting on eslint.
