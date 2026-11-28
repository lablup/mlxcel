# Technical Report: PR #3078 - Regenerate the bundled xla decode goldens

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Low

## Executive Summary

Three `mlxcel-xla` lib tests had failed on main since #3048, because the bundled `decode.mlir` assets were never regenerated after the decode KV cache moved into module globals. This PR regenerates the `decode.mlir` of all 15 registered families with the crate's own `ArchFixture::emit_all`. The prefill and ragged-logits assets are unchanged.

## 1. Problem Statement

#3048 made the single-sequence decode keep its KV cache in module globals (`@kcache` / `@vcache`), so the decode `@main` no longer takes or returns the cache. The emitter changed, but the committed goldens kept the old signature. As a result, `emitter::tests::from_json_reproduces_bundled_assets_byte_for_byte`, `validation::tests::emit_all_reproduces_registered_goldens` and `validation::tests::registered_fixtures_are_byte_exact` failed: for llama-3.2-1b the first mismatch was at line 2, with 1350 golden lines against 1361 emitted. No PR job runs `cargo test --workspace`, so CI did not report it (#3075).

## 3. Technical Decisions

The emitter is the reference, not the goldens. The signature change is the stated intent of #3048, and the regenerated files differ only in the decode module. Re-freezing goes through `ArchFixture::emit_all`, the same primitive the validation tests compare against, so the assets come from the code path under test rather than from a hand edit. `assets/llama-3.2-1b/README.md` now describes the new decode inputs.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 16 (15 `decode.mlir`, 1 README) |
| Source changes | none |
| Tests | `cargo test -p mlxcel-xla --lib -- emitter:: validation::`: 220 passed, 0 failed, 17 ignored |

Related: #3048 (module-global decode KV), #3075 (issue).
