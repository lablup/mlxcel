# Technical Report: PR #1429 - Refresh embedding and reranking CLI guidance

**Date**: 2026-08-26

**Status**: Completed

**Languages**: Rust, Markdown

**Risk Level**: Low

## Executive Summary

PR #1429 makes the recently shipped embedding and reranking interfaces discoverable from `mlxcel`, `mlxcel-server`, the README, and the maintained documentation set. It also turns `mlxcel arch` into a clearer architecture catalog by exposing public embedding-family names, explaining Qwen 3.6 and 3.8 aliases, and moving checkpoint-specific Muse details out of the architecture label.

## 1. Problem Statement

Embedding and reranking support had landed across multiple implementation PRs, but the public guidance had drifted: the detailed embedding document still described family ports as incomplete, the README lacked a runnable retrieval quick start, reranker environment variables were absent, and `mlxcel-server --help` did not explain standalone or side-model serving. At the same time, top-level help carried a checkpoint-specific Muse block and a hard-coded tensor-parallel family list that would age quickly.

The architecture catalog was exhaustive at the enum level but did not expose several public product names. In particular, Qwen 3.6 and Qwen 3.8 reuse the Qwen 3.5 implementation paths through `qwen3_5_moe` and `qwen3_5`, so users could incorrectly conclude that those versions were unsupported when no separate entries appeared.

Leaving these surfaces inconsistent creates operational risk rather than inference risk: users can choose the wrong server startup mode, overlook `/v1/rerank`, or misread an implementation alias as missing model support.

## 2. Change Summary

- Added `mlxcel embed` and `mlxcel rerank` quick-start examples and synchronized the documentation index, contributor guide, environment-variable table, supported-model page, and Unreleased changelog.
- Expanded `mlxcel-server --help` with embedding-only, cross-encoder-only, and chat plus side-model configurations, including the shared queue and timeout controls.
- Replaced model-specific top-level help with stable links to the supported-model and distributed-runtime documents.
- Added a serving-interface footer to `mlxcel arch`, mapped Qwen 3.6 to `qwen3_5_moe` and Qwen 3.8 to `qwen3_5`, and changed embedding entries to recognizable names such as LFM2.5-Embedding and Nemotron-3-Embed.
- Reduced Muse's architecture label to `Muse Glimmer 30B VLM`; precision, cache, feature, and checkpoint constraints remain in the supported-model documentation.
- Added rendered Clap help and architecture-catalog regression tests for both binaries.

## 3. Technical Decisions

### 3.1 Keep top-level help stable and route volatile details to maintained catalogs

Model qualification, supported precision, distributed constraints, and benchmark results change more frequently than the command surface. The top-level help now explains where to find those facts instead of embedding a snapshot for one checkpoint or a fixed list of tensor-parallel families, while `mlxcel arch` remains the local architecture inventory.

### 3.2 Present public version aliases without duplicating runtime variants

The runtime should continue using one `ModelType` per actual implementation path. The catalog display names now pair the public versions that share those paths, and the footer states the corresponding `model_type` keys, preserving implementation truth while answering the user's support question directly.

### 3.3 Test rendered help rather than only argument parsing

Existing parser tests proved that flags were accepted but could not detect missing descriptions or stale after-help text. The new tests render long help and the architecture catalog, assert the required commands, flags, names, endpoints, and documentation links, and reject the checkpoint-specific guidance removed by this PR.

## 4. Compatibility and Risk

- **Breaking changes**: None. Command names, flags, environment-variable behavior, endpoints, request schemas, and inference paths are unchanged.
- **New dependencies**: None.
- **Runtime impact**: None beyond human-readable help and architecture display strings.
- **Residual risk**: Future model aliases still require metadata and documentation updates, but regression tests now pin the current mappings and keep every registered `ModelType` visible.

## 5. Change Statistics

| Item | Value |
|------|-------|
| Files changed | 11 |
| Lines added | 292 |
| Lines deleted | 72 |
| Related commit | `e05bb5e` |

## 6. Validation

- `cargo fmt --all -- --check`
- `cargo clippy --fix --workspace --all-targets --allow-dirty -- -D warnings`
- `cargo test --bin mlxcel --bin mlxcel-server`: 214 passed
- `cargo test --test cli_help_consistency`: 25 passed
- `python3 scripts/ci/check_crate_versions.py`
- `python3 scripts/ci/check_kernel_dtype_keys.py`
- Manual inspection of built `mlxcel --help`, `mlxcel embed --help`, `mlxcel rerank --help`, `mlxcel arch`, and `mlxcel-server --help` output

## 7. Follow-up Actions

No required follow-up remains for this documentation pass. New model generations that reuse an existing architecture should add their public alias to `ModelType::metadata()`, the `mlxcel arch` alias note when needed, and the supported-model documentation in the same change.
