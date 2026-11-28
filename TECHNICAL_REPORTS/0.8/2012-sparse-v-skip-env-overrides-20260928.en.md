# Technical Report: PR #2012 - Honor MLXCEL_BIN and MODEL Overrides in measure_sparse_v_skip_rate.sh

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Shell
**Risk Level**: Low

## Executive Summary

PR #2012 makes `scripts/measure_sparse_v_skip_rate.sh` honor the `MLXCEL_BIN` and `MODEL` environment variables as fallback defaults, replacing two hardcoded literal assignments. This brings the script in line with sibling benchmark scripts such as `scripts/bench_block_width.sh`, letting it run against a non-default release binary or model store without editing the file.

## 1. Problem Statement

### 1.1 Background

`scripts/measure_sparse_v_skip_rate.sh` measures the sparse-V post-softmax attention skip rate (#377) across a set of decode contexts. Several sibling benchmark scripts already read `MLXCEL_BIN` with a fallback default so they can be pointed at an alternate build directory; this script did not.

### 1.2 Existing Issue

- **Hardcoded paths**: `MLXCEL` and `MODEL` were assigned fixed literals (`./target/release/mlxcel`, `models/qwen3-4b-4bit`). A contributor building outside the default `target/release` tree, or without the default Qwen3 checkpoint on disk, had to edit the script directly to run it.

### 1.3 Risk Assessment

Low. The script is a local benchmarking tool, not part of the request path or CI. Left unaddressed, the friction shows up only when a contributor wants to run this particular benchmark against a non-default build or model store.

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | 2 |
| Lines deleted | 2 |

- `MLXCEL="./target/release/mlxcel"` becomes `MLXCEL=${MLXCEL_BIN:-./target/release/mlxcel}`.
- `MODEL="models/qwen3-4b-4bit"` becomes `MODEL=${MODEL:-models/qwen3-4b-4bit}`.
- The existing positional-argument parsing loop (`*) MODEL="$1"; shift ;;`) is untouched and still runs after the default assignment, so a positional model argument continues to take precedence over the `MODEL` environment variable.
- Default behavior when neither variable is set is unchanged: the same `mlxcel not found` error against the same default path.

## 3. Technical Decisions

### 3.1 `${VAR:-default}` parameter expansion, not a new flag

**Context**: The script already accepts a positional model argument and `--contexts` / `--decode-tokens` / `--outdir` flags. The alternative, a script-local `--bin` flag, would have made the binary path configurable without touching the environment.

**Rationale**: Sibling scripts, such as `scripts/bench_block_width.sh`, already establish `MLXCEL_BIN` as the repository's convention for pointing bench scripts at a non-default binary. Reusing that variable through the same shell parameter-expansion idiom keeps the convention consistent across scripts instead of adding a flag that only this script would have.

**Trade-off**: An environment variable is easier to leave set by accident than an explicit flag, but this matches the pattern already used elsewhere in `scripts/`, so a contributor familiar with one bench script gets the same override for free in this one.

## 4. Validation

- `bash -n scripts/measure_sparse_v_skip_rate.sh` and `shellcheck`: clean.
- Stubbed `MLXCEL_BIN` binary plus fake model directories: `MLXCEL_BIN` honored, `MODEL` env override honored, and a positional model argument still wins over `MODEL`.
- Unset-variable behavior confirmed unchanged: same `mlxcel not found` error against the same default `./target/release/mlxcel` path as before.
- Not run: an actual benchmark against a real model checkpoint (shared GPU, out of scope for an env-var plumbing change).

## 5. Related Work

- Issue #1661: source issue for this change.
- Issue #377: introduced `scripts/measure_sparse_v_skip_rate.sh` itself.
- `scripts/bench_block_width.sh`: sibling script whose `MLXCEL_BIN` pattern this PR mirrors.
