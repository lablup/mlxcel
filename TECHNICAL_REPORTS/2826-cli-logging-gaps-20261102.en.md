# PR #2826: Recover offline CLI warnings and widen the startup-notice guard

**Date**: 2026-11-02
**Status**: Completed (PR open; closes #2725)

## Executive Summary

PR #2826 closes the two lost-diagnostic gaps left by #2724. Offline CLI commands now install a stderr subscriber filtered at `warn` when `RUST_LOG` is unset, so environment-fallback and deprecated-flag warnings reach the user without setup, while stdout stays byte-for-byte unchanged. The startup-notice guard test no longer follows a fixed file list or matches callees by bare name; it scans every compiled `.rs` under `src/`, records module paths and enclosing `impl` types, and resolves each call form separately. Three files change, +608/-56, of which about 570 lines are the guard and its fixtures.

## 1. Problem Statement

`init_cli_tracing` returned early unless `RUST_LOG` was set. Every `tracing::warn!` and `tracing::error!` reached by `generate`, `inspect`, `tune` and the other offline commands was discarded by default, so a user only saw them after already knowing to opt in.

The guard added by #2724 followed calls only through a fixed file list and compared callees by bare name. A pre-subscriber helper in an unscanned file could hide its `tracing::` call, and an unrelated same-named method in another module could produce a false positive.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/main.rs` | `init_cli_tracing` installs the subscriber for every non-serve command; new pure `cli_tracing_filter(Option<&OsStr>) -> EnvFilter` |
| `src/main_tests.rs` | Filter selection test for unset, `debug` and `error` `RUST_LOG` values |
| `src/server/startup_notice_tests.rs` | Whole-`src/` scan, `FnDef` records with module and `impl` type, `use` parsing, call-form resolution, two new fixtures |

## 3. Technical Decisions

### 3.1 Default filter is `warn`, not `error`

`EnvFilter::from_default_env()` falls back to `error` when `RUST_LOG` is unset. The whole point of the issue is the WARN lines that were being lost, so the explicit default is `warn`. A set `RUST_LOG` is passed through verbatim, preserving the previous directive syntax and keeping `autotune: applied ...` at `info` and therefore opt-in.

### 3.2 `serve` stays skipped

`server::startup` installs its own subscriber and would panic on a second global `init()`. The guard remains `matches!(command, Commands::Serve(_))`, and `try_init` still degrades a future double initialization to a no-op instead of aborting.

### 3.3 Call resolution is deliberately narrower than a crate-wide name match

A literal crate-wide method-name match for `.f(` produced 470 false positives on the real tree. For example, `into_startup_config` calling `.resolve()` matched an unrelated `SpeculativeDispatch::resolve`. The receiver type is not knowable from source text, so `.f(` resolves only to `impl` methods in the caller's own module. Files named `*_tests.rs` are excluded because they compile only under `#[cfg(test)]` and created bogus `env_fallback_*` roots.

Bare `f(` resolves to free functions in the same file or brought in by `use`. `module::f(` resolves to free functions in that module. `Type::f(` resolves to methods of that type. The roots in `is_root` are unchanged.

## 4. Verification

| Check | Result |
|---|---|
| `cargo test --lib startup_notice` | 8 passed, including the real-tree guard |
| `cargo test --bin mlxcel cli_tracing` | 1 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | clean |
| `rustfmt --edition 2024 --check` on the edited files | clean |

## 5. Learning Points

The guard's value depends on resolution accuracy in both directions. Widening the scan without narrowing the match would have traded a false negative for hundreds of false positives, and a guard that fails on the real tree gets disabled. The chosen rule keeps the two required fixtures and the real tree green.

## 6. Unverified

The default `warn` subscriber was verified through the pure filter helper, not by running an offline command and capturing stderr. A live check would also confirm the formatting layer does not write to stdout.

## 7. Risks

Low. The change affects diagnostics only. Stdout is untouched by construction because the subscriber writes to stderr, and `serve` keeps its existing path.
