# PR #2724: Surface pre-subscriber info notices and guard against new ones

**Date**: 2026-10-28 (written before merge; the PR is open as a draft)
**Status**: Implementation complete at 3f816e06. The implementation review found no CRITICAL or HIGH items. Filtered local test runs passed (`cargo test --lib startup_notice`: 6 passed). `make verify-rocm-held` and the unfiltered `cargo test --bin mlxcel` were still to run, and hosted CI was partly pending (cargo-fmt and cargo-deny passed; cargo-clippy and the link jobs had not finished) at the time of writing
**Languages**: Rust
**Risk Level**: Low (no resolved value changes; what changes is which lines reach the log, at what level, and the signatures of about a dozen public `env_fallback_*` helpers and `GrammarSourceArg::load`)

## Executive Summary

PR #2724 (closes #2722) finishes the work #2704 started. #2704 collected the pre-subscriber warnings from the server's startup path into `startup_notices: Vec<String>` and logged them at WARN once `start_server` had installed tracing. Two gaps remained. The "CLI flag takes precedence over the environment variable" lines were still `tracing::info!` calls made before any subscriber existed, so they were written to nothing. And nothing prevented a new `tracing::` call from being added to that path and silently disappearing the same way.

The PR replaces the string list with a typed `StartupNotice { level, message }` (levels Info, Warn, Error), converts every pre-subscriber log line on the server startup path into a notice at its original level, and adds a source-scan test that walks the call graph from the startup roots and fails on any logging macro it reaches. The guard found three more dropped lines that the issue had not listed (the superseded-model INFO line in both binaries and the `--grammar` parse error), and those are converted too. It also gives the KV cache mode precedence warning a single emission point: `resolve_kv_cache_mode` no longer logs, and each caller that runs after its subscriber exists logs it explicitly.

## 1. Problem Statement

### 1.1 Precedence lines were dropped

Both server binaries (`mlxcel-server` in `src/bin/mlx_server.rs` and `mlxcel serve` in `src/commands/serve.rs`) resolve flags and environment variables in `build_startup_input` and `ServerStartupInput::into_startup_config` before `start_server` calls `initialize_server_logging`. `mlxcel serve` is explicitly excluded from the CLI's own `init_cli_tracing` (`src/main.rs`), because `server::startup` installs its own subscriber and a second `init()` would panic. Any `tracing` macro on that path therefore has no subscriber.

#2704 rerouted warnings but left INFO lines alone. The issue counted 14 precedence `tracing::info!` calls in `src/server/cli_input.rs` plus one in `src/cli/turbo_args.rs` and two in `src/cli/speculative_args.rs`; `thinking_budget.rs` and `chat_template_kwargs.rs` had more. An operator who set both `--cache-type-k` and `LLAMA_ARG_CACHE_TYPE_K` with different values got no indication which one won.

### 1.2 No protection against regressions

#2704 fixed the warnings it found by hand. The pattern that caused the bug (call `tracing::warn!` in an env fallback helper, which looks correct and compiles) was still the natural thing to write, and the test for #2704 only checked the notices it already knew about.

### 1.3 Duplicate KV cache precedence warning

`resolve_kv_cache_mode` logged the "split `--cache-type-k/v` flags win over legacy `--kv-cache-mode`" warning itself, and `into_startup_config` also pushed it as a notice. The server calls `resolve_kv_cache_mode` up to three times (early validation before a model download, the memory preflight, and `into_startup_config`), and `generate` twice (preflight and run). On the server every direct log was lost; on the CLI it could be printed more than once.

## 2. Change Summary

### 2.1 `StartupNotice`

New module `src/server/startup_notice.rs`, re-exported from `mlxcel::server`:

- `StartupNoticeLevel::{Info, Warn, Error}` and `StartupNotice { level, message }` with `info`/`warn`/`error` constructors, `contains`, and `Display` (the message only).
- `ServerStartupInput`, `ServerStartupConfig`, and `ServeArgs` carry `Vec<StartupNotice>` instead of `Vec<String>`.
- `log_startup_notices` (`src/server/startup.rs`) matches on the level and calls `tracing::info!`, `warn!`, or `error!`. Its call site is unchanged: right after `initialize_server_logging` and ahead of the router-mode branch, so router and single-model servers both report.

Every existing `notices.push(String)` becomes `StartupNotice::warn(...)`, so #2704's warnings keep their WARN level.

### 2.2 Precedence lines as INFO notices

Every precedence `tracing::info!` in `cli_input.rs`, `turbo_args.rs`, `speculative_args.rs`, `thinking_budget.rs`, and `chat_template_kwargs.rs` now pushes `StartupNotice::info(...)` with the same text. Helpers that only had precedence lines had no notice list, so they gain a `notices: &mut Vec<StartupNotice>` parameter: `env_fallback_cache_type_k/v`, `kv_quant_scheme`, `apc_hash`, `draft_model`, `log_file`, `embedding_model`, `reranker_model`, `lang_bias`, `draft_kind`, and `chat_template_kwargs`. Both binaries pass their list.

### 2.3 Lines the guard found

- The superseded-model INFO line (`superseded_model_notice`) in `mlx_server.rs`'s `build_startup_input` and in `serve.rs`'s `run_serve_async` is pushed as an INFO notice.
- `GrammarSourceArg::load` (`src/cli/grammar_args.rs`), called from `into_startup_config`, used `tracing::error!` for a GBNF grammar that fails to parse. It now takes `notices` and pushes an ERROR notice from `unparseable_gbnf_notice`. The behavior it reports is unchanged: as in llama.cpp b10621, the server still starts and every constrained request is refused with the parser's diagnostic. A grammar that fails only on a `<token-text>` terminal (which needs the model's vocabulary) produces no notice.

### 2.4 Source-scan guard

`src/server/startup_notice_tests.rs` (522 lines, almost all of it the scanner) adds:

- `startup_notices_guard_finds_no_tracing_before_the_subscriber`. It reads `cli_input.rs`, every `src/cli/*_args.rs`, `thinking_budget.rs`, `chat_template_kwargs.rs`, `model_source.rs`, `bin/mlx_server.rs`, and `commands/serve.rs`. Roots are every function named `env_fallback_*`, `into_startup_config`, or `build_startup_input`. A breadth-first walk follows any identifier in a body that names a function defined in those files and reports each `trace!`/`debug!`/`info!`/`warn!`/`error!`/`event!` it reaches, with the file, line, function, and the root it came from. The test also asserts that roots were found in each expected file and that exactly two `build_startup_input` roots exist, so renaming a root cannot quietly empty the scan.
- `startup_notices_guard_flags_a_tracing_call_in_an_env_fallback_helper`, which splices `tracing::warn!` into the real `env_fallback_kv_bits` source and asserts the scan reports it. This is the issue's acceptance criterion, run on every test pass instead of by hand.
- A synthetic case that pins transitive calls, comment and string blanking, and the `#[cfg(test)]` module exclusion.

Before scanning, `code_only` replaces comments (nested block comments included), the contents of normal and raw string literals, and char literals with spaces while keeping offsets and newlines, so a macro name in a doc comment or a `{` in a format string does not confuse the brace matcher or produce a false hit.

### 2.5 One emission point for the KV cache precedence warning

- `resolve_kv_cache_mode` no longer logs.
- New `log_kv_cache_mode_precedence_notice` in `turbo_args.rs` logs `kv_cache_mode_precedence_notice` through `tracing::warn!`. `generate` (the main run, not the preflight), `inspect`, and the in-process CLI server (`resolve_cli_kv_cache_mode`) call it once.
- `bench_decode` has no subscriber, so it prints the notice with `eprintln!` next to its existing mode-substitution line.
- The server keeps its single notice from `into_startup_config`; the early-validation and preflight calls to `resolve_kv_cache_mode` stay silent.

### 2.6 Tests

- `startup_notices_are_logged_once_a_subscriber_exists` (`src/server/startup_tests.rs`) sets a flag and a differing environment variable for 21 precedence cases, runs the helpers with no subscriber, then installs a capturing subscriber, calls `log_startup_notices`, and asserts each precedence line appears exactly once at INFO, the existing warning stays one WARN line, and the grammar error is one ERROR line. The issue's acceptance criterion requires it to fail on main, where the precedence lines are never captured.
- `startup_notices_reach_the_startup_config` checks that order and level survive `into_startup_config`.
- The grammar test checks one ERROR notice for malformed GBNF and none for a token terminal.
- Call sites in existing tests (`cli_input_tests.rs`, `speculative_args_tests.rs`, `chat_template_kwargs_tests.rs`, `grammar_args_tests.rs`, `main_tests.rs`) are updated for the new signatures.

## 3. Technical Decisions

### 3.1 A level per notice instead of a second list

The alternative was a separate `startup_info_notices` list next to the warnings. That splits ordering (the INFO line explaining which value won and the WARN line about the ignored value would be logged in two groups) and needs another field in three structs for each new level. One list of tagged items keeps the original order, and the ERROR case from `--grammar` fit without further plumbing.

### 3.2 Collect, do not buffer the subscriber

Another option was to install a buffering subscriber at process start and replay into the real one. That would catch every pre-subscriber line, including ones outside the scanned files, but it couples the binaries to a replay layer, interacts with the `--log-file` and log-format options that `initialize_server_logging` reads from the very config being built, and still needs care about double `init()`. The explicit list keeps the data flow visible in the function signatures: a helper that can report something takes `&mut Vec<StartupNotice>`, and the compiler makes every caller pass one.

### 3.3 Source scan over a runtime check

A runtime test can only cover the inputs it constructs, which is exactly how the #2704 gaps survived. Scanning source text covers every branch of every helper, including ones no test drives. The cost is that it is a heuristic parser, not `syn`: it resolves callees by bare name and does not know about modules, macros that expand to logging, or `use` aliases. The design leans toward over-reporting (any same-named function in a scanned file is followed), and the self-test plus the "roots must exist" assertions keep it from silently scanning nothing.

### 3.4 Leave `resolve_kv_cache_mode` pure

Making the resolver side-effect-free lets the server call it three times without three log lines, and moves the decision about where the line goes (notice, `tracing`, or stderr) to the caller that knows whether a subscriber exists.

## 4. Remaining Items and Unverified Parts

- **Scan scope.** Callees outside the scanned files are not followed. `into_startup_config` reaches code in other modules (for example `crate::lora::multi` and `crate::lang_bias`); none of those has a pre-subscriber `tracing` call today, but the guard would not catch one.
- **Roots are by name.** Code that runs before the subscriber but is not under an `env_fallback_*`, `into_startup_config`, or `build_startup_input` root is not scanned. Two examples exist at 3f816e06: `run_serve_async` in `serve.rs` (its one line was converted by hand in this PR, but the function itself is not a root), and `main` in `mlx_server.rs`, which logs `tracing::info!("autotune: applied {var}={value} from the tactic cache")` before `start_server`. That autotune line is still dropped; it is outside #2722's scope and only fires with `MLXCEL_AUTOTUNE` and a tuned entry.
- **Bare-name callee resolution** can follow a common name such as `load` or `new` from an unrelated call. This only produces false positives.
- **Test coverage gap.** The 21-case acceptance test drives `env_fallback_cache_type_k` but not `env_fallback_cache_type_v`. Both go through `apply_optional_string_env_fallback`, so the gap is in the test, not the behavior.
- **CLI visibility.** `generate` and `inspect` only install a subscriber when `RUST_LOG` is set (`init_cli_tracing`), so their KV cache precedence warning is still invisible by default. This was already true before the PR; only `bench_decode` (stderr) and the server (startup notice) show it unconditionally.
- **API change.** About a dozen `pub fn env_fallback_*` helpers and `GrammarSourceArg::load` gained a parameter. All in-tree callers are updated; out-of-tree users of `mlxcel::server` would need the same change.
- **Pending verification.** `make verify-rocm-held`, the unfiltered `RUST_MIN_STACK=16777216 cargo test --bin mlxcel`, and the remaining hosted CI jobs had not completed when this report was written.

## 5. Learning Points

- A logging macro called before the subscriber is installed is a silent no-op, and code review rarely catches it because the call looks correct. When startup has a "before logging exists" phase, give that phase an explicit output channel and make it part of the function signatures.
- A test that checks known messages proves the fix; a test that scans for the bug pattern prevents the next one. The scan here found three more instances on its first run.
- Text-level source scans need a self-test that injects a real violation into real source and a check that the roots still exist, otherwise a rename or a parser bug turns the guard into a test that always passes.
- Resolver functions that are called several times along a startup path should not log; let the single caller that owns the output decide.

## 6. Related

- Issue #2722; follows PR #2704 (issue #2697) and `TECHNICAL_REPORTS/2704-env-fallback-startup-warnings-guard-20261028.en.md`.
- #2664 (CORS origin variables), #1350 (KV cache mode substitution per model family), #1445 (early KV cache type validation).
