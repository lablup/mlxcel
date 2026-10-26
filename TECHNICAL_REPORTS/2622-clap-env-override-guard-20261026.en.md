# PR #2622: Forbid clap env bindings on overrides_with flag pairs

**Date**: 2026-10-26
**Status**: Implemented and verified locally (draft PR, not yet merged)
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2622 (issue #2618) is the follow-up to #2612, which removed clap `env` bindings from six llama-server `--x` / `--no-x` pairs because clap 4.6.7 turns `VAR=... --other-half` into a parse error. This PR does two things. It adds a test-time guard, `mlxcel::cli::env_bound_override_args`, that walks a clap `Command` tree and lists every argument combining `env` with `overrides_with`; both binaries assert that the list is empty, so a new pair cannot reintroduce the bug silently. It also closes the strictness gap #2612 left open: `LLAMA_ARG_MODELS_AUTOLOAD` now goes through the same `apply_boolish_pair_env` reader as the other five pair variables, so an invalid value stops startup with the same message, and the single-letter spellings `y`/`t`/`n`/`f` are accepted.

## 1. Problem Statement

Two gaps remained after #2612.

- **No guard against reintroduction.** The defect in #2609 came from a combination that looks harmless in a derive attribute: `env = "LLAMA_ARG_X"` on one half of a pair whose halves `overrides_with` each other. clap counts an env-sourced value as present, registers each override partner as a direct conflict (`clap_builder` `parser/validator.rs`), and only removes the overridden half for a command-line occurrence. Nothing in the test suite or `make verify-llama-compat` would catch a future pair written the old way. The compat manifest checker reads JSON, not clap definitions, so it could not see the attribute either.
- **Two env readers with different rules.** After #2612, five pair variables were read by `cli::apply_boolish_pair_env`, which uses clap's `BoolishValueParser` vocabulary and returns an error on an unknown value, and both binaries turned that error into a startup failure. `LLAMA_ARG_MODELS_AUTOLOAD` was read by `parse_env_bool`, which rejects `y`/`t`/`n`/`f`, and only logged a warning on a bad value before keeping autoload on. llama-server b10621 rejects an invalid pair value at startup (`docs/llama-server-compat.md`), so the autoload behaviour was the outlier.

## 2. Change Summary

| Area | Change |
|---|---|
| Guard helper | `src/cli/mod.rs` gains `pub fn env_bound_override_args(&Command) -> Vec<String>` (`#[doc(hidden)]`). It recurses through subcommands, and for each argument with `get_env()` set it collects partners from both directions: arguments whose override list names it, and arguments it names in its own list. Each finding is formatted as `<command path> <arg id> (env <VAR>) <-> <partners>`. |
| Override list reader | `arg_overrides(&Arg)` extracts the ids from the `overrides: [...]` field of the `Arg` `Debug` output with a small quoted-string scanner (handles `\` escapes). It panics when the field is missing or the list is unterminated. |
| Guard self-tests | `cli::env_override_guard_tests`: env on the overriding half, env on the overridden half, no findings for a pair without env or env without a pair, and subcommand walking. |
| Binary guard tests | `no_arg_combines_clap_env_with_overrides_with` in `src/main_tests.rs` (`mlxcel` `Cli::command()`, all subcommands) and in the `tests` module of `src/bin/mlx_server.rs`. Both assert an empty list and print the offenders on failure. No allowlist. |
| Autoload reader | `env_fallback_models_autoload` in `src/server/cli_input.rs` now returns `anyhow::Result<()>`. It maps the `bool` field onto the `Option<bool>` shape `apply_boolish_pair_env` expects (`enable_was_set.then_some(*value)`, with `disable_was_set` as `negated`) and writes back only when a value was resolved. |
| Shared error | New private `invalid_bool_pair_env((var, raw))` builds `"<VAR> has an invalid boolean value \"<raw>\""`; `apply_bool_pair_env_bindings` uses it too, so the six variables share one message. |
| Call sites | `build_startup_input` (`mlxcel-server`) and `run_serve_async` (`mlxcel serve`) propagate the result with `?`. |
| Tests | `src/server/cli_input_tests.rs`: `models_autoload_env_ignores_an_unparseable_value` becomes `models_autoload_env_rejects_an_unparseable_value` (exact error text, default untouched, either flag still bypasses an invalid variable); new `models_autoload_env_accepts_boolish_spellings` (`y`, `t`, `YES`, `n`, `f`, `Off`). Existing callers in the three test files add `.expect(...)`. |

## 3. Technical Decisions

### 3.1 A Rust test over the real `Command`, not a manifest lint

#2612 suggested a lint in `make verify-llama-compat` that would reject `env_binding: clap` on `--no-*` entries. The PR instead inspects the clap `Command` both binaries actually build. The manifest only covers llama-server compatibility flags and depends on someone recording the binding correctly; the `Command` is the source of truth and covers every argument, including mlxcel-only pairs that will never appear in the manifest. The check runs in the ordinary `cargo test` binaries, so no new verify target is needed.

### 3.2 Reading a private field through `Debug`, and failing loudly

clap exposes `get_env()` publicly but has no getter for `overrides`. The options were to reimplement the knowledge in a parallel list (which drifts), to match on parse errors for synthetic env values at runtime (slow, and needs every arg exercised), or to read the `Debug` output. The PR reads `Debug` and makes the fragile part explicit: a missing or unterminated field panics with a message pointing at `cli::arg_overrides`, and the self-tests pin that a synthetic `env` + `overrides_with` pair is still detected. A clap upgrade that renames or reshapes the field therefore breaks the test run instead of quietly returning an empty list that would make the binary guards pass.

### 3.3 Check both directions of the override

The derive code for these pairs puts `overrides_with` on both halves today, but clap applies the conflict even when only one side declares it. The helper therefore reports an env-bound argument whether it overrides or is overridden. The self-tests build pairs where only the positive half declares the override and put the env binding on each half in turn, covering both lookup paths.

### 3.4 No allowlist

The issue states that no legitimate `env` + `overrides_with` case exists. An allowlist would invite a future exception for the very pattern the guard exists to stop. If one is ever needed, it can be added with a reason at that time.

### 3.5 Adapt autoload to the shared reader instead of adding a second strict reader

`models_autoload` is a plain `bool` with `default_value_t = true`, so it cannot use `Option<bool>` presence the way the five compat pairs do. Rather than give it its own strict parser, the PR translates its state into the shared reader's inputs: `--models-autoload` on the command line becomes `Some(current)`, `--no-models-autoload` becomes `negated = true`, and the env value is written back only when the reader produced one. The vocabulary, the error, and the precedence rules then come from one place, and the message is factored into `invalid_bool_pair_env` so the six variables cannot drift apart.

## 4. Validation

From the PR body, run locally:

- `cargo test --profile test-fast --lib cli::`: 291 passed, including the 4 guard self-tests.
- `cargo test --profile test-fast --lib server::cli_input`: 155 passed.
- `cargo test --profile test-fast --bin mlxcel --bin mlxcel-server`: 316 + 39 passed.
- `cargo test --profile test-fast --test cli_help_consistency`: 27 passed.
- `cargo clippy ... -D warnings`, `cargo fmt --all -- --check`, `make verify-llama-compat`, `make verify-env-var-docs`: pass.
- Negative demonstration: with `env` temporarily re-added to `mlxcel serve --models-autoload` and to `--cache-prompt`, both binary guard tests failed and named the three offending args. With the bindings removed, both pass.
- Manual: `LLAMA_ARG_MODELS_AUTOLOAD=maybe` makes both `mlxcel-server` and `mlxcel serve` exit with the shared error.

Full `make verify` was not run in the implementation worktree and is left to the merge gate, which the third acceptance criterion of #2618 requires.

## 5. Known Gaps

- **`Debug` scan starts at the first `overrides: [` substring.** In clap 4.6.7 the `help` and `long_help` fields are printed before `overrides`. Help text containing the literal `overrides: [` would be matched first and could yield a wrong list, including an empty one, without triggering the panic. No current help text contains it; anchoring the search on `, overrides: [` after the `settings` field would narrow this further.
- **Only `overrides_with` is checked.** `conflicts_with` combined with `env` (for example `LLAMA_ARG_CORS_ORIGINS` against `--allowed-origins`) has the same "env counts as present" behaviour, but it is a documented mutual exclusion rather than an override pair, and #2618 put it out of scope.
- **Global arguments.** The helper looks at `Command::get_arguments()` on the unbuilt command and matches partners within one command. An env-bound global argument at the root overridden by an argument declared in a subcommand would not be reported. No such argument exists today.
- **Whitespace handling changed for autoload.** `parse_env_bool` trimmed the value; `parse_boolish` does not. `LLAMA_ARG_MODELS_AUTOLOAD=" true"` was accepted between #2612 and this PR and is now a startup error, matching the other five variables and clap's own parser.
- **Behaviour change for deployments.** A deployment that sets `LLAMA_ARG_MODELS_AUTOLOAD` to an unrecognized value previously started with autoload on and a warning; it now fails to start. This is intended (b10621 parity) but is user-visible.

## 6. Learning Points

- When a library hides the state a lint needs, reading its `Debug` output is acceptable for a test-only guard if the parser fails closed and a self-test pins the format against a known positive case.
- A guard that returns "no findings" must be tested for its ability to find something; otherwise a format change makes it pass forever.
- When two fields of different shapes (`bool` with a default, `Option<bool>`) need the same env semantics, translating one into the other's inputs keeps a single reader and a single error message, which is easier to keep consistent than two parsers.
