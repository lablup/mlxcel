# PR #2612: Read LLAMA_ARG_* toggle pair variables after parsing

**Date**: 2026-10-26
**Status**: Implemented and verified locally (draft PR, not yet merged)
**Languages**: Rust, JSON
**Risk Level**: Low

## Executive Summary

PR #2612 (issue #2609) fixes a parse failure in both server binaries: `LLAMA_ARG_MODELS_AUTOLOAD=<any value> mlxcel serve --no-models-autoload` exited with clap's "cannot be used with '--models-autoload'" error instead of letting the flag win. The cause is how clap 4 treats an `env`-bound argument that also has an `overrides_with` partner. The audit the issue asked for found the same defect on five more llama-server compatibility pairs: `--cache-prompt`, `--cont-batching`, `--context-shift`, `--cache-idle-slots` and `--kv-unified`. All six now read their `LLAMA_ARG_*` variable after parsing, and only when neither half of the pair is on the command line. The resulting precedence matches llama-server b10621: an explicit flag, either half, beats the environment, and the environment alone still applies.

## 1. Problem Statement

llama-server exposes many booleans as a `--x` / `--no-x` pair plus an `LLAMA_ARG_X` variable. mlxcel modelled these with clap's `overrides_with` between the two halves and an `env = "LLAMA_ARG_X"` attribute on the positive half. That combination is broken in clap 4.6.7:

- clap registers every `overrides_with` partner as a direct conflict (`gather_arg_direct_conflicts` in clap_builder's validator).
- clap counts a value that came from the environment as present.
- Only a command-line occurrence removes the overridden half before validation.

So with the variable set, `--no-x` meets a "present" `--x` and the validator reports a conflict. The override never runs. A deployment that sets the variable globally (a container image, a systemd unit) could not turn the feature off for one invocation, which is exactly the case the `--no-x` flag exists for. The bug predates #2584.

## 2. Change Summary

| Area | Change |
|---|---|
| `--models-autoload` (both binaries) | `env` removed from the clap arg in `src/main.rs` (`ServeArgs`) and `src/bin/mlx_server.rs` (`ServerArgs`). New `env_fallback_models_autoload(value, enable_was_set, disable_was_set)` in `src/server/cli_input.rs` reads the variable with `parse_env_bool` when neither flag was given, and logs a warning and ignores an unparseable value. `resolve_compat_toggle` still resolves the pair afterwards. |
| Five `Option<bool>` pairs | `env` removed from `--cache-prompt`, `--cont-batching` (`CacheCompatArgs`), `--context-shift` (`ContextCompatArgs`), `--cache-idle-slots` and `--kv-unified` (`SlotCompatArgs`). Each group gains `apply_env_bindings()`, built on a shared `cli::apply_boolish_pair_env(var, &mut value, negated)`. |
| Shared reader | `apply_boolish_pair_env` returns early when `value.is_some() || negated`, otherwise parses the variable with clap's `BoolishValueParser` vocabulary (`y/yes/t/true/on/1`, `n/no/f/false/off/0`, case-insensitive) and returns `(var, raw)` on anything else. |
| Aggregator | `server::apply_bool_pair_env_bindings(cache, context, slot)` runs the three groups and turns an invalid value into `"<VAR> has an invalid boolean value \"<raw>\""`. |
| Wiring | `mlxcel serve`: in `run_serve_async`, right after the settings-endpoint fallback, before the memory preflight (which reads `--kv-unified`) and before the Web UI router override of `--models-autoload`. `mlxcel-server`: in `build_startup_input`, next to `env_fallback_cors_credentials`. `cache_args::from_env()` now calls `apply_env_bindings()` after its empty-argv parse. |
| Compat manifest | The six entries in `compat/llama-server/b10621/router-models.json` and `runtime-and-context.json` change from `env_binding: clap` to `env_binding: runtime`, with `env_test: src/server/cli_input_tests.rs`. Doc comments still name each variable so `make verify-llama-compat` and `make verify-env-var-docs` keep matching. |
| Tests | `src/main_tests.rs`: `mlxcel_serve_models_autoload_env_yields_to_either_flag` (true/false/1/0 against both flags, plus env alone). The existing `mlxcel_serve_no_models_autoload_reaches_the_resolved_toggle` no longer unsets the variable, and the unused `ScopedEnv::unset` is removed. `mlx_server.rs`: `models_autoload_env_does_not_conflict_with_either_flag`, including env alone through `build_startup_input`. `src/server/cli_input_tests.rs`: models-autoload precedence and unparseable value; for the five pairs, env plus `--no-*`, falsey env against the positive flag, env alone, and an invalid value reported by name. |

## 3. Technical Decisions

### 3.1 Read the variable after parsing instead of working around clap

There were two other options. Dropping `overrides_with` would make `--x --no-x` on one command line a conflict too, which breaks llama-server's "last flag wins" behaviour that scripts rely on. Keeping `env` and catching the conflict error would mean string-matching clap errors. Reading the variable after parsing keeps clap responsible for the command line and keeps env precedence in code the project controls. It also follows a pattern the codebase already had: `--cors-credentials`, `--slots` and `--warmup` never used clap `env`, and `env_fallback_cors_credentials` was the template the issue pointed at.

### 3.2 Two detection mechanisms, chosen by field type

The five compat pairs store the positive half as `Option<bool>` with `num_args = 0..=1`, so after parsing `Some(_)` already means "the command line said something", and the negative half is a plain `bool`. `apply_boolish_pair_env` therefore needs no argv inspection. `models_autoload` is a plain `bool` with `default_value_t = true`, so parsing cannot distinguish "absent" from "given". It uses `long_cli_flag_was_set`, which scans the process argv for `--name` or `--name=`, as the CORS fallback does. The same signature also makes the function easy to drive in tests that cannot control the test harness argv: the `mlx_server` test passes the booleans directly.

### 3.3 Keep the old value domain for the five pairs

A variable that worked before should mean the same thing now. The flags still use `BoolishValueParser` for their optional value, so the env reader reproduces that vocabulary exactly rather than reusing `parse_env_bool` (which lacks `y`, `t`, `n` and `f`). An invalid value is a startup error naming the variable, as clap's binding was. `models_autoload` instead follows its CORS sibling: `parse_env_bool`, with a warning and fall-through to the default on a bad value. This asymmetry is deliberate in the PR but is worth knowing about (see Known Gaps).

### 3.4 Apply the fallbacks early in `mlxcel serve`

The values are consumed before `ServerStartupInput` exists: the memory preflight reads `--kv-unified`, and the Web UI path can override `--models-autoload` for router mode. Applying the fallbacks at the top of `run_serve_async` means both see the environment-derived value, as they did when clap filled the field. Applying them later would silently change preflight results for anyone who sets `LLAMA_ARG_KV_UNIFIED`.

### 3.5 Record the change in the compat manifest

The b10621 manifest states how each variable is bound. Changing `clap` to `runtime` and naming the test file keeps the verifier honest: a `runtime` binding needs a test that exercises it, and `env_test` points at one.

## 4. Validation

From the PR body, run locally (no hosted CI involvement claimed):

- `cargo test --profile test-fast --bin mlxcel --bin mlxcel-server`: 315 + 38 passed.
- `cargo test --profile test-fast --lib cli::`: 287 passed. `--lib server::cli_input`: 154 passed.
- `cargo clippy --profile test-fast --bin mlxcel --bin mlxcel-server --lib --tests -- -D warnings`: clean.
- `cargo fmt --all -- --check`, `make verify-llama-compat`, `make verify-env-var-docs`: pass.

The acceptance criteria in #2609 (both flags against true and false values, env alone, both binaries, a `ScopedEnv` test in `src/main_tests.rs`, compat verification) are each covered by a named test or check above.

## 5. Known Gaps

- **Strictness differs between the two readers.** An invalid `LLAMA_ARG_MODELS_AUTOLOAD` is warned about and ignored (autoload stays on), while an invalid value for the five other variables stops startup. `LLAMA_ARG_MODELS_AUTOLOAD=y` or `=t` is also rejected by `parse_env_bool`, whereas the pairs accept them. Whether the old clap binding on that `bool` field accepted those spellings was not checked; if it did, they are a small regression for that one variable.
- **`long_cli_flag_was_set` scans raw argv.** A token that equals `--models-autoload` but is really the value of another option, or that follows a `--` separator, would be counted as the flag. This is inherited from the existing helper and shared by the CORS fallback; no realistic invocation hits it.
- **Out of scope: `--cors-origins`.** `LLAMA_ARG_CORS_ORIGINS` uses `conflicts_with = "allowed_origins"`, so the variable also conflicts with `--allowed-origins`. That pair is documented as mutually exclusive rather than an override pair, so it was left as is.
- **No guard against reintroduction.** Nothing stops a future `--x` / `--no-x` pair from adding `env` alongside `overrides_with` again. A lint in `make verify-llama-compat` that rejects `env_binding: clap` on entries with a `--no-*` spelling would close this.
- **Not exercised end to end.** The tests drive parsing and startup-input construction; no server was started with the variables set.

## 6. Learning Points

- In clap 4, `overrides_with` is implemented as a conflict that the command-line occurrence order resolves. An env-sourced value takes part in the conflict but not in the resolution, so `env` and `overrides_with` on the same pair do not compose.
- When a flag field is `Option<bool>`, `Some` after parsing is a reliable "the user said something" signal. A `bool` with a default loses that information, which forces argv inspection.
- When moving a value reader out of a library, copy the library's accepted vocabulary instead of reusing a nearby helper that looks equivalent. The two boolean parsers in this codebase differ by four spellings.
