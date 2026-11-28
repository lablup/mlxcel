# PR #2668: Read the CORS origin variables after parsing so they no longer trip clap's conflict check

**Date**: 2026-10-27
**Status**: Open (draft), local verification complete
**Languages**: Rust, Markdown, JSON
**Risk Level**: Low (startup argument handling; CORS policy resolution unchanged)

## Executive Summary

`--cors-origins` and `--allowed-origins` are mutually exclusive in clap, and both bound an environment variable through clap's `env` attribute. clap 4.6.7 counts an env-sourced value as present when it checks `conflicts_with`, so setting one variable and passing the other flag was a parse error in both `mlxcel serve` and `mlxcel-server` (issue #2664). PR #2668 removes the clap `env` binding from both arguments and reads `LLAMA_ARG_CORS_ORIGINS` and `MLXCEL_ALLOWED_ORIGINS` after parsing, with a command-line flag winning over the other side's variable. The notice for an ignored variable is carried to `start_server` and logged after the tracing subscriber exists, so it is visible. The env-binding guard in `src/cli/mod.rs` now also reports env-bound arguments on either side of a `conflicts_with` pair.

## 1. Problem Statement

Reproduced on main (4246aa87) with `target/release/mlxcel serve`, in both directions:

```
$ LLAMA_ARG_CORS_ORIGINS=http://a.example mlxcel serve --allowed-origins http://b.example ...
error: the argument '--allowed-origins <ORIGINS>' cannot be used with '--cors-origins <ORIGINS>'
$ MLXCEL_ALLOWED_ORIGINS=http://a.example mlxcel serve --cors-origins http://b.example ...
error: the argument '--cors-origins <ORIGINS>' cannot be used with '--allowed-origins <ORIGINS>'
```

The same pair exists in `src/bin/mlx_server.rs` and `src/main.rs`. The operator never typed the conflicting combination: an inherited variable was enough to stop startup. The existing guard `env_bound_override_args` (#2618) only looked at `overrides_with` pairs, so it did not catch this shape.

## 2. Change Summary

| Area | Change |
|---|---|
| Argument definitions | `cors_origins` and `allowed_origins` drop clap's `env` in both binaries. Their doc comments say the variables are read at runtime, and `--help` no longer prints `[env: ...]` for them. |
| Runtime fallbacks | New `env_fallback_cors_origins` and `env_fallback_allowed_origins` in `src/server/cli_input.rs`, called from `src/commands/serve.rs` and `src/bin/mlx_server.rs` after parsing and before model resolution, with `long_cli_flag_was_set` reporting each flag. |
| Notices | The helpers return the "ignored" warning instead of logging it. Both binaries collect it into `ServerStartupInput::cors_origin_notices`, `into_startup_config` carries it into `ServerStartupConfig::cors_origin_notices`, and `start_server` calls `log_cors_origin_notices` right after `initialize_server_logging`, ahead of the router-mode branch. |
| Guard | `env_bound_override_args` adds `arg_conflict_partners`, which uses clap's public `Command::get_arg_conflicts_with` in both directions. Entries read `<path> <id> (env VAR) conflicts with <ids>`. |
| Docs and compat | `docs/environment-variables.md` describes the runtime reading and the precedence. The b10621 manifest entry `transport-tls-cors.json` moves to `env_binding: runtime` with an `env_test`. |

## 3. Technical Decisions

### 3.1 Precedence

| Command line | `LLAMA_ARG_CORS_ORIGINS` | `MLXCEL_ALLOWED_ORIGINS` | Result |
|---|---|---|---|
| `--cors-origins` | any | set | Flag applies; allow-list variable ignored with a warning |
| `--allowed-origins` | set | any | Flag applies; b10621 variable ignored with a warning |
| neither | set | unset | Variable applies as before |
| neither | unset | set | Variable split on `,`, then the existing origin validation |
| neither | set | set | Startup error naming both variables |
| both flags | any | any | clap parse error, unchanged |

A flag's own variable is unused when the flag is given, which matches clap's former behavior. Both variables without a flag is an error because the two rules are mutually exclusive and nothing tells which one the operator meant; the old clap conflict also failed in that case. A non-UTF-8 value is a startup error, as clap's binding was.

### 3.2 Why the notice is returned rather than logged

The fallbacks run before `start_server` installs the tracing subscriber, and `mlxcel serve` skips `init_cli_tracing`, so a `tracing::warn!` inside them reached nothing. The first commit did exactly that, and the implementation review reproduced the silent start. This matters most for `--cors-origins '*'` with `MLXCEL_ALLOWED_ORIGINS` set: the allow-list gives way to a wildcard, which is the intended precedence but should not happen without a message. The second commit returns the text, threads it through the startup structs (via a skipped `ServeArgs` field for `mlxcel serve`), and logs it once logging exists. Other `env_fallback_*` warnings were left as they are to keep the change scoped.

### 3.3 Guard extension

clap enforces a conflict in both directions whichever side declares it, so `arg_conflict_partners` collects the arguments a given argument conflicts with and the arguments that declare a conflict with it. The binaries' existing tests assert that the guard list is empty, so a new env binding on a conflicting argument fails the build's tests.

### 3.4 Security notes

The security review found no CRITICAL or HIGH issues. With neither flag, behavior is as before. Env-sourced allow-lists go through the same `parse_allowed_origins` validation as the flag. `MLXCEL_ALLOWED_ORIGINS=` still fails startup, and `LLAMA_ARG_CORS_ORIGINS=` still becomes an empty literal, not `*`. When both fields end up populated, `CorsPolicy::resolve` picks the non-empty allow-list, which fails closed. `long_cli_flag_was_set` matches `--flag value` and `--flag=value`; neither binary enables `infer_long_args` or aliases these flags, and the only `allow_hyphen_values` arguments are the integer `--parallel` / `--n-parallel`.

## 4. Validation

- Per binary: the flag wins over the other side's variable in both directions, a lone variable still applies, two flags still conflict, and both variables fail startup naming both. The `mlxcel-server` both-variables check runs in a child process so it cannot race tests that call `build_startup_input` without the env lock.
- `src/cli/mod.rs`: guard unit tests for an env binding on the declaring side and on the conflicted side, plus a no-env case.
- Notices: each helper returns the notice for the overridden variable (library and both binaries), `into_startup_config` carries it, and `log_cors_origin_notices` emits a `WARN` to a captured subscriber. Stubbing the notice or the logging step fails these tests.
- Red on main: with the clap `env` attributes restored, the per-binary env tests and both guard tests fail (the guard lists both CORS arguments). With the guard extension stubbed out, the two new `cli::` tests fail.
- Green: `cargo test --profile test-fast` for `--lib cli:: server::cli_input server::cors`, `--bin mlxcel`, `--bin mlxcel-server`, `--test cli_help_consistency`, `--test llama_compat_manifest`; clippy with `--lib --tests --bins -D warnings`; fmt; `make verify-env-var-docs verify-llama-compat`.

## 5. Known Gaps and Follow-ups (not yet filed)

- **Other pre-subscriber warnings**: the remaining `env_fallback_*` helpers in `src/server/cli_input.rs` still call `tracing::warn!` before the tracing subscriber exists, so their warnings are lost the same way the CORS notice was. The `kv_cache_mode_notices` comment documents the same limitation. A general fix would route all startup notices through `start_server` as this PR does for CORS.
- **Guard coverage**: `arg_conflict_partners` in `src/cli/mod.rs` reads only per-argument `conflicts_with`. Conflicts declared through an `ArgGroup`, `multiple(false)` group membership, and `exclusive` arguments are not reported, so an env binding in those shapes would slip past the guard. None exist today.
- **Test flag detection**: the CORS test helper `apply_cors_origin_env` detects flags with `argv.contains`, so the `--cors-origins=value` form is covered only by the tests of `long_cli_flag_was_set` itself, not by the CORS precedence tests.
