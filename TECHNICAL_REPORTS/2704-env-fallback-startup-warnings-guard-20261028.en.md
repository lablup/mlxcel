# PR #2704: Show env fallback startup warnings and close CLI env-guard gaps

**Date**: 2026-10-28 (written before merge; the PR is open as a draft)
**Status**: Implementation complete; local tests passed on ROCm (gfx1151); hosted CI pending at the time of writing
**Languages**: Rust
**Risk Level**: Low (no behavior change in what values are resolved; changes where warnings go and what the CLI guard detects)

## Executive Summary

PR #2704 (closes #2697) fixes three gaps left by #2622 and #2668 in how `mlxcel serve` and `mlxcel-server` read environment variable fallbacks.

1. Every warning raised by an `env_fallback_*` helper or by `ServerStartupInput::into_startup_config` was emitted with `tracing::warn!` before `start_server` installed the tracing subscriber, so it was written to nothing. These warnings are now collected into one `startup_notices: Vec<String>` and logged once, at WARN, immediately after `initialize_server_logging`. This replaces the two earlier ad hoc lists (`cors_origin_notices` from #2664 and `kv_cache_mode_notices` from #1350).
2. The `env_bound_override_args` guard, which keeps clap `env` bindings off arguments that clap would treat as conflicting, now also sees conflicts that clap derives from argument groups and from `exclusive`.
3. The CORS `--flag=value` spelling is tested end to end, using a new `long_cli_flag_in` that applies the exact matching rule of `long_cli_flag_was_set`.

The PR is 15 files, +1065 / -368, in two commits (`36e9b8e`, `07a4e40`).

## 1. Problem Statement

### 1.1 Warnings logged before the subscriber exists

Both server binaries resolve their arguments in two stages before serving: the binary calls a set of `env_fallback_*` helpers (in `src/bin/mlx_server.rs::build_startup_input`, `src/commands/serve.rs`), then `ServerStartupInput::into_startup_config` validates and lowers the input. Only after that does `start_server` call `initialize_server_logging`. Any `tracing::warn!` in the earlier stages has no subscriber and is dropped silently.

For an operator this means a typo such as `LLAMA_ARG_KV_BITS=eight` or `LLAMA_ARG_ENDPOINT_SLOTS=maybe` was ignored with no trace in the log, and the server started with defaults. The same applied to warnings inside `into_startup_config` (for example `--cache-ram` together with `--prompt-cache-capacity-bytes`, an out-of-range `--adaptive-decay`, `--api-prefix` combined with an API key).

Two earlier fixes had already hit this for one case each: #1350 added `kv_cache_mode_notices` for the KV cache mode substitution, and #2664 added `cors_origin_notices` for an ignored CORS origin variable. Each had its own field and its own logging point.

### 1.2 The env-binding guard missed derived conflicts

clap counts an env-sourced value as present. If an argument with `env = "VAR"` conflicts with another argument, `VAR=... --other` becomes a parse error even though the user typed only one of them (#2609, #2664). `env_bound_override_args` in `src/cli/mod.rs` lists such arguments, and tests on both binaries assert the list is empty. Before this PR it looked only at `overrides_with` and `conflicts_with`. clap's validator also derives conflicts from non-`multiple` groups, from a group's own `conflicts_with`, and from `exclusive(true)`. An env binding added in any of those shapes would have passed the guard and broken parsing.

### 1.3 CORS `--flag=value` untested

The CORS test helpers decided "was the flag given" with `argv.contains(&"--cors-origins")`, so the `--cors-origins=http://a.example` form was never exercised end to end, although the binaries handle it through the prefix match in `long_cli_flag_was_set`.

## 2. Change Summary

| Area | Change |
|---|---|
| Env fallback helpers (`src/server/cli_input.rs`, `src/cli/speculative_args.rs`, `src/server/thinking_budget.rs`) | Every helper that warned takes `notices: &mut Vec<String>` and pushes the message instead of logging: KV bits, group size, skip-last-layer, endpoint slots, settings endpoint, CORS credentials, lang-bias byte fragments, prompt cache enabled / TTL / snapshot TTL, APC enabled, every `Option<usize>` helper (through `apply_optional_usize_env_fallback`), reasoning budget, draft max, draft block size. |
| `into_startup_config` | `mut self`; takes the incoming list with `std::mem::take`, then appends its own warnings: `--cache-ram` precedence, invalid `--reasoning-budget`, `--timeout` migration, `--api-prefix` with an API key, reasoning-flag versus `--chat-template-kwargs` collisions (`apply_reasoning_template_kwargs`), out-of-range `--adaptive-decay` (`sanitize_adaptive_decay`), the split versus legacy KV cache flag precedence, and the #1350 KV mode substitution. |
| KV cache flag precedence (`src/cli/turbo_args.rs`) | New `kv_cache_mode_precedence_notice` returns the message. `resolve_kv_cache_mode` still logs it for non-server callers; the server additionally pushes it as a notice. |
| Data model | `startup_notices: Vec<String>` on `ServerStartupInput`, `ServerStartupConfig`, and `ServeArgs` (`#[arg(skip)]`) replaces `cors_origin_notices` and `kv_cache_mode_notices`. |
| Logging point (`src/server/startup.rs`) | `log_startup_notices` (renamed from `log_cors_origin_notices`) logs every entry at WARN right after `initialize_server_logging`, before the router-mode branch. The separate #1350 loop later in `start_server` is removed. |
| `--spec-type none` | Both binaries push the "ignoring the configured draft model" and "ignoring --draft-kind prompt-lookup" warnings to the list. |
| CLI guard (`src/cli/mod.rs`) | `env_bound_override_args` builds a clone of the command first; `arg_conflict_partners` is split into `direct_conflicts` (own conflicts, group conflicts, non-multiple group siblings), `unroll_group_or_arg` (group ids to member args, nested groups included), `group_conflicts` (reads the private list from `Debug`), and a shared `parse_debug_id_list`. Exclusive arguments are reported. |
| Flag detection (`src/server/cli_input.rs`) | `long_cli_flag_in(args, name)` exposes the rule behind `long_cli_flag_was_set`, which now delegates to it. Re-exported from `mlxcel::server`. |
| Tests | One end-to-end capture test for 20 warning paths, a `--spec-type none` notice test, six guard self-tests, two `--flag=value` CORS tests, and signature updates across existing tests. |

## 3. Technical Decisions

### 3.1 An out-parameter instead of return values or a global buffer

The issue allowed either returning or pushing the notice. The PR chose `&mut Vec<String>` on every warning helper. Three alternatives were available:

- **Return `Option<String>` per helper.** This is what the CORS helpers from #2664 already do (they return `Result<Option<String>>`), and those were kept as they are. For the rest, a return value is easy to discard with `let _ =` or by forgetting to collect it, and some helpers can emit more than one warning (`env_fallback_kv_skip_last_layer` tries two variables, `apply_reasoning_template_kwargs` can warn per key).
- **A process-global buffer or a buffering subscriber installed early.** This would have avoided touching every signature, but it hides the data flow, makes tests that run in parallel share state, and an early subscriber would need to be replaced by the configured one (format, log file, level), which `initialize_server_logging` is not built for.
- **Out-parameter (chosen).** Adding a required parameter makes every call site a compile error until it passes a list, so no caller can silently drop notices. The list rides on the existing input and config structs, so it reaches `start_server` with no new plumbing.

The cost is a wide signature change on public functions re-exported from `mlxcel::server`. Any out-of-tree caller has to update; inside the repository every call site was updated, and tests that do not care pass `&mut Vec::new()`.

### 3.2 One list, logged once, before the router branch

Merging `kv_cache_mode_notices` and `cors_origin_notices` into one `startup_notices` means one field, one logging function, and one place to add the next warning. Logging it directly after `initialize_server_logging` and ahead of the router-mode branch makes router servers report these warnings too. Previously the #1350 notices were logged later inside the single-model path, after the router `return` and after the `--models-preset` and sampler-order checks, so a router start never reached that loop and a start that failed one of those checks exited without showing them. The trade-off is ordering: the KV substitution notice now appears near the top of the log rather than next to the `kv_cache_mode = ...` INFO line. A comment at the old location points readers to `log_startup_notices`.

### 3.3 Only warnings moved; INFO lines did not

The "env var is set but the CLI flag takes precedence" messages in the same helpers are `tracing::info!` and stay as they are, so they are still emitted before the subscriber and still lost. The PR states this explicitly. The judgment is that a flag winning over a variable is expected behavior, while an invalid value being ignored is the case an operator needs to see. Moving the INFO lines would have meant either logging them at WARN or carrying a level with each notice.

### 3.4 Building the command before scanning it

clap's derive form `#[arg(group = "x")]` records membership on the argument; clap copies it into the `ArgGroup` only in `Command::build`, and a group created this way is not `multiple`. Scanning an unbuilt command would miss exactly the form the binaries use. `env_bound_override_args` therefore clones and builds the command first. Building is idempotent and the clone keeps the caller's command untouched.

### 3.5 Mirroring clap's validator, including private fields via `Debug`

`direct_conflicts` follows clap's `gather_arg_direct_conflicts` in `clap_builder/src/parser/validator.rs`: the argument's own conflicts, each containing group's conflicts with group ids unrolled into members, and the siblings of each non-multiple containing group. `arg_conflict_partners` adds the reverse direction (anything whose direct conflicts name this argument) and every `exclusive` argument, mirroring `validate_exclusive`. An env-bound argument that is itself exclusive is reported with the relation `exclusive`.

clap exposes no getter for a group's `conflicts` list. The PR reads it from the group's `Debug` output, the same technique already used for an argument's `overrides`. The search starts after `, args: [` so a group id containing the text cannot match. If clap changes the format, the guard panics with a message naming the function to update, rather than returning an empty list and passing. The id-list parser was factored into `parse_debug_id_list` and is shared by both readers. `ArgGroup::is_multiple` takes `&mut self` in clap 4.6, hence `group.clone().is_multiple()`.

### 3.6 Testing the real matching rule

Rather than fixing the test helpers' `contains` checks in place, the PR made the binaries' rule callable on an arbitrary argv (`long_cli_flag_in`) and had both the production function and the test helpers use it. The tests now exercise the same code as the binaries, so a future change to the matching rule cannot drift from what the tests check.

## 4. Verification

- `env_fallback_warnings_are_logged_once_a_subscriber_exists` sets invalid values for 18 environment variables with no subscriber installed, runs every helper and `into_startup_config` (with an out-of-range `--adaptive-decay` and conflicting KV cache flags), then runs `log_startup_notices` under a capturing `tracing_subscriber`. It asserts that each of the 20 messages appears on a WARN line and that the number of WARN lines equals 20, which also catches a duplicated notice.
- Guard self-tests cover a non-multiple group declared on the group and on the argument, a group conflicting with an argument in both directions, a group conflicting with another group, an exclusive env argument, and an env argument beside an exclusive one. The real-CLI guard checks in `src/main_tests.rs` and `src/bin/mlx_server.rs` still report an empty list, so no current argument trips the extended guard.
- Local runs reported in the PR on ROCm (gfx1151): `cargo clippy --lib --tests --bins -- -D warnings`, `cargo test --lib server::cli_input` (161 passed), `cargo test --lib cli::` (301 passed), `cargo test --lib server::thinking_budget`, `cargo test --bin mlxcel-server` (45 passed), `cargo test --bin mlxcel` (321 passed). This report did not rerun them.

## 5. Remaining Gaps and Risks

- **INFO precedence lines are still lost** (see 3.3). If operators need them, the notice type would have to carry a level.
- **Nothing forces new warnings onto the list.** The out-parameter protects existing helpers. A new `tracing::warn!` added to a pre-subscriber path will be dropped again unless the author knows the rule. A lint or a test that runs the full startup input path under a capturing subscriber and asserts nothing is logged before `start_server` would close this; the PR does not add one.
- **Duplicate emission for the KV flag precedence if the ordering changes.** `resolve_kv_cache_mode` still calls `tracing::warn!` itself, and the server also pushes the same message as a notice. Today the first call has no subscriber, so the operator sees the message once. If `into_startup_config` were ever called under an active subscriber, the message would appear twice.
- **Message text changed slightly.** Several messages switched from manual quoting (`"{raw}"`) to `{raw:?}`. For ordinary values the output is identical; values with quotes or control characters are now escaped. Log scrapers that match the exact old text for unusual inputs would see a difference.
- **`Debug`-format coupling to clap.** Both `arg_overrides` and `group_conflicts` depend on clap's derived `Debug` layout. A clap upgrade that changes it will panic in the guard tests, which is the intended failure mode, but it will need a code update at that point.
- **Public API change.** The `env_fallback_*` functions re-exported from `mlxcel::server` have new signatures. No out-of-tree caller is known.

## 6. Learning Points

- In a Rust binary that installs its tracing subscriber late, every `tracing` call before that point is a no-op. Treat early configuration code as needing an explicit message channel, not logging.
- A required `&mut Vec<_>` parameter is a cheap way to make "do not drop this" a compile-time property across many call sites.
- clap's derive group membership exists only after `Command::build`, and clap derives conflicts from groups and `exclusive` that are not visible through `get_arg_conflicts_with`. Any static analysis of a clap command should build it first and mirror the validator, not the builder API.
- When a test helper re-implements a production rule, expose the production rule over explicit inputs and call it from the test instead.

## 7. References

- Issue #2697 (this fix), PR #2704
- Earlier work: #1350 (KV mode substitution notices), #2609 and #2618 (env binding on override pairs, guard introduced), #2622, #2664 and #2668 (CORS origin variables and their notices)
- clap 4.6.1, `clap_builder/src/parser/validator.rs` (`gather_arg_direct_conflicts`, `validate_exclusive`)
