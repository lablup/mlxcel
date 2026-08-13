# Technical Report: PR #1128 - fix(server): wire --dry-sequence-breaker through to the sampler

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium (a previously inert flag becomes functional, and an unrepresentable value now fails startup)

---

## Executive Summary

PR #1128 closes issue #1103, the third and last of the #1102 / #1109 / #1103 chain. Both server binaries expose `--dry-sequence-breaker`; the value flowed from the CLI into `ServerStartupConfig` and stopped. There was no tokenization step, no `ServerConfig` field, no fallback in `build_server_generate_options`, and no `/props` entry. The flag was accepted, stored, and never read.

The wiring is not a copy, because the flag takes token strings and the sampler takes token IDs. The conversion needs the tokenizer, which is not loaded when the server config is built, so the resolution point had to be chosen deliberately. A breaker that cannot be represented as one token now fails startup rather than being dropped.

---

## 1. Problem Statement

### 1.1 Background

The three issues in this chain are the same defect seen from three positions. #1102 was a per-request breaker list surviving the request layer and then being dropped by the greedy branch of `build_sampling_config`. #1109 was the flag's own name differing between the two server binaries. #1103 is the operator-facing half: the flag never reached the sampler from any direction, so there was no server-wide default for a request to inherit.

### 1.2 Existing Issues

- **The trail ended at a struct field.** `grep -n dry_sequence_breakers src/server/startup.rs` returned the declaration and the `Default` initializer, and nothing else. `ServerConfig` declared `default_dry_multiplier`, `default_dry_base`, `default_dry_allowed_length` and `default_dry_penalty_last_n`, and stopped there.
- **One line in `build_server_generate_options` did not match its four neighbours.** `dry_sequence_breakers: overrides.dry_sequence_breakers.unwrap_or_default()` where every adjacent DRY field reads `unwrap_or(config.default_*)`. The shape difference is the defect made visible: there was no server default to fall back on, so a request that omitted the field always got an empty vector no matter what the operator passed.
- **The failure was silent in both directions, and the second direction is the dangerous one.** Nothing told the operator the flag was inert. And because the breakers terminate the DRY backward match rather than enable anything, running DRY without them lets `match_len` grow past the intended boundary, so the penalty applied is at or above what was configured. An operator who sets `--dry-multiplier 0.8 --dry-sequence-breaker '\n'` gets a stronger penalty than those numbers describe, not a weaker one.
- **`/props` could not report it.** `src/server/routes/props.rs` listed the four DRY fields it had. The one endpoint an operator would use to confirm what the server resolved could not answer the question.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| DRY tuned on the server behaves unlike its configuration, and the difference is attributed to the model | Medium | Medium |
| An operator confirms the flag "took" because startup was clean and `/props` showed the other four DRY fields | Low | High |
| The flag is removed later as dead surface, breaking scripts that pass it | Low | Low |

---

## 2. Technical Review

### 2.1 The Resolution Point Was the Real Design Question

`ServerConfig` is built by `build_server_config`, which runs before the model's tokenizer is loaded. The conversion from token strings to token IDs needs that tokenizer. Three placements were possible:

1. Tokenize in `build_server_config`. Rejected: it would have to load a tokenizer of its own, duplicating work and giving the function a filesystem dependency it does not currently have.
2. Move the tokenizer load earlier. Rejected: `run_server` has an early return for `serve_remote_pipeline_stage` before the current load site, so moving the load would make a remote pipeline stage load a tokenizer it never uses.
3. Leave the field empty in `build_server_config` and fill it in `run_server` immediately after `load_tokenizer` returns. Chosen.

The third has a real hazard: a field that is invalid between construction and a later assignment. It is mitigated by precedent and by documentation rather than by a type. The same shape already exists in this function for `config.pipeline_parallel_runtime`, and in `request_options` for `sampling.loop_detection` and `sampling.xtc_special_token_ids`, each of which is also resolved after the value that determines it becomes visible. The field's doc comment on `ServerConfig` says where it is filled, and the `build_server_config` site carries a comment saying why it is empty there.

### 2.2 Why Single-Token Is Not a Policy Choice

The sampler's breaker check is `config.dry_sequence_breakers.contains(&window[p1])` over a `Vec<i32>`. A multi-token breaker has no representation in that type at all. llama.cpp supports multi-token breakers by building a map from token to candidate tails; matching that would be a sampler change, not a wiring change, and is out of scope for an issue about a flag that is not read. Given the data model, the only choices were to reject, to drop silently, or to expand into several breakers that the operator did not ask for. Rejecting is the only one that cannot mislead.

A string that encodes to zero tokens is the same class of failure and is handled identically. The error names the offending string, reports the count it actually got, and names the flag.

### 2.3 The Escape Handling Is Load-Bearing, Not a Convenience

This is the part of the change the issue did not ask for and the implementation could not omit. `--dry-sequence-breaker '\n'` reaches the process as the two characters `\` and `n`: no common shell expands an escape inside single quotes, and `"\n"` is the same two characters in POSIX shells. The flag's help text has advertised `"\n"` and `"\t"` as its examples since it was added.

So without escape handling, the fail-closed rule would reject the flag's own documented usage at startup. That is a worse outcome than the bug being fixed: the flag went from silently inert to refusing to boot on the example in its own help. The alternative failure is quieter and worse, if the literal two-character `\n` happens to be a single token in some vocabulary, in which case the server would install a breaker that has nothing to do with newlines.

The rule interprets `\n`, `\t`, `\r` and `\\`, and preserves every other backslash sequence exactly as typed. Preserving unknown sequences rather than rejecting them keeps a breaker that genuinely contains a backslash expressible, and `\\` is the escape hatch for a literal backslash that precedes one of the four.

### 2.4 `Some(vec![])` and `None` Must Not Collapse

The fallback is `unwrap_or_else(|| config.default_dry_sequence_breakers.clone())`, so an absent request field inherits the server default and an explicitly empty request list turns it off for that request. That distinction is the reason for `unwrap_or_else` rather than a check on emptiness, and `an_explicitly_empty_request_breaker_list_disables_the_server_default` pins it, because collapsing the two would make the server default impossible to opt out of per request.

### 2.5 The Test Fixture Had to Be Built Rather Than Borrowed

`MlxcelTokenizer::stub_with_byte_fallback()` was the obvious fixture and is unusable here. Its BPE model has no merges, so `Hello` tokenizes to nothing rather than to its vocabulary entry, and it cannot express "exactly one token" for anything but a byte-fallback character. The tests build a tokenizer whose vocabulary is single characters (`a`, `b`, newline, tab, space) with no merges and no byte fallback, which makes all three outcomes the resolver must distinguish reachable without a checkpoint: one token, more than one, and none. `fixture_tokenizer_behaves_as_the_tests_assume` pins those encodings so a fixture change fails there, with a message saying so, instead of making the real assertions look wrong.

---

## 3. Technical Decisions

### 3.1 Wire It Rather Than Remove It

| Option | Pros | Cons |
|---|---|---|
| **Chosen: wire it** | llama.cpp compatibility is a stated goal and the flag is in the help text of two binaries; the silent-and-stronger failure mode is removed | Changes generation output for deployments that already pass the flag, and can now refuse to start |
| Remove the flag | Smallest diff; no startup-failure risk | Breaks scripts that pass it, needs a `CHANGELOG` note anyway, and abandons a documented llama.cpp-compatible knob |
| Wire it but warn instead of failing on a bad breaker | Never refuses to start | Reproduces the original defect in miniature: a warning in a startup log is exactly what nobody reads, and the resulting penalty is again stronger than configured |

The third option is the tempting one and is the reason the issue specified the posture explicitly ("should fail startup with a clear message rather than being silently dropped or expanded"). A flag whose misconfiguration is only a log line is a flag whose misconfiguration ships.

### 3.2 Put the Resolver in Its Own Module

`src/server/startup.rs` is already large, and the resolver is a pure function of a tokenizer and a string list with its own failure taxonomy. `src/server/cors.rs` is the established shape for this in the tree: a small private module that validates operator input, names the offending value, and carries its own `#[path]` test module. Following it keeps the two validators recognisable as the same kind of thing.

### 3.3 Extract `default_generation_settings` Instead of Testing Through `AppState`

`/props` gained a field, and the reported field set is a contract: it is what an operator reads to confirm what the server resolved. Asserting it through the axum handler would need an `AppState`, which needs a model. Extracting the payload construction into a function of `&ServerConfig` makes the field set assertable directly, and the three tests cover the resolved value, the present-but-empty case, and all five DRY fields together.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 12 (3 new) |
| New module | `src/server/dry_breakers.rs` (136 lines) |
| Tests added | 17 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Resolver | `src/server/dry_breakers.rs` (new) | `resolve_dry_sequence_breakers` and `unescape_breaker`; fail-closed on a breaker that is not exactly one token, on a flag that produced nothing usable, and on a token id that does not fit `i32` |
| Config | `src/server/config.rs` | `default_dry_sequence_breakers: Vec<i32>`, documented as filled by `run_server` rather than by `build_server_config`, and its `Default` |
| Startup | `src/server/startup.rs` | `build_server_config` leaves the field empty with the reason; `run_server` resolves it after `load_tokenizer` and logs the resolved IDs |
| Request path | `src/server/request_options.rs` | `unwrap_or_default()` becomes `unwrap_or_else(|| config.default_dry_sequence_breakers.clone())` |
| Endpoint | `src/server/routes/props.rs` | `dry_sequence_breakers` added; payload construction extracted into `default_generation_settings` |
| Help text | `src/main.rs`, `src/bin/mlx_server.rs` | Byte-identical on both binaries (required by the #1109 parity assertion): server-wide default, per-request override, single-token requirement, interpreted escapes |
| Tests | `dry_breakers_tests.rs`, `props_tests.rs`, `request_options_tests.rs` | 17 tests across the resolver, the `/props` field set, and the default/override/disable triple |
| Changelog | `CHANGELOG.md` | `### Fixed` entry recording both behavior changes, including that startup can now fail |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `470ecf5b` | fix | fix(server): wire --dry-sequence-breaker through to the sampler |

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib server::dry_breakers`: 11 passed.
- `--lib server::routes::props`: 3 passed.
- `--lib server::request_options`: 38 passed, up from 35.
- `--lib server::cli_input`: 93 passed.
- `--lib execution::sampling`: 3 passed.
- `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.

### Pre-existing Failure, Not From This Change

`server::startup::muse_glimmer_startup_guard_tests::muse_glimmer_startup_allows_baseline_and_keeps_video_disabled` fails when run with its siblings and passes in isolation. `muse_glimmer_startup_rejects_xla_backend_selection` sets `MLXCEL_BACKEND=xla` and restores it under a crate-wide env lock that the baseline test does not take, so the baseline test can observe the variable inside that window. The file was last touched by #1101 and is not in this diff.

### Not Covered

No real-checkpoint validation. Whether a given breaker string is one token is a property of the loaded vocabulary, so a checkpoint test would assert a fact about that checkpoint rather than about the resolver. The acceptance criteria are all at the configuration boundary, and that is where the tests sit.

No end-to-end assertion that the penalty value changes. That needs a model and a token history and would be seed-dependent; #1102 records the same reasoning for the same reason.

### Follow-up

Multi-token breakers remain unsupported, by data model rather than by decision. Supporting them means changing `SamplingConfig::dry_sequence_breakers` from `Vec<i32>` to a token-to-tails structure, matching what llama.cpp does, and is a sampler change rather than a wiring change.

The scope comment at `src/commands/generate.rs` says the CLI penalty is stronger than the same settings produce on the server. That is now true in a way it was not before this PR, since the server default finally exists; the comment is accurate as written and needs no change, but it is worth knowing that its truth value moved.
