# Technical Report: PR #1125 - fix(cli): make the drifted server flag spellings work on both binaries

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust (clap attributes, help text, tests), Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1125 closes issue #1109. `mlxcel serve` and `mlxcel-server` are two hand-maintained clap definitions of the same server sharing 112 flag spellings. Four had drifted, and the worst case had no shared spelling at all: `--n-parallel` worked only on `serve`, `--parallel` only on `mlxcel-server`, so a command line copied between the two failed to parse even though both flags read the same `LLAMA_ARG_N_PARALLEL` environment variable.

Three of the four are repaired by adding the missing `visible_alias`, which changes no primary spelling. The fourth, the DRY sequence breakers, needed a primary chosen because the two binaries disagreed on the name itself. The durable half is `tests/cli_help_consistency.rs`, which now compares the whole long-name surface of the two binaries against a three-name allowlist of deliberate exceptions, so a flag added to one binary and forgotten on the other fails immediately.

---

## 1. Problem Statement

### 1.1 Background

The two server surfaces are not one flattened clap group. `ServeArgs` lives in `src/main.rs` and `ServerArgs` in `src/bin/mlx_server.rs`, and they are maintained by hand against each other. The shared flag GROUPS (`TurboKvCacheArgs`, `SpeculativeArgs`) are flattened and therefore cannot drift, and `tests/cli_help_consistency.rs` already pinned those. Everything outside a flattened group had nothing holding it aligned.

The repository had already met this problem once and solved it correctly. The drafter flags are aliased symmetrically so either spelling works on either binary, and the doc comment at `src/main.rs` states the reason outright: "so commands copied between the two binaries work unchanged". `--draft-max` / `--draft` got the same treatment. What was missing was any mechanism to notice the next flag that did not.

### 1.2 Existing Issues

| Concept | `mlxcel serve` accepted | `mlxcel-server` accepted | llama-server |
|---|---|---|---|
| parallel slots | `--n-parallel` only | `--parallel` only | `--parallel` |
| predict cap | `--n-predict` only | `--predict`, `--n-predict` | `--n-predict` |
| LoRA adapter | `--adapter`, `--lora` | `--lora` only | `--lora` |
| DRY breakers | `--dry-sequence-breakers` | `--dry-sequence-breaker` | `--dry-sequence-breaker` |

- **The parallel row is the only one where neither spelling worked on both.** A copied command line fails with `error: unexpected argument '--parallel' found`. Because `LLAMA_ARG_N_PARALLEL` works on both, an operator who switches to the environment variable to get past it removes the symptom without recording the cause, and the next person to read the script sees a deployment that avoids a flag for no stated reason.
- **The help text had already forked to match the drift.** `mlxcel serve` said "Use `--n-parallel 1` (or `--no-batch`) for single-slot serving"; `mlxcel-server` said "Use `--parallel 1` (or `--no-batch`) to restore single-slot sequential serving". Same paragraph, two wordings, each edited to match its own binary's spelling. That divergence is what made the flag-name drift look normal instead of looking like a bug: a reader comparing the two help outputs sees two paragraphs that differ in several places and has no reason to single out the flag name.
- **`README.md` advertises llama-server compatibility as a migration aid.** Three of the four broke that promise on `mlxcel serve` specifically.
- **Nothing would have caught the fifth one.** The existing consistency test covers two flattened groups. A concept that lives directly on the two structs was outside every invariant in the suite.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A command line copied between the binaries fails, and the failure is attributed to a version difference or a typo | Medium | High |
| An operator works around the gap with `LLAMA_ARG_N_PARALLEL` and the flag gap stays undiscovered | Low | High |
| A fifth flag diverges and lands unnoticed | Medium | Medium |

---

## 2. Technical Review

### 2.1 What Changes Behavior and What Does Not

Three of the four repairs are pure additions: `visible_alias = "parallel"` on `ServeArgs::n_parallel`, `visible_alias = "n-parallel"` on `ServerArgs::parallel`, `visible_alias = "predict"` on `ServeArgs::n_predict`, `visible_alias = "adapter"` on `ServerArgs::lora`. A `visible_alias` adds an accepted spelling and renders it in `--help`; it removes nothing. Every command line that parsed before still parses.

The DRY breaker row is the only edit that changes a primary spelling, and only on `mlxcel serve`. Its clap field is `dry_sequence_breakers`, so `#[arg(long)]` derived `--dry-sequence-breakers` from the Rust identifier. Making the singular primary requires writing `long = "dry-sequence-breaker"` explicitly. The Rust field name is unchanged, so nothing downstream of parsing is affected: `src/commands/serve.rs`, `src/server/cli_input.rs`, and `src/server/startup.rs` all keep reading `dry_sequence_breakers`.

### 2.2 Coverage of the Spelling Change

The plural is retained as `visible_alias` on both binaries, so the change is invisible at the command line. Two places in the tree named the plural in prose and were updated to name the primary: the help sentence on `SamplingOptions::dry_multiplier` added by #1118, and the scope comment at `src/commands/generate.rs:921`. Both statements remain true either way (the plural still parses); naming the primary is what keeps the help text pointing at the spelling the binary documents.

### 2.3 The Assertion Had To Be Proved Non-Vacuous

A cross-binary contract test that passes is worth nothing unless it fails on a real regression, and this one has three ways to fail silently: the entry matcher could find no entry, the alias parser could return an empty list, or the description extractor could return an empty string for both binaries. Each is checked:

1. `signature_long_name_accepts_clap_signatures_and_rejects_prose` pins the matcher against the four signature shapes clap renders (long only, long plus value name, short plus long plus value name, and a value name with a repetition marker) and against four shapes it must reject, including a description line that starts with `--parallel` and a hyphen-bulleted prose line.
2. `accepted_spellings_and_description_split_an_entry_correctly` pins the splitter against a rendered entry and against its mirror image: a different primary, a different clap-derived value name, and the opposite alias annotation, with the same description. Those two must compare equal on both the spelling set and the description, which is exactly the comparison the contract makes across binaries.
3. `dropping_a_shared_flag_alias_would_fail_the_spelling_parity_assertion` takes the live `mlxcel-server --help`, cuts the rendered `[alias: --n-parallel]` annotation out of the `--parallel` entry, and asserts the accepted set collapses to `["--parallel"]`. This is the same idiom the pre-existing `removing_the_rendered_alias_annotation_leaves_the_alias_undocumented` test uses on the drafter flags.

A one-off live mutation confirmed the end-to-end path: removing `visible_alias = "n-parallel"` from the clap definition and rebuilding made `shared_server_flags_accept_the_same_spellings_on_both_binaries` fail with `left: ["--parallel"], right: ["--n-parallel", "--parallel"]`, and restoring it returned the suite to green.

### 2.4 A Named List Alone Would Not Have Delivered What the Issue Asked For

The first draft of this change asserted only a named list of six concepts. That catches a REGRESSION of a pairing someone already thought to write down. It cannot catch what the issue actually asked for, a FUTURE divergence: a flag added to one binary tomorrow is simply absent from the list, and the suite stays green.

The reason the issue offered the named list as a fallback was an assumption that a whole-surface comparison would be too noisy, because the two binaries legitimately differ. Measuring it killed that assumption. After this change the two surfaces share 134 spellings and differ by exactly three: `--estimate-memory` and `--force` on `mlxcel serve` (both subcommand-shaped one-shot actions), and `--version` on `mlxcel-server` (`mlxcel` carries it at the top level instead). A three-name allowlist is small enough to read, so the whole-surface assertion is the one that ships.

Both invariants are kept, because they fail on different things. The surface comparison sees a set of names, not which names are two spellings of one concept: if `--parallel` were dropped from `mlxcel-server` and `--n-parallel` from `mlxcel serve` in the same change, the two surfaces would stay equal and only `SHARED_SERVER_FLAG_GROUPS` would notice. The named list is the regression guard; the surface comparison is the new-divergence guard.

---

## 3. Technical Decisions

### 3.1 Singular `--dry-sequence-breaker` as the Primary on Both Binaries

| Option | Pros | Cons |
|---|---|---|
| **Chosen: singular primary on both, plural aliased on both** | Matches llama-server, which is why these flags carry `LLAMA_ARG_*` env vars at all; `mlxcel-server` already used it, so only one binary changes; the two binaries end up sharing one primary, which is stronger than what the drafter flags achieve | `mlxcel serve` users see a different primary in `--help` than before, though their existing command lines still work |
| Plural primary on both, singular aliased | Only `mlxcel-server` changes | Diverges from llama-server on the flag whose entire reason for existing is llama-server compatibility |
| Keep opposite primaries and alias both ways, as the drafter flags do | Consistent with the nearest precedent; no primary changes at all | The drafter compromise exists because mlx-lm and llama-server each have an established spelling and neither should be demoted. No mlx-lm spelling exists here, so there is nothing to balance and the compromise would preserve a difference for no reason |

The third option is the one worth arguing about, because it is what the neighbouring code does. The reason not to follow it: the drafter flags have two upstream conventions competing, and `--dry-sequence-breaker` has one. Copying the compromise where there is no conflict would leave the two binaries permanently rendering different primaries in `--help` to no end.

### 3.2 Assert the Prose, Not Only the Spellings

Spelling parity alone would have left the forked `--n-parallel` paragraph in place, and that paragraph is the mechanism by which the drift stayed invisible: two wordings of the same text make a reader stop comparing. `SHARED_SERVER_FLAG_DESCRIPTIONS` therefore requires identical prose for the four non-drafter concepts.

The two drafter groups are deliberately excluded rather than forced into agreement. Their descriptions name each binary's own primary and alias roles, and those roles are opposite by design, so identical prose there would make one of the two statements false. Recording the exclusion with its reason is the point; a silent omission would read as an oversight.

### 3.3 Exclude the Signature and Annotations By Construction

`flag_description` drops the signature line and every line beginning with `[`. That is not a convenience: the signature line necessarily differs (different primary spelling, and clap derives the value name from the Rust field name, so `--parallel <PARALLEL>` versus `--n-parallel <N_PARALLEL>`), and the `[env:]` / `[default:]` / `[alias:]` annotations necessarily differ too. Excluding them structurally means the comparison cannot be defeated by a hand-maintained allowlist drifting out of date, and each binary keeps its own primary spelling and alias without weakening the contract.

### 3.4 Anchor the Entry on the Long Name, Not the Full Signature

The pre-existing `flag_help_entry` matches a complete rendered signature such as `--draft-model <PATH>`, which is right when the value name is part of the contract. It cannot be reused here, because the value name is derived from the Rust field name and therefore differs between binaries for the same concept. `flag_entry_by_long_name` anchors on the long name alone but still requires the whole line to have signature shape, so prose cannot anchor an entry. The shared body-slicing loop was factored into `entry_body` so the two finders cannot disagree about where an entry ends.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 6 |
| Lines added | +587 |
| Lines deleted | -16 |
| clap attributes changed | 5 |
| Tests added | 13 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| clap definition | `src/main.rs` | `visible_alias` on `n_parallel` and `n_predict`; DRY breakers get `long = "dry-sequence-breaker"` plus the plural alias; the `--n-parallel` help paragraph is reconciled with `mlxcel-server`'s; the #1118 sentence names the new primary |
| clap definition | `src/bin/mlx_server.rs` | `visible_alias` on `parallel` and `lora`; DRY breakers get the plural alias and the shared doc comment; three parse-level alias tests and a clap name-uniqueness guard |
| Code comment | `src/commands/generate.rs` | The DRY scope comment names the new primary spelling |
| Unit tests | `src/main_tests.rs` | Four parse-level alias tests on `mlxcel serve`, including `--adapter` / `--lora`, which was already symmetric, plus a clap name-uniqueness guard |
| Integration tests | `tests/cli_help_consistency.rs` | The whole-surface comparison with its two allowlists, `SHARED_SERVER_FLAG_GROUPS` and `SHARED_SERVER_FLAG_DESCRIPTIONS`, three contract tests, five guard tests, six new helpers, and the `entry_body` refactor |
| Docs | `docs/CONTINUOUS_BATCHING.md` | No longer presents `--n-parallel` and `--parallel` as one spelling per binary |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `36b617b8` | fix | fix(cli): make the drifted server flag spellings work on both binaries |

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --bin mlxcel tests::serve_`: 13 passed.
- `cargo test --profile test-fast --features cuda --bin mlxcel-server tests::`: 13 passed.
- `cargo test --profile test-fast --features cuda --test cli_help_consistency`: 25 passed, up from 17 on the base.
- `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings`: clean. `--bins` is included because both binary crates change.
- `cargo fmt --all -- --check`: clean.
- Two live mutations, each restored and re-run green afterwards. Removing `visible_alias = "n-parallel"` from the clap definition makes the named-list parity assertion fail with `left: ["--parallel"], right: ["--n-parallel", "--parallel"]`. Removing `--force` from `SERVE_ONLY_FLAGS` makes the whole-surface assertion fail with `left: {"--estimate-memory", "--force"}, right: {"--estimate-memory"}`, which is the shape a genuinely new one-sided flag would produce.

### Not Covered

No end-to-end check that a server actually starts with each spelling. The parse-level tests assert the resolved field value, which is the boundary the flag name controls; everything past `clap::Parser` reads the Rust field name and cannot tell which spelling produced it.

Short forms are out of scope and remain divergent: `mlxcel-server` carries `-c` for `--ctx-size` and `-n` for `--predict`, and `mlxcel serve` has neither, so `mlxcel-server -m X -c 4096 -n 256` still does not copy across. Both letters are free on `mlxcel serve`, so closing it is cheap, but it is a different table from the one issue #1109 enumerated and the help text no longer claims otherwise. The contract compares long names only.

Description parity covers 4 of the 134 shared spellings. Prose drift exists elsewhere and is unguarded: `--prompt-cache-enabled`, for one, says "when the CLI flag is absent" on `mlxcel serve` and "is not explicitly provided" on `mlxcel-server`. Not introduced here, and a concrete measure of what the description subset leaves out.

### Follow-up

Issue #1103 wires the DRY breaker flag's effect. It was blocked on this decision only in the sense that it should not re-decide the primary name; with the singular settled on both binaries, the remaining work there is tokenization and the server-side default. The scope comment in `src/commands/generate.rs` claims the CLI penalty is stronger than the server's; that claim is only fully true once #1103 makes the server flag functional, and the comment should be revisited then.
