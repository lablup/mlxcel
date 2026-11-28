# Technical Report: PR #1118 - docs: explain CLI DRY runs without sequence breakers

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust (comment and clap help only)
**Risk Level**: Low

---

## Executive Summary

PR #1118 closes issue #1108. The CLI exposes `--dry-multiplier`, so DRY can be switched on from the command line, but `resolved_cli_sampling_params` hardcodes `dry_sequence_breakers: Vec::new()`. With no breakers the backward match never stops at a newline or punctuation boundary, so `match_len` keeps growing and the penalty is stronger than the same nominal numbers produce on the server. Nothing said so, in the code or in the help text.

This is the documentation fix the issue arrived at after investigation, not the feature the issue originally proposed. Thirteen added lines, all comment or doc-comment. No `--dry-sequence-breakers` CLI flag, no behavior change.

---

## 1. Problem Statement

### 1.1 Background

Issue #1108 began as a flag-parity request: the server has `--dry-sequence-breakers` and the CLI does not, so add it. The investigation recorded in the issue killed that framing. Comparing the CLI's `SamplingOptions` against the server's per-request sampling fields shows the CLI omits nine knobs, not one, including `frequency_penalty`, `presence_penalty`, `xtc_probability`, `xtc_threshold`, `logit_bias`, `repetition_context_size`, `stop` and the loop-detection fields. A narrow CLI sampling surface is therefore deliberate, and "the server has a flag the CLI lacks" is not on its own an argument for anything.

### 1.2 Existing Issues

- **One of the five hardcoded fields is not what the other four are.** `resolved_cli_sampling_params` hardcodes five sampling fields in a contiguous block. Four of them (`frequency_penalty: 0.0`, `presence_penalty: 0.0`, `xtc_probability: 0.0`, `xtc_threshold`) are honest "feature off" defaults: the feature really is off, because no CLI flag can turn it on. `dry_sequence_breakers: Vec::new()` is the exception, because `--dry-multiplier` can turn DRY on. Once it is on, the empty vector is not a disabled feature. It is the feature running in a configuration the user cannot change.
- **The resulting behavior difference is invisible.** With no breakers, the backward match in `src/lib/mlxcel-core/src/sampling.rs` never terminates at a boundary, so `match_len` keeps growing and the penalty term is larger than the server produces from identical settings. A user who tunes `--dry-multiplier` with `mlxcel run` and then deploys those numbers to `mlxcel-server` gets different output from the same values, with nothing in either help text explaining why.
- **The investigation itself was at risk of being lost.** Nothing in the code recorded that the empty vector is a scope decision. The next reader would have re-derived the entire nine-knob comparison to reach the same conclusion.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Sampling values tuned on the CLI behave differently in production, and the difference is attributed to the model or the seed | Medium | Medium |
| A future contributor reads the empty vector as an oversight and adds the flag without the scope discussion | Low | Medium |
| A reader groups the DRY line with the four "feature off" neighbours and concludes DRY is off on the CLI | Low | Medium |

---

## 2. Technical Review

### 2.1 The Diff Is Provably Non-Executable

Thirteen added lines: ten `//` comment lines in `src/commands/generate.rs` and three `///` doc-comment lines in `src/main.rs` (two added, one existing line re-punctuated). No executable statement changed, no field value changed, no clap attribute changed. `cargo fmt --check` is clean, and `rustfmt` does not reflow comments at the project's settings.

### 2.2 Coverage of the One Help Sentence

`SamplingOptions` is flattened into both `GenerateArgs` and `RunArgs`, and the chat path shares the same `SamplingConfig` assembly. One sentence added to the `--dry-multiplier` help therefore covers `mlxcel generate`, `mlxcel run` and `mlxcel chat` without repetition. That coverage is why a single help string was sufficient and why no `docs/` page needed a parallel note.

### 2.3 The Second `dry_multiplier` Was Deliberately Untouched

`src/main.rs` has a second `dry_multiplier` field around line 1245, on `ServeArgs`. `ServeArgs` already exposes `--dry-sequence-breakers` sixteen lines below it. That field is the server side of the very asymmetry being documented, so appending the CLI caveat to it would have made the server's own help text describe a limitation the server does not have. It was left alone.

---

## 3. Technical Decisions

### 3.1 Document the Asymmetry Instead of Removing It

| Option | Pros | Cons |
|---|---|---|
| Add `--dry-sequence-breakers` to `SamplingOptions` | Removes the asymmetry outright | Expands a sampling surface that is deliberately a subset; needs tokenizer-time tokenization and a new unit gate; inert at the CLI's default `--temp 0.0` until #1102 lands, since the greedy branch of `build_sampling_config` drops the breakers |
| Change the CLI default to the llama.cpp breaker set | Matches the widely expected behavior | A silent behavior change to existing CLI invocations, which is exactly what a docs issue must not ship |
| **Chosen: comment plus one sentence of user-facing text** | Removes the surprise at zero behavior risk; records why the scope decision was made | The asymmetry itself remains, so the CLI still cannot express server-equivalent DRY |

The issue explicitly left the flag available as a separate feature issue, including the acceptance shape it would need: a unit assertion that a parsed flag reaches `ResolvedSamplingParams.dry_sequence_breakers` as token IDs, rather than an output-diff assertion, which is model- and seed-dependent and makes a flaky gate. That path stays open; this PR does not foreclose it.

### 3.2 Put the User-Facing Sentence in clap Help, Not in `docs/`

The clap help is what a user sees at the moment they are choosing a `--dry-multiplier` value, which is the moment the surprise occurs. A `docs/` page is read earlier or not at all. Keeping the change inside `src/` also keeps it in the same file as the flag it qualifies, so a future flag change and its caveat cannot drift apart into separate files.

### 3.3 Explain Why This Field Differs From Its Neighbours

The comment does not just state that the CLI runs DRY without breakers. It states why that line is unlike the four beneath it: those are off because nothing can turn them on, and this one is not, because `--dry-multiplier` can. Without that contrast, the next reader has to redo the comparison to know whether the empty vector is intentional. The comment exists to make the investigation non-repeatable work.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 2 |
| Lines added | +13 |
| Lines deleted | -1 |
| Executable lines changed | 0 |
| Tests added | 0 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Code comment | `src/commands/generate.rs` | Ten-line comment above `dry_sequence_breakers: Vec::new()` recording the scope decision, why it differs in kind from the four "feature off" neighbours, and the resulting `match_len` behavior |
| User-facing help | `src/main.rs` | `--dry-multiplier` help on `SamplingOptions` gains one sentence: CLI DRY matches across all boundaries, and the server's `--dry-sequence-breakers` has no CLI equivalent |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `33322d66` | docs | docs: explain CLI DRY runs without sequence breakers |

---

## 5. Validation and Follow-up

### Passed

- `cargo fmt --check` clean.
- `python3 scripts/ci/check_cross_repo_refs.py` passes (no bare `#NNN` added).
- Diff reviewed line by line as comment-and-help-text-only: every added line matches `^\s*(///|//)`.

### Not Covered

Compilation was not run in the implementation worktree, which had no `target/` directory and would have triggered a full MLX source build for a change that alters no executable line. Multi-line doc comments on clap fields are already used in the same struct (for example on `seed`), so the pattern is established rather than novel.

### Follow-up Candidates

- The flag itself, as a feature issue, with the acceptance shape recorded in 3.1. It would be inert at the CLI's default `--temp 0.0` until #1102 lands.
- The eight other sampling knobs the CLI omits are undocumented in the same way, but none of them shares this failure mode: each is genuinely unreachable from the CLI, so no user can turn one on and then be surprised by its fixed configuration. Only the `--dry-multiplier` and `dry_sequence_breakers` pair has a switch without its companion setting.
