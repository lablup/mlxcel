# Technical Report: PR #1124 - fix(server): thread dry_sequence_breakers through the greedy branch

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

PR #1124 closes issue #1102. `build_sampling_config` branches on `temperature <= 0.0`. The sampled branch forwarded all five DRY fields; the greedy branch forwarded four and let `dry_sequence_breakers` fall through to `SamplingConfig::greedy()`, which sets `Vec::new()`. DRY is not gated on temperature, so a server request at `temperature: 0` with a positive `dry_multiplier` ran the penalty with no breakers.

One line of production code, plus the two test assertions that had pinned the defect. The second of those, in `request_options_tests.rs`, was not named in the issue and was found by running the neighbouring module; it failed on the fix, which is what confirmed the defect end to end rather than only at the helper's boundary.

---

## 1. Problem Statement

### 1.1 Background

`src/execution/sampling.rs` is the one place where the CLI and the server agree on how request fields map onto greedy versus sampled generation. Its `build_sampling_config` has two struct literals, one per branch, and each has to list every field it wants carried; whatever a branch omits falls to the `..SamplingConfig::greedy()` or explicit tail. That shape makes an omission invisible: nothing in the type system distinguishes "deliberately reset" from "forgotten".

The greedy branch already carried a comment for exactly this hazard. `xtc_probability` and `xtc_threshold` are threaded through with a note saying XTC "is a logits pre-processing step applied regardless of temperature (like the repetition/DRY/frequency/presence penalties above)". The comment names DRY as being in that same class. The four scalar DRY fields were threaded accordingly. The fifth was not.

### 1.2 Existing Issues

- **The omission does not disable DRY, it strengthens it.** `dry_sequence_breakers` is not a knob that turns something on. It is the termination condition of the backward match in `apply_dry_penalty` (`src/lib/mlxcel-core/src/sampling.rs:846`, `if config.dry_sequence_breakers.contains(&window[p1]) { break; }`). An empty vector means the loop never breaks at a boundary token, so `match_len` grows past the window the caller intended and `dry_multiplier * dry_base.powi(match_len - dry_allowed_length)` comes out exponentially larger than requested. The failure mode is a penalty stronger than configured, not a feature that quietly does nothing.
- **The request was accepted as valid.** `dry_sequence_breakers` is a documented per-request field on both request shapes (`src/server/types/request.rs:601` and `:978`), reaching `src/server/request_options.rs:239` through `routes/chat.rs:1198` and `routes/native_completion.rs:285`. Nothing rejected, warned, or logged. `{"temperature": 0, "dry_multiplier": 0.8, "dry_sequence_breakers": [198]}` returned a 200 with output shaped by a penalty the caller never asked for.
- **Two tests asserted the defect as the contract.** `src/execution/sampling_tests.rs:70` asserted `Vec::<i32>::new()` at `temperature: 0`. It carried no comment justifying the reset, unlike the XTC assertion five lines below which does explain itself, and the same test asserted none of the four DRY fields the branch did forward. That asymmetry is what marks it as transcription rather than intent: a deliberate contract would have said so, and would have stated the whole branch rather than one field of it.
- **`temperature: 0` is the common case for the affected workload.** DRY is reached for by operators fighting repetition loops, and repetition-loop debugging is usually done at greedy temperature to remove sampling as a variable. The one branch that dropped the field is the branch such a user is most likely to be on.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Output at `temperature: 0` differs from the request's intent and the difference is attributed to the model or the prompt | Medium | Medium |
| DRY parameters tuned at `temperature > 0` behave differently when the same request is replayed at `temperature: 0` | Medium | Medium |
| A future reader takes the pinned test assertion as a deliberate "DRY is greedy-inert" decision and propagates it | Low | Low |

---

## 2. Technical Review

### 2.1 Scope of the Behavior Change

The change is inert unless DRY is already running. `apply_dry_penalty` is called only when `config.dry_multiplier > 0.0 && !token_history.is_empty()` (`src/lib/mlxcel-core/src/sampling.rs:389`), and the default `default_dry_multiplier` is `0.0` (`src/server/config.rs:560`). A request that does not enable DRY sees an identical `SamplingConfig` before and after. A request that enables DRY and sets no breakers also sees no change, because the field resolves to an empty vector from `unwrap_or_default()` either way. Only the requests that set both change, and they change toward what they asked for.

### 2.2 Greedy Determinism Is Untouched

The concern worth stating explicitly is whether adding a field to the greedy branch can make greedy non-deterministic. It cannot. `SamplingConfig::greedy()` supplies `top_k: 1` and `top_p: 1.0`, the change does not touch either, and DRY is a logits pre-processing step that runs before selection. The regression test re-asserts `top_k` and `top_p` alongside the breakers so a future change cannot restore the breakers by weakening the greedy contract instead.

### 2.3 The Second Test Was the Real Confirmation

Issue #1102 named one test (`sampling_tests.rs:70`). Running the neighbouring module surfaced a second: `build_server_generate_options_applies_request_overrides` (`src/server/request_options_tests.rs:129`) constructs `temperature: Some(0.0)`, `dry_multiplier: Some(0.9)`, `dry_sequence_breakers: Some(vec![1, 2])`, which is the bug report's exact request shape, and asserted `Vec::<i32>::new()` at line 172. It failed on the fix with `left: [1, 2], right: []`.

That matters for the adjudication the issue asked for. The issue offered two readings, an omission or a deliberate "DRY is greedy-inert" decision, and asked whoever picked it up to record which. A deliberate decision would have been recorded once, at the helper, and the assertions at the two layers would agree because one is the reason for the other. Instead the same unexplained assertion appears independently at both layers, transcribed from the branch rather than derived from an intent. Combined with the greedy branch's own comment naming DRY as temperature-independent, the omission reading is the only self-consistent one.

### 2.4 Adjacent Paths Not Touched

`src/lib/mlxcel-xla/src/sampler.rs:218` runs the same breaker-terminated match against its own `params.dry_sequence_breakers`, populated at `src/server/batch/xla_worker_admission.rs:593` by cloning the already-built `sampling.dry_sequence_breakers`. It therefore inherits the fix rather than needing a parallel one. `src/commands/generate.rs:922` hardcodes `Vec::new()` on the CLI path with a comment recording that as a scope decision (#1118); unchanged here.

---

## 3. Technical Decisions

### 3.1 Forward the Field Rather Than Drop the Other Four

| Option | Pros | Cons |
|---|---|---|
| **Chosen: forward `dry_sequence_breakers` in the greedy branch** | Makes an accepted request behave as documented; consistent with the branch's own XTC comment; inert unless DRY is enabled | Changes output for existing `temperature: 0` + `dry_multiplier > 0` + breakers callers, which is the intended correction |
| Drop all five DRY fields from the greedy branch (make DRY greedy-inert) | Internally consistent; one clear rule | Silently disables a penalty for every existing `temperature: 0` caller who enabled it, a strictly larger behavior change with no evidence anyone wanted it; contradicts `sampling.rs:389`, which gates DRY on the multiplier alone |
| Reject the field at the request layer when `temperature <= 0` | Fails loudly instead of silently | Breaks working requests; the field is valid at any temperature; solves a documentation problem with a 400 |

The issue itself set the bar: the drop reading "would change existing behavior for anyone running `temperature: 0` with `dry_multiplier > 0`, so it needs a stronger argument". No such argument exists in the tree. The evidence points the other way.

### 3.2 State the Whole DRY Contract in the Greedy Test

The greedy test now asserts all five DRY fields rather than the one that changed. The reason is the defect's own origin: a per-field struct literal fails silently when a field is dropped, and a test that checks a subset of the fields cannot catch the next drop. Asserting the complete set makes the branch's DRY contract legible in one place, which is what the issue asked for in its third acceptance criterion.

### 3.3 Add a Named Regression Test Rather Than Relying on the Widened Assertion

`build_sampling_config_keeps_dry_sequence_breakers_at_zero_temperature` restates the reported request (`dry_multiplier: 0.8`, breakers `[198]`) as its own test. The widened greedy-defaults assertion would catch a regression, but its name describes greedy defaults, so a failure would not tell the next reader what was broken. A named test carries the bug report into the failure message.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 3 |
| Lines added | +41 |
| Lines deleted | -2 |
| Production lines changed | 1 (plus 6 comment lines) |
| Tests added | 1 |
| Test assertions corrected | 2 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Sampling policy | `src/execution/sampling.rs` | Greedy branch forwards `dry_sequence_breakers`; comment records that the breakers are the DRY match-termination condition, so omitting them inflates the penalty rather than disabling it |
| Unit tests | `src/execution/sampling_tests.rs` | Greedy test asserts all five DRY fields with a comment stating the contract; new named regression test for the reported request |
| Unit tests | `src/server/request_options_tests.rs` | The end-to-end override test asserted an empty breaker vector for a request that set `[1, 2]`; corrected with a comment naming the greedy branch |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `6d8badc7` | fix | fix(server): thread dry_sequence_breakers through the greedy branch |

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib execution::sampling`: 3 passed.
- `cargo test --profile test-fast --features cuda --lib server::request_options`: 35 passed.
- `cargo fmt --all -- --check`: clean.
- Pre-fix reproduction confirmed: `build_server_generate_options_applies_request_overrides` failed with `left: [1, 2], right: []` against the unmodified assertion, which is the defect observed from the server's own option-building layer rather than from the helper under test.

### Not Covered

No generation-level assertion that the penalty value changes. Such a test would need a model and a token history, and would be seed- and checkpoint-dependent; the configuration-level assertions pin the contract that was actually broken. The `mlxcel-xla` sampler path inherits the corrected vector by clone and was not separately exercised.

### Follow-up

`--dry-sequence-breakers` is parsed at startup and never reaches the sampler at all (#1103), which is the operator-facing half of the same gap: this PR makes a per-request value survive the greedy branch, and #1103 makes a server-wide default exist in the first place. The flag's spelling diverges between the two server binaries (#1109).
