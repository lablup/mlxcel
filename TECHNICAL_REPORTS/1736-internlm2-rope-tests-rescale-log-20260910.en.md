# Technical Report: PR #1736 - fix(internlm2): add family rope_scaling tests and NTK rescale log

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (module tests, clippy and fmt green; two real-checkpoint runs on `mlx-community/internlm2_5-7b-chat-4bit`; no corrected long-context reference exists for this family, so the rotation itself stays unverified)
**Languages**: Rust
**Risk Level**: Low (no arithmetic changed; the single hot-path edit is a level check guarding a debug-only diagnostic)

---

## Executive Summary

Issue #1320 asked for InternLM2 to honor `rope_scaling` and `rope_traditional`. By the time this branch opened, the code already did: PR #1389 had wired `ModelArgs` and `Attention` to the shared `DynamicNtkRope` schedule while closing #1324, and that covers every step of #1320's implementation plan. What remained was the issue's other half, which is family-level tests proving the config plumbing reaches the helper, a debug line reporting the rescaled base, and a run against the real checkpoint.

The diagnostic is the piece with lasting value. A dropped `rope_scaling` block produces the correct schedule for every sequence inside `max_position_embeddings`, so nothing shorter than a 32768-token prompt can observe the defect, and until this PR a running binary said nothing about the base it was rotating at. A 34021-token prompt now emits exactly one line at debug level, `seq_len=34021 base_eff=1077737.0`, which is the closed form evaluated at that length.

---

## 1. What #1320 asked for, and what was left of it

### 1.1 The implementation plan was already on main

The issue's plan is four steps: declare `rope_scaling` on `ModelArgs`, replace `Attention`'s `rope_dims` / `rope_base` pair with a `DynamicNtkRope`, replace the two hard-coded `fast_rope` calls with `self.rope.apply(...)`, and extend the helper's `// Used by:` comment. All four had already merged in #1389, which fixed both InternLM families in one pass because a single helper serves both and both had inherited the same wrong expression from mlx-lm (`rope_scale = 1 / factor if rope_type == "linear" else 2.0`, then reused as the NTK factor).

Checking that before writing any code was the load-bearing step. The issue body still reads as untouched, and following it literally would have produced either a no-op diff or a second parse path for the same `rope_scaling` block.

### 1.2 What was left

The issue's validation section names five unit tests and asks for a real-checkpoint run with a debug log of `base_eff` at the first forward past `max_position_embeddings`. #1389 shipped three tests (the checkpoint block parses, an absent block resolves to the unscaled schedule, an unimplemented scheme fails the load) and none of the log. This PR adds five tests to `src/models/internlm2_tests.rs`, one to `src/models/dynamic_ntk_rope_tests.rs`, and the diagnostic itself.

---

## 2. Why the family needs config tests of its own

InternLM2's defect sat one stage earlier than InternLM3's. InternLM3 parsed its block and then mis-resolved it, so a helper test could see the wrong answer. InternLM2 never declared the field, so `{"type": "dynamic", "factor": 2.0}` was discarded by serde before any schedule existed, and no amount of testing `DynamicNtkRope` in isolation would have caught it. Only a test that starts from a `config.json` excerpt and goes through `ModelArgs::rope()` covers that stage.

The five new cases are deliberately scoped to the plumbing rather than to the arithmetic, which `dynamic_ntk_rope_tests.rs` already covers for the same geometry:

| Test | What it pins |
|---|---|
| `a_linear_block_reaches_the_helper_as_an_inverse_factor_scale` | `{"type": "linear", "factor": 4.0}` resolves to `scale() == 0.25`, with `base_for` unmoved at 56 and at 65536 |
| `rope_traditional_reaches_the_helper` | `rope_traditional: true` arrives at `DynamicNtkRope::traditional()`, and a silent config keeps serde's `false`, which is what every public checkpoint runs |
| `the_checkpoint_configs_dynamic_schedule_matches_the_issue_table` | `base_for` at 56, 32768, 40000 and 65536, against the issue's own table within 1e-3 relative |
| `the_rope_type_spelling_resolves_identically_to_type_for_this_family` | both key spellings give `Dynamic { factor: 2.0 }`, so a conversion that renames the key cannot silently change the schedule |
| `a_dynamic_block_without_a_factor_is_a_load_error_naming_the_family` | the rejection message contains `internlm2`, so an operator with several checkpoints loaded learns which one is malformed |

`traditional()` is a new accessor added for the second of these. It is the only public API this PR adds.

---

## 3. The rescale diagnostic

`DynamicNtkRope::apply` now computes `base_eff` into a local, hands the same value to `fast_rope` as before, and offers it to a debug event on the way. Four decisions in that path are worth recording.

### 3.1 Deduplicated per schedule, not per process

A process can hold more than one model: the server's `--models-dir` routing, the pipeline stage executors, and the tensor-parallel ranks all put several checkpoints behind one binary. A process-wide `Once` would let the first checkpoint past the boundary suppress the line for every later one. `rope_utils::report_unusable_rope_scaling_once` had already reached this conclusion and keys its set on `(model_label, rope_type)`.

`DynamicNtkRope` is `Copy` and carries no label, so the key here is the schedule itself: `(dims, base.to_bits(), max_position_embeddings, factor.to_bits())`. Raw bits rather than the `f32` values, so the tuple is `Ord` without a total-order wrapper. That makes the key an identity on bit patterns rather than on values, and `from_scaling` screens `factor` through `is_usable_scalar` but leaves `base` unscreened, so a `-0.0` or NaN base could split one schedule across two lines. No real checkpoint produces either, and the cost if one did is a duplicate log line.

Being `Copy` also means the per-layer rebuild is a distinct value with identical parameters, and the test asserts that such a rebuild does not log again.

### 3.2 The level check sits in `apply`, not in the helper

Past the boundary, the dedup lock would otherwise be taken once per call, and `apply` runs twice per layer per forward. The steady state after the line has been emitted is a lookup that can only answer "already present": roughly 40ns uncontended, a few microseconds per forward against a decode step of tens of milliseconds at these context lengths. Small, but paid on every long-context step by every run, including runs with no subscriber installed.

`apply` therefore asks `tracing::enabled!(tracing::Level::DEBUG)` before touching the lock. The check could not go inside `log_dynamic_rescale_once`, because the unit tests call that function directly, with no subscriber, and assert on its return value.

### 3.3 The guard declares no fields while the event declares six

This is the one accepted gap. `tracing::enabled!` with no fields and the `tracing::debug!` it guards agree on level and target but not on fields, so an `EnvFilter` directive that selects on a field (`RUST_LOG=[{seq_len}]=debug`, say) would enable the event and not the guard, and the line would go missing with no indication. Every documented way in is a level directive instead: `-v` and `--verbosity 4` both expand to one through `server::logging::filter_directive_for_verbosity`, and the issue's own recipe is `RUST_LOG=debug`. The gap is recorded in the source comment rather than closed.

### 3.4 The logged base is a snapshot, not the base in force

The key omits `seq_len` on purpose, so a request that crosses the boundary and then grows further rescales without logging again. A reader who takes the line as "the base this run used" will be wrong for any request that kept going. At 34021 tokens the base moves about 62 per token, so the drift is quick. The doc comment states this, because the alternative (keying on `seq_len` too) would emit a line per distinct length and turn a validation aid into per-step output.

One smaller point: the mutex guard is dropped before the event fires, so a panicking subscriber cannot poison the lock, and the lock is recovered with `unwrap_or_else(|err| err.into_inner())` anyway. The set stays a valid set either way, and losing every later line is the worse outcome.

---

## 4. Three documentation corrections

**A checkpoint directory that does not exist.** `models/internlm2-7b-4bit` appeared in the module header, in both test files and in `docs/supported-models.md`, and it came from the issue body, which names it too. The directory is `models/internlm2_5-7b-chat-4bit`. Anyone following the issue's validation command would have hit a missing path with nothing explaining why.

**An entry claiming the wrong standard of evidence.** The `docs/supported-models.md` entry covers InternLM 2 and 3 together and credited both with InternLM3's evidence: token-exact against a corrected mlx-lm 0.31.3 greedy reference, pinned by `tests/causal_prefill_greedy_parity.rs`. InternLM2 has no such reference. The entry now records what its own checkpoint does establish, which is that a 34021-token prompt enters the dynamic branch and rescales to the documented base, and states that the accompanying byte-identity pair is a control rather than a parity result.

**An f32 / f64 comparison read at the wrong length.** The entry's previous parenthetical compared the closed form at 34020 against the f32 value printed at 34021 and reported the difference as a precision gap. It is a length difference: f64 at 34021 gives 1077736.99, so the f32 the code computes agrees to every digit the line prints, and 34022 gives 1077799.125, which the same f32 evaluation reproduces bit for bit. The value now travels with its length in the entry.

---

## 5. Validation, and what it does not establish

### 5.1 Gates

| Gate | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib models::internlm2` | 8 passed |
| `cargo test --profile test-fast --features metal,accelerate --lib models::dynamic_ntk_rope` | 17 passed |
| `cargo clippy --lib --tests`, `cargo fmt --all --check` | clean |
| CI on the PR (13 checks) | pass |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | runs in `nightly-verify.yml`, not on this PR |

The last row is one of #1320's acceptance criteria. `ci.yml` runs the scoped clippy and fmt jobs, so the workspace gate is satisfied by the nightly job rather than by this PR's checks.

### 5.2 Real checkpoint

`models/internlm2_5-7b-chat-4bit`, greedy, against a binary built from this branch's parent commit:

| Prompt | Tokens generated | Result |
|---|---|---|
| 89 tokens | `-n 64` | byte-identical, MD5 `5a6d6133` |
| 34021 tokens | `-n 32` | byte-identical, MD5 `69a12bff` |

That pair is a control, not a parity result. The parent commit already carries #1389's wiring, so the two arms differ only by the debug diagnostic, and byte-identity is what it should show: the diagnostic is inert with respect to the output. It is reported as such rather than as evidence about the rotation.

The long-context run was repeated against the final binary, after the `tracing::enabled!` gate was added, so the diagnostic is confirmed to still reach a `RUST_LOG=debug` run and not merely to have done so before that commit. The 34021-token prompt emits exactly one line, `seq_len=34021 base_eff=1077737.0`, and the generation answers the question the prompt ends with in fluent, on-topic text about Rayleigh scattering.

### 5.3 What is not established

The rotation itself. InternLM3 is held to a corrected mlx-lm reference because one exists: stock mlx-lm 0.31.3 rotates every token at twice its true position on a `dynamic` checkpoint, so the reference had to be built by replacing its per-layer rope module and nothing else. No equivalent reference was built for InternLM2 here. The standing claim for this family is that the dynamic branch is entered at the documented length and computes the documented base, and no more than that.

---

## 6. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 5 |
| Lines added | 293 |
| Lines removed | 10 |
| Tests added | 6 (5 family-level, 1 dedup) |

### Changes by file

- `src/models/dynamic_ntk_rope.rs`: the `traditional()` accessor, `log_dynamic_rescale_once` with its per-schedule dedup set, and the `tracing::enabled!` gate in `apply`. The rotary arithmetic is unchanged; `base_for(seq_len)` is hoisted into a local and passed to the same `fast_rope` call.
- `src/models/internlm2_tests.rs`: five tests covering `ModelArgs::rope()`, plus a header note recording which tests came with #1324 and which with #1320.
- `src/models/dynamic_ntk_rope_tests.rs`: `the_rescale_log_fires_once_per_schedule_not_once_per_process`, and a file-level comment reserving the geometries that test depends on.
- `src/models/internlm2.rs`: a module-header paragraph separating #1389's wiring from #1320's tests and log, and the corrected checkpoint directory.
- `docs/supported-models.md`: the InternLM entry's evidence claim and the `base_eff` figure.

### Related commits

| Hash | Type | Subject |
|---|---|---|
| `c26adb2` | fix | add family rope_scaling tests and NTK rescale log |
| `9d92894` | fix | key the NTK rescale log per schedule |
| `a2317d0` | perf | skip the dedup lock when debug logging is off |
| `a1accf3` | docs | state what the long-context run does and does not show |
| `1aba962` | docs | correct four claims in the rescale log's comments |
| `4d2911a` | docs | correct the f64 base_eff figure for InternLM2 |

### Related issues and PRs

Closes #1320. Depends on #1324 and its implementation in #1389, which supplied `src/models/dynamic_ntk_rope.rs` and the wiring this PR tests.

---

## 7. Follow-up

**No long-context parity reference for InternLM2.** Building one takes the same shape as InternLM3's: mlx-lm 0.31.3 with its per-layer rope module replaced by the corrected schedule, compared on identical prompt ids. Until then `docs/supported-models.md` says so in the entry, which is the mitigation, not a fix.

**The field-filter blind spot in section 3.3.** A user who filters on fields loses the line silently. Closing it means declaring the event's fields in the `enabled!` guard, or dropping the guard and accepting the lock.

**The dedup set couples test ordering.** `log_dynamic_rescale_once` writes to a process-global set, and the unit test asserts on first-crossing behavior, so any later test that crosses `max_position_embeddings` on a reserved geometry would make the first assertion depend on execution order. The mitigation is a comment naming the reserved geometries. A reset hook behind `#[cfg(test)]` would be sturdier if this file grows more tests that log.

### Transferable lesson

An issue can be most of the way closed by a PR filed under a different number, and nothing in the issue will say so. #1320 read as untouched while its whole implementation plan sat on `main`, because #1389 landed the shared helper for #1324 and wired both families at once. The check that caught it was cheap: read the current code before the plan. The failure it avoided is not a wasted afternoon but a duplicate parse path for one `rope_scaling` block, which is the exact class of defect the shared helper was introduced to prevent.
