# Technical Report: PR #1770 - test: fix settings-schema roster and drafter-refusal drift

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: post-merge workspace gate on main `58dc9693`
**Status**: Pre-merge (both repaired suites pass under their own filters, clippy and fmt clean; the workspace gate is re-run on main after this merges and is expected to report only #1769)
**Languages**: Rust
**Risk Level**: Low (no arithmetic and no request path; two roster entries, three match arms, one assertion substring, one test rename)

---

## Executive Summary

The workspace gate, `cargo test --workspace --profile test-fast --features metal,accelerate`, run against merged main at `58dc9693` after the 2026-09-10 batch of merges, reported 10981 passed and 3 failed. Two of the three are cross-file contract drift and this PR repairs both. #1759 added `video_max_frames` and `video_fps` to `ServerConfig` without adding them to the classification roster that the management settings schema is built from. #1751 reworded the DFlash drafter refusal, updated the unit test that asserts it, and left an integration test under `tests/` asserting a substring the new wording no longer contains.

Neither defect sits inside the diff of the PR that caused it, and neither was reachable from a test scope chosen by reading that diff. Both PRs were green on their own heads when they merged. That asymmetry is what this report is about: a per-PR scope is derived from the files the PR changed, while a contract violation lives in the file it did not change.

The third failure, `fp8_block_requantize_matches_direct_path`, is not drift. It is a tolerance derivation exposed by a host toolchain upgrade, it is deliberately untouched here, and it is tracked in #1769.

---

## 1. Two contract violations, neither inside a diff

### 1.1 A field on `ServerConfig` is not yet a knob on the settings API

`src/server/runtime_settings.rs` publishes the management settings surface behind `GET /v1/settings` and `GET /settings`. `schema()` does not reflect over `ServerConfig`. It iterates `CLASSIFIED_SERVER_CONFIG_FIELDS`, a hand-maintained `&[&str]` whose doc comment states the intent plainly: every field declared on `ServerConfig`, in source order, so that "a future field cannot silently miss the settings schema."

The word doing the work is *silently*. Because `schema()` iterates the roster rather than the struct, a field absent from the roster is not an error at any level; it is simply not published. A server started with `--video-max-frames 8` answers `GET /v1/settings` with a schema and a current-values map that never mention the flag the operator set, and nothing in the process says so.

`server_config_schema_classifies_all_105_fields` is the guard for that. It re-reads `config.rs` at compile time through `include_str!`, parses the `pub` field names out of the `ServerConfig` declaration, and asserts three things: the declared count, the roster equal to the declared list including order, and the schema's own name list equal to the roster mapped through `api_name`. #1759 pushed the declared count to 107 while the roster stayed at 105, and the first assertion fired with the message it was written to give, `ServerConfig field count changed`.

Repairing it is four edits, not one. `video_max_frames` and `video_fps` go into the roster between `vision_cache_size` and `lang_bias_config`, which is where `config.rs` declares them, then into `read_only_reason` (both under `SCHEDULER_REASON`), `read_only_value` (both `json!`), and `read_only_kind` (`Int` and `Float` respectively).

### 1.2 An assertion on error prose, in the one target the author's scope did not build

`dflash_drafter_not_standalone_error` in `src/models/detection.rs` is the shared refusal every entry point raises when a checkpoint directory turns out to be a speculative drafter rather than a standalone model, so that the offline `-m` path, server startup and the stage loaders give one message instead of three different weight-lookup symptoms (#1170).

The message has been reworded twice. #1751 (`7fae0b56`) turned `is a DFlash speculative drafter checkpoint` into `is a DFlash-family speculative drafter checkpoint (Qwen 3.5 DFlash or LFM2 DSpark)`. #1762 (`fb12866a`) reworded the parenthetical again to add the Muse Glimmer assistant.

Three files assert on that message by substring. Two of them, `src/models/detection_tests.rs` and `src/commands/generate_tests.rs`, were moved to `DFlash-family speculative drafter` by #1751 itself and were therefore untouched by #1762, because the second reword changed only the list inside the parentheses. The third, `tests/speculative_dispatch.rs:485`, still asserted `DFlash speculative drafter`, the wording it was given when the test was written in #1170 (`865bede2`), 434 commits earlier.

The substring was chosen loosely on purpose, so that rewording the message would not break the test. It broke anyway, because the reword inserted `-family` inside the substring rather than around it. The repair moves this assertion onto the same substring the other two already use, which is now the one span that has survived a reword.

---

## 2. Why a per-PR scope could not see either

Both PRs ran the right tests for their own diffs. #1759 touched 34 files including `src/server/config.rs`, and neither `runtime_settings.rs` nor `runtime_settings_tests.rs` was among them. #1751 touched 29 files across `src/models/`, `src/lib/mlxcel-core/src/drafter/` and `src/server/batch/`, and nothing under `tests/`.

The two misses have different shapes, and the distinction matters when deciding what a narrow scope should be.

**A filter miss.** `runtime_settings_tests.rs` is `#[cfg(test)] mod tests` inside `runtime_settings.rs`, so it is in the same `--lib` target as the `config.rs` change that broke it. A scope of `--lib server::chat_request`, `--lib server::startup` or any other module filter chosen from the diff excludes it. A bare `cargo test --lib` would have run it. The test was one filter argument away.

**A target miss.** `tests/speculative_dispatch.rs` is a separate integration binary of the root `mlxcel` package. No `--lib` invocation ever builds it, whatever the filter. As `CLAUDE.md` records under Test scope, `--all-targets` without `--workspace` does not compile a member's test target either, and a bare `cargo test` resolves to `-p mlxcel` and never builds `mlxcel-core`. The test was outside the target set, not outside the filter.

Neither miss is carelessness. A scope selected by asking "what did I change" cannot contain "what elsewhere asserts about what I changed", because the answer to the second question is exactly the set of files the first excludes. Finding it takes either a tree-wide search for the changed symbol before choosing the scope, or a run wide enough not to need one.

There is a second reason a per-PR run could not have caught these even at full width: neither defect existed on either PR's head. #1759's branch was correct against the roster it was written against. So was #1751's assertion against the message on its base. The drift is a property of the merged state, and the merged state exists for the first time on main. A gate that runs once per PR, before the merge, is measuring a tree in which the contract still holds.

That leaves exactly one layer where the merged state and the full target set meet, which is the workspace gate run on main after a batch of merges. It is not a CI check: GitHub CI does not run the Metal test suite at all, so neither repaired suite has ever executed there. Both defects would have shipped otherwise, and the visible one, the settings roster, would have shipped a management API that omits two knobs the same release advertises on the CLI and in `docs/llama-server-compat.md`.

---

## 3. The third failure is not drift

`models::fp8_block::fp8_block_tests::fp8_block_requantize_matches_direct_path` also fails on `58dc9693`, and is left alone.

It is deterministic, not a flake: three consecutive single-test runs reproduce `element 4: mxfp8 error 0.50390625 exceeded 0.3002931 (group max 4.8046875)` character for character, because the fixture is seeded and its input bytes and bf16 block scales are identical on every host.

It is also not a drift symptom in the sense of section 1. The same test asserts three things in order, and the two byte-identity assertions pass: the packed mxfp8 plane and the scale plane produced through `requantize_block_fp8_weights` still match `quantize_weights_with_mode` on the directly quantized tensor, exactly. What fails is the third assertion, an accuracy bound on MLX's own mxfp8 quantize and dequantize round trip, which is not the relationship the test's name describes.

The bound is `group_max / 16.0 + f32::EPSILON`, justified in the source comment as a round-to-nearest half-ulp argument for E4M3's four significant bits. The observed error is `0.50390625 / 4.8046875 = 0.10488` of the group maximum, between half an ulp (`2^-4`) and one full ulp (`2^-3`). That is the signature of a quantizer rounding toward zero rather than to nearest, which is a claim about MLX's arithmetic on this backend rather than about anything the 2026-09-10 merges did.

What changed is the host, not the tree: this machine moved to macOS 27.0 and Xcode 27 on 2026-09-10, after #1742 merged, and CI has never exercised this test on Apple GPU generation 17. Folding it in here would mean widening a numeric bound to make a suite green, which is the failure mode `CLAUDE.md` names under "Changes that move the numbers". #1769 puts the decision criterion first instead: run the same seeded fixture on an M1 Ultra and let the two hosts agreeing or disagreeing decide whether this is a backend divergence or a bound that never held. That is a separate standard of evidence, and it keeps its own PR.

---

## 4. What the repair encodes

### 4.1 Four registration sites, one loud

Adding a read-only field to the settings surface means touching four places in `runtime_settings.rs`. Only one of them complains on its own.

| Site | What it decides | If a new field misses it |
|---|---|---|
| `CLASSIFIED_SERVER_CONFIG_FIELDS` | whether the field is published at all | Silent. `schema()` iterates the roster, so the field is absent from `GET /v1/settings` and from `current()`. |
| `read_only_value` | the value published | Loud. `unreachable!("read-only ServerConfig field missing from schema: {field}")`, but only reachable once the roster entry exists. |
| `read_only_kind` | the declared type | Silent. `_ => KnobKind::Str`, so `video_max_frames` would be published as a string knob. |
| `read_only_reason` | why it cannot be patched | Silent. `_ => MODEL_REASON`, so a scheduler knob would claim to be fixed by the loaded model provider. |

The count test covers rows one and two, the second only indirectly: it builds `schema(&ServerConfig::default())`, so a roster entry with no value arm panics inside the test rather than in a running server. It does not cover rows three and four. It asserts names, order, mutability, uniqueness and that every read-only spec carries a non-empty reason, and it never asserts which kind or which reason. The `Int` and `Float` choices this PR makes for the two video fields are therefore correct and unasserted, which is section 7's first follow-up.

### 4.2 The count is in the test name on purpose

Renaming `server_config_schema_classifies_all_105_fields` to `..._107_fields` alongside the three constants looks like churn and is not. A diff that moves 105 to 107 in three places and leaves the name alone reads as a mechanical bump; renaming the test puts the new number in the one line a reviewer sees before the body. The cost is that every field addition renames a test, which is the intended friction.

---

## 5. Validation

| Gate | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib server::runtime_settings` | 8 passed |
| `cargo test --profile test-fast --features metal,accelerate --test speculative_dispatch` | 22 passed |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| CI on the PR | 13 checks, 10 pass and 3 skipping for untouched paths |

None of the ten passing checks runs either repaired suite, because GitHub CI does not run the Metal test suite. The evidence that these two tests pass is the local run; the evidence that they stay passing is the next workspace gate on main.

That gate cannot run on the merged result until this merges. With #1769 still open it should then report exactly one failure, and anything else means a third drift the 2026-09-10 run did not reach.

---

## 6. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 3 |
| Lines added | 15 |
| Lines removed | 7 |
| Tests added | 0 (two existing tests repaired, one renamed) |

### Changes by file

- `src/server/runtime_settings.rs`: `video_max_frames` and `video_fps` added to `CLASSIFIED_SERVER_CONFIG_FIELDS` in `ServerConfig` source order, to `read_only_reason` under `SCHEDULER_REASON`, to `read_only_value` as `json!` passthroughs, and to `read_only_kind` as `Int` and `Float`.
- `src/server/runtime_settings_tests.rs`: the roster test renamed to `server_config_schema_classifies_all_107_fields`, with its declared-count, schema-length and uniqueness assertions moved from 105 to 107.
- `tests/speculative_dispatch.rs`: the `-m <drafter>` refusal assertion moved to `DFlash-family speculative drafter`, matching `detection_tests.rs` and `generate_tests.rs`.

### Related commits

| Hash | Type | Subject |
|---|---|---|
| `fc6b10d4` | test | fix settings-schema roster and drafter-refusal drift |

### Related issues and PRs

Refs #1322, #1339, #1343, #1769. Repairs drift introduced by #1759 (`41a0adc9`) and #1751 (`7fae0b56`); the assertion in `tests/speculative_dispatch.rs` dates to #1170 (`865bede2`).

---

## 7. Follow-up

**Kind and reason are unasserted for every read-only field.** `read_only_kind` and `read_only_reason` both fall through to a default, so a field can be published with the wrong type or the wrong justification and every current test passes. Extending the roster test to pin the kind and the reason class per field would close it, at the cost of a table that has to be maintained alongside the roster. The narrower version, asserting only that no read-only field lands on the `KnobKind::Str` fallback unless it is declared to, is cheaper and catches the case that occurred here.

**`config.rs` does not point at the roster.** The obligation is documented once, on `CLASSIFIED_SERVER_CONFIG_FIELDS`, which is the file an author adding a field does not open. The field-level doc comments on `video_max_frames` and `video_fps` are thorough about the fallback path and say nothing about the settings surface. A short note on the `ServerConfig` struct doc naming the roster would put the pointer where the change starts, which is the same reasoning `docs/code-guidelines.md` gives for the `// Used by:` convention.

**Error-prose assertions have no roster of their own.** Three files assert on `dflash_drafter_not_standalone_error` and nothing connects them to it. The message's doc comment could name them, the way a shared function names its callers, so the next reword has a list to check rather than a grep to remember.

**The window between a merge and the next gate.** These defects lived on main from 2026-09-10 until the gate ran, and that window is bounded by how often the gate runs rather than by anything either PR could have done. Narrowing it means running the workspace gate on each merge to main, or getting a Metal runner into CI, which would also cover #1769's class. #1769 itself is open, with the M1 Ultra arm of its decision criterion not yet run.

---

### Transferable lesson

A test correctly scoped to a diff cannot, in principle, cover the contract that diff breaks somewhere else. The two failures here are the two ways that plays out: one test was in the right target under the wrong filter, the other in a target the invocation never built. Both authors ran a defensible scope and both defects reached main anyway, because what they violated was not a property of either branch. It was a property of the merged tree, which neither branch had.

So a wide gate is not redundancy over the narrow ones; it measures a different object. The narrow run answers whether the change works. The wide run on merged main is the only one that answers whether the tree still agrees with itself, and it is the cheapest place to learn that two knobs are missing from an API before an operator finds out by setting one.
