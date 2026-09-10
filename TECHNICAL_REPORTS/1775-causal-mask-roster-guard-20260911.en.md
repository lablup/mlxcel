# Technical Report: PR #1775 - chore(core): fix create_causal_mask roster count and add a drift guard

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed
**Languages**: Rust (doc comments and a unit test), Markdown
**Risk Level**: Low (no production code path changes; the new test is `#[cfg(test)]` only)

---

## Executive Summary

The `// Used by:` rosters in `src/lib/mlxcel-core/src/utils.rs` are how a contributor answers "what breaks if I change this helper", and `create_causal_mask` is the worked example `docs/code-guidelines.md` points at. Its count said 45 caller files under `src/models` when the grep it documents returned 49. The command it printed searched all of `src` and returned 64. Six sibling rosters had drifted as well, and two of them named callers that never reach the function.

This PR corrects all seven rosters from fresh greps at the base commit `8ad62d4b` and from each call site. It narrows the printed command to the population the count describes, and adds a unit test that fails with an actionable message whenever the stated count and the tree disagree.

---

## Problem Statement

The count was last right at `f0bf3a2c` (#1121, 44). Five caller files arrived afterwards, and only one of them was ever named in the prose. The issue traced the drift commit by commit. Nothing enforced the roster: `make verify` checks crate versions, CUDA kernel dtype keys and the llama-server manifest, but not rosters. So each new family that called the mask helper silently made the discovery mechanism wrong.

Review of the first commit found a worse failure than a stale number, a roster that lists non-callers:

- **`create_causal_mask_with_window`** named Gemma2, Gemma3 and Qwen3. Gemma2 and Qwen3 call `causal_attention` with window `0`, which takes the plain causal branch. Gemma3 never calls `causal_attention`; its window mask comes from `create_sliding_window_prefill_mask`. The families that actually reach the function through a nonzero window are Baichuan, Cohere2, Cohere2MoE, CohereCompass, Exaone4, ExaoneMoE, Gemma3n, Gemma4, IQuestLoopCoder, Laguna, Mellum, Ministral3, MuseGlimmer and Step3P5.
- **`softplus`** counted the cxx declaration of `ffi::softplus` in `lib.rs` and Apertus's own scalar `fn softplus(x: f32)` as callers. It also missed that `pub use ffi::*` makes `mlxcel_core::softplus` the raw FFI function. Mamba, Mamba2, Jamba, RecurrentGemma, FalconH1, GraniteMoeHybrid, NemotronH, Plamo2 and the two audio attention files call that directly and never touch the wrapper. Only GatedDelta, Laguna and DeepSeekV4MoE call `utils::softplus`.

A contributor reading those rosters before changing either helper would have checked the wrong models.

---

## Change Summary

- **Rosters.** `create_causal_mask` states 49 and assigns every file to a named group: Laguna joins the hybrid group, CohereCompass the sliding-window group, and the GLM4-MoE-Lite MTP drafter and the LFM2 DSpark drafter fold into their families. The "outside `src/models`" sentence now covers every grep hit, including a test-only use in `cache.rs` and a comment mention in the Qwen3.5 MTP drafter. The six sibling rosters distinguish direct callers from callers that arrive through a wrapper or the shared `causal_attention` dispatch, and they name the grep's false matches as such. `repeat_kv` and `softcap` were already accurate and are untouched.
- **Regeneration command.** It is now `grep -rln '\bcreate_causal_mask(' src/models --include='*.rs' | grep -v 'tests\.rs$'`, which returns the stated number. The copy of it in `docs/code-guidelines.md` matches, and the guidelines' worked example writes `N` instead of a count that would go stale on its own.
- **Guard test** `utils::tests::create_causal_mask_roster_count_matches_repo`:
  - It walks `src/models` with `std::fs`, recursing on `DirEntry::file_type()` so it never follows a symlink, the same as `grep -r`.
  - It counts files containing `create_causal_mask(` at a word boundary and excludes files whose names end in `tests.rs`, which drops `diffusion_gemma/tests.rs`.
  - It reads the stated count from a fixed sentence prefix in `include_str!("utils.rs")` and asserts equality.
  - On failure it prints the stated and measured counts, the sorted file list, the file and line to edit, and the command to re-run. The line number is computed from the marker's position, so it does not go stale.
  - It skips only when the workspace's own `utils.rs` is unreachable from the computed root, as in a vendored build, and otherwise asserts that `src/models` exists. I/O errors panic with the path, not with a misleading drift message. It adds no dependency.

---

## Technical Decisions

**A unit test, not a `make verify-*` script.** The count is one integer behind a fixed prefix, and `verify-test` already runs on every PR. A script would need a new Makefile target and CI job for one integer. The script-shaped alternative is `scripts/ci/check_kernel_dtype_keys.py`, and it was rejected for this.

**A preceding `:` still counts.** The issue asked to exclude a match preceded by `_`, an alphanumeric character, or `:`. Excluding `:` would measure 35, because 14 files call the helper as `utils::create_causal_mask(`. The test would then disagree with the grep definition the doc comment freezes. The implementation follows `grep`'s `\b` semantics, and a comment at the rule explains why, so nobody "fixes" it to match the issue text.

**Count only, not family names.** The sibling rosters name families rather than counts, so there is no integer to compare. A general `// Used by:` checker (absence detection, a family-to-path map) is #1141's design question and stays out of scope.

---

## Validation

- The regeneration command returns 49 at the base with this shell's `grep`, BSD `/usr/bin/grep` and `rg`.
- Negative proof: with the count edited to 48 and then 50, the guard fails with the full actionable message; restored to 49, it passes.
- On the first commit, the workspace gate (123 binaries, 11,015 passed, 0 failed, 359 ignored), workspace clippy with `-D warnings`, fmt, and `make verify-versions` / `verify-kernel-dtype-keys` / `verify-llama-compat` all pass. The review-fix commit touches only the doc comments and the test module of the same file, and was re-verified with the full `mlxcel-core` lib suite and workspace clippy.

---

## Learning Points

- **A roster that names non-callers is worse than an incomplete one.** An incomplete roster makes a reader look further. A wrong one sends them to check Gemma2 and Qwen3 for a change that cannot affect them, and to skip the fourteen families it does affect. The count guard cannot catch this; only reading each call site's arguments does. The window roster's errors were found that way in review.
- **The definition has to be the same in three places.** The sentence, the printed command and the test each encode what counts as a caller file. They drifted apart once (the command said `src`, the sentence meant `src/models`), and this PR makes the test fail whenever the other two disagree.

---

## Follow-ups

- #1141 remains open for the general `// Used by:` checker over `utils.rs`, `layers.rs`, `switch_layers.rs` and `gated_delta.rs`. Its criterion "passes on `main` as it stands" is now true again for `utils.rs`.
