# PR #2957: Bounding prefill positions at the C entry points

**Date**: 2026-11-11
**Status**: Completed (closes #2954)

## Executive Summary

PR #2936 bounded decode coordinates in the Rust wrapper and at the C decode entry points, but the four prefill C entry points still forwarded caller-supplied positions to the graph unchecked. The emitted prefill graph indexes the `[context_capacity, rot]` RoPE table with those values and clamps nothing, so a position outside `[0, context_capacity)` is an out-of-bounds device read, not an error the graph can report. This PR adds one shared `xla_check_prefill_positions` helper, calls it before any IREE call from `xla_llama_prefill`, `xla_llama_prefill_slot_logits`, `xla_llama_prefill_diagnostics_slot`, and the shared embeddings prefill implementation, adds the missing NULL checks, and proves the boundary with a real-device test that drives the C entry points directly. 4 files, +283/-2.

## 1. Problem Statement

Decode was fixed in #2936: `mrope_decode_coordinate` and `mrope_slot_coordinates` return a typed `MropeCoordinateError::OutOfRange`, and the C decode entry points repeat the bound so a direct C caller cannot bypass the Rust guard. Prefill was left out of that PR because it is a separate set of entry points and the review scoped the change to decode.

The prefill graph reads every position in the buffer, not only the first `real_len`, because the padded tail reaches the same RoPE gather. `xla_llama_prefill_slot_logits` already checks every token id for exactly that reason, but no prefill entry point checked position values. `xla_llama_prefill` additionally dereferenced `c`, `tokens`, `positions`, and `out_token` without NULL checks, and `xla_llama_prefill_diagnostics_slot` NULL-checked the context and outputs but not `tokens` or `positions`.

The risk is a device read outside the RoPE table from an unvalidated integer. The Rust API normally synthesizes canonical positions, so the bug is only reachable by a C caller or by a Rust caller that reaches the C boundary with explicit positions; those callers are the ones the guard must protect.

## 2. What Changed

### 2.1 One shared position guard (`src/lib/mlxcel-xla/csrc/xla_iree.c`)

`xla_check_prefill_positions(caller, positions, count, axes, context_capacity)` walks every axis and every index of the axis-major M-RoPE layout (`axis * count + index`) and returns 1 on the first value outside `[0, context_capacity)`, printing the caller, the axis, the index, the value, and the capacity. All `count` entries are checked even when `real_len < count`, so the padded tail cannot reach the graph with a bad value.

The helper is called from four places, each before any IREE call:

- `xla_llama_prefill` with `lp` positions and 1 or 3 axes.
- `xla_llama_prefill_slot_logits` with `lp` positions and 1 or 3 axes.
- `xla_llama_prefill_diagnostics_slot` with `lp` positions and 1 or 3 axes.
- `xla_llama_prefill_embeddings_impl`, the shared implementation behind the nine `xla_llama_prefill_embeddings*` wrappers, using the descriptor's `context_capacity` count and the explicit `position_mode`.

### 2.2 NULL checks

`xla_llama_prefill` now validates `c`, `tokens`, `positions`, and `out_token` before touching the context. `xla_llama_prefill_diagnostics_slot` adds `tokens` and `positions` to its existing contract check.

### 2.3 Test-support forwarders and one device test

The public Rust API synthesizes canonical positions and `prepare_for_mode` already rejects explicit out-of-range ones, so a device test cannot reach the C boundary through the normal wrappers. Two `MLXCEL_XLA_TEST_SUPPORT` forwarders, `xla_test_prefill` and `xla_test_prefill_slot_logits`, call the real entry points with caller-supplied positions, exposed to Rust as `IreeLlama::test_prefill_with_positions` and `IreeRaggedLlama::test_prefill_slot_logits_with_positions`.

The new `#[ignore]` test `prefill_c_entry_points_reject_out_of_range_positions` in `tests/xla_ragged_hardening.rs` feeds the same tokens with one out-of-range value and expects failure, then feeds the same tokens with every value in range and expects success:

- 1D: the last padded-tail slot is set to -1 and then to `CAPACITY`, with `real_len` 3, proving the tail is checked.
- 3-row M-RoPE: the third axis' first slot is set to -1 and the first axis' padded tail to `CAPACITY`, proving every axis is checked.

## 3. Technical Decisions

### 3.1 One helper instead of four inline loops

Four copies of the same axis-major loop would drift. The helper carries the layout comment, the bound, and the message format once, and each call site passes only what differs: the caller name, the count, the axis count, and the capacity.

### 3.2 Check at C, not only at Rust

The Rust wrapper already rejects explicit out-of-range positions, but the C API is exported and reachable without Rust. The #2936 review established this rule for decode, and prefill follows it: the guard lives at the boundary that can be called directly, and the Rust check remains as the earlier, friendlier failure.

### 3.3 A test-support forwarder instead of widening the public API

Adding a public "explicit positions" parameter to the Rust prefill API would change a stable signature to test a C guard. The two forwarders compile only under `MLXCEL_XLA_TEST_SUPPORT` and call the same C functions production uses, so the test exercises the real boundary without expanding the public surface.

### 3.4 Check the padded tail

`real_len` bounds how much of the sequence is semantically real, not how much of the buffer the graph touches. The RoPE gather consumes the whole `positions` buffer, so bounding only the first `real_len` entries would leave the same out-of-bounds read reachable at index `real_len..lp`. The loop covers the full count, matching the existing token-range check.

## 4. Validation

Environment: AMD Strix Halo (`gfx1151`) with the pinned IREE ROCm distribution `iree-rocm-3.12.0rc20260721`; every cargo and GPU command ran under `scripts/rocm_gpu_guard.sh --hold` with the standard ROCm environment.

- `cargo fmt --all -- --check`: pass.
- `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings`: pass.
- `cargo test -p mlxcel-xla --lib validation::tests::emit_all_reproduces_registered_goldens`: 1 passed; no golden was edited (`git diff --stat src/lib/mlxcel-xla/assets` is empty).
- `cargo check --features xla-iree,test-utils --test xla_ragged_hardening`: pass.
- `cargo test --release --features xla-iree,test-utils --test xla_ragged_hardening -- --ignored --nocapture`: 5 passed, 0 failed (the new test plus the four #2936 gates). The rejection lines name the caller, axis, index, value, and capacity, for example `xla_llama_prefill: position[axis=0 index=7]=-1 outside [0,8)` and `xla_llama_prefill_slot_logits: position[axis=2 index=0]=-1 outside [0,8)`.
- Pre-merge contract gate in the unit worktree: the project's XLA lib tests and `llama_compat_manifest`, plus the kernel-port-dispatch and native-env-writes audits.

## 5. Limits

- The C guard bounds positions, not token ids. `xla_llama_prefill` has no `vocab` parameter, so it cannot check an embedding index; `xla_llama_prefill_slot_logits` keeps its existing token-range check. A future `xla_llama_prefill` token guard would need the vocabulary, which is a signature decision outside this issue.
- The device test covers the 1D `xla_llama_prefill` and the M-RoPE `xla_llama_prefill_slot_logits` boundaries. The diagnostics and embeddings variants share the same helper and the same axis rule, but their wrappers are not separately exercised on device.
- The test-support forwarders are compiled only with `MLXCEL_XLA_TEST_SUPPORT`; the device suite remains `#[ignore]`d for ordinary CI, following `tests/xla_ragged_hardening.rs`.

## 6. Change Summary

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | +283 |
| Lines deleted | -2 |
| Commits | 1 |
| Device gates | 5 passed |

| File | Change |
|------|--------|
| `src/lib/mlxcel-xla/csrc/xla_iree.c` | Shared position guard, four call sites, NULL checks, two test-support forwarders |
| `src/lib/mlxcel-xla/src/iree.rs` | Test-support externs and wrapper methods |
| `src/lib/mlxcel-xla/src/lib.rs` | Re-export `IreeLlama` under test-support |
| `tests/xla_ragged_hardening.rs` | Real-device boundary test |
