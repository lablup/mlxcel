# PR #3491: Apply YaRN rope scaling in the Qwen 3.5 MTP drafter layer (issue #3479)

**Date**: 2026-11-27

**Status**: Merged as `944dbfc2d`. 15 files, +664 / -143. Validated locally on an M3 Ultra; remote CI was cancelled by maintainer decision. No YaRN Qwen 3.5 checkpoint was available, so the acceptance gain was not measured.

**Risk Level**: Low (output stays exact because the target verifies every token; non-YaRN configs are unchanged, checked by identical text and acceptance)

## Executive Summary

The Qwen 3.5 MTP drafter rotated Q and K with plain `rope_theta` even when its `rope_parameters` declared YaRN, while the target used the YaRN divisor table and attention factor. The drafter therefore proposed tokens from a different position encoding than the one the target verifies with, which lowers acceptance on long-context YaRN checkpoints. The drafter now receives the same table and factor, built by the same function as the target's.

## 1. Problem Statement

- **Position encoding mismatch.** Upstream's MTP head reuses the target's YaRN-aware attention layer. mlxcel's `Qwen35MtpAttention` in `mlxcel-core` called `fast_rope(.., rope_base, ..)` unconditionally.
- **No visible failure.** Speculative decoding is exact by construction, so greedy text was correct. The cost was only a lower acceptance rate, which no test checked.
- **Layering.** The YaRN table builder (`Qwen35Config::yarn`) lived in the binary crate, and the drafter is loaded by `mlxcel-core`, which cannot depend on the binary crate.

## 2. Technical Decisions

### 2.1 The binary crate installs the table after core loads the drafter

| Option | Pros | Cons |
|---|---|---|
| Move the YaRN builder into `mlxcel-core` | drafter self-contained | moves `rope_utils` and the `--yarn-*` override plumbing across crates |
| Duplicate the builder in core | small diff | two copies of the YaRN math that can drift |
| **Chosen: core exposes `set_yarn_rope`, the binary's `drafter_loader` installs the table** | one builder; the CLI, server drafter slot and `speculative_bench` already go through `drafter_loader::load_drafter` | a direct call to core `load_drafter` gets no YaRN |

`Qwen35Config::yarn`'s body became the free function `qwen35_yarn(rope_parameters, rope_dims, rope_theta, max_position_embeddings)`. The target calls it from its config, and `drafter_loader::qwen35_mtp_yarn` calls it with the drafter's own geometry, so `--yarn-*` overrides reach both. A block that names `yarn` but cannot build a table fails the drafter load with the target's error message instead of falling back silently.

### 2.2 One lane-scaling function for target and drafter

transformers' YaRN multiplies cos and sin by the attention factor, so only the rotated lanes of a partial-rotary head are scaled. `fast_rope_with_freqs` has no such multiplier, so the caller scales the first `rope_dims` lanes before rotating. That code lived privately in `Qwen3NextAttention`; it is now `mlxcel_core::rope_lanes::scale_rotary_lanes`, and both the target and the drafter call it. A factor of exactly 1.0 returns a copy without arithmetic.

```rust
fn rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
    match &self.rope_freqs {
        Some(freqs) => ffi::fast_rope_with_freqs(
            &scale_rotary_lanes(x, self.rope_dims, self.rope_attention_factor),
            self.rope_dims, false, 1.0, offset, freqs),
        None => ffi::fast_rope(x, self.rope_dims, false, self.rope_base, 1.0, offset),
    }
}
```

### 2.3 Drafter config reads `max_position_embeddings`

YaRN's table depends on the original context length. The drafter text config did not parse it, so it now does, and the tests check that the drafter's table equals the target's bit for bit on the SenseNova YaRN config.

## 3. Change Summary

| Area | Change |
|---|---|
| `mlxcel-core/src/rope_lanes.rs` (new) | `scale_rotary_lanes` |
| `mlxcel-core/src/drafter/qwen3_5_mtp/{layer,model,config}.rs` | optional YaRN table and factor, `set_yarn_rope`, `yarn_rope`, `config()`, `max_position_embeddings` |
| `mlxcel-core/src/drafter/qwen3_5_mtp/yarn_tests.rs` (new) | rotation with and without a table, lane scaling |
| `src/models/qwen3_5.rs` | `qwen35_yarn` free function, `Qwen35Config::yarn` delegates |
| `src/models/drafter_loader.rs` | `load_qwen35_mtp`, `qwen35_mtp_yarn` |
| `src/models/qwen3_next.rs` | uses `scale_rotary_lanes` instead of its own copy |
| `src/models/qwen3_5_mtp_yarn_tests.rs` (new) | drafter table equals target table, loader installs it, bad YaRN block fails the load |

| Commit | Message |
|---|---|
| `de04e847f` | fix: apply YaRN rope scaling in the Qwen3.5 MTP drafter layer |

## 4. Validation

| Check | Result |
|---|---|
| `cargo test -p mlxcel-core ... drafter::qwen3_5_mtp rope_lanes` | 26 passed, existing cases unchanged |
| `cargo test -p mlxcel ...` `qwen3_5` / `qwen3_next` / `rope_utils` / `drafter_loader` | 62 / 23 / 36 / 3 passed |
| Greedy MTP generation, `qwen3.8-27b-4bit` with `qwen3.8-27b-mtp-4bit` (no YaRN), 64 tokens | identical text and acceptance before and after (28 rounds, 35 / 56 accepted, 0.625) |
| clippy `-D warnings` on both crates, fmt | clean |

## 5. Follow-up

- Measure acceptance on a YaRN Qwen 3.5 checkpoint with long prompts once one is available; the expected gain is unverified.
- A caller that uses `mlxcel_core::drafter::load_drafter` directly gets a drafter without YaRN. All current callers go through the binary's wrapper; a new entry point must too.
