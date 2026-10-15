# PR #2328: Validate the quantization scheme in the Qwen 3.5 MTP drafter loader

**Date**: 2026-10-16
**Status**: Implemented; unit revert check and real-checkpoint load verified on CUDA (GB10) before merge; end-to-end MTP speculative generation and the 5-bit Markov path on a real drafter not verified (see "Not verified and follow-up")
**Risk**: Low (a validation step on a load path that previously aborted, and one bit-depth list replaced by the shared constant)

## Summary

Issue #1790 fixes a process abort when a Qwen 3.5 MTP drafter checkpoint declares an unsupported quantization scheme.

1. `Qwen35MtpDraftModel::quantize_dense_projections` (`src/lib/mlxcel-core/src/drafter/qwen3_5_mtp/model.rs`) now returns `Result`. Once its candidate list is non-empty it calls `validate_affine_quantization_bits` and `validate_affine_quantization_group_size`, and `from_path` maps the error to `DrafterError::Config`.
2. The DFlash Markov head (`drafter/dflash/markov.rs`) takes its bit-depth list from `SUPPORTED_AFFINE_BITS`, so a 5-bit factor loads.
3. The `Used by:` lists in `layers.rs` name the new callers.

## Problem

`quantize_dense_projections` computed with `group_size` and `bits` from the checkpoint config without checking them. `group_size: 0` panics on the `%`. An unsupported `bits` or `group_size` makes MLX's affine quantize kernel throw `std::invalid_argument`, which crosses the cxx bridge as a process abort instead of a load error. The Markov head kept its own `[2, 3, 4, 6, 8]` list, which omitted 5 bits although the kernel supports it, so a valid 5-bit factor was refused.

## Technical decisions

- Validate only when the candidate list is non-empty. A pre-converted checkpoint that the function skips is not refused over values it never uses, and `validate_bounds` is unchanged.
- Reuse the two shared affine validators rather than adding a drafter-specific check, so the accepted schemes match `split-mtp` and the other loaders. The error is prefixed "Qwen 3.5 MTP drafter config".
- Replace the Markov head's private list with `SUPPORTED_AFFINE_BITS` instead of adding 5 to it, so the two lists cannot drift again. An unexplainable packed width is still refused.
- Make `quantize_dense_projections` `pub(super)` so the tests call it directly with a config that carries an explicit `quantization` block.

## Changes

- `src/lib/mlxcel-core/src/drafter/qwen3_5_mtp/model.rs`: `Result` return, the two validator calls, and the `from_path` mapping.
- `src/lib/mlxcel-core/src/drafter/dflash/markov.rs`: `SUPPORTED_AFFINE_BITS` replaces the local constant.
- `src/lib/mlxcel-core/src/layers.rs`: doc comments only.
- `src/lib/mlxcel-core/src/drafter/qwen3_5_mtp/tests.rs`: `quantize_dense_projections_rejects_unsupported_quantization` (bits 7 and 1, group_size 0, 48 and -64), `quantize_dense_projections_only_validates_when_it_would_quantize`, and an ignored real-checkpoint test `real_drafter_loads_and_refuses_an_unsupported_scheme`.
- `src/lib/mlxcel-core/src/drafter/dflash/markov_tests.rs`: `a_five_bit_markov_factor_loads`, which also checks a 7-bit-width factor is still refused.

## Behavior change

An MTP drafter config with an unsupported `bits` or `group_size` (including 0) now fails the load with `DrafterError::Config` and a message that lists the supported values. A 5-bit DFlash Markov factor now loads. Valid checkpoints load as before.

## Verification

- Revert check: with the two fixes reverted, `quantize_dense_projections_rejects_unsupported_quantization` and `a_five_bit_markov_factor_loads` fail; restored, they pass.
- Real drafter `mlx-community/Qwen3.8-27B-MTP-bf16` (and target `mlx-community/Qwen3.8-27B-4bit`) downloaded: the shipped drafter loads unchanged. Scratch copies with `bits = 7` and with `group_size = 0` return `DrafterError::Config` ("quantization bits (7) must be one of 2, 3, 4, 5, 6, 8", "quantization group_size (0) must be one of 32, 64, 128") instead of aborting.
- DFlash end to end through `mlxcel-server` with `qwen3.5-4b-4bit` and `qwen3.5-4b-dflash` (`--draft-kind dflash`, temperature 0): the drafter lazy-loaded and the request completed with 48 tokens over 15 DFlash rounds. No throughput numbers were taken.
- clippy with `-D warnings`, fmt and the contract scripts pass.

## Not verified and follow-up

- End-to-end MTP speculative generation was not run: `mlxcel generate --draft-kind mtp` is Metal-only by an existing gate, so on this CUDA box the drafter was exercised only through the load tests.
- The 5-bit Markov path is covered only by the unit test, because the local DFlash drafter has no Markov head.
