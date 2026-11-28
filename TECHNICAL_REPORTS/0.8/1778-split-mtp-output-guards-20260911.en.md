# Technical Report: PR #1778 - fix(split-mtp): fix refusal text, --force shards, --q-bits validation

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (CLI guards and one shared validator; the drafter written by `split-mtp` is byte-identical to the pre-PR output)

---

## Executive Summary

The PR #1753 report listed three defects in `mlxcel split-mtp` among its follow-ups. The block-size refusal printed two runs of 14 spaces. `--force` left another checkpoint's weight shards next to the drafter. `--q-bits` accepted widths MLX cannot quantize, and the process aborted after the tensor work was done. This PR fixes all three. Review of the first version found five more gaps in the same guards, all fixed here too. The guard now refuses to write beside any safetensors file it did not write. The group size is validated as well as the bit width. Checks that change nothing on disk run before the first check that does.

On the real `glm-4.7-flash-bf16` checkpoint, the drafter this PR writes is byte-identical to the one from the pre-PR binary. The two refusal paths exit 1 without changing a file.

---

## Problem Statement

- **Refusal text.** A wrapped string literal in the `MAX_BLOCK_SIZE` refusal had lost its `\` continuations, and rustfmt then joined the lines, so the indentation became part of the message.
- **`--force` and foreign weight files.** `--force` removed a stale `model.safetensors.index.json` and nothing else. The loader's `glob_safetensors` reads every `*.safetensors` in a directory when there is no index. Shards of a previous checkpoint, left beside the drafter's single `model.safetensors`, would load as part of the drafter.
- **Unsupported widths.** The only check before quantizing was the shared load-time bounds check, `validate_quantization_params`, which accepts `1..=32`. MLX's affine quantize supports 2, 3, 4, 5, 6 and 8, and rejects anything else with `std::invalid_argument`. That call goes through a cxx bridge that does not return `Result`, so the throw was a `std::terminate`: `--q-bits 1` aborted the process after the layer had been read and partly processed.

Review of the first fix found:

- Gemma 4 kept its own copy of the bit set (`SUPPORTED_OVERRIDE_BITS`) instead of the new shared constant.
- `--q-group-size` had no check, so `--q-bits 4 --q-group-size 16` aborted the same way.
- The shard predicate matched only `model-*.safetensors`, so `consolidated.safetensors` bypassed it.
- An unreadable output directory was treated as having no shards.
- `prepare_output_dir` ran before the `--q-bits` check and before the output-equals-source refusal. With `-m X -o X --force` on a single-file source, it deleted the source's own stale index before the refusal ran.

---

## Change Summary

- **Shared validators in `mlxcel-core::layers`.** Adds `SUPPORTED_AFFINE_BITS` (2, 3, 4, 5, 6, 8) and `SUPPORTED_AFFINE_GROUP_SIZES` (32, 64, 128), with `validate_affine_quantization_bits` and `validate_affine_quantization_group_size`. The group sizes are the ones the pinned MLX `mlx/ops.cpp` accepts. `infer_quantization_bits` and Gemma 4's override path read the shared bit set. `validate_quantization_params` is unchanged (#929).
- **Producer-side check in `mlxcel-surgery`.** `split_mtp` validates both parameters before it touches the weight map. `refuse_output_is_source` is public so the CLI can run it early.
- **CLI ordering.** `preflight_checks` (bits, group size, output equals source) changes nothing on disk and runs first. `prepare_output_dir` runs only after it passes.
- **`--force` contract.** `prepare_output_dir` refuses and names the file when the output directory holds any `*.safetensors` other than `model.safetensors`. `find_foreign_safetensors_file` returns an `io::Result`, so a listing error stops the run instead of reading as "clear". A stale index is still removed when nothing foreign remains.
- **Messages.** The block-size refusal is one sentence. Messages from the core validators name the parameter ("quantization bits"), and split-mtp adds the flag name. An earlier doc claim that the six widths are the only ones affine packing can represent was wrong: 7 packs as well, but MLX has no kernel for it. The wording now says only what MLX implements.
- **PR #1753 report.** The three follow-up bullets are marked resolved, with the final scope of the guard.

---

## Technical Decisions

**Refuse, do not delete.** `--force` means "overwrite the files this tool writes", not "empty the directory". A shard in the output directory may be the only copy of a checkpoint that the user pointed at by mistake. Deleting it would be irreversible. Refusing costs one manual removal. The refusal names the file so the user can see what is at stake.

**Screen every foreign safetensors file, not a name pattern.** The loader decides what it reads by extension, so the guard uses the same rule. A name-based list would have to guess every other converter's naming, and `consolidated.safetensors` shows the guess fails.

**Validate at the producer, in a shared module.** The supported sets are MLX's, not split-mtp's, and Gemma 4 already carried a second copy. One definition in `layers.rs` serves both callers. Its tests accept every value in each set and refuse sampled values outside it: 1, 7, 9, 16 and 32 for bits, and 16, 256 and 1 for group size. The producer is the only place a bad width can fail cleanly: once the value reaches MLX's quantize call through a non-`Result` bridge, the only outcome is a process abort.

**Leave the load-time bound alone.** `validate_quantization_params` guards against out-of-range values in configs the loader reads (#929). Narrowing it to MLX's current kernel set would reject checkpoints before MLX is asked. That is a separate decision with its own blast radius, and this PR does not make it.

---

## Validation

- **Gate.** On 059ea43e the workspace gate passed: exit 0, 123 binaries, 11,038 passed, 0 failed. Clippy, fmt, `verify-versions`, `verify-kernel-dtype-keys` and `verify-llama-compat` are clean, and CI is green.
- **Revert arms.** Every new or strengthened test was run with its fix reverted and failed by name. With only the load-time `1..=32` bound left, `--q-bits 1` does not fail with a message. It aborts the test binary: `std::invalid_argument: [quantize] The requested number of bits 1 is not supported`, SIGABRT. The refusal test uses an empty `WeightMap`, which proves the check runs before any tensor work.
- **Real run.** A release binary at 059ea43e against `glm-4.7-flash-bf16` (62.5 GB):
  - `--q-bits 4` writes the drafter: 54 tensors, 0.72 GB, `glm4_moe_lite_mtp`, 4-bit affine, group size 64. `model.safetensors` and `config.json` are byte-identical to the output of the #1772 merge binary.
  - `--force` over a planted `model-00001-of-00003.safetensors` exits 1 and names the file. Every file's name, size, mtime and inode is unchanged.
  - `--q-bits 7` exits 1 in under a second with `quantization bits (7) must be one of 2, 3, 4, 5, 6, 8`, and `drafter2` is never created.

---

## Learning Points

- **Test the rendered message, not the literal.** rustfmt joining a string whose `\` continuation was lost produces valid code and a broken message. The block-size test now asserts there is no run of two spaces.
- **A precondition MLX enforces by throwing is a process abort here.** Through a `UniquePtr`-returning bridge, `std::invalid_argument` becomes `std::terminate`. The only graceful place to reject the value is before the call. The same pattern turned a config value into an abort at first inference in earlier issues.
- **Order checks by side effect.** A guard that mutates must come after every guard that does not. Otherwise a refusal that looks correct in isolation runs after the damage it exists to prevent.
- **Match the guard to the consumer's rule.** The loader reads by extension. A guard that screens by name protects against the names someone thought of.

---

## Follow-ups

- The output writes follow symlinks, and the marker check does not look at tokenizer file names. `exists()` also treats a broken symlink as absent. In a shared directory someone else created, a planted `tokenizer.json` symlink would be truncated even without `--force`. Checking and then writing also leaves minutes in which a second concurrent run passes the same guard. Writing into a temporary directory and renaming it into place closes both. This predates PR #1778.
- The Qwen 3.5 MTP drafter (`drafter/qwen3_5_mtp/model.rs`) passes `bits` and `group_size` from its config to `quantize_weights_with_mode` without the new validators.
- The Markov drafter's supported bit list omits 5.
