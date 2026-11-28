# PR #3423: prompt lookup verify rounds on first-generation Inkling

**Date**: 2026-11-25
**Status**: Draft PR open (closes #3422). Validated on an M3 Ultra (512 GB) with `inkling-0.6b-a0.6b` and `inkling-small-mlx-4bit`. The narrow lib suites, clippy and the contract checks pass.

## Executive Summary

`--prompt-lookup` and the server's token-only verify rounds refused first-generation Inkling. Its caches are model-owned, it opts out of padded prefill, and it did not claim it could rewind a verify block. PR #3419 had already built an exact rewind for MTP: inside the verify scope, each short convolution records its whole input window, and `keep_verified_prefix` rebuilds the state of any accepted prefix from that window. The token-only round never entered that scope, so it recorded nothing, and the model had no `rewind_decode_appends`.

This PR connects the two. The text from a temperature-0 prompt-lookup run equals classic decode on both checkpoints, and the per-model survey's engine path is equal at every width and prompt length it tried. A copy-heavy reply runs 1.14x classic on Inkling-Small and 2.12x on the 0.6B checkpoint.

## 1. Mechanism

- **Scope.** `forward_with_sequence_id` enters `verify_block_scope()` when the row kernel is on and `row_invariant::is_verify_block(shape)` holds. That check means a speculative append (the engine's `DecodeLookaheadAppendScope`) of one sequence with 2 to 16 rows. Inside the scope the dense products, convolutions, experts and attention take the row-invariant forms from #3409 and #3280, so every row equals a decode step. Each convolution also saves its `[1, kernel - 1 + L, C]` window. An ordinary prefill does not run as a speculative append, so it is unchanged.
- **Gate.** `supports_speculative_verify_rewind` returns `row_kernel_enabled()`. `prompt_lookup_unsupported_reason` and `Engine::verify_rounds_unsupported` then admit the family through their existing `model_rewinds` branch. The shared gate in `prompt_lookup.rs` is untouched. With `MLXCEL_INKLING_ROW_INVARIANT=0` a block is neither exact nor recorded, so the gate still refuses.
- **Rewind.** `rewind_decode_appends(seq, n)` calls the new `inkling_layer::drop_newest_verify_rows`. The cache does not know the kernel size, but the state the forward left behind holds `kernel - 1` rows, so the block length is the window's rows minus the state's rows. The function first checks that every layer holds windows and that all layers agree on the length, and that `n` does not exceed it. Only then does it call `keep_verified_prefix(appended, appended - n)` on each layer. `keep_verified_prefix` now accepts `kept == 0`: the KV tail is trimmed by the whole block, and each convolution state becomes the window's first `kernel - 1` rows, which is the state from before the block. The engine needs this case because `failed_round` and `warm_up_verify_widths` unwind every append. MTP always keeps at least one row, so its behavior is unchanged.
- **Wide blocks.** `--prompt-lookup-max-draft` allows up to 64 proposals, which makes a block of up to 65 rows. A speculative append wider than 16 rows now records its window too, even though it runs outside the row-invariant scope. A rejected tail of such a block therefore rewinds correctly, but the block itself is not byte-identical to the chain. Without that recording, a block that wide would fail its rewind and end the request.
- **Wiring.** `InklingVlModel` (the public checkpoints' wrapper) delegates both hooks. The CLI and the server scheduler both go through `Engine::verify_round`, so neither needed code of its own. `supports_padded_prefill` stays `false` and `trim_state` keeps clearing the convolution state. Its comment now says it serves padded prefill only, and line 66 of the #3419 reports no longer calls it the prompt-lookup rewind.

## 2. Exactness

The engine path of `survey_block_vs_chain` returned `NotRun` for every model-owned layout. It now also runs on `LoadedModel::Inkling` and `InklingVLM`, whose `make_caches` resets the default state that both arms use. Inkling 2 is still excluded, so its verdicts do not change.

| Checkpoint | Widths | Prompts | Engine verdict |
|---|---|---|---|
| inkling-0.6b-a0.6b | 2, 4, 8 | 8, 600, 2100 | equal (9 of 9) |
| inkling-small-mlx-4bit | 2, 4, 8, 16 | 8, 600, 2100 | equal (12 of 12), wide quantized kernel |

## 3. Throughput

`mlxcel generate --temp 0`, ABAB with 4 cycles under the GPU lock, while other worktrees were building. The tok/s figures include prefill. Every pair produced identical text.

| Checkpoint, prompt | Classic | Prompt lookup | Ratio | Tokens per forward |
|---|---|---|---|---|
| Small, copy (550 tokens) | 42.12 | 48.00 | 1.14x | 2.26 |
| Small, copy, cut at 400 tokens (reasoning only) | 42.29 | 44.00 | 1.04x | 1.89 |
| 0.6B, repeated passage (50 tokens) | 235.52 | 500.41 | 2.12x | 6.13 |
| 0.6B, reply nothing to draft from (52 tokens) | 234.24 | 190.85 | 0.81x | 0.98 |

On Small the ratio is well below the tokens per forward, because an exact verify block runs its experts and attention per position. Its cost per block was not measured separately.

## 4. Tests

- `tiny_tests::prompt_lookup_gate_follows_the_row_kernel`: the gate returns `None` with the row kernel on and `Some` with it off. It fails on the previous code, where the default `supports_speculative_verify_rewind` is `false`.
- `prompt_lookup_tests::a_rewound_verify_round_matches_the_single_token_chain_on_metal`: widths 4 and 10, every `kept` from 0 to `width - 1`. Each case runs a verify block as a speculative append, rewinds through `rewind_decode_appends`, and takes two decode steps. The logits must equal, byte for byte, those of a model that decoded the kept tokens one at a time.
- `a_rewind_without_a_recorded_block_is_refused_without_mutation`: a rewind is refused after a plain step, for `n` past the block, on a second rewind and for an unknown sequence. The state still matches the chain afterwards.
- `a_verify_block_past_the_row_invariant_scope_still_rewinds`: a 20-row block, 15 rows dropped, checked against the chain within 2e-4.

## 5. Not done

- The server path was not exercised end to end. It shares the CLI's gate and round code.
- On the 0.6B checkpoint, the instruction-style copy prompt yields no drafts, because the model does not continue code. Its throughput row uses a prompt whose reply repeats the prompt.
