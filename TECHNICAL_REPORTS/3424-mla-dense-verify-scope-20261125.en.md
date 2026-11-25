# PR #3424: Exact token-only verify for the remaining MLA and dense families

**Date**: 2026-11-25
**Status**: Draft PR open (closes #3421). Validated on an M3 Ultra (512 GB) with `deepseek-v2-lite-chat-4bit-mlx`, `afm-4.5b`, `k2-horizon-0.9b`, `olmoe-1b-7b-0125-4bit`, `kimi-linear-48b-a3b-instruct-4bit` and both `longcat-flash-lite` checkpoints; narrow lib suites, clippy and the contract checks pass

## Executive Summary

PR #3420 made a token-only verify block (prompt lookup, a full-model drafter) byte-identical to single-token decode on the DeepSeek V3 family, Xing4.0 and the GLM-4.7-Flash engine path, behind `MLXCEL_MLA_ROW_INVARIANT=1`. This PR brings DeepSeek V3.2, DeepSeek V2, `Llama3Model` (K2 Horizon), Arcee and OLMoE under the same opt-in, and records why Kimi Linear, Kimi K3 and LongCat stay out: the token-only round refuses them, so no verify block forms. The default path is unchanged.

## 1. Which families form a block

`prompt_lookup_unsupported_reason` refuses Kimi Linear and Kimi K3 (model-owned KDA and MLA state, no verify rewind) and both LongCat variants (one model-owned slot and an n-gram tail no trim reaches). The survey reports "the token-only verify round does not take this model" on every cell of the three local checkpoints. The issue asked whether Kimi Linear's recurrent KDA layers already match the chain: they never see a token-only block, so the question does not arise, and their block behavior was not measured. One test per family pins the refusal. DeepSeek V2 takes the round and has the same single-query/block split in its latent cache, so it is in.

## 2. Changes

| Family | What a verify block does inside the scope |
|---|---|
| DeepSeek V3.2 | Each row runs `single_query_attention` (the decode step's DSA attention, extracted from `forward`) over its prefix of the stored `[latent, indexer key]` window, with its own indexer query and top-k. Indexer, attention, MLP, gate and LM head products are row-invariant; float experts run per position |
| DeepSeek V2 | Each row appends and attends alone, on the latent cache (`MlaLatentCache::attend`) or on per-head K/V |
| `Llama3Model`, Arcee | `FusedQKVLinear`'s split goes through `row_invariant::linear_whole_block`, the layer's own forward outside a scope. Inside a scope the fused rope launchers and the compiled SwiGLU MLP give way to the composed graph in every step, and a block attends one row at a time through `KVCache::attend` |
| OLMoE | Routing and experts run per position, so each takes the fused single-token MoE kernel; attention per row |

The per-row MLA blocks count one path-stat entry per layer through `mla::stats::as_one_call`; before, a width-K block recorded K single-query calls.

## 3. Results

Engine-path survey, widths 2, 4, 8 at 8, 600 and 2100 tokens (9 cells):

| Checkpoint | Off (both kernels) | On, wide | On, narrow |
|---|---|---|---|
| deepseek-v2-lite-chat-4bit-mlx | differs, layer 0 output | differs | equal |
| afm-4.5b | differs, layer 0 output (layer 1 in one cell) | equal | not needed |
| k2-horizon-0.9b | differs, layer 0 output | equal | not needed |
| olmoe-1b-7b-0125-4bit | differs, layer 0 output | differs | equal |

The remaining wide-kernel difference on quantized checkpoints is MLX's `qmv_wide`, as on the DeepSeek V3 family.

Decode ABAB against `origin/main`, variable unset, 4 cycles, identical tokens: K2 Horizon 211.50 against 211.80 tok/s, OLMoE 410.21 (421.24 without a cold first run) against 421.21, V2-Lite 173.28 against 174.58.

## 4. Tests

- `deepseek_v32::verify_tests`, `deepseek_v2::verify_tests`: tiny models (dense and float MoE layers; V3.2 at top-k 4 and 64, V2 on both cache layouts), widths 2, 4, 8, 12, with and without the matvec kernel; a test that the block differs without the scope; a path-count test.
- `dense_verify_block_tests`: Llama 3, Arcee and OLMoE, float and 4-bit (the 4-bit half under `qmv_wide` off, restored on drop).
- `deepseek_v3::verify_block_tests`: the path count is now one per block and layer, and decode counts are checked too.
- `the_token_only_verify_round_refuses_this_model` for Kimi Linear and both LongCat variants; `as_one_call_records_the_first_path_once` in mlxcel-core.

## 5. Not done

- `deepseek-v3.2-4bit` (352 GB) was not loaded; V3.2 is covered by the tiny model only.
- The diagnostic captures layer outputs only, so the first difference names a layer, not the op inside it. All candidate ops were replaced together.
- Prompt-lookup throughput with the variable on was not measured for these families.
