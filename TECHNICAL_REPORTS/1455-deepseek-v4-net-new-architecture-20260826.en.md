# Technical Report: PR #1455 - DeepSeek-V4 as a net-new architecture

**Date**: 2026-08-26
**Author**: mlxcel maintainers
**Status**: Completed; all three real-checkpoint gates pass on `mlx-community/DeepSeek-V4-Flash-4bit`, and the generated text is byte-identical before and after the review hardening
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1455 implements issue #523: DeepSeek-V4 support, ported from the in-tree reference at `references/mlx-vlm/mlx_vlm/models/deepseek_v4/`. It adds ten files under `src/models/deepseek_v4*.rs`, a real-checkpoint integration test, and the usual registration surface.

The report worth writing here is not "a model was added". It is why the *first* attempt at this issue could not work, and what that says about how a family should be classified before any code is written. PR #592 implemented `DeepSeekV4Model` as a thin wrapper over `DeepSeekV3Model` plus two deltas, on the premise that V4 is a V3 variant. It was closed unmerged. V4 shares almost nothing structural with V3: it replaces the residual stream, replaces MLA, replaces the routing rule, and changes the rank of the hidden state carried between blocks. A wrapper cannot load a real V4 checkpoint, let alone run one.

## 1. Problem Statement

### 1.1 Background

mlxcel carried `deepseek`, `deepseek_v2`, `deepseek_v3` and `deepseek_v32` as four separate implementations sharing only the generic core layers. The natural inference from that sequence is that `deepseek_v4` is a fifth member of the same family, differing by config values and a feature or two.

The inference is wrong, and the config alone does not make that obvious. `deepseek_v4`'s `config.json` carries familiar keys (`q_lora_rank`, `qk_rope_head_dim`, `n_routed_experts`, `routed_scaling_factor`, `norm_topk_prob`, `topk_method: "noaux_tc"`) alongside unfamiliar ones (`hc_mult`, `hc_sinkhorn_iters`, `compress_ratios`, `num_hash_layers`, `o_groups`, `index_topk`). Reading it as "V3 plus some extras" is a defensible first pass. Reading the reference implementation is what refutes it.

### 1.2 Why the V3-wrapper premise fails

Five components distinguish V4, and each one is load-bearing rather than optional.

**HyperConnections replace plain residual connections.** The state carried between blocks is rank-4 `[B, L, hc_mult, D]` with `hc_mult = 4`, not `[B, L, D]`. The embedding output is broadcast to four copies at entry and a `HyperHead` collapses it back before the final norm. Each sublayer runs a learned gate that produces a `pre` vector, a `post` vector and an `hc_mult x hc_mult` mixing matrix, and the mixing matrix is softmaxed and then Sinkhorn-normalised for 20 alternating rounds. There is no place in a V3 block to put this: it changes the shape of the thing flowing through the stack.

**A pooled-KV `Compressor` replaces MLA.** V3 compresses KV through a low-rank latent per token. V4 pools whole windows of tokens into single compressed rows, with the window size selected per layer by `compress_ratios`, and keeps a separate small local KV window alongside. These are different mechanisms, not different parameterisations of one mechanism.

**HiSA sparse selection.** Sparse layers own an `Indexer` with its own compressor, which scores pooled rows and returns a top-`index_topk` selection feeding a split softmax that shares one log-normalizer with the local KV.

**Hash-routed MoE with `sqrtsoftplus` gating.** The first `num_hash_layers` layers do not route by scoring at all: expert indices come from a `tid2eid` lookup table indexed by the raw token ids. V3's group-limited softmax routing is absent.

**A `MultiLinear` grouped output projection**, with an inverse RoPE applied to the attention output before it.

### 1.3 Risk assessment

The failure mode of this port is not a crash. Every one of the five components, done the plausible-but-wrong way, produces finite, correctly-shaped, fluent output:

- Sinkhorn with the wrong iteration count or with the row and column normalisations transposed still returns a valid doubly-stochastic-ish matrix.
- The overlap compressor (ratio 4) and the simple compressor (ratio 128) have identical output shapes.
- Gathering routing weights from the bias-corrected scores instead of the unbiased ones misweights every routed contribution while leaving the model coherent. This is the same contract `bailing_moe`, `afmoe` and `klear` document.
- An indexer that selects the wrong pooled rows degrades long-context recall without affecting short prompts at all.

That is why this port is gated on component-level parity fixtures rather than only on end-to-end coherence, and why the real-checkpoint gate that matters most is the long-context one.

## 2. Technical Decisions

### 2.1 Port the pure-ops Sinkhorn path, not the fused Metal kernel

The reference ships two implementations of the HyperConnection gate: a fused `mx.fast.metal_kernel` (`_hc_sinkhorn_collapse_kernel`) and a pure-ops path (`_hc_ops` / `_hc_split_sinkhorn_ops`). The reference itself uses the ops path when training and when Metal is unavailable, so the ops path is the correctness baseline and the kernel is an optimisation of it.

This port implements the ops path only. Writing a Metal kernel for a first-time architecture port would mean debugging a novel algorithm and a novel kernel simultaneously, with no independent reference for either. The kernel is recorded as a follow-up.

### 2.2 Model-owned heterogeneous per-layer cache state

`LanguageModel::forward` takes `caches: &mut [KVCache]`, one homogeneous entry per layer. V4 needs a rotating window KV cache plus one or two pooling caches per layer, with the count varying by the layer's `compress_ratio`. `PoolingCache` has no Rust equivalent and its semantics (a remainder buffer, distinct prompt and decode branches, emission only on full windows) do not fit `KVCache`.

Rather than widen the trait, the port uses `ModelOwnedSequenceState<T>` from `src/models/model_owned.rs`, the same escape hatch `mamba2`, `gemma3`, `afmoe` and `qwen3_next` use for non-KV or heterogeneous state, and returns placeholder `KVCache::new()` entries for trait compatibility. This keeps a model-specific cache shape out of the shared trait, at the cost of the model not participating in the paged and quantized KV paths.

### 2.3 One attention struct across three kinds, not three structs

The reference declares `LocalAttention`, `CompressedAttention` and `SparseCompressedAttention` as three classes with a factory. They share their entire projection set and differ only in whether a compressor exists, whether an indexer exists, and which branch the forward pass takes. The port uses one struct with a kind discriminant, which keeps the shared weight loading in one place and makes the three forward branches visibly adjacent.

### 2.4 Read the per-path quantization override table rather than trusting the top level

The real checkpoint declares `{group_size: 64, bits: 4, mode: "affine"}` at the top level and then 641 per-module-path overrides. 129 of those, all the routed expert projections, are `{group_size: 32, bits: 4, mode: "mxfp4"}`. The other 512 restate the top-level pair.

A loader that read only the top level would apply affine/64 to mxfp4/32 expert tensors. The port reads the override table. It also validates every declared pair, because a mode string MLX cannot parse becomes an uncatchable `std::terminate` at the first forward rather than a load error.

### 2.5 Truncate `compress_ratios` to `num_hidden_layers` rather than requiring equality

The real checkpoint ships **44** `compress_ratios` entries for **43** layers. The reference's `__post_init__` truncates to the first `num_hidden_layers`, which drops a trailing `0` and makes the final layer ratio 4, a sparse layer.

Two plausible readings both fail here. Requiring `len == num_hidden_layers` rejects the real checkpoint at load. Taking the *last* 43 entries shifts the whole schedule by one and builds the wrong attention kind in every layer, which loads cleanly and produces fluent nonsense. The port follows the reference and pins the behaviour with an assertion on index 42.

### 2.6 Strict weight coverage as a load-time gate

`from_weights` fails the load naming any checkpoint tensor that maps onto no module path, and any module parameter that finds no tensor. For a first-time port of an architecture with two tensor-name planes, a silent fallback on an unmatched key is the difference between "this checkpoint is unsupported" and "this checkpoint runs with a randomly-initialised submodule".

## 3. Implementation Details

| File | Contents |
| --- | --- |
| `deepseek_v4.rs` | Config parsing and validation, block and model assembly, model-owned sequence state, `LanguageModel` impl |
| `deepseek_v4_hyper.rs` | `HyperConnection`, `HyperHead`, `hc_expand`, float32 Sinkhorn ops path |
| `deepseek_v4_rope.rs` | Inf-padded proportional RoPE, Yarn correction, inverse tables, `freq_scale` division |
| `deepseek_v4_compress.rs` | `PoolingCache`, simple and overlap compressors, pooled-visibility mask helpers |
| `deepseek_v4_indexer.rs` | HiSA indexer: decode fast path, batched tiled path, flat fallback |
| `deepseek_v4_attention.rs` | The three attention kinds, sinks, inverse output RoPE, `MultiLinear` output path, split-softmax sparse attention |
| `deepseek_v4_moe.rs` | Hash and biased routing, `sqrtsoftplus`, limited SwiGLU, per-path expert quantization |
| `deepseek_v4_sanitize.rs` | Both tensor-name planes, `mtp.*` drops, strict coverage check |
| `deepseek_v4_tests.rs` | 39 unit tests |
| `tests/deepseek_v4_real_model.rs` | Real-config preflight plus three ignore-gated real-checkpoint gates |

### 3.1 The Sinkhorn order

The ordering is not symmetric and is easy to get wrong. The reference is: softmax over the last axis, add `hc_eps`, then an **initial column normalisation**, then `hc_sinkhorn_iters - 1` rounds of (row normalisation, column normalisation). The port matches this exactly, and a float64 fixture replays the reference's own arithmetic to confirm it rather than asserting a property that many orderings would satisfy.

### 3.2 The overlap compressor

At ratio 4 the compressor runs in overlap mode: `out_dim = 2 * head_dim`, the kv and gate tensors are split on the feature axis, the first half is shifted one window back with a zero prefix (and a `-inf` gate prefix), and the halves are concatenated on the window axis before the softmax. At ratio 128 the simple path applies. The two produce identically shaped output, so a test that only checks shapes would not distinguish them; both are pinned against float64 replays of the reference.

One consequence is worth recording because it looks like a bug: the shift applies within each ready batch, so a window completed at a decode step or a chunk boundary sees the zero prefix rather than its predecessor's values. That is reference behaviour, not a porting artifact, and it makes ratio-4 pooled rows batching-sensitive by design.

### 3.3 Threading `input_ids` through every block

Hash routing needs the raw token ids at every MoE layer, because the expert indices come from `tid2eid[input_ids]`. This is a signature change through the whole block loop rather than a local detail, and it is the reason the MoE forward takes an extra argument that no other family in the tree needs.

## 4. Test Strategy

The unit suite is built around the observation in section 1.3: the failure mode is plausible output, not a crash. So the tests replay the reference's arithmetic in float64 over small fixtures and compare numerically, rather than asserting shape or finiteness.

- Reference-math fixtures: Sinkhorn gates, `hc_expand`, Yarn tables, RoPE table composition and forward/inverse identity, both compress functions, limited SwiGLU, `sqrt(softplus(x))`.
- Contract tests: biased selection with unbiased weighting, `tid2eid` routing taking indices from the table, HiSA hierarchical selection agreeing with the flat fallback on tie-free inputs.
- Rejection tests with positive controls: hostile config values, quantization overrides, load-time shape checks, incomplete legacy expert planes, strict coverage.
- Integration-shaped: a tiny three-kind end-to-end model, window-aligned chunked-prefill parity, cache-barrier neutrality.

## 5. Real-Checkpoint Results

Checkpoint: `mlx-community/DeepSeek-V4-Flash-4bit`, 43 layers, 2481 tensors, roughly 151 GB on disk. Host: M3 Ultra, 512 GB.

```
cargo test --test deepseek_v4_real_model --release --features metal,accelerate -- --ignored --nocapture

[deepseek-v4] long-context prompt: 2234 tokens
[deepseek-v4] long-context answer: " The Pacific Ocean is the largest ocean on Earth. The Pacific"
test deepseek_v4_real_model_long_context_hits_sparse_and_compressed_paths ... ok
[deepseek-v4] prompt: "The capital of France is"
[deepseek-v4] greedy continuation (24 tokens): " Paris. The capital of France is Paris. ..."
test deepseek_v4_real_model_loads_and_generates_coherently ... ok
test deepseek_v4_real_model_decode_crosses_pooling_windows ... ok

test result: ok. 3 passed; 0 failed; finished in 80.37s
```

The long-context gate is the one that carries weight. At 2234 tokens a ratio-4 layer's pooled count passes `index_topk` (512), so the sparse layer takes the HiSA selection and split-softmax branch rather than the dense concat fallback, and it still has to retrieve a fact from early in the context. Short-prompt coherence says nothing about the indexer, because short prompts never reach it.

The repetition in the short greedy continuation is ordinary temperature-0 decoding on a base checkpoint. It is not evidence of a porting defect and should not be reported as clean output either.

### 5.1 A verification note on the checkpoint itself

The local checkpoint arrived as a partial download, and completing it produced two false-completion signals worth recording, because both would have produced a "successful" run on corrupt weights:

1. A **file-count check passed while four shards were truncated.** `curl -C -` combined with `--retry` does not reliably resume: on an internal retry it re-opened the output file and truncated it, so shards that had reached several GB silently reset toward zero while remaining present on disk.
2. A **byte-count check nearly passed on corrupted data.** A fetcher that used `curl -w '%{http_code}'` while appending the response body to the file wrote the three literal bytes `206` into the middle of each shard at every retry boundary, because `--write-out` goes to stdout. This was caught only because one shard landed exactly three bytes over its declared size.

The reliable check is both of: every file matching the upstream declared byte count, and every shard self-consistent against its own safetensors header (`filesize == 8 + header_len + max(data_offsets)`). Anyone re-running these gates on a freshly fetched checkpoint should verify both before trusting a failure as a porting bug.

## 6. Validation Summary

| Gate | Result |
| --- | --- |
| `cargo test --lib models::deepseek_v4` | 39/39 |
| `cargo test --lib models::deepseek` (whole family) | 68/68, no regression |
| `cargo test --lib models::detection` | 43/43 including the new arm |
| `cargo test --lib loaded_model` / `models::switch_layers` | 5/5, 23/23 |
| `cargo clippy --lib --tests -- -D warnings` | exit 0 |
| `cargo fmt --check` | clean |
| Weight coverage against the real index | 2481 tensors, 0 missing, 0 unknown |
| Quantized plane consistency | 641/641 satisfy `packed_in * 32 == bits * num_groups * group_size` |
| Real-checkpoint gates | 3/3 |

The real-checkpoint gates were run twice: once before the review hardening in section 7, and once after. The generated text was byte-identical across both runs. That matters more than either run alone, because `PoolingCache::update_and_fetch` changing from a deep copy to a borrow is an aliasing change in the decode path, and aliasing bugs pass unit tests and corrupt long decodes.

## 7. Hardening Applied During Review

**Out-of-bounds read from checkpoint data (HIGH).** `tid2eid` had its shape validated but never its contents. V4 is the only routing path in the tree whose expert indices come from a checkpoint tensor rather than from an `argpartition` over a score row, so the "indices are in range by construction" argument covering `bailing_moe`, `afmoe` and `klear` does not reach it. Those indices feed `take_along_axis` and `gather_qmm`, and MLX gathers fold negative indices but never clamp. Now bounded at load.

**Pooled cache deep copy (HIGH, performance).** `PoolingCache::update_and_fetch` materialised the entire pooled buffer on every call, including the no-op branch that dominates decode (3 of every 4 steps at ratio 4, 127 of 128 at ratio 128). Roughly 54 MB per decoded token at 8k context. Now returns a borrow, matching the reference.

**Unevaluated graph accumulation (MEDIUM, performance).** The indexer's pooling cache accumulated one `slice_update` node per decode step, never forced, each pinning that step's hidden state, because its only consumer is discarded on two of the three sparse branches. `eval_state` after the layer loop mirrors `KVCache::eval_state`.

**Unguarded checkpoint data reaching MLX (MEDIUM).** A missing shape check on `e_score_correction_bias`, an unguarded 2-D `wo_a` reshape, architecture scalars bounded below but not above, and `index_block` / `index_keep` unvalidated while being multiplied in `i32`.

## 8. What Remains Unverified

- **No token-exact comparison against the reference.** The reference is a Python/MLX implementation in-tree, but no harness exists to run it against this port on identical inputs, and the checkpoint is too large to make an ad-hoc comparison cheap. Validation is component-level parity plus end-to-end behaviour, not output equality.
- **Batch sizes above 1** are untested on the real checkpoint. The ratio-4 overlap compressor is batching-sensitive by reference design (section 3.2), so this is the most likely place for a batch-dependent discrepancy.
- **Context beyond ~2200 tokens.** The sparse path is exercised, but not at the 1M `max_position_embeddings` the config declares, and not across a `RotatingKVCache` wrap at long context.
- **Quantization modes other than the shipped mixed mxfp4/affine layout.** Only one real checkpoint exists.
- **Speculative decode, prefix-cache reuse, and the server batch path** were not exercised against this family.

## 9. Follow-up Actions

- The reference's fused HC Sinkhorn+collapse Metal kernel, as an optimisation over the ops path now in place.
- MTP drafting; `mtp.*` tensors are currently dropped at load and `num_nextn_predict_layers` is ignored.
- Tensor-parallel and pipeline sharding (the reference's `shard()`).
- A `forward_last_logits` override. Without one, a prefill chunk materialises full `[1, 2048, 129280]` logits to sample a single row, roughly 530 MB in bf16.
- Revisit the f32 promotion in `sparse_pooled_attention`, which costs roughly 2x on the model's largest intermediates where the reference stays in the activation dtype. Deferred because changing it moves numerics.
- Issues #549 (HiSA) and #550 (shared-expert `swiglu_limit` clamp) were filed under the rejected V3-wrapper scope and are subsumed by this port. They remain open for a human to close.

## References

- Issue #523, and the closed PR #592 whose premise it corrects
- Reference implementation: `references/mlx-vlm/mlx_vlm/models/deepseek_v4/`
- `PoolingCache` reference: `references/mlx-vlm/mlx_vlm/models/cache.py`
- `docs/adding-models.md`, Text Model Checklist and Quantization Parameter Bounds
- `docs/supported-models.md`, the `deepseek_v4` entry
