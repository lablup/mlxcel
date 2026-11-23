# PR #3251: Paradigma Limite text model port

**Date**: 2026-11-23
**Status**: PR open as draft (closes #2360); verified on M3 Ultra with the real checkpoint

## Executive Summary

Paradigma Limite (`model_type: limite`, `LimiteForCausalLM`, `paradigma-inc/limite-1b-violetto`) is a 48-layer dense decoder that mlxcel refused with `Unsupported model type: limite`. The architecture combines nine features that are each easy to get almost right: a weightless RMSNorm whose epsilon comes from the dtype, bf16-rounded scalar folds, a partial interleaved RoPE, a 1025-key sliding window, NoPE global layers, value embeddings, XSA, a per-head attention gate, four residual lambdas per layer, a MUDD mixer on two layers, and a sigmoid-softcapped tied head. Most of these load every tensor and still produce plausible text when implemented slightly wrong, so the port was checked against mlx-vlm's `mlx_vlm/models/limite` by teacher-forced logit traces rather than by reading generated text.

This PR adds the model with upstream-exact numerics, wires it into detection, the config-backed loader, `LoadedModel`, the registry and the docs, and gives it model-owned per-sequence caches with native batched decode, `last_row_logits` and per-layer KV mode reporting. Over 2560 traced positions of `docs/architecture.md`, mlxcel disagrees with the reference on 0 of 814 decided positions, and a negative arm that drops the MUDD mix makes the reference disagree with itself on all 814, so the trace is sensitive to the features it is meant to check.

## 1. Change Summary

| Area | Change |
|---|---|
| `limite_layers.rs` | Numerics: dtype-eps RMSNorm, `fold`, in-dtype erf GELU, `mudd_combine`, the attention block (value embeddings, XSA, gate), SwiGLU MLP, residual lambdas, and the sliding/global `Cache` enum. |
| `limite_rope.rs` | Partial interleaved RoPE with cos/sin tables built once per forward and shared by every RoPE layer. |
| `limite.rs` | Loader, MUDD history and tap wiring, the decoder stack, the f32 sigmoid-softcapped head, and cache construction per KV mode. |
| `limite_config.rs` | `ModelArgs`, the convention-string checks and the refusals, tap-index and softcap helpers, EOS list. |
| `limite_shapes.rs` | Load-time shape validation of every checkpoint tensor. |
| `limite_wrapper.rs` | Model-owned per-sequence caches, batched decode, `last_row_logits`, KV mode report, release by sequence id. |
| `sanitize.rs` | Skip the bf16 to f16 load cast for `limite`; list it as f16-fragile below Ampere. |
| Integration | `detection.rs`, `mod.rs`, `loaded_model.rs`, `registry.rs`, `model_metadata.rs`, `memory_estimate.rs`, tensor-parallel fallback name, shared coverage test tables. |
| Tooling and docs | `scripts/limite_oracle_trace.py` (mlx-vlm reference trace with `--no-prefix-sharing` and `--drop-mudd`), `docs/supported-models.md` entry and KV mode row. |
| Tests | `limite_tests.rs` (59 unit tests) plus entries in detection, model-owned KV mode, mixed-cache batched decode and coverage tests. |

22 files, +2484 / -6.

## 2. Architecture

### 2.1 Layer block

Each layer computes `attn_in = rms_norm(x)` (or the MUDD mix on mixing layers), runs attention, then applies the residual with four learned scalars:

```
x   = resid_attn * base + post_attn * attn(attn_in)
out = resid_mlp  * x    + post_mlp  * mlp(rms_norm(x))
```

`base` is `x` on ordinary layers and the MUDD residual mix on mixing layers. The lambdas are 0-dim tensors cast to the activation dtype once at load. The MLP is SwiGLU.

### 2.2 Attention: sliding and global

Q, K and V come from projections whose learned `qkv_scale` and `o_scale` are folded into the weights at load. Q and K get a weightless RMSNorm, and the softmax scale is fixed at 0.1 rather than `1/sqrt(head_dim)`.

- **Sliding layers** see `sliding_window + 1` = 1025 keys: key `k` is visible from query `q` when `k >= q - 1024`, inclusive. They hold a `RotatingKVCache(1025)` and apply the partial RoPE.
- **Global layers** (every fourth layer, from `global_layers`) are NoPE and full causal on a growing `KVCache`. A finite `global_window` is refused.

Prefill uses the shared `prefill_sliding_window_mask` and `prefill_causal_mask` helpers; a mask-less multi-token forward goes through `maskless_prefill_attention`; batched decode uses the shared `attend_batched_rows`.

### 2.3 Value embeddings

On `ve_layers`, a second embedding table `model.value_embeds` is gathered with the same token ids, reshaped to `[B, L, ve_stored_heads, ve_dim]`, zero-padded to `head_dim`, trimmed to the KV head count, scaled by `gate_scale * sigmoid(attn_in[..., :channels] @ ve_gate.T)`, and added to V before the Q/K norm and before the cache write, so the cache and the attention both see the adjusted values.

### 2.4 XSA

On `xsa_layers`, the attention output has its component along the (GQA-repeated) value direction removed: `v_hat = v / max(||v||, 1e-4)` in f32, then `y -= tanh(xsa_alpha) * <y, v_hat> * v_hat`, with the subtraction cast back to the activation dtype.

### 2.5 Attention gate

After XSA, each head's output is multiplied by `gate_scale * sigmoid(attn_in[..., :channels] @ gate.T)` before `o_proj`.

### 2.6 MUDD mixer

Layers 24 and 47 read a learned per-position mix of earlier layer outputs instead of `x`. `hist[0]` is the normalized embedding and `hist[j]` is the output of layer `j - 1`. For each mixing layer the weights are `gelu(rms_norm(x) @ dense1.T) @ dense2[layer].T + bias[layer]`, one weight per tap, and the mix is `sum_i w_i * hist[tap_i]`. The attention way feeds `rms_norm(mix)` to attention; with `mudd_mlp` a second way supplies the residual base. Only the tapped history slots are retained, and they are dropped at the end of each forward.

### 2.7 Head

After an optional tanh `final_softcap`, the final weightless norm and the tied embedding produce raw logits, which are cast to f32 and returned as `23 * sigmoid((raw + 5) / 7.5)`. Those bounded values are what the sampler receives, as in the reference. Greedy decoding is unaffected by the bound; temperature sampling sees a flatter distribution, and that is the model as published. `last_row_logits` applies the head to the sampled row only.

## 3. Parity Traps and How Each Was Handled

| Trap | Wrong version that still runs | Handling |
|---|---|---|
| Dtype epsilon | A fixed `rms_norm_eps`, or the f16 cast changing `finfo.eps` from 0.0078125 to 0.00098 | `finfo_eps(dtype)` per call; the load cast to f16 is skipped for `limite`, so the checkpoint keeps its bf16 activations. Unit test on the eps per dtype. |
| bf16 fold | Multiplying the f32 scalar into the weight, then rounding | `fold` rounds the scalar to the weight dtype first (`scale.astype(W.dtype) * W`). Unit test against a hand bf16 rounding. |
| Partial interleaved RoPE | Rotating all 64 pairs, rotating half-split lanes, or f32 cos/sin | 32 of 64 adjacent-lane pairs rotate, the rest have zero frequency; odd-lane sine negated; cos/sin computed in f32 and rounded to bf16. The unit test fails if all 64 pairs rotate. |
| 1025-key window | A 1024-key window (`k > q - 1024`) or a 1024-slot rotating cache | Window and cache are `sliding_window + 1`. Unit test: after 1100 + 1 tokens the sliding cache shows 1025/1025/1025 keys and a global cache 1101. |
| NoPE globals | Rotating global layers (identity at position 0, so short tests miss it) | `uses_rope` is false on global layers; full causal mask. |
| Softmax scale | `1/sqrt(head_dim)` = 0.125 | Read from the config, default 0.1. |
| Value embedding order | Adding after the Q/K norm or after the cache write | Added to V right after the projection, before the cache write. |
| MUDD accumulation | A stacked contraction (`einsum` over taps), which rounds differently in bf16 | Strict left-to-right adds in the activation dtype. Unit test: 256 + 1 + 1 in bf16 gives 256, not 258. |
| GELU dtype | `mlxcel_core::gelu`, which computes in f32 and casts back | `gelu_erf` computes in the input dtype, as MLX `nn.gelu` does. |
| Head | Raw logits, or a bf16 softcap | f32 sigmoid softcap; any other softcap kind is refused. |

The config also carries convention strings (`rope_style`, `sliding_window_convention`, `mudd_hist_convention`, `rms_norm_eps_mode` and others). Each is checked against the one convention this port implements, so a future checkpoint that changes a convention is refused at load instead of being run with the wrong arithmetic.

## 4. Integration Points

- **Detection and loading**: `ModelType::Limite` in `detection.rs`, the config-backed loader path, and `LoadedModel::Limite(LimiteWrapper)` delegated through `delegate_language_model!`.
- **Registry**: `mlxcel arch` lists the family with its KV modes; memory estimation and the tensor-parallel fallback map the type to `limite` (tensor and pipeline parallel are not wired).
- **Caches**: model-owned per sequence, like Mellum and Laguna. Batched decode is native, the scheduler's KV bound applies, and release goes through `release_sequence_state_by_id`.
- **KV modes**: `--kv-cache-mode` is honored on the global layers and reported per layer. The 1025-key sliding layers store FP16 under every mode, because 1025 is not a multiple of 32; the docs table records this.
- **Tooling**: `scripts/limite_oracle_trace.py` scores the `logit_trace` chunk plan with mlx-vlm so `scripts/compare_logit_traces.py` can compare the two TSVs.

## 5. Review-Driven Changes

### 5.1 Load-time shape validation (security review)

Two problems came from trusting checkpoint shapes:

- **HIGH**: `model.value_embeds` had no row-count check. A table with fewer rows than `embed_tokens` lets valid token ids gather out of bounds on the GPU, and MLX does not bounds-check gathers.
- **MEDIUM**: any tensor with the wrong shape threw a C++ exception through the non-`Result` FFI and terminated the process at load or at the first forward.

`limite_shapes.rs` now checks the projections, MLP, gates, scalars, `xsa_alpha`, the MUDD tensors and both embedding tables (`[vocab_size, ...]`) before any op runs, and returns a load error naming the tensor.

### 5.2 Config refusals

`validate` additionally refuses `num_hidden_layers: 0` (which panicked on `layers[0]`), a `ve_stored_heads` that neither equals 1 nor reaches the KV head count (no other value broadcasts), and, after the implementation review flagged it, `mudd: false` with `mudd_tap_idx` set. Upstream applies the mix from `mudd_tap_idx` regardless of the flag, so mlxcel skipping it would silently diverge; refusing the combination is cheaper than guessing. The released checkpoint declares `mudd: true` and is unaffected. These join the existing refusals of `rope_per_layer: true`, `mlp_type: relu2`, a non-sigmoid softcap, a finite `global_window` and quantized exports.

### 5.3 RoPE table hoisting (performance review)

The cos/sin tables were rebuilt for every layer and for both Q and K. `limite_rope.rs` builds them once per forward from the position array (`[1, L]` for one sequence, `[B, L]` for batched decode with per-row offsets) and shares them across all RoPE layers. The residual lambdas are cast at load instead of per call. Greedy output is byte-identical, and decode went from about 99.4 to 111.3 tok/s on a test-fast build.

## 6. Validation

### 6.1 Unit tests

`--lib limite`: 59 passed, covering the dtype eps, the bf16 fold, the partial RoPE, the MUDD order, the 1025-key cache, the config and shape refusals, the shared model-owned contracts, int8 KV and last-row logits. The `coverage_tests` (18), `models::registry`, `models::detection_tests`, `models::mixed_cache_batched_decode_tests`, `models::model_owned_kv_mode_tests` and `models::last_row_logits` selectors pass, as do clippy `-D warnings`, fmt and the contract scripts.

### 6.2 Logit trace

M3 Ultra, bf16 checkpoint, corpus `docs/architecture.md`. A position is decided when the reference's top-two gap is at least 2. Plan `64 40 8 1536` is 64-token chunks behind up to 1536 tokens of context; chunks 17 to 39 have more than 1025 tokens of context, so the sliding window is exercised.

| Arms | Positions | Decided disagreement | Max delta on ref top-1 |
|---|---|---|---|
| ref (one forward per chunk) vs mlxcel, `64 40 8 1536` | 2560 | 0 / 814 | 0.63 |
| ref (prefix-shared forward) vs mlxcel, same plan | 2560 | 1 / 815 | 2.74 |
| ref vs ref, shared vs per-chunk | 2560 | 1 / 814 | 3.79 |
| ref vs mlxcel width 1 (decode), `1 1300 8 1536` | 1300 | 1 / 402 | 3.54 |
| mlxcel `MLXCEL_TRACE_BATCH_ROWS=1` vs `=4`, `64 36 8 1536` | 2304 | 0 / 666 | 0.58 |
| ref vs ref `--drop-mudd` (negative arm) | 2560 | 814 / 814 | |

The single disagreement in three rows is always corpus position 1250, where the reference's two forward shapes already disagree with each other (3672 vs 87, gaps 4.05 and 3.49). mlxcel agrees with the per-chunk reference in both its prefill and decode shapes. Perplexity is 797.3 for the reference and 796.2 for mlxcel; the absolute value is high because the head is bounded to `(0, 23)`.

The negative arm matters: replacing the MUDD mix on layers 24 and 47 with plain `rms_norm(x)` flips every decided position. A trace that passed with that feature removed would prove nothing about it.

### 6.3 Generate and server

- `mlxcel generate -m models/limite-1b-violetto -p "Hello" -n 50 --temp 0` runs at about 90 tok/s. The model is a math reasoner with a fixed system prompt, so the output is its reasoning channel ("We need to parse the user's request..."). `-p "What is 17 * 23?" -n 600` derives 340 + 51 = 391.
- `mlxcel-server` served an 1821-token chat prompt and a concurrent 45-token prompt; both finished with `finish_reason: length` (120 and 200 tokens), and the short one reasoned to 42.

## 7. Open Items

- MUDD history slots are held to the end of the stack rather than freed after their last tap (LOW, not fixed).
- `eos_token_id` values are narrowed from i64 to i32 without a range check (LOW, not fixed).
- The 1025-key sliding layers cannot use quantized KV storage; only the global layers benefit from `--kv-cache-mode`.
- Tensor and pipeline parallel are not wired for this family.
- The performance number in 5.3 is from a test-fast build; the release-build generate run measured about 90 tok/s with a different prompt and length, so the two are not directly comparable.

## 8. Learning Points

- A feature that is the identity at position 0 or for short inputs (NoPE globals, a 1024 vs 1025 window) needs a test that reaches the case where it differs: two offsets, or a sequence past the window.
- When a model's arithmetic depends on dtype (here the norm epsilon), the generic load-time f16 cast is a semantic change, not a precision change.
- Order of accumulation is part of the model in bf16. A stacked contraction and a left-to-right sum are both correct in exact arithmetic and give different tokens.
- A logit trace needs a negative arm. Removing one feature and watching the agreement collapse shows the trace can see that feature.
- Two forward shapes of the reference itself can disagree on a near-tie position; compare against both before calling a single mismatch a bug.
