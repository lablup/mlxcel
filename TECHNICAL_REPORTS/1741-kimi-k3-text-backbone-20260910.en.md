# Technical Report: PR #1741 - feat(models): add the Kimi K3 text backbone

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Pre-merge (25 family tests plus the shared-module suites, clippy and fmt clean; two layer-truncated local copies load, prefill, decode and trace finite logits on an M5 Max; the full 93-layer model is not reachable on any hardware this project has, and the tokenizer is #1338)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a new family, plus three edits inside code other families share: the gated-delta gate, `SwitchGLU`'s activation, and `kimi_linear`'s `kv_b_proj` decomposition, which this PR lifts out of that module and puts a guard in front of)

---

## Executive Summary

Moonshot's Kimi K3 (`model_type: "kimi_k3"`, `text_config.model_type: "kimi_linear"`) is a 2.8T-parameter, 93-layer decoder: 69 Kimi Delta Attention layers, 24 gated NoPE MLA layers, a latent MoE with 896 experts on a 3584-wide projection of the 7168-wide hidden state, and Attention Residuals over 12-layer blocks. `src/models/kimi_linear.rs` already implements the KDA plus absorbed-MLA hybrid and the sigmoid router, so this PR adds `src/models/kimi_k3.rs` for the five mechanisms that module does not have, borrows the three pieces it does (`ShortConv1d`, `MultiLinear`, the `kv_b_proj` decomposition), and reads the published compressed-tensors `mxfp4-pack-quantized` experts by reinterpreting them into MLX's mxfp4 layout at sanitize time rather than converting them.

Three things in this PR are worth more than the port itself. The security pass turned 23 config-versus-checkpoint disagreements from mid-forward MLX aborts into load errors that name a tensor and a field, and rewrote the Attention Residual mix twice to keep a 4096-token residual window out of float32 residency. `decompose_kv_b_proj`, extracted here so both families share one copy, gained the reshape cross-check that Kimi Linear had been running without since it was written. And porting the state-reset line verbatim exposed that it silently truncates a chunked prefill: every chunk after the first discards the recurrent state its predecessors built, with no error and no shape mismatch. Kimi K3 opts out of chunked prefill; #1749 carries the same defect in `kimi_linear` and five other families.

---

## 1. Five mechanisms `kimi_linear.rs` does not have

### 1.1 One projection and one convolution where there were three of each

Kimi Linear's KDA layer runs `q_proj`, `k_proj` and `v_proj` separately, each followed by its own `ShortConv1d`. Kimi K3 ships one `qkv_proj` of shape `[3P, hidden]` and one depthwise `qkv_conv` over the concatenated `3P = 3 * 96 * 128 = 36864` channels, split into per-head q / k / v after the convolution. The checkpoint stores the six tensors separately, so the fusing happens in `sanitize_kda_layer`: the three `[C, 1, K]` PyTorch conv weights are transposed to MLX's `[C, K, 1]` and concatenated on axis 0, and the three projections are concatenated the same way for `.weight`, `.scales` and `.biases` alike. The forward pass then does one matmul and one convolution per KDA layer.

The gate is the other half. Kimi Linear computes `g = exp(-exp(A_log) * softplus(a + dt_bias))`, which reaches zero for large `a` and can decay the recurrent state to nothing. Kimi K3 declares `gate_lower_bound = -5.0` and computes

```text
g = exp(gate_lower_bound * sigmoid(exp(A_log) * (a + dt_bias)))
```

which is the `safe_gate` form of FLA's KDA kernels and bounds every decay to `(e^-5, 1)`. `A_log` ships as 128 entries for 96 heads on the published checkpoint and is sliced at sanitize. The output gate is full rank (`g_proj` at `[P, hidden]`) rather than the low-rank `g_b_proj(g_a_proj(x))` pair, selected by `linear_attn_config.use_full_rank_gate`; both forms are implemented because the config field exists.

One detail is easy to get subtly wrong and is spelled out in `KimiK3TextConfig::qk_norm_eps`. The reference runs FLA's in-kernel l2norm, `x / sqrt(sum(x^2) + 1e-6)`, while this tree has `rms_norm`. With `D = head_dim`, `sum(x^2) = D * mean(x^2)`, so the l2norm equals `D^-0.5 * rms_norm(x, eps = 1e-6 / D)`. The epsilon divided by the head dimension is not cosmetic: at `D = 128` it is the difference between `1e-6` and `7.8e-9` inside the square root. Kimi Linear uses mlx-lm's plain `1e-6` and is left alone, so the two families deliberately differ here.

### 1.2 q-LoRA and an output gate on an MLA path that rotates nothing

The MLA layers keep Kimi Linear's absorbed form unchanged: `embed_q` and `unembed_out` are derived from `kv_b_proj` at sanitize time, the `(kv_latent, k_pe)` pair lives in one `KVCache`, decode scores in the latent space and prefill expands the latent to per-head k and v. What K3 adds around it is a query compressed through `q_a_proj` at rank 1536, an RMSNorm, and `q_b_proj`; no rotation on the `q_pe` or `k_pe` halves; and `sigmoid(g_proj(x))` multiplied into the attention output before `o_proj`.

The RMSNorm epsilon is `1e-6`, not the config's `rms_norm_eps` of `1e-5`. The reference builds both `q_a_layernorm` and `kv_a_layernorm` as `KimiRMSNorm(rank)` and takes the class default rather than the config field, so following the config here would be wrong on exactly the two norms that sit inside the query and key-value compression.

`mla_use_nope` is rejected when false rather than implemented. A rotated variant is a different attention, and a config that declares one would otherwise load and produce fluent text from the wrong positions.

### 1.3 SiTU, and the one thing it costs

Every gated MLP in this family uses SiTU rather than SwiGLU:

```text
situ(up, gate) = (beta * tanh(gate / beta) * sigmoid(gate)) * (linear_beta * tanh(up / linear_beta))
```

at `beta = 4.0` and `linear_beta = 25.0`, computed in float32 and cast back, with the second factor reducing to `up` when `linear_beta` is null. `hidden_act` other than `"situ"` fails at load.

The routed experts need it too, which is the shared-code edit: `SwitchGLU` gained a `SwitchGluActivation` field that defaults to `SwiGlu`, and `SwitchGLU::with_activation` is the only way to leave that default. The cost is stated where it happens. `forward_fused_kernel` returns `None` for any non-SwiGLU activation, because that kernel fuses `silu(gate) * up` and nothing else, so Kimi K3's experts always take the `gather_qmm` path. On this checkpoint that costs nothing today, since the fused kernel is affine-only and these experts are mxfp4, but the guard is keyed on the activation rather than on the quantization so a future affine conversion of the same family cannot slip into a kernel computing the wrong nonlinearity.

### 1.4 The experts run on a projection of the hidden state, not on the hidden state

Layers 1 through 92 route through a latent. `routed_expert_down_proj` takes the 7168-wide residual to 3584, 16 of 896 experts run there, the weighted sum passes through `routed_expert_norm` and `routed_expert_up_proj` back to 7168, and two always-on shared experts run on the full width and are added. Sizing the experts on the latent is what makes 896 of them affordable: each `gate_proj` plane is `[3072, 3584]` rather than `[3072, 7168]`.

The router is sigmoid over the gate logits in float32. Selection uses `scores + e_score_correction_bias`; the weights are taken from the bias-free `scores` by `take_along_axis`, then renormalized against their own sum plus `1e-20` and scaled by `routed_scaling_factor`. Grouped routing is refused rather than implemented: `num_expert_group` and `topk_group` must both be 1, which the published config satisfies.

### 1.5 Attention Residuals need no cache, and that is the whole design

Every `attn_res_block_size` layers the running residual is frozen into a block list, and each attention and MLP sublayer reads a softmax mix of the frozen blocks and the current partial sum instead of the residual itself:

```text
logit_k = (raw_k @ w_eff) * rsqrt(mean(raw_k^2) + eps)      for each stored block
logit_p = (partial @ w_eff) * rsqrt(mean(partial^2) + eps)
out     = sum_k softmax(...)_k * raw_k + softmax(...)_p * partial
```

with `w_eff = res_norm.weight * res_proj.weight` folded into one `[D, 1]` float32 column at load, so the mix is one gemv per stored block. At a block-start layer the residual restarts (`partial = y` rather than `x + y`), because `x` was just frozen into the list; layer 0 stores the embeddings.

The blocks are per-token rows of shape `[B, T, D]`, and a token's mix reads only that token's own frozen states, never another position's. A decode step therefore computes its own blocks from its own embedding at the block-boundary layers, exactly as the reference forward does, and nothing carries across calls. That is why `ResidualBlocks` is a local of `run_layers` and there is no cache variant for it. It is also why they become a pipeline-parallel problem rather than a free ride: a stage starting at layer 24 needs the blocks stored by layers 0 and 12 on an earlier node, which is item 4 of #1734.

---

## 2. Borrowing from `kimi_linear.rs` instead of copying it

Three pieces moved rather than being duplicated. `MultiLinear` and `ShortConv1d` became `pub(crate)`, the latter gaining a `new` that wraps an already-sanitized `[channels, kernel, 1]` weight so the fused instance can be built from the concatenated conv. `decompose_kv_b_proj` was lifted out of `KimiLinearModel::sanitize_weights` into a free function both sanitizers call, carrying its `#1026` missing-biases error and its `#958` quantization-parameter bound with it.

Two edits reach past both families, and both are no-ops for existing callers by construction rather than by inspection:

| Edit | Existing callers | Why it cannot drift |
|---|---|---|
| `gated_delta::compute_g_lower_bounded` and `gated_delta_update_with_lower_bound` | Qwen3Next, Qwen 3.5, Kimi Linear | `lower_bound = None` returns `compute_g(...)` itself, so it is the same graph rather than an equivalent one. `gated_delta_update` is now a one-line forwarder. |
| `SwitchGLU::with_activation` | every MoE family in the tree | Both constructors set `SwitchGluActivation::SwiGlu` explicitly, and the fused decode kernel declines anything else, so nothing changes for a caller that never calls the setter. |

The bound only changes how the float32 gate is computed before `gated_delta_ops`. The Metal gated-delta kernel and the ops fallback both take a plain float32 gate either way, so neither path was touched.

---

## 3. The mxfp4 experts are reinterpreted, not converted

compressed-tensors `mxfp4-pack-quantized` stores one quantized linear as `weight_packed` (E2M1 codes, two per byte, uint8 `[out, in / 2]`) and `weight_scale` (E8M0 block scales, uint8 `[out, in / 32]`). On the published shards that is `[3072, 1792]` and `[3072, 112]` for `w1` / `w3`, and `[3584, 1536]` and `[3584, 96]` for `w2`. MLX's native mxfp4 wants uint32 words holding eight codes each, low nibble first, with the same uint8 E8M0 scales and no biases.

Those are the same bytes. `stack_expert_plane` stacks the per-expert planes on a new leading axis and calls `view(UINT32)`, which divides the trailing axis by 4 and leaves the buffer alone: `[896, 3072, 448]` for `gate_proj` and `up_proj`, `[896, 3584, 384]` for `down_proj`. Little-endian byte order is exactly the low-nibble-first code order MLX's mxfp4 kernels read, which `mxfp4_repack_matches_scalar_dequant` pins against a scalar E2M1 dequantization rather than against the assumption.

Two operational details matter at 896 experts across 92 layers. Planes are stacked one at a time, evaluated, and their per-expert sources dropped before the next plane, so the peak is one stacked plane rather than two copies of the layer. And the plane vector is grown rather than reserved with `Vec::with_capacity(num_experts)`, because `num_experts` is an unbounded `config.json` field and reserving it turns an absurd declaration into a capacity-overflow panic instead of the named shortfall error the loop already produces.

`expert_quantization` decides the mode from the data, not the config: a uint8 `.scales` plane is mxfp4's signature and pins `(group_size 32, bits 4)` regardless of what the MLX `quantization` block says, a float scales plane follows the config's pair with the mode inferred from the presence of biases, and no scales means dense.

The last piece is not in this family's code at all. `config_has_quantization_metadata` in `src/models/sanitize.rs` is the predicate the Apple Silicon bf16 to f16 conversion keys on, and Kimi K3 declares nothing at the top level: its only declaration is a compressed-tensors `quantization_config` nested under `text_config`. Teaching that predicate to look for the HuggingFace spelling at either level is what keeps the attention, shared experts, dense MLP, router and `lm_head` planes (everything the published `ignore` list names) uniformly bf16 around uint8 and uint32 expert planes.

---

## 4. `config.json` is untrusted input, and MLX aborts rather than returns

Several MLX calls on this path take the process down on a bad argument instead of returning an error, because an MLX C++ throw crossing the cxx bridge is an uncatchable `std::terminate`. Every one of them here takes checkpoint data, `config.json` values, or both. The review commits turned each into a load-time refusal that names the tensor and the field.

| MLX call | Arguments come from | Failure without a guard | Guard |
|---|---|---|---|
| `reshape` in `decompose_kv_b_proj` | `kv_b_proj` plus four config dimensions | Abort mid-sanitize with no key named; or a width that divides but is not `kv_lora_rank` survives and aborts inside the absorbed MLA matmul on the first forward | Element-count check spelling out `num_heads * (qk_nope + v_head) * kv_lora_rank` |
| `stack` over per-expert planes | Checkpoint | Abort mid-sanitize | `check_uniform_shapes`, naming the offending expert index |
| `view(UINT32)` | Packed trailing axis | Abort when the axis is not a whole number of uint32 words | Explicit positive-multiple-of-4 check |
| `concatenate` in the q/k/v fuse | Three conv or projection planes | Abort mid-sanitize on a partially converted plane set | `check_concat_compatible` |
| `argpartition(kth = k - 1)` | `num_experts_per_token` | Index past the score axis on the first routed token, with the checkpoint already resident | `validate()` bounds `k` to `[1, num_experts]` on both load paths |
| `reshape` of the KDA gate to `[B, T, H, D]` | `g_proj` width | Abort on the first forward | `check_axis` at load |
| `matmul` in `attn_res_mix` | `res_proj` and `res_norm` widths | A pair that agrees with itself but not with `hidden_size` survives load and aborts on the first forward | `load_attn_res_weight` checks both against `hidden_size` before multiplying them |

`kimi_k3.rs` carries 23 such cross-checks (15 `check_axis`, 8 `check_numel`), and `hidden_size` is covered transitively through the `model.norm.weight` element count, so input-axis mismatches fail at load as well. A missing key is deliberately not an error in either helper: whichever loader needs it reports it by name.

The Attention Residual mix was rewritten twice for the same reason, and the second rewrite is the more instructive one.

- **First form.** Stack the values into `[B, T, n + 1, D]` and fold with one matmul. Same arithmetic up to summation order, but it allocates a contiguous float32 copy of every stored block on top of the blocks themselves. At `D = 7168` with the full 8-block window that is about 0.9 GB of transient for a 4096-token prefill.
- **Second form.** Drop the stack, fold per term, but promote each block to float32 once and read that array twice, from the logit matmul and from the weighted term. Every consumer of the softmax runs after every logit, so all of those promotions stay resident from the first loop until the fold. Same 0.9 GB.
- **Shipped form.** Hand each block to `matmul` and `multiply` in its stored dtype and let MLX promote against the float32 `w_eff` and the float32 probability column inside each op. Bit-identical arithmetic, because that is the same `astype` the code was writing by hand, but each promotion is now an op-local temporary MLX frees when the op is done.

The general shape is worth keeping: an `astype` you write yourself is a graph node with a lifetime that spans every consumer, while the same promotion performed inside an op is a temporary. Writing it by hand looks like sharing work and is really extending a lifetime.

Findings reported and left open are listed in section 8.

---

## 5. A prefill that discards its own state, silently

`forward_for_sequence` opens with a reset keyed on the token count:

```rust
if seq_id.is_none() && seq_len > 1 {
    self.sequence_state.replace_internal(self.make_layer_caches());
}
```

That reset is load-bearing. It is what makes the cache-less `LanguageModel::forward` entry point safe to call for two unrelated prompts in a row: a multi-token forward carrying no sequence id is read as the start of a new prompt.

`supports_chunked_prefill` defaults to true, and `chunked_prefill_last_logits` calls `forward_last_logits` once per chunk with no sequence id, which lands on `forward` and therefore on the reset. Every chunk after the first discards the KDA conv and SSM state and the MLA latent pair that its predecessors built. Nothing raises: no error, no warning, no shape mismatch. The logits are computed from the last `MLXCEL_PREFILL_CHUNK` tokens (2048 by default) and look like any other logits. With `max_position_embeddings` at 1048576 on this family, a prompt past one chunk is the headline case rather than an edge case.

Kimi K3 overrides `supports_chunked_prefill` to false. That is the conservative half of the fix: it gives up the per-chunk prefill memory bound of #672, so transients become one graph over the whole prompt, and it keeps the answer correct. The complete fix is to let `forward_for_sequence` tell a new prompt from a continuation by something other than the token count, which is a change to the shared `ModelOwnedSequenceState` contract and has to be checked against every family that resets state inside a forward.

The audit in #1749 walked all 42 `replace_internal` call sites under `src/`. Seven are in-forward, all with the identical guard: `kimi_k3` (mitigated here), `kimi_linear` (#1749 itself), and `qwen3_next`, `rwkv7`, `bailing_moe_linear`, `recurrent_gemma` and `afmoe`, which need their own follow-ups because each needs its own validation checkpoint. The server path is not affected in any of them, since it drives `forward_for_sequence` with a real `SequenceId` and never enters the branch. This is a CLI and benchmark defect, and it has been in `kimi_linear.rs` since chunked prefill landed in #672 and #674.

---

## 6. One distributed claim corrected, one gap filled

**Pipeline parallelism does not run this family, and the docs said otherwise.** What this PR adds is `kimi_k3_per_layer_bytes` in the partition profile, which prices the dense layer 0, the cheaper q-LoRA MLA layers at every fourth position and the 15 to 16 GB MoE layers separately, and counts the routed experts at 4 bits plus one E8M0 byte per 32 weights on the latent width regardless of the `bits_per_weight` that describes the dense planes. That is a planning input. There is no `StageFamily` variant and no stage executor, so `--pp-size` and `--pp-layers` refuse a `kimi_k3` model, and `docs/distributed.md` and `docs/supported-models.md` now say so instead of describing PP as working. #1734 owns the executor, the Attention Residual transport across a stage boundary, the uint32 wire dtype, and the three-node run.

**Tensor parallelism had a fall-through with a plausible-looking failure.** Without an entry in the replicate list, `generate_shard_plan` hands a `kimi_k3` model the generic transformer plan, which claims to shard `q_proj`, `k_proj`, `v_proj` and `gate_proj` on a model whose projections are `qkv_proj`, `kv_a_proj_with_mqa`, `embed_q` and `switch_mlp`. The family is now replicated beside `kimi_linear`, and `kimi_k3_is_replicated` pins that rather than leaving it to the next person to rediscover.

---

## 7. Validation

Host: M5 Max, 128 GB, Metal, shared with seven concurrent build agents. Every row below was re-run on the binary built from the final commit.

### 7.1 Unit tests and lints

`cargo test --profile test-fast --features metal,accelerate --lib models::kimi_k3` passes 25 tests: SiTU against a scalar reference, the lower-bounded gate's range, fused QKV against separate projections, decode against prefill, absorbed q-LoRA MLA against an unabsorbed reference, latent MoE against a per-expert loop, the AttnRes mix against a scalar softmax, block accounting at `attn_res_block_size = 2`, the mxfp4 repack against scalar E2M1 dequantization, causality, config parsing of the published `config.json`, EOS resolution, sequence-state isolation, and a negative test for every load-time refusal the review commits added.

The shared modules were run scoped: `models::gated_delta` (6), `models::switch_layers` (24), `models::detection` (61), `distributed::tensor_parallel::plan_generator` (28), `distributed::pipeline::partition_profile` (7), `cli::turbo_args` (19). `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` are clean.

### 7.2 Two truncated checkpoints, and why there are two

The full checkpoint is about 1.4 TB resident at 4 bits, so the real-model gate runs on `models/kimi-k3-8l-mxfp4`, a local copy of `moonshotai/Kimi-K3` with `text_config.num_hidden_layers` lowered from 93 to 4, `model.safetensors.index.json` rewritten to the 7 shards holding layers 0 to 3 plus the embeddings, final norm and `lm_head`, and `metadata.total_size` corrected to the 54.46 GiB those shards actually contain. Both originals are kept beside the edited files.

Four layers is where the published schedule first contains one of every layer kind: the dense MLP at layer 0, KDA at layers 0 to 2, MoE at layers 1 to 3, and the first MLA layer at index 3 (1-based 4, the first entry of `full_attn_layers`). Each further layer is a repeat costing about 17 GB of MoE experts, so the memory went into a second variant instead of more depth.

| Gate | Command | Result |
|---|---|---|
| Pre-load budget | `mlxcel inspect -m models/kimi-k3-8l-mxfp4` | 54.46 GiB weights, 1.88 GiB KV at 8192 tokens, 67.69 GiB total, FITS with 53.91 GiB headroom |
| Load, prefill, decode | `mlxcel generate -m models/kimi-k3-8l-mxfp4 -p "def fib(n):" -n 32` | loaded in 5.3 s, 45.41 GB resident, 49.71 GB peak, 32 tokens in 0.60 s (53.66 tok/s), exit 0, no NaN, no abort |
| Finite logits | `examples/logit_trace models/kimi-k3-8l-mxfp4 <corpus> 32 2 8 0` | 64 teacher-forced positions, 512 top-k logits, 0 non-finite; NLL 10.35 to 17.23 (mean 13.65), top-k logits 6.66 to 10.38 |
| Multi-block Attention Residuals | the same two commands on `models/kimi-k3-4l-attnres2` | 32 tokens at 54.61 tok/s, 64 positions traced, 0 non-finite; NLL 10.56 to 18.19 (mean 13.85) |

`models/kimi-k3-4l-attnres2` is the same 7 shards by symlink with `attn_res_block_size` lowered from 12 to 2. That is the only way to reach the multi-entry softmax on real weights: at the published block size of 12, any copy small enough to load stores exactly one block, so `n = 1` and the mix never exercises the loop. At block size 2, layers 0 and 2 each freeze a block and layer 3's mix scores three entries against real `res_proj` and `res_norm` weights. The 13-layer copy that would exercise it naturally is about 220 GB.

Both traces and both generations reproduce to the digit on a second run, which is also the determinism check.

The generated text is multilingual noise in both runs. That is what 4 of 93 layers produces, compounded by the generic tiktoken fallback reading `tiktoken.model` with the wrong pre-tokenization regex until #1338 lands. The gate is the load, the shapes, the absence of NaN and the finiteness of the logits. It is not the text, and the report says so rather than presenting a decode speed as if it were a quality result.

---

## 8. What is not established

**The full model.** Not run, and not runnable: about 1.4 TB at 4 bits against a largest available host of 128 GB, and no pipeline stage executor in the tree even if three 512 GB nodes appeared. #1734 carries both.

**Fluent output.** Nothing in this PR demonstrates it. Two of the acceptance criteria on #1334 are explicitly unchecked for this reason.

**Token-exactness against any reference.** mlx-lm has no `kimi_k3`, so there is no external oracle to diff greedy ids against, and the truncated runs compare only against themselves and against the scalar references in the unit tests. #1734's second acceptance criterion is the layer-by-layer float32 reference trace that would close this.

**The workspace gate.** `cargo test --workspace` and `cargo clippy --workspace --all-targets` were not run on this host, because concurrent `mlxcel-core` test binaries abort each other (#1008). The scoped suites above plus CI cover them.

**Cross-check gaps reported and left open (MEDIUM).** `lm_head.weight` rows against `vocab_size`; `routed_expert_up_proj` against the latent width; `moe_intermediate_size` against the stacked expert planes; `intermediate_size` against the dense MLP. `num_hidden_layers` is unbounded, so a hostile `config.json` turns the `0..num_hidden_layers` loop in `sanitize_weights` into a hang and `from_weights`'s `Vec::with_capacity` into a capacity-overflow panic.

**Two costs measured only by arithmetic.** The residual window holds `ceil(num_hidden_layers / attn_res_block_size)` full-width copies of the residual stream for the whole forward, which is 8 blocks and about 470 MB in bf16 for a 4096-token prefill at the published config, with no chunking for this family. Decode pays two mixes per layer, each one gemv plus one broadcast multiply per stored block, roughly 1.5k extra small ops per token at 93 layers and 8 blocks. Neither has been profiled on a checkpoint deep enough to show them.

**The leftover-expert drop path.** `sanitize_moe_layer` drops anything still under `experts.` after stacking, on the theory that a compressed-tensors export may carry sidecars such as `weight_shape`. The published shards carry only `weight_packed` and `weight_scale` per expert, so that path never fires on them and is exercised only by unit tests.

**Batching, padded prefill and the vision tower.** `supports_batching` and `supports_padded_prefill` are both false, for the same reasons as Kimi Linear. The vision tower is #1342.

---

## 9. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 29 |
| Lines added | 5087 |
| Lines removed | 117 |
| New modules | 3 (`kimi_k3.rs` 2086, `kimi_k3_sanitize.rs` 488, `kimi_k3_tests.rs` 1796) |
| Tests added | 25 family tests, 1 partition-profile test, 1 TP plan test, 1 detection test, 1 sanitize-policy test |

### Changes by area

- `src/models/kimi_k3.rs`: the config and its `validate`, the KDA and MLA blocks, the dense SiTU MLP, the latent MoE, the Attention Residual mix and its per-layer and model-level weights, the decoder layer, the model shell and its `LanguageModel` impl, and the 23 config-versus-checkpoint cross-checks.
- `src/models/kimi_k3_sanitize.rs`: the `language_model.` scope pass, MTP and out-of-range layer drops, legacy residual key spellings, the KDA conv and projection fuse, the MLA decomposition call, the MoE rename and the mxfp4 expert stack. Idempotent by construction.
- `src/models/kimi_linear.rs`: `MultiLinear` and `ShortConv1d` opened to the crate, `ShortConv1d::new` added, `decompose_kv_b_proj` extracted and given its reshape cross-check.
- `src/models/gated_delta.rs`: `compute_g_lower_bounded` and `gated_delta_update_with_lower_bound`.
- `src/models/switch_layers.rs`: `SwitchGluActivation`, `situ_activation`, `SwitchGLU::with_activation`, and the fused-kernel opt-out.
- `src/models/sanitize.rs`: `config_has_quantization_metadata` documented and tested against the nested compressed-tensors spelling.
- `src/distributed/`: the Kimi K3 partition profile, the TP replicate-list entry and its fallback architecture label, with tests for both.
- Registration: detection, registry, metadata, `LoadedModel`, the special weight loader, memory estimate, the hybrid-SSM prompt-cache roster, and the MLA-latent cache family list.
- Docs: `docs/supported-models.md` (the family entry), `docs/distributed.md` (PP is a planning input only), `docs/turbo-kv-cache.md` and `src/cli/turbo_args.rs` (the MLA-latent quantization exclusion).

### Commits

| Hash | Type | Subject |
|---|---|---|
| `d7ae018c` | feat | add the Kimi K3 text backbone |
| `4fcddfd6` | fix | bound the router, cross-check config against the checkpoint |
| `9828b5ca` | fix | name the two remaining checkpoint-driven MLX aborts |
| `9c8ad3cd` | fix | bound the residual mix and the kv_b_proj decompose |
| `a6993da1` | fix | opt out of chunked prefill, correct the distributed claims |
| `b1e826e5` | docs | add the family to the shared-function and KV-cache rosters |
| `a424acbf` | test | pin the tensor-parallel plan as replicated |

### Related issues

Closes #1334, a sub-issue of epic #1331 alongside #1338 (tokenizer and XTML chat rendering) and #1342 (MoonViT3D vision tower). Files #1734 (pipeline stage executor and full-model validation) and #1749 (chunked prefill discards recurrent state in `kimi_linear` and five other families). Carries the #1026 missing-biases error and the #958 quantization-parameter bound into the shared `decompose_kv_b_proj`. Gives up the #672 per-chunk prefill memory bound for this family.

---

## 10. Follow-up

**#1734 is the blocker for everything this PR could not check.** The full-model run, fluent output, and the layer-by-layer float32 reference trace all live there, and so does the Attention Residual transport question: a stage starting at layer `s` needs `ceil(s / 12)` frozen blocks that were computed on an earlier node.

**#1749 should not stop at `kimi_linear`.** Five more families carry the identical in-forward reset with the default `supports_chunked_prefill`. Whoever takes the shared fix should remove Kimi K3's override in the same change, so the two Kimi families do not drift.

**Close the MEDIUM cross-check gaps.** The four remaining config-versus-checkpoint widths in section 8 have the same failure shape as the ones already closed, and the unbounded `num_hidden_layers` is the one that turns a hostile config into a hang rather than an error.

**One number in the docs is wrong.** The `docs/supported-models.md` entry gives the stacked `down_proj` plane as `[896, 3584, 192]`. The published shards store `w2.weight_packed` as uint8 `[3584, 1536]`, and the uint32 view divides the trailing axis by 4, so it is `[896, 3584, 384]`. The `[896, 3072, 448]` for `gate_proj` and `up_proj` is right.

### Transferable lesson

The most valuable defect in this PR is not in the code it adds. `forward_for_sequence`'s reset was copied from `kimi_linear.rs` verbatim, and it has been silently truncating chunked prefills in that module since #672. What surfaced it was having to write down, for a new caller, what the condition `seq_id.is_none() && seq_len > 1` actually means. It means "infer the caller's intent from the shape of its argument", and a shape carries no intent: two unrelated callers can send the same shape for opposite reasons, and one of them is `chunked_prefill_last_logits` sending it once per chunk. The failure is silent precisely because the inference is type-correct.

The rule that follows is cheap to apply. When a port copies a line whose purpose is not local, do not translate it. Explain it to the new caller in prose, then check every entry point that can produce the state the explanation assumes. Here that took one grep for `supports_chunked_prefill` and one for `replace_internal`, and it turned one family's port into an audit of seven.
