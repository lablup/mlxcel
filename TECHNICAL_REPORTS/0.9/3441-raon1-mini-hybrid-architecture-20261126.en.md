# PR #3441: Raon-1 mini (raon1_mini) hybrid architecture (issue #3225)

**Date**: 2026-11-26

**Status**: PR open (closes #3225). Validated on an M3 Ultra. 33 files, +4324 / -42.

## Executive Summary

The PR adds the KRAFTON Raon-1 mini architecture (`raon1_mini`, `KRAFTON/Raon-1-mini` at revision `6691a076`). It runs from the original bf16 + MXFP8 checkpoint in `mlxcel generate`, `mlxcel-server` and `mlxcel arch`. The model combines four pieces that mlxcel did not have together: a four-stream Raon_mHC residual, RDA gated-delta linear attention layers, gated NoPE MLA layers shared with Kimi K3, and a sigmoid-routed top-6 MoE with SiTU activations and MXFP8 experts.

Greedy output matches the Transformers reference token for token on a short prompt and on an 871-token prompt (43 tokens). Teacher-forced traces show 0 of 578 decided positions disagreeing on a 4 x 256 corpus, and 1 of 549 on the long prompt, at a row where the reference disagrees with itself between bf16 and f32. Server batched decode is native: aggregate decode reaches 29.9 / 75.0 / 127.0 tok/s at 1 / 2 / 4 concurrent requests.

## 1. Problem Statement

### 1.1 Background

Issue #3225 asked for Raon-1 mini support at steering alpha 0 (the reference at `set_reasoning_effort("max")`), matching `modeling_raon1_mini.py`. The issue body wrote out the forward pass from a reading of that file, including where each value is rounded.

### 1.2 What the port had to cover

- **Raon_mHC**: the residual is carried as 4 streams `[B, L, 4, D]`. Each sublayer site computes per-token pre, post and residual-mix coefficients; the mix matrix is built on a Helmert chart so it is doubly stochastic. This is not DeepSeek-V4's Sinkhorn mHC.
- **RDA**: 18 gated-delta layers with f32 short convs, L2-normalized q and k (eps 1e-6), a write ratio, a lower-bounded per-dimension decay, and a `[B, 40, 128, 128]` f32 recurrent state.
- **Gated NoPE MLA**: 7 layers, the only ones that hold KV.
- **MoE**: 128 MXFP8 experts, top-6 sigmoid routing with a correction bias, a shared expert, and SiTU (`4 tanh(g/4) sigmoid(g) * 25 tanh(v/25)`) in both experts and dense MLPs.
- Five reasoning-effort steering tensors that must load (f32) but are not applied yet (#3435).

### 1.3 Risks

| Risk | Impact | Likelihood |
|------|--------|------------|
| Rounding placed differently from the reference drifts logits and flips greedy tokens on long prompts | High | High (this happened, see 3.2) |
| KV estimate wrong for a hybrid MLA family, so server context sizing and prompt-cache budgets are off | High | High (80x overestimate before the fix) |
| Malformed config or checkpoint panics or aborts across the cxx bridge instead of failing by name | High | Medium |
| Server decode does not scale with concurrency | Medium | High (per-row fallback before the fix) |

## 2. Technical Review

### 2.1 Security

The config and checkpoint are untrusted input. The security pass (999aa995e) fixed these HIGH findings:

| Issue | Severity | Status |
|-------|----------|--------|
| `validate_layer_partition` sized a bitmap from `num_hidden_layers`; `usize::MAX` overflowed and large values allocated memory from the config alone | High | Fixed |
| Zero widths accepted (RDA `head_dim` 0 gives an `inf` query scale), unbounded `raon_mhc_streams`, dimensions above i32 truncated by `as i32` | High | Fixed |
| `stack_mxfp8_experts` stacked experts of different shapes, which throws inside MLX and aborts the process | High | Fixed |
| Fewer than two mHC streams (the Helmert basis is empty) | Medium | Fixed |
| Projection input widths (RDA, MLA linears, expert `down_proj`) are checked only on output rows, so a malformed checkpoint can still abort in the first matmul | Medium | Open (same as Kimi K3) |
| The oracle script imports the checkpoint's Python files; it records their digests but does not pin them | Low | Open |

The vendor MLX export and the NVFP4 exports are refused by name (#3437). The check runs on the raw JSON before deserialization and again in `validate()` on the flattened extras, so the config-backed weight route is covered.

### 2.2 Performance

| Area | Before | After |
|------|--------|-------|
| Aggregate decode, 1 / 2 / 4 concurrent requests | 29.8 / 30.7 / 29.8 tok/s | 29.9 / 75.0 / 127.0 tok/s |
| KV estimate per token | 25 layers x 40 heads x 320 elements | 7 layers x 576 latent elements (about 80x smaller) |
| Single-stream decode | n/a | 32.9 tok/s |
| Load peak memory | n/a | equal to active memory, 32.8 GB |

The f32 RDA state adds about 47 MB of state traffic per token over bf16 (18 layers x 40 x 128 x 128 x 4 bytes, read and written), well under 1 ms at M3 Ultra bandwidth. Single-stream decode is about 30 ms per token against a weight-read floor near 8 ms (about 6.5 GB per token); the step is dispatch-bound (see section 8).

### 2.3 Compatibility

- **Breaking changes**: none. Kimi K3 MLA is refactored behind `NopeMlaSpec` / `KimiK3MLAAttention::from_spec` with unchanged behavior. Other gated-delta families keep the standard kernel.
- **New dependencies**: none. The oracle script needs transformers (validated with 5.16.1).
- Tensor parallel, DFlash2 (#3436), NVFP4 (#3437) and Raon-1 nano (#3438) are out of scope. `raon_mhc.rs` and `raon_rda.rs` take no Raon-1-mini config types so nano can reuse them.

### 2.4 Code Quality

New modules are split by concern: `raon_mhc.rs`, `raon_rda.rs`, `raon1_mini_moe.rs`, `raon1_mini_config.rs`, `raon1_mini_sanitize.rs`, `raon1_mini_lm.rs`, `raon1_mini.rs`, each with a test file. Tests cover the Helmert basis, doubly stochastic residuals, aggregate and mix, the layer partition, SiTU, the RDA ratio and decay (both bound forms), the MoE router, NoPE MLA order invariance, sanitize, MXFP8 native matmul against dequantize-then-matmul, prefill against chunked prefill and decode with a left-padded row, batched decode against per-sequence decode, config validation, detection, export refusals, and a 0.998 decay surviving 64 steps in f32.

## 3. Technical Decisions

### 3.1 Follow the reference, not the issue body, where they disagree on rounding

**Context:** The issue body described the forward pass, but on several points the reference rounds differently. Examples: the issue said the MoE output is `sum_k(experts.f32 * w_k) + shared.f32`, rounded once, while `Raon1MiniMoE.forward` rounds the routed sum to the model dtype and then adds the shared expert in that dtype. The reference also rounds the L2-normalized q to bf16 before the `head_dim ** -0.5` scale and keeps `beta` in bf16, and the checkpoint ships three separate convs (`q_conv1d`, `k_conv1d`, `v_conv1d`) rather than one joint `qkv_conv1d`.

| Option | Pros | Cons |
|--------|------|------|
| Follow the issue body | Matches the written spec | Each misplaced rounding adds bf16 noise that the trace comparison would flag |
| **Chosen: follow `modeling_raon1_mini.py` at `6691a076`** | Bitwise-comparable intermediates, traces can be held to zero disagreement | The issue body becomes partly stale |

**Rationale:** The acceptance test is agreement with the reference, so the reference decides. The three convs are concatenated in sanitize into one depthwise conv in q, k, v order, which is exact for a depthwise conv and saves two dispatches per layer.

### 3.2 Keep the RDA decay and recurrent state in f32

**Context:** After the first commit, the long-prompt trace still showed p99 / max logit delta 1.125 / 10.25. The cause was found by comparing against an f32 reference forward layer by layer: with bit-identical bf16 q, k, v and beta at layer 0, mlxcel's RDA core output was at 0.045 relative error against the f32 reference, while the reference's own bf16 forward was at 0.0022. The error was in the recurrence, not in its inputs.

The standard Metal gated-delta kernel casts the f32 decay `g` and the incoming state to the activation dtype and stores the outgoing state in it. In bf16 a value just below 1 has a step of 2^-8, so a decay of 0.998 becomes 0.996. Long-memory heads multiply that decay once per token, so the error compounds with prompt length. The reference keeps the log decay and the state in fp32.

| Option | Pros | Cons |
|--------|------|------|
| Standard kernel (bf16 decay and state) | Shared with Qwen3Next and the Kimi families, less state traffic | Decay drifts once per token on long prompts |
| Make every gated-delta family f32 | One path | Changes output of families validated against bf16 references |
| **Chosen: `gated_delta_ops_precise` for RDA only** | Matches the reference; reuses the existing precise kernel path (no casts, f32 state) | About 47 MB more state traffic per token |

**Result:** p99 / max logit delta went from 1.125 / 10.25 to 0.75 / 2.0, perplexity delta from +0.475% to +0.337%. The 4 x 256 corpus stayed at 0 of 578. A unit test checks that a 0.998 decay survives 64 steps with an f32 state; it fails on the standard kernel.

### 3.3 Fix the oracle's presence penalty before chasing a model bug

**Context:** Greedy decoding of the 871-token prompt flipped one token: the reference wrote "Linux and Windows", mlxcel "Linux/Windows". The oracle passed `presence_penalty=None` to `Raon1MiniForCausalLM.generate`, which only overrides the generation config when the argument is not None, so `generation_config.json`'s value of 1.5 stayed active and `_get_logits_processor` applied it to every step. `/` had already appeared in "Linux/CUDA", the penalty took 1.5 off it, and its 1.125 lead over " and" was gone.

**Decision:** pass 0.0, which skips the processor (f7c94abcc). The reference then emits the same 43 tokens as mlxcel. The reference arm was the bug, and comparing reference against reference is what exposed it.

### 3.4 Treat row 866 as reference-side bf16 noise

**Context:** On the long prompt plus continuation, 1 of 549 decided positions still disagrees: row 866, the token after `<|im_start|>assistant`. To decide which side was wrong, the reference was run in f32 there. f32 picks `\n`, the same token as mlxcel, with a gap of 0.48. The reference's bf16 variants (prompt only, cached prefill, longer context) agree with each other, so the bf16 reference is consistently off at that row, not randomly.

**Decision:** exempt row 866 from the zero-disagreement criterion and record why. Chasing it further would mean matching the reference's bf16 error, not the model.

### 3.5 Size the KV cache as a latent pair in the MLA layers only

**Context:** The config has `kv_lora_rank`, so `kv_arch` fell to the DeepSeek V2 decompressed branch and sized all 25 layers at 40 heads x 320 elements per token, about 80 times the real cache. Server context sizing and prompt-cache snapshot budgets read that estimate.

**Decision (45dee535d):** `raon1_mini` joins `MLA_LATENT_CACHE_FAMILIES`, and `kv_arch` counts MLA layers from `rda_config.rda_layers`, as it does from Kimi's `kda_layers`. It is not added to `QUANTIZED_LATENT_FAMILIES`, so an int8 latent request resolves to fp16 until an int8 trace is run. The fixed RDA state (about 47 MB per sequence) is not counted, which is the estimator's convention for recurrent layers.

The same commit fixed the decay for a null `gate_lower_bound`: it went through `compute_g`, which puts `exp(A_log)` outside the softplus, while the reference computes `-softplus(exp(A_log) * (a + dt_bias))`. Both published configs set -5.0, so their output is unchanged.

### 3.6 Native batched decode

**Context:** The first commit declared `BatchedDecodeKind::PerRowFallback`, like other recurrent hybrids. The server then ran one full forward per active sequence and read every weight, including the experts and the 260K-row head, once per row. Aggregate throughput stayed at about 30 tok/s regardless of concurrency.

**Decision (48a466487):** follow Kimi Linear and Kimi K3. RDA conv and recurrent states of the rows are stacked into one `[B, ...]` state per step and split back. MLA layers go through `KimiK3MLAAttention::forward_batched_decode` against each row's own latent cache, so rows of different lengths share the step. Multi-token steps, masks, and mixed fresh and started rows still run one row at a time. The two mHC sites and the FFN are shared between both paths through `forward_with_mixer`.

**Trade-off:** two of four concurrent greedy prompts matched their solo runs exactly; the other two diverged after about 280 characters, the usual bf16 effect of batched matmul shapes.

### 3.7 Use the MXFP8 experts as stored

The checkpoint's expert weights are uint8 E4M3 with uint8 E8M0 scales per 32 inputs, which is MLX's native `mxfp8` layout. Sanitize views the weights as uint32, renames `weight_scale_inv` to `scales`, and stacks the experts into `switch_mlp`, without changing a byte. A non-uint8 weight or scale, or experts of different shapes, is a load error.

## 4. Implementation Details

```
embed -> broadcast to 4 streams
  per layer:
    mHC(attn site): pre, post, R  -> x = norm(sum_s pre[s]*streams[s])
    mixer: RDA (18 layers, f32 state) | gated NoPE MLA (7 layers, latent KV)
    streams = R @ streams + post * branch      (f32, rounded once)
    mHC(mlp site) -> SiTU MLP or MoE (routed rounded, then + shared)
final: norm(mean over streams) -> lm_head
```

Main files: `src/models/raon_mhc.rs`, `src/models/raon_rda.rs`, `src/models/raon1_mini_moe.rs`, `src/models/raon1_mini.rs`, `src/models/raon1_mini_lm.rs`, `src/models/raon1_mini_config.rs`, `src/models/raon1_mini_sanitize.rs`; `src/models/gated_delta.rs` (new `gated_delta_ops_precise`); `src/models/kimi_k3.rs` (`NopeMlaSpec`); `src/execution/kv_arch.rs`; `scripts/raon1_mini_oracle_trace.py` (`generate` and `trace` modes, presence penalty forced to 0).

```rust
pub fn gated_delta_ops_precise(q, k, v, g, beta, state, mask)
    -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    gated_delta_ops_with_parity(q, k, v, g, beta, state, mask, false, true)
}
```

## 5. Learning Points

- **Find a precision bug by fixing the inputs.** Feeding bit-identical bf16 inputs to one layer and comparing its output against both an f32 and a bf16 reference separates "my inputs differ" from "my operator rounds differently". The 20x gap (0.045 against 0.0022) pointed straight at the recurrence.
- **bf16 cannot hold decays near 1.** With 8 bits of mantissa the spacing just below 1 is 2^-8 (about 0.0039). A per-token product of such decays turns a one-ulp error into a different memory horizon.
- **Check the oracle before the model.** A `None` argument that silently keeps a config default (here a 1.5 presence penalty) made the reference not greedy. Run the reference against itself (f32 against bf16, cached against uncached) before attributing a flip to the port.
- **Reference noise is a valid finding.** When the reference's f32 run agrees with the port and its bf16 variants agree with each other, the disagreement belongs to the reference's precision, and it can be exempted with evidence.

## 6. Related Issues

- #3225: Raon-1 mini support (closed by this PR)
- #3435: apply reasoning-effort steering vectors
- #3436: DFlash2 for Raon
- #3437: NVFP4 exports
- #3438: Raon-1 nano

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 33 |
| Lines added | +4324 |
| Lines deleted | -42 |

| Hash | Type | Message |
|------|------|---------|
| `028e5edea` | feat | add Raon-1 mini (raon1_mini) hybrid architecture |
| `4ed234d71` | fix | keep the Raon RDA decay and state in float32 |
| `45dee535d` | fix | size Raon-1 mini KV as a latent pair in its MLA layers |
| `f7c94abcc` | fix | run the Raon-1 mini greedy oracle without presence penalty |
| `999aa995e` | fix | refuse oversized Raon-1 mini geometry before it aborts |
| `48a466487` | fix | batch Raon-1 mini decode natively instead of per row |
| `a9e60c6fa` | test | cover the Raon-1 mini router, detection and export refusals |
| `8fb7c34f7` | docs | describe Raon-1 mini thinking control and final behavior |

## 8. Follow-up Actions

### Known gaps

- `presence_penalty` from `generation_config.json` is not read; the default stays 0.0 for all families. Reading it changes every family that sets it.
- Single-stream decode is dispatch-bound: each of the 50 mHC sites issues about 30 small ops per token, and each RDA layer runs 9 separate projections. Candidates: compile-fuse mHC `route` / `aggregate` / `mix`, and fuse q/k/v (and `b_proj` / `rda_w_proj`) into one matmul at load.
- Projection input widths are not checked at load (same gap as Kimi K3).
- Fast RMSNorm rounds `x * inv` to bf16 before the gamma multiply, where the reference rounds once (all families).
- `compute_g_lower_bounded` adds a bf16 `a` to the f32 `dt_bias` before casting, and the CUDA overlay rounds that sum to bf16 (shared with Kimi K3; Metal is exact).
- int8 latent KV for `raon1_mini` waits for an int8 trace.

## Appendix: Validation (M3 Ultra)

- `mlxcel arch` lists Raon-1 mini. The issue's `generate` command answers `42`; with the template's `enable_thinking=false` rendering through `--no-chat-template` it emits `42<|im_end|>`, the same ids as the reference.
- Greedy against Transformers (bf16, CPU, transformers 5.16.1, `set_reasoning_effort("max")`): identical on the short prompt and the 871-token prompt (43 tokens).
- Teacher-forced traces (`compare_logit_traces.py --decided 1`): 4 x 256 tokens, 0 of 578 decided positions disagree, perplexity +0.19%. Long prompt plus continuation: p99 / max logit delta 0.75 / 2.0, perplexity +0.34%, 1 of 549 disagrees (row 866, see 3.4).
- `mlxcel-server` streaming: thinking on puts the reasoning in `reasoning_content` and `42` in `content`; `chat_template_kwargs.enable_thinking=false` gives `42` with no reasoning.
- Raon unit tests, clippy, fmt, the `verify-*` contract targets and the full local `make verify` gate pass.
