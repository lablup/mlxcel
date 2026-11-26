# PLaMo 2 on OpenXLA/IREE (#3347)

PLaMo 2 (`model_type: plamo2`) alternates a Mamba2-variant mixer with attention. With the default `mamba_step = 2`, layer `i` is Mamba when `i % 2 != 1`, so pfnet/plamo-2-1b has eight Mamba blocks and eight attention blocks. Attention layers outside `full_attention_idx` use a 2048-token sliding window. Every block has an MLP. KV rows exist only for attention layers and recurrent state only for Mamba layers, so the existing hybrid contract (`layer_mixers`, `attn_layer_index`, `state_layer_index`) carries the model without new runtime code.

## Block structure

- Four norms per block: `pre_mixer_norm`, `post_mixer_norm`, `pre_mlp_norm` and `post_mlp_norm`. Each computes `(offset + weight) * rmsnorm(x)`, with offsets 1, 1/5, 1 and 1/5^1.5. The final norm has offset 1. Offsets are added to the weights in the graph, and the shared `GemmaFf` four-norm placement then runs, so other families' graphs are unchanged. The f32 value of 1/5^1.5 is 0.08944272.
- Attention: fused `qkv_proj`; weightless per-head RMSNorm on Q and K with eps fixed at 1e-6; a learned `[heads, head_dim]` scale; split-half RoPE using `rope_local_theta` on local layers and `rope_theta` on full layers.
- Mamba mixer (`SsmKind::Plamo2`, `ssm_plamo2.rs`), built from the shared conv and selective-scan helpers without touching the SSD internals or Mamba1:
  - `in_proj` output is per-head interleaved: the first P lanes of each head are z, the next P are x.
  - Depthwise causal conv with K=4, no bias, then SiLU.
  - `bcdt_proj` is split into `[B N, C N, dt R]` with R = max(64, hidden/16).
  - Each piece gets a weighted RMSNorm (plain weight, no offset), then `dt_proj`.
  - Mamba2 scan with one group, then `out_proj(y * silu(z))`. There is no gated norm.

## Sliding window

HF keeps keys with `age <= attention_window_size` (it builds the mask with `triu(diagonal=-window)`). All three XLA local masks keep `age < sliding_window`, so Plamo2 stores `sliding_window = W + 1`. No shared mask code changed.

The probe uses an attention-only PLaMo 2 with one local and one full layer, hidden 32, head-specific Q/K scales and distinct norm weights. A boundary token with a large V gain moves logits by about 0.4. Prompts of 2047 to 2051 tokens plus six greedy steps push the boundary across age 2048 in prefill, decode and ragged decode. The same weights are also loaded with W = 2047 and W = 2049. Each fixture must follow the float64 reference with its own mask (inclusive, exclusive or one-wider), and every variant's trajectory must differ from the others. Worst delta is 4.2e-6 on local-task and 5.4e-6 on CUDA. The nearest wrong variant differs by 5.9e-3 at the first step and by tokens after that.

## RoPE scaling bug

The first probe run was off by about 0.24 at 2047 tokens, where no window bites. The config inherited `RopeScaling::Llama3` (factor 32, original context 8192) from the Llama-3.2 base. That scaling changes only the low-frequency bands, so the short fixtures and the 45-position oracle could not see it. The config now uses plain RoPE, and a unit test asserts it. A float32 rerun of the reference differed from float64 by 4e-6, which rules out ill-conditioning as the cause. Falcon-H1 (`falcon_h1_config.rs`) builds on the same base without overriding `rope` and needs the same check.

## dt clamp

MLX eager clamps `softplus(dt + dt_bias)` to (0.001, 100); the HF CPU path does not. The emitter and the PyTorch reference follow MLX. Half the synthetic heads use `dt_bias = -9`, so the clamp floor is active. Removing the clamp moves logits by 4e-3 to 4e-2, which the 3e-4 tolerance catches. Padded prefill rows get `dt = 0` after the clamp.

## Oracle

The HF remote code imports CUDA-only `mamba_ssm` and `causal_conv1d`. The 40-token oracle is therefore the MLX eager CPU run: F32 weights, revision 92c75fd6eea9018bcb9c33ee8921589febe071fa, prompt "The capital of France is". Structural independence comes from `plamo2_oracle.py`, a float64 reimplementation of the HF formulas that loads weights by HF names and runs a naive sequential scan.

## Reproduce

```sh
$MLXCEL_ORACLE_PYTHON spike/openxla/plamo2_oracle.py --out /tmp/plamo2-fixtures
export MLXCEL_PLAMO2_FIXTURES=/tmp/plamo2-fixtures MLXCEL_XLA_DEVICE=local-task MLXCEL_PLAMO2_MODEL=/path/to/plamo-2-1b MLXCEL_PLAMO2_ORACLE=$PWD/spike/openxla/plamo2_reference/oracle.json
cargo test --profile test-fast --features xla-iree,test-utils --test xla_plamo2 -- --ignored --nocapture --test-threads=1
```

On GB10, wrap CUDA runs in `gpu-lock run --tag <purpose> --`; on ROCm use `scripts/rocm_gpu_guard.sh --hold --`.

## Results (GB10, f32)

| Gate | local-task | CUDA |
|---|---|---|
| Real-checkpoint 40-token oracle | 40/40 | 40/40 |
| Hybrid synthetic parity (single, ragged, slot recycle) | worst 6.9e-6 | worst 1.0e-5 |
| 2048-window probe (prefill, decode, ragged) | worst 4.2e-6 | worst 5.4e-6 |
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24 --oracle` | not run | exit 0 |
| CLI vs server (4 HTTP + 4 SSE) | not run | 8/8 text and token counts |

ROCm `hip` is not verified here.

## Scope limits

Rejected rather than approximated: quantized checkpoints, MoE fields (`n_expert`, `k_expert`, `sparse_step`), `eval_attention_n_bit` and `eval_mlp_n_bit`, `image_token_id`, `rope_scaling`, `use_predefined_initial_state`, and `mamba_step < 2`. The prompt cache and APC are refused for the model type.
