# Jamba on OpenXLA/IREE (#3349)

Jamba interleaves Mamba1 blocks with attention blocks (`attn_layer_period` / `attn_layer_offset`) and alternates dense and MoE feed-forwards (`expert_layer_period` / `expert_layer_offset`). Every block has a feed-forward, so both mixer kinds take the general layer path: `x1 = x + mixer(input_layernorm(x)); out = x1 + ffn(pre_ff_layernorm(x1))`. Attention blocks use no RoPE. The KV cache covers only the attention blocks (2 rows on the 3B checkpoint) and the recurrent state covers the Mamba1 blocks (26 rows). Pure SSM layers in other families keep their FFN-free layout through `is_ffn_free_ssm_layer`.

## Weighted dt/B/C norms

`SsmKind::Mamba1` now carries `BcdtNorm::{None, Weightless, Weighted}`. Falcon-Mamba stays `Weightless`, and its golden is byte-identical. Jamba selects `Weighted` and loads `mamba.dt_layernorm.weight`, `mamba.b_layernorm.weight` and `mamba.c_layernorm.weight`. The checkpoint cross-check validates every `mamba.*` tensor including these vectors, `pre_ff_layernorm`, and the dense or stacked expert weights. `mamba_conv_bias = true` and `mamba_proj_bias = false` follow the config.

## MoE routing

Routing follows Hugging Face: softmax over every expert, then top-k, with no renormalization of the selected weights. The MLX eager reference (`src/models/jamba.rs`) takes softmax over the top-k logits only, which renormalizes; that path is unchanged. The XLA loader accepts the stacked mlx-lm `switch_mlp` expert layout. A raw per-expert HF MoE checkpoint is refused at load (`jamba_reference/cpu-raw-hf-moe-refused.log`). The real 3B checkpoint has `num_experts = 1`, so it is dense and unaffected.

## Softmax-MoE decode compile flags

With `--iree-dispatch-creation-enable-aggressive-fusion`, IREE 3.12.0rc20260721 fuses the router softmax into the tuple-reduction top-k argmax and cannot lower the result on llvm-cpu or CUDA (`'vector.store' op write affecting operations on global resources are restricted to workgroup distributed contexts`). This is not Jamba-specific. Compiling the existing `qwen2-moe-tiny` and `olmoe-tiny` `decode_ragged_logits_b4.mlir` goldens on llvm-cpu fails with the flag and succeeds without it (rechecked on main `c5afefbd`). `decode_compile_flags` therefore omits the flag when `Config::has_softmax_moe()` is true. Sigmoid routing (Nemotron-H, #3411) uses a single-result argmax and keeps the flag. Emitted graphs are unchanged. The flag choice is covered by `softmax_moe_selects_the_unfused_decode_compile` in the default lib tests; the iree-gated `softmax_moe_decode_drops_aggressive_fusion` is type-checked only, because the `mlxcel-xla` lib-test binary cannot link the IREE runtime on its own.

## Synthetic MoE fixture

The real checkpoint carries no MoE, so `spike/openxla/jamba_moe_fixture.py` (seed 3352) generates the only MoE evidence: four blocks covering Mamba1+MoE, attention+dense, Mamba1+dense and attention+MoE, four experts, top-2. The router and the dt/B/C norm weights are randomized. The generator asserts that every MoE block selects all six expert pairs across the reference trajectories, that resetting the norm weights to one moves the logits by 0.22, and that renormalizing the top-k weights moves them by 0.02. Both shifts are far above the 3e-4 parity tolerance, so neither omission can pass.

```sh
$MLXCEL_ORACLE_PYTHON spike/openxla/jamba_moe_fixture.py --out /tmp/epic564-models/jamba-moe-tiny
MLXCEL_JAMBA_FIXTURE=/tmp/epic564-models/jamba-moe-tiny MLXCEL_XLA_DEVICE=local-task \
  cargo test --profile test-fast --features xla-iree,test-utils --test xla_jamba \
  jamba_moe_hf_synthetic_parity -- --ignored --nocapture --test-threads=1
```

## Results (GB10, IREE 3.12.0rc20260721, f32)

First acceptance ran on runtime source `76a3dc4c`. After rebasing onto main `01e6516b` (Falcon-H1 plain RoPE #3452, Plamo2 #3449), the integrated runtime source `b9b5545a` repeated every row below on CUDA and the synthetic parity on local-task (`jamba_reference/integrated-*`). It also passed the Plamo2, Falcon-H1, Mamba1/Mamba2 SSM and Nemotron-H CUDA suites, kept every registered golden byte-exact, and was clippy clean with `--all-targets`. Commands and binary hashes for both runs are in `jamba_reference/provenance.json`.

| Check | Result |
|---|---|
| `validate_arch.sh --device cuda --batch 4 --requests 8 --maxcap 24 --oracle` on the 3B checkpoint | exit 0; goldens, 40/40 token-exact, batch reference-exact (170 tokens) |
| CLI vs native HF fp32 oracle, cuda | 40/40 |
| Server vs CLI displayed text, cuda | 4/4 non-streaming and 4/4 SSE, token counts equal |
| Synthetic MoE parity (single, ragged, slot recycle) | local-task max abs 2.0e-6, cuda 3.6e-6 (tolerance 3e-4) |
| Synthetic MoE CLI, `xla_oracle_check`, `xla_batch_bench` B=4 N=8 | token-exact and reference-exact on both devices |

The 3B model runs at about 6.4 tok/s single-sequence and 5.2 tok/s batched in f32 on GB10. These are acceptance runs, not performance measurements.

## Not verified

ROCm `hip` (external replay requested on the PR), a real MoE Jamba checkpoint, raw per-expert HF MoE layouts, quantized Jamba, and real Qwen2-MoE / OLMoE checkpoints after the decode flag change (their ragged decode did not compile with the flag before this change).

## External ROCm acceptance handoff

```sh
eval "$(scripts/iree/setup-rocm.sh --env)"
export MLXCEL_XLA_DEVICE=hip MLXCEL_XLA_ROCM_TARGET=gfx1151 MLXCEL_XLA_PRECISION=f32
scripts/rocm_gpu_guard.sh --hold -- scripts/xla/validate_arch.sh --model models/hf/jamba-3b \
  --device hip --batch 4 --requests 8 --maxcap 24 --oracle spike/openxla/jamba_reference/hf-f32-oracle.json
```

Use the original HF checkpoint `ai21labs/AI21-Jamba-Reasoning-3B` revision `7524370414253301341370beba35e3d549cbf5f3`, not an MLX conversion.
