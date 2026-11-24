# Falcon-H1 on OpenXLA/IREE (#3346)

The raw Hugging Face layout uses parallel attention and Mamba2 branches from one normalized input. Their sum enters the first residual; a separately normalized SwiGLU MLP enters the second residual. This is distinct from the existing Cohere `parallel_block` (attention and MLP). The shared model-layer dispatcher applies the Falcon-H1 ordering to token prefill, single-sequence decode, and ragged decode. Every layer has both a compact KV row and a recurrent-state row. Existing recurrent-model guards disable prefix-hash prompt caches and default execution to f32.

The loader reads `model.embed_tokens`, `model.final_layernorm`, and per-layer `self_attn`, `mamba`, `feed_forward`, `input_layernorm`, and `pre_ff_layernorm`. `mamba_d_head` is the canonical SSM head width; `mamba_head_dim` is accepted as an alias when the canonical key is absent. Explicit `mamba_d_ssm` and `intermediate_size` take precedence over expansion factors. Attention and SSM head widths are independent.

Original HF weights have not folded muP. Embeddings and final logits receive their own factors; a tied head reads the original embedding matrix. Branch input factors precede their projections, the key factor precedes RoPE/cache writes, and branch output factors follow their output projections. The five SSM channel factors scale contiguous `[z,x,B,C,dt]` segments after the packed input projection. The MLP gate factor precedes SiLU and the down factor follows the down projection. Optional grouped SSM normalization supports both gating orders; absent normalization uses `y * silu(z)`. Input and output SSM biases use separate HF flags.

The first supported checkpoint is `tiiuae/Falcon-H1-Tiny-90M-Instruct` revision `e6389502a0b12cd8da894b395ba5bf7436873b16`. Its SSM width is 768, hidden width 512, attention head width 64, SSM head width 32, and SSM normalization is disabled. Sparse attention schedules, sliding windows, non-default RoPE scaling, and an explicit serialized `mup_vector` are rejected rather than approximated; HF derives its nonpersistent vector from `ssm_multipliers`.

## Reproducible independent gate

Generate three small checkpoints and fresh-prefix HF fp32 oracle trajectories using the pinned Transformers 5.12.1 CPU environment:

```sh
$MLXCEL_ORACLE_PYTHON spike/openxla/falcon_h1_oracle.py --out /tmp/falcon-h1-fixtures
MLXCEL_FALCON_H1_FIXTURES=/tmp/falcon-h1-fixtures MLXCEL_XLA_DEVICE=local-task \
  cargo test --profile test-fast --features xla-iree,test-utils --test xla_falcon_h1 \
  falcon_h1_hf_synthetic_parity -- --ignored --nocapture --test-threads=1
```

The ignored device test compares single-sequence tokens and four concurrent ragged trajectories with independent HF logits (absolute tolerance `3e-4`) and verifies slot re-admission. Fixtures use non-unit values for every multiplier, two SSM groups, unequal branch dimensions, both gating orders, no norm, and independent bias flags. On a GB10 host, wrap a CUDA invocation in `gpu-lock run --tag issue3346 --`; on ROCm use `scripts/rocm_gpu_guard.sh --hold --`.

These synthetic gates do not replace real-checkpoint acceptance: 40 oracle-exact CLI tokens, batch 4 / requests 8, four concurrent server completions, and the requested HIP `validate_arch.sh` gate remain mandatory before marking the issue complete.
