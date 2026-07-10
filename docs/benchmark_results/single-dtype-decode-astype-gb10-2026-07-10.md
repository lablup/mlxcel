# Single-dtype decode graph: per-token AsType inventory on CUDA (GB10)

Issue #636. Host: GB10 (NVIDIA GB10, SM 12.1), CUDA backend, MLX pin per `main` at 70b2b29. All counts from `MLXCEL_TRACE_ASTYPE`, which walks the first decode step's unevaluated `(token, logprob)` graph and counts `AsType` nodes (traversal only, no extra eval). Cross-checked against `MLXCEL_EXPORT_DECODE_DOT` (`grep -c AsType` on the DOT) which agrees at 0.

## Headline finding

On CUDA the greedy quantized decode graph is already single-dtype: the three inventory models emit **0 AsType nodes per token**. The ~773 AsType/token figure in `moe-decode-gap-investigation.md` was measured on Apple/Metal (M1 Ultra); the CUDA-only `src/lib/mlx-cpp/patches-cuda/dtype.cpp` promotion patch (`bf16 + fp32 -> bf16`) already collapses the diffuse scalar/constant promotions that produced it. The counter tool merged here makes that state measurable and guards it against regression.

The only remaining reducible AsType source on CUDA is the temperature-sampler chain, addressed by building the fused sampler's scalar constants in the logit dtype.

## Before / after AsType per decode step

| Model | Path | Before | After | Reduction |
|-------|------|-------:|------:|----------:|
| llama-3.1-8b-4bit | greedy (temp 0) | 0 | 0 | already single-dtype |
| qwen3-8b-4bit | greedy (temp 0) | 0 | 0 | already single-dtype |
| qwen3-30b-a3b-4bit | greedy (temp 0) | 0 | 0 | already single-dtype |
| qwen3-8b-4bit | temp 0.8 + top-k 40 + top-p 0.9 | 4 | 1 | 75% |
| qwen3-8b-4bit | temp 0.8 + min-p 0.05 | 4 | 1 | 75% |

The residual `1` on the sampling path is the intrinsic `u32 -> f32` inside `mlx::core::random::categorical` (uniform bit-draw to float for the gumbel key), not a logit dtype round-trip.

On CUDA a natively-f16 checkpoint now runs temperature sampling in f16 rather than the previous f32 upcast. This is exactly the single-dtype behavior the issue asks for on CUDA, and it is within fp16 tolerance: an f16 softmax over well-separated logits on an inherently stochastic categorical draw. The scalar-dtype change is gated on `!metal::is_available()`, so Metal keeps its bare-f32-scalar behavior (the unpatched f16+f32 rule upcasts the chain to an f32 softmax) and its temperature-sampling numerics are unchanged.

## Inventory: where the AsType come from

| Source | Model class | Before | Disposition |
|--------|-------------|-------:|-------------|
| Model body (weights/norms/rope/attention) on CUDA | quantized dense + MoE | 0 | Already single-dtype via the `patches-cuda/dtype.cpp` bf16 promotion patch and the consistent-dtype quant path. Nothing to reduce. |
| Model body, bf16-native checkpoints on CUDA | dense bf16 | 0 | Single-dtype bf16; CUDA keeps bf16 (native ALUs). Optional f16 normalization available via env, no AsType change. |
| Fused sampler scalars (temperature, top-k/top-p/min-p sentinels) | any, temperature sampling | 3 (`f32 -> bf16`) | Removed on non-Metal backends: scalars built in the logit dtype so the CUDA promotion patch inserts no per-step conversion. Bit-identical for bf16 logits. Metal keeps the bare f32 scalars unchanged (gated on `!metal::is_available()`), so its numerics are untouched. |
| `random::categorical` uniform draw | any, temperature sampling | 1 (`u32 -> f32`) | Intrinsic to random generation; left as-is. |
| RMSNorm fp32 accumulation | gemma family (f16-fragile) | 52 (`f16 <-> f32`, 2 per layer) | Intentionally kept: gemma norms + softcap need the wider accumulation. On the f16-fragile exception list. |

## Quality: greedy parity

40-token greedy (temp 0), `MLXCEL_CUDA_F16_NORMALIZE` off vs on, generated text compared byte-for-byte:

- llama-3.1-8b-4bit: identical (quantized, env is a no-op).
- qwen3-30b-a3b-4bit: identical (quantized, env is a no-op).
- llama-3.1-8b-bf16: identical (bf16 vs f16-normalized both produce the same 40 tokens on this healthy family).

## Decode tok/s A/B (env on vs off)

`mlxcel-bench-decode --prompt x --prompt-tokens 512 --max-tokens 128`, two runs each:

| Model | off (bf16) | on | Note |
|-------|-----------:|---:|------|
| qwen3-30b-a3b-4bit | 92.44 / 90.59 | 92.84 / 91.63 | quantized: env is a no-op, within run-to-run noise |
| qwen2.5-0.5b-bf16 | 208.05 / 207.79 | 206.82 / 207.33 | bf16 vs f16 within noise, confirming CUDA native bf16 (no f16 throughput win) |

## Why CUDA f16 normalization is opt-in, not default

The single-dtype objective is already met in bf16 on CUDA, and CUDA has native bf16 compute, so casting bf16 -> f16 yields no AsType reduction and no throughput gain while narrowing dynamic range. The load-time f16 path (`MLXCEL_CUDA_F16_NORMALIZE`) is therefore off by default, with a conservative f16-fragile exception list (gemma, cohere/command-r, apertus, gpt-oss, and any softcap/`logit_scale` config) that keeps bf16 even when enabled. Metal/Apple Silicon numerics are untouched: the always-on Apple bf16 -> f16 policy is unchanged and the new path is gated on the CUDA backend.
