# MoE decode performance gap investigation (mlxcel vs mlx-lm)

Apple M1 Ultra (128 GB, macOS 26.5.1), MLX commit `a6ec712`, mlx-lm 0.31.3.
Decode throughput measured with `mlxcel-bench-decode` vs `scripts/bench_mlxlm.py`
on identical checkpoints: raw continuation prompt, 100 tokens after a warmup
pass, temp-0 greedy (so both runtimes emit identical tokens), best-of-3 with
cooldown. This document consolidates issues #266, #267, #268, #269.

## 1. Summary of the sweep

| Family | Checkpoint | mlxcel | mlx-lm | ratio |
|--------|-----------|-------:|-------:|------:|
| llama (dense) | llama-3.1-8b-4bit | 109.3 | 109.4 | 1.00× |
| granite (dense) | granite-3.3-2b | 184 | 186 | 0.99× |
| lfm2-moe | LFM2-8B-A1B-4bit | 184 | 184 | 1.00× |
| qwen3 (dense) | qwen3-8b-4bit | 75.3 | 83.8 | 0.90× |
| apertus (dense) | Apertus-8B-2509-4bit | 70 | 85 | 0.83× |
| seed_oss (dense) | Seed-OSS-36B-4bit | 17.1 | 20.3 | 0.84× |
| nemotron-h (hybrid+MoE) | nemotron-h-30b-4bit | 55 | 92 | 0.60× |
| qwen3_moe | qwen3-30b-a3b-4bit | 47.2 | 68.7 | 0.69× |
| dots1 | dots.llm1.inst-mixed-4-6bit | 13.2 | 25.7 | 0.51× |

Vanilla Llama/Granite dense is at parity. The gap grows with the number of
*extra small ops per layer* a model runs on top of that baseline: QK-norm
(qwen3), xIELU (apertus), and most of all the MoE router/combine/expert path.

## 2. SSM hybrids: fixed (#266, #267)

The three Mamba2 hybrids (falcon_h1, granitemoehybrid, plamo2) decoded 3-5×
slower than mlx-lm because each SSM mixer forced an **unconditional**
`eval(&result)` per layer, an M5-Max-NaN workaround that was never hardware
gated. Gating it on `hardware::is_m5_neural_accelerator()` (PR #266) restored
Falcon-H1 to ~3.8×, granite-h-tiny ~2.5×, plamo2 ~1.4× faster, output
byte-identical. nemotron_h carried the same eval (PR #267), but its decode runs
a fully-fused C++ path (`forward_fused`), so that gate only helped prefill;
nemotron's decode gap is MoE (below).

## 3. The MoE decode gap (#268): GPU-bound dtype/op overhead

Profiled `qwen3-30b-a3b` (the fast-iterating MoE proxy) with the
`mlxcel-gpu-profiling` hooks.

- **Step 0 (pipeline):** `async_eval` (GPU) 20.3 ms/tok vs `forward` (Rust graph
  build) 0.8 ms/tok → ~95% GPU-bound, not FFI/dispatch. `MLX_MAX_OPS_PER_BUFFER`
  swept 100-1000 is flat → not command-buffer-gap bound.
- **Step 1 (bandwidth):** ~3B active params at 4-bit ≈ 1.7 GB/tok; 47 tok/s ×
  1.7 GB ≈ 80 GB/s effective ≈ 16-20% of M1 Ultra GEMV peak. The GPU is mostly
  idle, spending its time on many small kernels rather than streaming weights.
  (MoE per-expert GEMVs underutilize the GPU on both runtimes; mlxcel worse.)
- **Step 2 (op-count diff), one decode token, 48 layers:**

  | Op | mlxcel | mlx-lm | excess |
  |----|------:|------:|------:|
  | **AsType** | **773** | **0** | **+773** |
  | Slice | 288 | 144 | +144 |
  | Reshape | 435 | 336 | +99 |
  | ExpandDims | 144 | 96 | +48 |
  | QuantizedMatmul | 145 | 241 | −96 |
  | total nodes | 3476 | 2505 | +971 |

mlx-lm emits zero dtype conversions; mlxcel emits 773/token (~16/layer), ~80% of
the node excess.

### Why the AsType are not a quick fix

Traced every candidate source:

- The Rust hot path has ≤6 explicit `astype` (all in the KV-cache turbo module).
- `quantized_linear_forward` (C++) calls `mlx::core::quantized_matmul` with no
  astype; `multiply_scalar`/`divide_scalar` already build the scalar in the
  input dtype.
- `convert_quant_scales_bf16_to_f16` promotes **all** `.scales`/`.biases`
  (including the quantized embedding) bf16→f16, and weights are bf16→f16 too, so
  activations and quantized params are **consistently f16**: no per-op dtype
  mismatch.
- The compiled activations (`compiled_swiglu_activation`, GeGLU, GELU) end in
  `astype(result, x.dtype())` but inside `mlx::core::compile`, so they fuse into
  one kernel and appear as `Compiled` nodes, not AsType.
- The fallback `rms_norm`/rope (which do cast) are not used; the model runs
  `fast_rms_norm`/`fast_rope`.

So the 773 AsType are diffuse MLX-internal type promotions (scalar/constant
arithmetic), not a single dominant source, and per MLX's eval-time elementwise
fusion they largely collapse into adjacent kernels. This is the same class as
the prior Gemma3n bf16 decode-gap investigation, where removing the AsType excess
measured only ~+2.6%; the static node count overstates GPU kernels.

## 4. Dense gaps (#269): same overhead, milder

apertus (0.83×) and seed_oss (0.84×) have no model-specific bug. llama-8b is at
parity, qwen3-8b (QK-norm) is 0.90×, apertus (QK-norm + xIELU) 0.83×. mlxcel's
xIELU is actually leaner than mlx-lm's (precomputed post-softplus scalars vs
runtime `softplus` on GPU arrays). The residual tracks the per-layer extra
small-op count, the same per-op execution overhead as the MoE case, just
smaller. Subsumed by #268.

## 5. Conclusion and realistic paths

The remaining gap is a single underlying phenomenon: mlxcel's per-op execution
overhead, amplified by op density. Vanilla dense ≈ parity; QK-norm/xIELU add a
little; MoE (router + combine + per-expert gather, plus the diffuse AsType) adds
the most. It is GPU-bound and the GPU is underutilized, but there is no single
fusable lever, confirmed by exhausting the AsType sources and by the Gemma3n
precedent.

Closing it meaningfully requires a deliberate effort, not a one-line change:

1. **Fused decode-MoE kernel**: collapse router scoring + top-k gather + the
   per-expert gate/up/down + weighted-sum into fewer Metal dispatches so the GPU
   stops idling between small kernels. Highest potential, largest effort
   (Metal/MLX-level), validate RMS < 5e-3.
2. **AsType reduction campaign**: eliminate the diffuse dtype promotions to keep
   the whole decode graph in one dtype like mlx-lm. Low risk per change, but the
   Gemma3n precedent caps the expected payoff (~few %); measure each step on
   qwen3-30b-a3b (cheap, 16 GB).
3. **Accept the gap** as the documented mlxcel characteristic on high-op-density
   decode (simpler models already run at 0.9-1.0× of mlx-lm).

Validation harness for any attempt: re-bench qwen3-30b-a3b per change, confirm on
dots.llm1 + nemotron-h, greedy temp-0 output byte-identical.
