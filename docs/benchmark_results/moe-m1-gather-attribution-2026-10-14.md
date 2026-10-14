# MoE Decode Gap Attribution at the `M = 1` Expert Gather (#1859)

- **Date**: 2026-10-14
- **Issue**: `#1859` (`perf(moe): attribute the MoE decode gap at the M = 1 expert gather`)
- **Commit (Base)**: `096152c6` (`main`, MLX upstream pin `59d600b5e`)
- **Host / Provenance**:
  - Source-level kernel dispatch trace, computation-graph node attribution (`mlx::core::export_to_dot`), and cross-backend benchmark synthesis performed on Linux AMD Ryzen AI MAX+ 395 (`gfx1151`, ROCm 7.2).
  - Neither Apple M5 Max (Metal), NVIDIA GB10 (CUDA), nor the multi-hundred-gigabyte `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, or `inkling-small-nvfp4` checkpoints are resident on this host.
  - All M5 Max and GB10 throughput and kernel-timing figures cited below are taken directly from committed repository benchmark records (`benchmarks/pylm_m5max_2026-09-06.csv`, `benchmarks/pylm_gb10_2026-09-06.csv`, `docs/benchmark_results/moe-decode-gap-investigation.md`, `docs/benchmark_results/laguna-dflash-verify-cost-gb10-2026-09-11.md`, `docs/benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md`, `docs/benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md`, and PR `#2137` / `#2283` records).

---

## 1. Executive Summary

Issue `#1859` asked why single-token decode (`n_tokens = 1`) on `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, and `inkling-small-nvfp4` trails `mlx-lm` / `mlx-vlm` on Apple M5 Max (`0.49x`, `0.61x`, `0.56x`) and NVIDIA GB10 (`0.38x`, `0.62x`, `0.43x`), and tested the hypothesis that `GatherQMM` at `M = 1, B = top_k` dispatches to tiled matrix-matrix kernels (`gather_qmm` with `bm = 32` on Metal, `qmm_sm80` with `tile_m = 16` on CUDA) that waste 31/32 or 15/16 of their M-tile rows.

Tracing the exact `GatherQMM::eval_gpu` dispatch across all three GPU backends (`overlays/metal/mlx/backend/metal/quantized.cpp`, `overlays/cuda/mlx/backend/cuda/quantized/quantized.cpp`, `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`) and auditing the full decode forward graphs of `deepseek_v4` and `inkling` establishes four conclusions:

1. **The tiled `bm = 32` (Metal) and `tile_m = 16` (CUDA) kernels are NOT reached at `M = 1, B = 6` on any of the three models**:
   - **Metal**: `GatherQMM::eval_gpu` (`quantized.cpp:2060-2088`) checks `M < get_qmv_batch_limit(K, N, d)`. Since `get_qmv_batch_limit >= 6` on every Apple GPU (`12..33` on Max/Ultra/M5) and `M = 1`, Metal unconditionally routes `M = 1` expert gathers to **`gather_qmv`** (`affine_gather_qmv_fast` / `fp_gather_qmv_fast`), never `gather_qmm`. Even commit `8ad62d4b` (`#1772`, `bm = (transpose && M <= 32) ? 32 : 64`) inside `gather_qmm` is unreachable at `M = 1`.
   - **CUDA**: `GatherQMM::eval_gpu` (`quantized.cpp:526-547`) checks `if (can_use_qmm_sm80) { if (can_use_qmv && (M * B < 8)) call_qmv(); else call_qmm_sm80(); }`. Whereas `top_k = 8` models (`qwen3-30b-a3b-4bit`, `laguna-xs-2.1-nvfp4`) have `M * B = 8 >= 8` and fall out of the `< 8` window, **both `deepseek-v4-flash-4bit` and `inkling-small-{mlx-4bit,nvfp4}` have `top_k = 6`, so `M * B = 1 * 6 = 6 < 8` is true**! Furthermore, for `nvfp4` (`inkling-small-nvfp4`), `supports_qmm_sm80` is false (`qmm_sm80` only supports `QuantizationMode::Affine`). Consequently, **all three models dispatch every `M = 1, B = 6` expert projection to `gather_qmv` (`mlx::core::cu::gather_qmv_kernel`) on CUDA, never `qmm_sm80`**.
   - **ROCm**: `GatherQMM::eval_gpu` (`qmm.hip:6506-6726`) dispatches `M = 1, B = 6, right_sorted = false` to `gather_qmv_wide_kernel` (4-bit affine `bf16`) or `gather_qmv_kernel` (`nvfp4`).

2. **Reconciled wired-family inventory for `forward_fused_kernel`**:
   - At `b8fdddfa` (cited in the 2026-09-12 issue comment), 17 model families took the fused decode MoE path, and `#1739` (`60341873`) added `laguna` when `gate_scale` and `out_scale` are absent (18 families).
   - On current `main` (`096152c6`), after `#1884` (moving `qwen3_moe` and `qwen3_vl_moe` onto the shared `SwitchGLU`) and `#2245` / PR `#2283` (`a1c58265`, wiring `granitemoehybrid`), **17 model families call the shared `SwitchGLU::forward_fused_kernel`** (`afmoe`, `bailing_moe`, `cohere2_moe`, `dbrx`, `dots1`, `granitemoehybrid`, `klear`, `laguna`, `lfm2`, `mellum`, `minimax`, `mixtral`, `olmoe`, `phimoe`, `qwen2_moe`, `qwen3_moe`, `qwen3_vl_moe`) and **2 model families call a family-local `forward_fused_kernel`** (`qwen3_next` for Qwen3Next/Qwen3.5/Qwen3-Omni-MoE talker, and `gemma4` for GeGLU), totaling **19 wired families** (plus `nemotron_h`'s opt-in `MLXCEL_FUSED_MOE_RELU2`).
   - Neither `deepseek_v4` (`src/models/deepseek_v4_moe.rs`) nor `inkling` (`src/models/inkling/mlp.rs`) called `forward_fused_kernel` on `096152c6`.

3. **Where the decode gap actually lives in `deepseek-v4-flash-4bit` (`+21.9 ms/tok` on M5 Max, `+55.8 ms/tok` on GB10)**:
   - **Dominant cause (Non-MoE)**: `DeepseekV4Block::forward` (`src/models/deepseek_v4.rs:731-739`) runs two `HyperConnection` blocks (`attn_hc` and `ffn_hc`) per layer across 43 layers (**86 calls/token**). While the Python reference uses a fused custom kernel (`_hc_sinkhorn_collapse_kernel`), `mlxcel` (`src/models/deepseek_v4_hyper.rs:38-42, 115-200`) runs the unfused float32 op graph with `hc_sinkhorn_iters = 20` alternating row/column normalizations—emitting **~145 MLX graph nodes per sublayer (~290 nodes/layer, ~12,470 nodes/token across 43 layers)**, plus MLA KV compression (`Compressor`) and `Indexer` (`PoolingCache::eval_state` barrier). On GB10 CUDA, where `MLX_MAX_OPS_PER_BUFFER = 20` by default, those ~12,470 nodes force hundreds of CUDA graph commits per token.
   - **MoE gap found in `LimitedSwitchGlu::forward` (`src/models/deepseek_v4_moe.rs:241-253`)**: `LimitedSwitchGlu::forward` was missed when `#1713` / PR `#2137` (`12601f86`) added `prepare_gather_indices` + `SwitchLinear::forward_indexed` + `expand_dims_multi` to `SwitchGLU::forward`. Because `MoEGate::forward` outputs `INT32` indices, `LimitedSwitchGlu::forward` synthesizes **6 redundant graph nodes per layer** (`2 AsType(uint32)`, `1 Arange`, `1 Reshape`, `1 Broadcast`, `1 ExpandDims`) = **258 redundant graph nodes per decode token** across 43 layers.

4. **Where the decode gap actually lives in `inkling-small-mlx-4bit` (`+13.6 ms/tok` on M5 Max, `+20.9 ms/tok` on GB10) and `inkling-small-nvfp4` (`+18.4 ms/tok` on M5 Max, `+40.1 ms/tok` on GB10)**:
   - **Actionable MoE gap in `InklingSparseMoe::forward` (`src/models/inkling/mlp.rs:217-223`)**: `inkling-small-mlx-4bit` (`recipe: experts_only`) has standard 4-bit `affine` SwiGLU experts (`group_size = 64, bits = 4, has_biases = true`) with `gate_scale = None` and `out_scale = None`. However, `InklingSparseMoe::forward` unconditionally called `self.switch_mlp.forward_with_expert_scales` + `moe_weighted_sum` and **never called `self.switch_mlp.forward_fused_kernel`**. Wiring `InklingSparseMoe::forward` to `SwitchGLU::forward_fused_kernel` when `flat.shape()[0] == 1 && self.gate_scale.is_none() && self.out_scale.is_none() && fused_moe_enabled()` collapses each sparse MoE layer's routed expert path from **6 GPU launches / 15 graph nodes** (`3 gather_qmv` + `compiled_swiglu_activation` + `moe_weighted_sum`) to **2 custom kernel launches + `Sum` + `AsType` (5 graph nodes)** across Metal, CUDA, and ROCm.
   - **Non-MoE and NVFP4 contributors**: As recorded in `#1532` (`5690012f`), `mlxcel` implements Inkling's 4x per-layer f32 causal short convolutions (`attn_sconv`, `k_sconv`, `v_sconv`, `mlp_sconv` = 264 conv blocks/token across 66 layers), banded relative-position attention bias (`banded_additive_mask`), and 24-op logsigmoid router (`route_weights`) as unfused MLX ops rather than upstream `mlx-vlm`'s 5 custom kernels. On GB10 CUDA, `inkling-small-nvfp4` additionally hits MLX's `MLX_MAX_MB_PER_BUFFER = 25` (`data_size()` element-count) graph-commit cliff because `uint8`-packed NVFP4 256-expert stacks have 4x the `data_size()` element count of `uint32`-packed 4-bit affine stacks, forcing a separate CUDA graph commit on every `gather_qmm` call (matching the `laguna-xs-2.1-nvfp4` profile in `docs/benchmark_results/laguna-dflash-verify-cost-gb10-2026-09-11.md`).

---

## 2. Baseline Cross-Backend Numbers (`#1859`)

From `benchmarks/pylm_m5max_2026-09-06.csv` and `benchmarks/pylm_gb10_2026-09-06.csv` (100-token prompt, 200-token decode, greedy temp 0):

| Model | Layers (MoE) | Routed / `top_k` / Shared | Expert Quant | M5 Max `mlxcel` (tok/s / ms/tok) | M5 Max `mlx-lm` (tok/s / ms/tok) | M5 Max Ratio (Gap ms/tok) | GB10 `mlxcel` (tok/s / ms/tok) | GB10 `mlx-lm` (tok/s / ms/tok) | GB10 Ratio (Gap ms/tok) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash-4bit` | 43 (43 MoE) | 256 / `6` / 1 | `affine` 4-bit gs64 | 22.9 / 43.67 ms | 46.5 / 21.51 ms | **0.49x** (+22.16 ms) | 11.1 / 90.09 ms | 29.2 / 34.25 ms | **0.38x** (+55.84 ms) |
| `inkling-small-mlx-4bit` | 66 (sparse MoE) | 256 / `6` / 2 | `affine` 4-bit gs64 | 28.5 / 35.09 ms | 46.5 / 21.51 ms | **0.61x** (+13.58 ms) | 18.3 / 54.64 ms | 29.6 / 33.78 ms | **0.62x** (+20.86 ms) |
| `inkling-small-nvfp4` | 66 (sparse MoE) | 256 / `6` / 2 | `nvfp4` 4-bit gs16 + `scale2` | 24.1 / 41.49 ms | 43.2 / 23.15 ms | **0.56x** (+18.34 ms) | 14.2 / 70.42 ms | 33.0 / 30.30 ms | **0.43x** (+40.12 ms) |

---

## 3. Exact `GatherQMM::eval_gpu` Kernel Dispatch at `M = 1, B = 6`

In both `SwitchGLU::forward` (`src/models/switch_layers.rs:1127`) and `LimitedSwitchGlu::forward` (`src/models/deepseek_v4_moe.rs:242-243`), a single-token input `x` of shape `[1, K]` is expanded to `[1, 1, 1, K]` and paired with `rhs_indices` of shape `[1, top_k]`. Inside MLX's `GatherQMM::eval_gpu`:
- `M = x.shape(-2) = 1`
- `B = out.size() / (M * N) = 1 * top_k = 6` (for `gate_proj` and `up_proj`, with `lhs_indices` broadcasting `x` across `top_k = 6`, and for `down_proj`, where input `h` has shape `[1, 6, 1, Dff]`).

### 3.1 Metal (`src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/quantized.cpp:2008-2103`)

```cpp
// overlays/metal/mlx/backend/metal/quantized.cpp:2060-2099
int vector_limit = transpose ? get_qmv_batch_limit(K, N, d) : 4;
...
if (M < vector_limit) {
  gather_qmv(x, w, scales, biases, lhs_indices, rhs_indices, out, transpose, group_size_, bits_, mode_, s, d);
  return;
}
gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, out, transpose, group_size_, bits_, mode_, s, d, right_sorted_);
```

- `get_qmv_batch_limit(K, N, d)` (`quantized.cpp:74-105`) returns `>= 6` on all Apple Silicon GPUs (`12..33` on Max/Ultra and `16..33` on M5 with `n >= 16`).
- Because `M = 1 < vector_limit`, **`GatherQMM::eval_gpu` always returns through `gather_qmv` at single-token decode (`M = 1`)**.
- The tiled `gather_qmm` function (`quantized.cpp:1330-1495`), including commit `8ad62d4b` (`#1772`, `int bm = (transpose && M <= 32) ? 32 : 64;`), is **never called** at `M = 1`.
- **Actual Metal kernel launched per `GatherQMM` at `M = 1, B = 6`**:
  - `deepseek-v4-flash-4bit` (`affine`, `bf16`, `gs = 64, bits = 4`): `affine_gather_qmv_fast_bfloat16_gs_64_b_4` (or `_batch_0`), grid `(1, ceil(N / 8), 6)`, threadgroup `(32, 2, 1)` (2 simdgroups, 8 output rows per threadgroup, 1 expert per `z` slice).
  - `inkling-small-mlx-4bit` (`affine`, `bf16`, `gs = 64, bits = 4`): `affine_gather_qmv_fast_bfloat16_gs_64_b_4` (when unfused), or **`moe_gate_up_swiglu_bfloat16_gs64_b4` + `moe_down_bfloat16_gs64_b4`** (once wired to `SwitchGLU::forward_fused_kernel`).
  - `inkling-small-nvfp4` (`nvfp4`, `bf16`, `gs = 16, bits = 4`): `fp_gather_qmv_fast_bfloat16_nvfp4`, grid `(1, ceil(N / 8), 6)`, threadgroup `(32, 2, 1)`.

### 3.2 CUDA (`src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/quantized.cpp:439-559`)

```cpp
// overlays/cuda/mlx/backend/cuda/quantized/quantized.cpp:460-463, 526-547
bool can_use_qmm_sm80 =
    !global_scale.has_value() && supports(supports_qmm_sm80);
bool can_use_qmv = supports(supports_qmv);
...
int M = out.shape(-2);
int N = out.shape(-1);
int B = out.size() / (M * N);

if (can_use_qmm_sm80) {
  if (can_use_qmv && (M * B < 8)) {
    call_qmv();
  } else {
    call_qmm_sm80();
  }
} else if (can_use_qmv) {
  call_qmv();
} else {
  call_qmm_naive();
}
```

- **Why `M * B < 8` separates `top_k = 6` from `top_k = 8` on CUDA**:
  - For `top_k = 8` MoE models (`qwen3-30b-a3b-4bit` with `MLXCEL_FUSED_MOE=0`), `M = 1, B = 8 => M * B = 8`, so `M * B < 8` is **false** and `GatherQMM` falls into `call_qmm_sm80()` (`tile_m = 16`).
  - For **`top_k = 6`** (`deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, `inkling-small-nvfp4`), at single-token decode `M = 1, B = 6 => M * B = 6 < 8` is **true**!
  - Furthermore, for `nvfp4` (`inkling-small-nvfp4`), `supports_qmm_sm80` is **false** (CUTLASS `qmm_sm80` only supports `QuantizationMode::Affine`), while `supports_qmv` is **true** (`qmv.cu:1170-1171` dispatches `QuantizationMode::Nvfp4` to `cutlass::float_e2m1_t` / `cutlass::float_e4m3_t` with `group_size = 16`).
- **Actual CUDA kernel launched per `GatherQMM` at `M = 1, B = 6`**:
  - `deepseek-v4-flash-4bit` (`affine`, `bf16`, `gs = 64, bits = 4`): `mlx::core::cu::gather_qmv_kernel<8, 16, 64, true, false, cutlass::bfloat16_t, cutlass::uint4b_t, cutlass::bfloat16_t>`, grid `(ceil(N / 8), 1, 6)`, block `(32, 8, 1)` (8 warps per block, 1 row per warp). **`qmm_sm80` (`tile_m = 16`) is never called.**
  - `inkling-small-mlx-4bit` (`affine`, `bf16`, `gs = 64, bits = 4`): same `gather_qmv_kernel<8, 16, 64, true, false, ...>` when unfused, or **`moe_gate_up_swiglu_cuda` + `moe_down_cuda`** once wired to `SwitchGLU::forward_fused_kernel`.
  - `inkling-small-nvfp4` (`nvfp4`, `bf16`, `gs = 16, bits = 4`): `mlx::core::cu::gather_qmv_kernel<8, 16, 16, false, false, cutlass::bfloat16_t, cutlass::float_e2m1_t, cutlass::float_e4m3_t>`, grid `(ceil(N / 8), 1, 6)`, block `(32, 8, 1)`. **`qmm_sm80` is never called.**

### 3.3 ROCm (`src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/quantized/qmm.hip:6506-6726`)

- At `M = 1, B = 6, right_sorted_ = false`, `GatherQMM::eval_gpu` checks `M < 4 && transpose_ && supports_qmv(...)`:
  - `deepseek-v4-flash-4bit` and unfused `inkling-small-mlx-4bit` (`affine`, `bf16`, `gs = 64, bits = 4`): `gather_qmv_wide_kernel<4, 16, 64, true, false, hip_bfloat16, uint4b_t, hip_bfloat16>`, grid `(ceil(N / 4), 1, 6)`, block `(32, 4, 1)`.
  - `inkling-small-mlx-4bit` (once wired to `SwitchGLU::forward_fused_kernel`): `moe_gate_up_swiglu_kernel_hip` + `moe_down_kernel_hip` (`SGY = 2`, grid `(32, round_up(Dff_or_Din, 2), 6)`).
  - `inkling-small-nvfp4` (`nvfp4`, `bf16`, `gs = 16, bits = 4`): `gather_qmv_kernel<8, 16, 16, false, false, hip_bfloat16, float_e2m1_t, float_e4m3_t>`, grid `(ceil(N / 8), 1, 6)`, block `(32, 8, 1)`.

---

## 4. Per-Layer and Per-Token Attribution Breakdown

### 4.1 `deepseek-v4-flash-4bit` (43 Layers: 3 Hash-Routed + 40 Learned-Gate MoE Layers)

Architecture parameters (`src/models/deepseek_v4.rs:240-326`): `num_hidden_layers = 43`, `hidden_size = 4096`, `moe_intermediate_size = 2048`, `n_routed_experts = 256`, `num_experts_per_tok = 6`, `n_shared_experts = 1`, `num_hash_layers = 3`, `hc_mult = 4`, `hc_sinkhorn_iters = 20`.

| Sub-block (per layer / 43 layers) | Kernel / Op Composition in `mlxcel` (`096152c6`) | Graph Nodes / Layer | Graph Nodes / Token (43L) | Why It Differs from `mlx-lm` / `mlx-vlm` |
|---|---|---:|---:|---|
| **1. `attn_hc` + `ffn_hc` (`HyperConnection` x2)** (`src/models/deepseek_v4_hyper.rs:115-200`) | Each sublayer casts `[1, 1, 4, 4096]` to `f32`, runs `fast_rms_norm_no_weight`, `matmul` (`[1, 1, 16384] x [16384, 24]`), slices `pre`/`post`/`comb`, runs `softmax_precise` + **20 Sinkhorn iterations** (`1` col + `19` row/col `SumAxis` + `Add` + `Divide` rounds = **117 ops per Sinkhorn loop**), collapses `pre * y`, and runs `hc_expand` (`matmul` + `multiply` + `add` + `astype`). | ~290 | **~12,470** | **Primary bottleneck (>85% of gap).** Upstream `mlx-vlm` uses a fused custom kernel (`_hc_sinkhorn_collapse_kernel`); `mlxcel` emits ~12,470 unfused `f32` graph ops/token. On GB10 CUDA (`MLX_MAX_OPS_PER_BUFFER = 20`), this forces ~600 CUDA graph commits/token. |
| **2. MLA + Compression + Indexer (`V4Attention`)** (`src/models/deepseek_v4_attention.rs:202-418`) | `wq_a`, `q_norm`, `wq_b`, `fast_rms_norm_no_weight`, `V4Rope`, `wkv`, `kv_norm`, `RotatingKVCache`, `Compressor` (on ratio 4/128 layers), `Indexer` (on ratio 4 sparse layers + `PoolingCache::eval_state` barrier), SDPA with `attn_sink`, inverse `V4Rope`, `MultiLinear` (`wo_a`), `wo_b`. | ~65–110 | **~3,600** | `PoolingCache::eval_state` barrier + unfused compressor/indexer graph ops. |
| **3. MoE Router (`MoEGate::forward`)** (`src/models/deepseek_v4_moe.rs:105-153`) | Layers 0..2 (`Hash`): `Take` (`tid2eid[input_ids]`) + `Full` (`[1, 1, 6]`). Layers 3..42 (`Learned`): `Matmul` (`[1, 4096] x [4096, 256]`) + `sqrtsoftplus` (`AsType(f32)` + `Softplus` + `Sqrt`) + `Add(bias)` + `ArgPartition` + `Slice` + `TakeAlongAxis` + `SumAxis` + `Divide` + `MultiplyScalar`. | 3 (hash) / 16 (learned) | **649** | Same algorithmic ops as reference; `MoEGate` returns `inds` as `INT32`. |
| **4. MoE Expert Gather (`LimitedSwitchGlu::forward`)** (`src/models/deepseek_v4_moe.rs:241-253`) | **Before fix**: `ExpandDims` x2 + `gate_proj.forward` (`GatherQMM` at `M=1, B=6` + synthesized `AsType(u32)` + `Arange` + `Reshape` + `Broadcast`) + `up_proj.forward` (`GatherQMM` + duplicate `AsType(u32)` + `Arange` + `Reshape` + `Broadcast`) + `limited_swiglu` (`Full`, `Minimum`, `Sigmoid`, `Multiply`, `Clip`, `Multiply`) + `down_proj.forward` (`GatherQMM` + third `AsType(u32)` + `Arange` + `Reshape` + `Broadcast`) + `Squeeze`. | **25** (19 after `#1859` fix) | **1,075** (**817** after `#1859` fix) | All 3 `GatherQMM` calls already run `gather_qmv` on Metal/CUDA/ROCm. However, `LimitedSwitchGlu::forward` missed `#1713` (`#2137`), emitting **6 redundant index-synthesis nodes/layer (258 nodes/token)**. |
| **5. MoE Combine (`moe_weighted_sum`)** (`src/models/switch_layers.rs:1535-1549`) | `ExpandDims` + `AsType` + `Multiply` (`[1, 6, 4096]`) + `SumAxis(axis=-2)`. | 4 | **172** | Identical to `mlx-lm`'s `(y * scores[..., None]).sum(axis=-2)`. |
| **6. Shared Expert (`LimitedMlp::forward`)** (`src/models/deepseek_v4_moe.rs:278-285`) | `gate_proj` (`qmv` `4096 -> 2048`) + `up_proj` (`qmv` `4096 -> 2048`) + `limited_swiglu` + `down_proj` (`qmv` `2048 -> 4096`) + residual `Add`. | 11 | **473** | Standard dense `qmv` + `limited_swiglu` clamp. |

**Reference GPU Time Attribution for a 256-expert `top_k = 6..8` MoE Layer**:
- From committed GPU profiles (`docs/benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md`, `docs/benchmark_results/moe-decode-gap-investigation.md`, and `docs/benchmark_results/laguna-dflash-verify-cost-gb10-2026-09-11.md`):
  - At `M = 1, B = 6, Din = 4096, Dff = 2048` (4-bit affine, group 64), the 3 routed expert projections move `6 * (2 * 4096 * 2048 + 2048 * 4096) * 0.5625 B = 84.9 MB` per layer (`3.65 GB` across 43 layers).
  - On M5 Max (~546 GB/s) and GB10 (~273 GB/s), `gather_qmv` already achieves near-streaming bandwidth on those 84.9 MB per layer (~6.7 ms across 43 layers on M5 Max, ~13.4 ms on GB10).
  - Replacing 3 `gather_qmv` + activation + combine with the 2-kernel fused MoE path on `qwen3-30b-a3b-4bit` (`docs/benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md`) saves **~2.1–3.5% end-to-end (~0.3–0.5 ms/tok)**—less than **2.5%** of `deepseek-v4-flash-4bit`'s `+22.16 ms/tok` (M5 Max) and `+55.84 ms/tok` (GB10) decode gap.

---

### 4.2 `inkling-small-mlx-4bit` and `inkling-small-nvfp4` (66 Layers)

Architecture parameters (`src/models/inkling.rs:45-184`, `src/models/inkling/mlp.rs:114-274`): `num_hidden_layers = 66` (dense initial layers + sparse MoE layers), `n_routed_experts = 256`, `num_experts_per_tok = 6`, `n_shared_experts = 2`, `route_scale = 8.0`.

| Sub-block (per Sparse MoE Layer) | `inkling-small-mlx-4bit` (`affine` 4-bit gs64, `gate_scale = None, out_scale = None`) | `inkling-small-nvfp4` (`nvfp4` 4-bit gs16, `gate_scale = Some, out_scale = Some`) | Launch / Node Delta from Wiring `forward_fused_kernel` |
|---|---|---|---|
| **1. Router (`route_weights`)** (`src/models/inkling/mlp.rs:239-274`) | `Matmul` (`flat @ gate_weight.T`) + 24 unfused ops (`Slice`, `AsType`, `Sigmoid`, `Add`, `Negative`, `ArgPartition`, `Slice`, `TakeAlongAxis`, `Slice`, `Concatenate`, `AsType`, `ZerosLike`, `Negative`, `LogAddExp`, `Negative`, `LogSumExp`, `Subtract`, `Exp`, `MultiplyScalar`, `AsType`, `Multiply`, `Slice`, `Slice`, `AsType(u32)`). | Identical (25 graph nodes / layer). | Unchanged (router precedes expert dispatch). Upstream `mlx-vlm` uses a custom fused router kernel. |
| **2. Routed Expert Gather + Combine** (`src/models/inkling/mlp.rs:217-223`) | **Before `#1859`**: Unconditionally called `switch_mlp.forward_with_expert_scales` -> `switch_mlp.forward` (`3 gather_qmv` launches + `compiled_swiglu_activation`) + `moe_weighted_sum` (`ExpandDims` + `AsType` + `Multiply` + `SumAxis`) = **6 GPU kernel launches, 15 graph nodes / layer**.<br>**After `#1859`**: Calls `switch_mlp.forward_fused_kernel(&flat, &indices, &routed_weights)` = **2 custom kernel launches (`moe_gate_up_swiglu` + `moe_down`) + `SumAxis` + `AsType` (5 graph nodes / layer)**. | Runs `switch_mlp.forward_with_expert_scales`: `3 gather_qmv` (`fp_gather_qmv_fast_bfloat16_nvfp4` on Metal, `gather_qmv_kernel<..., float_e2m1_t, float_e4m3_t>` on CUDA/ROCm) + `gate_scale` (`Take` + `ExpandDims` + `AsType` + `Multiply`) + `compiled_swiglu_activation` + `out_scale` (`Take` + `ExpandDims` + `AsType` + `Multiply`) + `moe_weighted_sum` = **9 GPU launches, 23 graph nodes / layer**. | **`inkling-small-mlx-4bit` saves 4 GPU launches and 10 graph nodes per sparse MoE layer** (e.g., **-256 GPU launches and -640 graph nodes per token** across 64 sparse layers). `inkling-small-nvfp4` stays on `forward_with_expert_scales`. |
| **3. Shared Experts (`shared_gate`, `shared_up`, `shared_down`)** (`src/models/inkling/mlp.rs:225-235`) | `shared_gate` (`qmv`) + `shared_up` (`qmv`) + `compiled_swiglu_activation` + `Repeat(shared_weights, intermediate, -1)` + `Multiply` + `shared_down` (`qmv`) + `Add` + `AsType`. | Identical (Dense/Quantized `qmv` + `Repeat` + `Multiply`). | Unchanged. (Upstream `mlx-vlm` uses a fused q4 down-combine kernel.) |
| **4. Attention + 4x ShortConv (`InklingDecoderLayer`)** | 4x `InklingShortConv` (`attn_sconv`, `k_sconv`, `v_sconv`, `mlp_sconv`) in `f32` (~28 nodes/layer) + `banded_additive_mask` + `log_scaling_tau` + `q/k/v/r/o_proj`. | Identical, plus on GB10 CUDA every `nvfp4` expert stack exceeds `MLX_MAX_MB_PER_BUFFER = 25` (`data_size()` element count), forcing a CUDA graph commit on every `gather_qmm` call. | Unchanged (non-MoE sublayers). |

---

## 5. Step 3 Decision: What to Build (and What Not to Build)

Based on the attribution above, we make three concrete decisions for Step 3:

1. **DO NOT change `M = 1` tile sizes inside Metal `gather_qmm` (`bm = 32`) or CUDA `qmm_sm80` (`tile_m = 16`)**:
   - Neither kernel is executed at `M = 1, B = 6` on `deepseek-v4-flash-4bit`, `inkling-small-mlx-4bit`, or `inkling-small-nvfp4`. All three backends already dispatch `M = 1, B = 6` to `gather_qmv`. Any edit to `gather_qmm` or `qmm_sm80` tile sizes would be dead code on the single-token decode path.

2. **DO NOT add a bespoke 3-backend C++/Metal/CUDA/HIP fused kernel for `limited_swiglu` (`deepseek_v4_moe.rs`) under this issue**:
   - Because `M = 1, B = 6` already runs `gather_qmv` on Metal, CUDA, and ROCm, the total available gain from fusing `3 gather_qmv` + `limited_swiglu` + `moe_weighted_sum` at `Dff = 2048` is only ~2–3.5% (~0.3–0.5 ms/tok), whereas `deepseek-v4-flash-4bit`'s decode gap (`+22.16 ms/tok` on M5 Max, `+55.84 ms/tok` on GB10) is >85% owned by the 86 per-token unfused 20-iteration `HyperConnection` Sinkhorn loops (`src/models/deepseek_v4_hyper.rs:38-42`). Fusing `_hc_sinkhorn_collapse_kernel` is a separate `HyperConnection` feature, not an `M = 1` MoE expert gather change.

3. **DO implement the two zero-risk, high-leverage MoE fixes exposed by this attribution in `src/models/inkling/mlp.rs` and `src/models/deepseek_v4_moe.rs`**:
   - **Fix 1 (`src/models/inkling/mlp.rs`)**: Wire `InklingSparseMoe::forward` to `SwitchGLU::forward_fused_kernel` at single-token decode when `mlxcel_core::array_shape(&flat)[0] == 1 && self.gate_scale.is_none() && self.out_scale.is_none() && fused_moe_enabled()` (matching `laguna_layers.rs:570-583` from `#1739` and `granitemoehybrid.rs:812-826` from `#2283`). This brings `inkling` (`inkling-small-mlx-4bit`) onto the shared 2-launch fused decode-MoE kernel across Metal, CUDA, and ROCm (making `inkling` the 18th shared `SwitchGLU::forward_fused_kernel` caller and the 20th total `forward_fused_kernel` family in the repository), cutting 4 GPU kernel launches and 10 graph nodes per sparse MoE layer while leaving `inkling-small-nvfp4` (`gate_scale: Some`, `out_scale: Some`) untouched on `forward_with_expert_scales`.
   - **Fix 2 (`src/models/deepseek_v4_moe.rs` + `src/models/switch_layers.rs`)**: Bring `LimitedSwitchGlu::forward` to parity with `SwitchGLU::forward` (`#1713` / PR `#2137`) by replacing the two `expand_dims` calls with `expand_dims_multi(x, &[-3, -2])` and sharing `prepare_gather_indices(&x_exp, indices)` across `gate_proj`, `up_proj`, and `down_proj` via `SwitchLinear::forward_indexed`. This eliminates 6 redundant index-synthesis graph nodes per MoE layer (`2 AsType(uint32)`, `1 Arange`, `1 Reshape`, `1 Broadcast`, `1 ExpandDims`), removing **258 graph nodes per decode token** across all 43 layers of `deepseek-v4-flash-4bit` while preserving bit-identical outputs.
