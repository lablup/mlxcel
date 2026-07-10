# Hybrid-SSM decode: CUDA port of the fused SSM update kernel (issue #631, GB10)

Date: 2026-07-10. Host: NVIDIA GB10 (Grace-Blackwell, sm_121 / cc 12.1), CUDA 13.0, MLX 0.32.1 (pin `57c66cac`).
Binary: `cargo build --release --features cuda`. Bench: `mlxcel-bench-decode` (64 decode tokens, short prompt), `mlxcel generate` (greedy parity).

## TL;DR

Hybrid SSM/attention models decoded at 0.29-0.36x of Metal on GB10 because the fused single-token SSM update kernel (`ssm_update_kernel`, "replaces ~55 individual ops with a single kernel call") was Metal-only: `ssm_kernel_available()` returned false off Apple, so every CUDA decode step ran the ~55-op `ssm_step` graph path per SSM layer. nsys on granite-4.0-h-350m showed thousands of 1-2 us elementwise/copy kernels per token with the GPU busy only ~30% of decode wall time. Porting the kernel to `mx.fast.cuda_kernel` (same thread mapping, `__shfl_down_sync` warp reduction instead of `simd_sum`) lifts granite-4.0-h-350m decode **62.8 -> 281.7 tok/s (4.5x)**; granite and falcon-h1 now beat their Metal M1 Ultra reference numbers and nemotron-h-30b reaches 0.96x of its Metal reference (from 0.45x). Greedy parity is byte-identical with the graph path on all three tested models; the mamba2 control is untouched.

## Root cause

- `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp` has a fused SSM decode kernel used by the hybrid models (granitemoehybrid, falcon_h1, plamo2, nemotron_h) for `seq_len == 1` decode with state. It was implemented only as `mx.fast.metal_kernel`, and `ssm_kernel_available()` hard-returned false on non-Apple builds.
- On CUDA each decode step therefore ran `ssm_step`: segsum-style decay, masked matmuls, and a long chain of small elementwise ops per SSM layer. These execute as thousands of 1-2 us kernel launches per token (nsys pre-change top offenders: `copy_v` 9800 instances, `binary_g_nd` 4984, `arange` 4036 over a 36-token run), leaving the GPU idle most of the wall time.
- Pure mamba2 (`src/models/mamba2.rs`) never used the fused kernel on either backend, which is why its CUDA/Metal ratio was ~1.0 while the hybrids collapsed: on Metal the hybrids ran the fused kernel, on CUDA they ran the graph.

## Fix

- `mlx_cxx_kernels.cpp`: added `SSM_CUDA_SOURCE`, a line-for-line CUDA port of the Metal SSM kernel via `mx.fast.cuda_kernel` (one warp per (batch*head, head_dim row); each lane owns `Ds/32` contiguous state columns; state update `state = dA*state_in + x*dt*B`, output `simd-sum(state*C) + x*D` via `__shfl_down_sync`). Selected at runtime by `metal::is_available()`, same pattern as the BitLinear and fused-MoE CUDA ports.
- `ssm_kernel_available()` now returns `mlx::core::cu::is_available()` off Apple. Kill switch for A/B and debugging: `MLXCEL_SSM_CUDA_KERNEL=0` forces the graph path.
- No model-side changes: all four callers (granitemoehybrid, falcon_h1, plamo2, nemotron_h) pick the kernel up through the existing `seq_len == 1 && ssm_kernel_available()` gate.

## Results (GB10, 64 decode tokens)

"before" = `MLXCEL_SSM_CUDA_KERNEL=0` (graph path) on the final binary; "after" = fused CUDA kernel:

| model | decode before | decode after | speedup | Metal M1 Ultra ref | CUDA/Metal ratio |
|-------|--------------:|-------------:|--------:|-------------------:|-----------------:|
| granite-4.0-h-350m-4bit | 62.8 | 281.7 | 4.5x | 219.5 | 1.28 |
| granite-4.0-h-tiny-4bit | 38.9 | 101.8 | 2.6x | 96.3 | 1.06 |
| falcon-h1-tiny-90m-4bit | 112.0 | 319.7 | 2.9x | 288.1 | 1.11 |
| nemotron-h-30b-4bit | 41.5 | 88.1 | 2.1x | 91.5 | 0.96 |
| plamo-2-1b (f32 checkpoint) | 33.0 | 43.7 | 1.3x | 107.1 | 0.41 |
| mamba2-1.3b-4bit (control) | 80.1 | 81.0 | 1.0x (unchanged) | 79.2 | 1.02 |

Acceptance criteria from #631:

- granite-4.0-h-350m-4bit >= 150 tok/s: met at 281.7 (1.28x the Metal reference; the issue's floor was ratio 0.68, target 0.8).
- mamba2 control unchanged; Metal untouched (the change is runtime-gated on the CUDA backend).
- Greedy parity: byte-identical for granite-4.0-h-350m, falcon-h1-tiny, and plamo-2-1b, graph path vs fused kernel.

Two models in the original issue table need scope notes:

- **plamo-2-1b** is the only f32 checkpoint in the family (`torch_dtype: float32`, unquantized). Post-change its decode is 88.5% `gemv_single<float>` (nsys), i.e. pure f32 weight-streaming. The GB10 bandwidth ceiling for a ~1B f32 model is ~68 tok/s (273 GB/s / ~4 GB per token), so the issue's 0.8x-of-Metal target (85.7 tok/s) exceeds what the hardware can do: the M1 Ultra reference rides 800 GB/s. At 43.7 tok/s GB10 runs at 64% of its own ceiling, above Metal's 53% of *its* ceiling; the remaining cross-backend gap is bandwidth, not software.
- **hunyuan-13b** is misclassified in the family: its config has no SSM/mamba keys at all (`HunYuanMoEV1ForCausalLM`, 64 experts, top-8 + 1 shared, 32 layers). It never touches the SSM path and is unchanged (15.4 tok/s). Its decode gap is MoE-decode territory (fused-MoE decode path / #636 single-dtype decode graph), not this issue.

## nsys evidence (granite-4.0-h-350m, 36-token decode, `--cuda-graph-trace=node`)

| | top kernels | GPU character |
|---|---|---|
| before | `qmv_kernel` 23.8% + 8.3%, then a long tail of thousands of 1-2 us `copy_v` / `binary_*` / `arange` / `copy_s` launches | ~30% GPU busy, launch/host bound |
| after | `qmv_kernel` 34.6% + 11.8% (the real weight GEMVs), `custom_kernel_ssm_kernel_cu` 7.0% (952 calls, 8.4 us avg) | weight-streaming bound, tiny-kernel tail gone |

## Reproduce

```bash
./target/release/mlxcel-bench-decode --model ./models/granite-4.0-h-350m-4bit --prompt "Write a long detailed story about a robot who learns to paint." --no-chat-template --max-tokens 64
MLXCEL_SSM_CUDA_KERNEL=0 ./target/release/mlxcel-bench-decode --model ./models/granite-4.0-h-350m-4bit --prompt "Write a long detailed story about a robot who learns to paint." --no-chat-template --max-tokens 64   # graph path
TMPDIR=$PWD nsys profile -o ssm_decode --cuda-graph-trace=node ./target/release/mlxcel-bench-decode --model ./models/granite-4.0-h-350m-4bit --prompt "..." --no-chat-template --max-tokens 32 --warmup-tokens 4
```
