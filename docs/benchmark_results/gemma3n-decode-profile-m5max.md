# Gemma3n decode profile on M5 Max (#329 follow-up)

Apple M5 Max (40-core GPU, 128 GB, macOS 26.5.1), MLX 0.31.2, mlx-vlm 0.4.4.
Companion to [`gemma3n-decode-profile.md`](gemma3n-decode-profile.md), which
profiled the same models on M1 Ultra for #329. Decode throughput measured with
`mlxcel-bench-decode` (raw continuation prompt, 100 tokens after a 20-token
warmup, `caffeinate -i`, best-of-3 with the machine cool). Pipeline and op-count
data from the `mlxcel-gpu-profiling` hooks.

M5 Max takes a different decode code path than M1 Ultra. It is detected as a
Neural-Accelerator part (`silicon_gen = M5`, `has_neural_accelerator = true`,
and macOS 26.5 satisfies `macos_supports_na`), so `use_fused_decode_path()` is
false and decode runs the per-op split path. The #60 fused decode path (stacked
AltUp plus the `gemma3n_mlp_forward` bridge) is the M1 Ultra path and is gated
off here.

## Verdict

PR #345's conclusion holds on M5 Max: no compiled fusion is justified for
Gemma3n decode. The supporting evidence differs from M1 Ultra in one place,
which is exactly why it was worth re-measuring.

1. Decode is 92 to 96% GPU-bound. The Rust graph-build step (`forward`), where
   an FFI-collapsing fusion would land, is 0.47 to 0.66 ms/token and stays
   hidden behind `async_eval`. Same picture as M1 Ultra, so the same argument
   against an FFI-collapsing fusion applies.
2. The `MLX_MAX_OPS_PER_BUFFER` lever that recovers +11 to 13% on M1 Ultra is
   flat on M5 Max (at most +1.7%, inside the best-of-3 noise band). The one knob
   that dominates a fusion on M1 Ultra does not exist on M5. The residual GPU
   idle on the 4-bit models (decode sits at 46 to 59% of the memory-bandwidth
   wall) is diffuse dispatch and structure overhead that neither the buffer knob
   nor, per the prior M5 study, any single fusion above ~2.6% recovers.
3. The #60 fused decode path, which the gate disables on M5, was re-measured
   neutral here (e4b-bf16 +0.84%, e4b-4bit +0.34%, both inside noise), not the
   historical -6.3% regression. Forcing it on buys nothing, so the validated
   split path stays the default. See section 5.

mlxcel also still leads mlx-vlm, the only Python runtime that loads these
multimodal checkpoints, by 1.14 to 1.17x (section 7), so there is no parity gap
pulling for a risky kernel.

## 1. Decode baseline (text and VLM)

Greedy, raw prompt, best-of-3, default `MLX_MAX_OPS_PER_BUFFER`. The M1 Ultra
column is from the companion document. Effective bandwidth is `decode tok/s ×
streamed GB/token` against a measured 478 GB/s GEMV peak (section 3).

| Checkpoint | M5 Max text | M5 Max image | M1 Ultra text | M5/M1 (text) | eff. GB/s (text) | util |
|------------|------------:|-------------:|--------------:|-------------:|-----------------:|-----:|
| gemma3n-e2b-4bit | 161.5 | - | 83.1 | 1.94x | 220 | 46% |
| gemma3n-e4b-4bit | 112.2 | 107.0 | 62.9 | 1.78x | 283 | 59% |
| gemma3n-e4b-bf16 | 40.4 | 39.0 | - | - | 362 | 76% |

M5 Max runs these 4-bit checkpoints at 1.8 to 1.9x M1 Ultra. The 4-bit models
sit well below the bandwidth wall, so decode there is not bandwidth bound; the
bf16 model, which streams ~3.5x the bytes per token, is closer to the wall.

## 2. Pipeline split: GPU-bound, not FFI-bound

`MLXCEL_PROFILE_PIPELINE_DETAIL`, per decode token:

| Checkpoint | forward (Rust graph build) | async_eval (GPU) | GPU share |
|------------|---------------------------:|-----------------:|----------:|
| e2b-4bit | 0.470 ms | 5.500 ms | 92.1% |
| e4b-4bit | 0.527 ms | 8.094 ms | 92.7% |
| e4b-bf16 | 0.660 ms | 23.271 ms | 96.3% |

`forward` is the whole Rust-side graph construction, every `mlxcel_core::*` FFI
call for the token, and it overlaps the previous token's GPU work through
`async_eval` lookahead. It never reaches the critical path, so there is no FFI
cost for a fusion to collapse.

## 3. Streaming floor and bandwidth utilization

Per-token streamed bytes were summed from the safetensors headers: matmul
weights plus the tied LM head, excluding the vision tower, the audio tower, and
the lookup-only per-layer embedding. The e4b-4bit split reproduces the companion
document's decomposition (vision 0.59 GB, audio 1.37 GB, per-layer embedding
1.34 GB, streamed 2.52 GB).

| Checkpoint | streamed GB/token | best decode tok/s | effective GB/s | util vs 478 GB/s |
|------------|------------------:|------------------:|---------------:|-----------------:|
| e2b-4bit | 1.36 | 161.5 | 220 | 46% |
| e4b-4bit | 2.52 | 112.2 | 283 | 59% |
| e4b-bf16 | 8.97 | 40.4 | 362 | 76% |

The 478 GB/s peak is the achievable f16/bf16 GEMV bandwidth on this box
(measured 478.1 f16, 478.4 bf16). Streamed bytes are an upper bound because
KV-sharing on the back half of the stack skips some k/v projections, so the true
utilization is slightly under these figures. The takeaway holds either way: the
4-bit decode paths leave the memory subsystem idle, which is the regime where a
fusion looks tempting, yet sections 4 and 5 show nothing recovers that idle.

## 4. Command-buffer batching is flat on M5 Max

`MLX_MAX_OPS_PER_BUFFER` controls how many ops MLX batches before committing a
command buffer (runtime, no rebuild). Best-of-3, text, 80 tokens:

| MLX_MAX_OPS_PER_BUFFER | e2b-4bit | e4b-4bit | e4b-bf16 |
|------------------------|---------:|---------:|---------:|
| default (unset) | 161.6 | 112.6 | 40.5 |
| 100 | 163.6 | 113.6 | 40.7 |
| 200 | 164.4 | 113.1 | 41.1 |
| 1000 | 163.5 | 114.2 | 40.7 |
| 2000 | 163.4 | 114.4 | 41.1 |

Every column moves at most +1.7% across the whole sweep, inside the run-to-run
noise, with no climb to a plateau. On M1 Ultra the same sweep on these models
rises +11 to 13% toward a plateau near 1000 (e2b 82.7 to 93.2, e4b 63.0 to
70.0). That lever is specific to M1 Ultra and must stay hardware-gated: raising
the default would help M1 Ultra and do nothing measurable on M5 Max. The earlier
M5 study (the source of the `mlxcel-gpu-profiling` skill) recorded larger
buffers as slower on M5; on this box with MLX 0.31.2 they are flat rather than
slower, but still offer no gain.

## 5. Fused vs split decode path on M5 Max

The gate at `use_fused_decode_path()` disables the #60 fused path on NA
hardware. To check whether that gate's rationale still holds, the path was forced
on and off through a temporary `MLXCEL_FORCE_FUSED_DECODE` /
`MLXCEL_FORCE_SPLIT_DECODE` override (reverted after measurement), best-of-3,
text, 100 tokens:

| Checkpoint | split (M5 default) | fused (forced) | delta |
|------------|-------------------:|---------------:|------:|
| e4b-bf16 (fused MLP bridge + stacked AltUp) | 40.31 | 40.65 | +0.84% |
| e4b-4bit (stacked AltUp only; bridge is bf16-only) | 112.20 | 112.58 | +0.34% |

Both deltas are inside the noise band, and the bf16 path did not hit the JIT
crash the earlier study warned about. So on this M5 Max with MLX 0.31.2 the fused
path is neutral, not the -6.3% regression the older comment recorded. The gate
stays because the fused path has no upside on M5 and the per-op split path is the
validated default; only the stated regression figure was out of date.

## 6. Text vs VLM: vision is a prefill cost only

| e4b-4bit | prefill tok/s | decode tok/s |
|----------|--------------:|-------------:|
| text prompt | 302 | 112.2 |
| image prompt | 2233 | 107.0 |

The vision tower runs once during prefill; the high image-prompt prefill rate
reflects the image's soft tokens processed in one pass (M1 Ultra reached 483
here, M5 Max 2233). Decode is the same per-layer structure either way. The small
image-decode delta is the longer KV context after ingesting the image soft
tokens, which is KV bandwidth, not a fusable small-op hotspot.

## 7. Parity with mlx-vlm

Both runtimes use the chat template so the prompt formatting matches; mlx-vlm
numbers are wall-clock `(tokens-1)/(t_last - t_first)` from
`mlx_vlm.stream_generate`. mlx-lm cannot load these multimodal checkpoints, so
mlx-vlm is the reference.

| Checkpoint | mlxcel | mlx-vlm | mlxcel / mlx-vlm | M1 Ultra ratio |
|------------|-------:|--------:|-----------------:|---------------:|
| e2b-4bit text | 158.2 | 135.9 | 1.16x | - |
| e4b-4bit text | 109.9 | 96.8 | 1.14x | 1.20x |
| e4b-4bit image | 107.2 | 91.7 | 1.17x | 1.23x |

mlxcel leads on M5 Max as it does on M1 Ultra. The margin is a little narrower
(1.14 to 1.17x against 1.20 to 1.23x) because mlx-vlm also benefits from the
faster hardware, but there is no gap that a Gemma3n-specific kernel would close.

## 8. Conclusion

Close the M5 Max question the same way #329 closed: no compiled fusion. Decode is
GPU-bound with FFI hidden, the command-buffer knob that wins on M1 Ultra is flat
here, the fused path the gate disables is neutral rather than harmful, and
mlxcel already leads mlx-vlm. The only stale fact was the -6.3% fused-path
regression in `src/models/gemma3n.rs`, corrected to neutral alongside this
document.

### Reproduce

```bash
# decode baseline (best-of-3, cool)
for i in 1 2 3; do caffeinate -i ./target/release/mlxcel-bench-decode \
  -m models/gemma3n-e4b-4bit -p "Hello, how are you today?" -n 100 \
  --warmup-tokens 20 --no-chat-template 2>/dev/null | grep Decode:; done

# command-buffer sweep
for OPS in 100 200 1000 2000; do MLX_MAX_OPS_PER_BUFFER=$OPS \
  ./target/release/mlxcel-bench-decode -m models/gemma3n-e4b-4bit \
  -p "Hello, how are you today?" -n 80 --warmup-tokens 20 \
  --no-chat-template 2>/dev/null | grep Decode:; done

# pipeline split
MLXCEL_PROFILE_PIPELINE=1 MLXCEL_PROFILE_PIPELINE_DETAIL=1 \
  ./target/release/mlxcel-bench-decode -m models/gemma3n-e4b-4bit -p Hi \
  -n 100 --warmup-tokens 20 --no-chat-template 2>&1 | grep PIPELINE

# achievable GEMV bandwidth (mlx, f16 ~= bf16 on M3+)
python3 -c "import mlx.core as mx, time
def bw(dt,N=16384,M=16384,it=300):
 W=mx.random.normal((N,M)).astype(dt);x=mx.random.normal((1,N)).astype(dt);mx.eval(W,x)
 [mx.eval(x@W) for _ in range(20)];t=time.perf_counter()
 [mx.eval(x@W) for _ in range(it)];return N*M*2/((time.perf_counter()-t)/it)/1e9
print(max(bw(mx.bfloat16) for _ in range(3)),'GB/s')"
```
