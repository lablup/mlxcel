# MLX CUDA: qmm_sm80 shared-memory epilogue race makes quantized matmul non-deterministic

Upstream bug report draft for ml-explore/mlx. Found and fixed in mlxcel (lablup/mlxcel#910); the fix is carried as a source overlay on the pinned MLX commit until it lands upstream.

## Summary

`qmm_sm80_kernel` (mlx/backend/cuda/device/qmm_sm80.cuh) has a shared-memory write-write race between the epilogue's C-tile staging and still-in-flight `cp.async` copies from the mainloop. `SharedStorage` is a union, so the epilogue C tile aliases the A/B pipeline slots, and the epilogue writes it without first retiring outstanding `cp.async` groups or re-converging the CTA. A late-landing async copy of A/B data overwrites freshly staged C values before the smem-to-gmem copy reads them, so the kernel publishes corrupted output on a timing-dependent subset of launches.

Observable effect: every quantized (tested with affine 4-bit) model decodes non-deterministically at temperature 0 on CUDA. Two runs of the same binary with the same prompt diverge bitwise, usually within the first few generated tokens, and the corrupted logits occasionally surface as garbage tokens (foreign-script injections, nonsense words) at positions where clean runs are confident by 14+ logits. Any `quantized_matmul` / `gather_qmm` dispatch with M >= 8 is affected (the `qmv` path used for M * B < 8 is clean); prefill is therefore always affected, and in MoE models the corrupted `gather_qmm` outputs also jitter the router scores so the selected expert set flips run to run.

Environment: GB10 (DGX Spark, sm_121), CUDA 13, Linux 6.17, MLX commit b7c3dd6d. The race is not architecture-specific in construction; the timing window will vary by device.

## Evidence

### bf16 vs 4-bit (same binary, temp-0 CLI decode, byte-diff of token streams)

| Model | Quant | Result |
|---|---|---|
| llama-3.1-8b-bf16 | none | deterministic (4/4 byte-identical) |
| llama-3.1-8b-4bit | affine 4-bit | non-deterministic |
| llama-3.2-1b-4bit | affine 4-bit | non-deterministic (3 distinct outputs in 4 runs of 128 tokens) |
| qwen3-4b-4bit | affine 4-bit | non-deterministic |
| gemma MoE 26b-a4b-4bit | affine 4-bit MoE | non-deterministic, occasional garbage tokens |
| qwen3-30b-a3b-4bit | affine 4-bit MoE | non-deterministic |

bf16 dense models never touch the qmm kernels (cuBLASLt path), which is why they are clean.

### Dispatch-boundary bisection (in-process, repeated identical forward, fresh KV caches, logits byte-hashed)

| Prefill length | Dispatch (QuantizedMatmul::eval_gpu) | Result over 24 iterations |
|---|---|---|
| 1 | qmv (M * B < 8) | deterministic (0/23 diverge) |
| 4 | qmv | deterministic (0/23) |
| 8 | qmm_sm80, CTA tile (16,128,64) | non-deterministic (23/23) |
| 32 | qmm_sm80 | non-deterministic (23/23) |

The flip from 0/23 to 23/23 exactly at M = 8 matches the `M * B < 8 ? qmv : qmm_sm80` dispatch boundary in quantized.cpp.

### Environment-variable discriminators (and why they misled)

| Config | Result | Interpretation |
|---|---|---|
| default | non-deterministic, high rate | race window wide under normal launch pressure |
| `MLX_USE_CUDA_GRAPHS=0` | still non-deterministic (~1/10 runs of 128 tokens) | not a graph-machinery bug; slower launch path narrows the window |
| graph-exec cache disabled (`MLX_CUDA_GRAPH_CACHE_SIZE=1` + thrash check off) | 0 divergence in 16 runs | per-commit graph instantiation slows the host enough to mask the race almost completely |
| `CUDA_LAUNCH_BLOCKING=1` | 0 divergence in CLI runs, but 1/23 iterations still diverged in a tight in-process loop | serialized launches keep the SM idle so the async copy nearly always lands before the epilogue store; suppression, not correctness |

The launch-blocking residual (1/23 in-process) is the decisive clue that this is an intra-kernel race rather than a host-side missing stream dependency: with fully serialized launches there is no host/device or cross-stream asynchrony left, yet the kernel still occasionally flips.

### compute-sanitizer racecheck (direct detection)

Running any M >= 8 quantized forward under `compute-sanitizer --tool racecheck` reports the hazard deterministically, pre-fix:

```
Error: Race reported between Write access at
  mlx::core::cu::qmm_sm80_kernel<(int)64, cutlass::half_t,
  cutlass::integer_subbyte<(int)4, (bool)0>, cutlass::half_t,
  cute::tuple<cute::C<(int)16>, cute::C<(int)128>, cute::C<(int)64>>>+0x2db0
and Write access at ...qmm_sm80_kernel<...>+0x5020 [32768 hazards]
RACECHECK SUMMARY: 214 errors
```

Post-fix: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.

## Root cause

File: `mlx/backend/cuda/device/qmm_sm80.cuh` at commit b7c3dd6d.

1. `SharedStorage` (line 18) is a union: the epilogue's `C` staging tile occupies the same shared memory bytes as the mainloop's `A`/`B` cp.async pipeline slots.
2. The mainloop only ever waits down to `K_PIPE_MAX - 2` outstanding cp.async groups (`cp_async_wait<K_PIPE_MAX - 2>()`, lines 235 and 249), and its final iteration issues one more (redundant, clamped to the last K tile) `fetch_gmem` (line 257). So when the k-tile loop exits, up to two cp.async groups into `sA`/`sB` are still in flight.
3. The epilogue (lines 265-272) immediately stages the accumulator tile into `sC`, which aliases those in-flight destinations, with no `cp_async_wait<0>()` and no `__syncthreads()` first:

```cpp
  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC_accu); i++) {
    tCrC(i) = Element(tCrC_accu(i));
  }
  copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);   // writes smem that cp.async may still write
  __syncthreads();
  copy_if(s2g_copy_c, tCpC, s2g_tCsC, s2g_tCgC);
```

When the DMA for the redundant final fetch lands after `r2s_tCsC` is written and before `s2g` reads it, the published C tile contains A/B bit patterns instead of results. The missing `__syncthreads()` before the C store is also a cross-thread write-after-read hazard: one warp can overwrite the union while another warp is still reading its final pipeline slot via `ldmatrix`.

## Minimal fix

```diff
   // Epilogue.
+  // The C tile below aliases the A/B pipeline slots (SharedStorage is a
+  // union), but up to K_PIPE_MAX-2 cp.async groups into sA/sB are still in
+  // flight here (the mainloop waits only down to K_PIPE_MAX-2 and its last
+  // iteration issues a redundant fetch_gmem of the final K tile). Retire
+  // them and re-converge the CTA before any thread overwrites the union.
+  cp_async_wait<0>();
+  __syncthreads();
   CUTE_UNROLL
   for (int i = 0; i < size(tCrC_accu); i++) {
     tCrC(i) = Element(tCrC_accu(i));
   }
   copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);
   __syncthreads();
   copy_if(s2g_copy_c, tCpC, s2g_tCsC, s2g_tCgC);
```

Measured cost on GB10: none observable (llama-3.2-1b-4bit CLI decode 256 vs 259 tok/s run-to-run noise; prefill-heavy shapes unchanged within noise). With the fix, 20/20 temp-0 CLI runs of 128 tokens are byte-identical and match the `CUDA_LAUNCH_BLOCKING=1` output, dense and MoE.

Note for deployments: MLX's on-disk PTX cache is keyed by module name only, so a source-level fix does not invalidate previously cached `qmm_sm80_*.ptx` under `$TMPDIR/mlx/<version>/ptx/`. Ship the fix with a cache-key change (or clear the cache) so stale kernels are not reused.

## Minimal standalone repro

Value-level (probabilistic, timing-dependent; the per-call flip rate is high on a busy GPU but a single matmul in an idle loop flips rarely, so drive it hard or run a model forward):

```python
import mlx.core as mx

mx.random.seed(0)
x = mx.random.normal((16, 2048)).astype(mx.float16)   # M=16 >= 8 -> qmm_sm80
w = mx.random.normal((2048, 2048))
wq, scales, biases = mx.quantize(w, group_size=64, bits=4)

ref = None
for i in range(10000):
    y = mx.quantized_matmul(x, wq, scales.astype(mx.float16),
                            biases.astype(mx.float16),
                            transpose=True, group_size=64, bits=4)
    mx.eval(y)
    import numpy as np
    b = np.array(y, copy=False).tobytes()
    if ref is None:
        ref = b
    elif b != ref:
        print("bitwise mismatch at iteration", i)
        break
else:
    print("no mismatch (window not hit; see racecheck below)")
```

Deterministic detection (does not depend on hitting the timing window):

```sh
compute-sanitizer --tool racecheck python repro.py
# pre-fix: "Race reported between Write access ... qmm_sm80_kernel ..." hazards
# post-fix: 0 hazards
```

Model-level repro: any 4-bit model, temp 0, fixed prompt of at least 8 tokens, two runs, byte-diff the token streams.

## Affected surface

- `QuantizedMatmul` with M * B >= 8 on CUDA (prefill, batched decode, speculative verify).
- `GatherQMM` batched paths that dispatch `qmm_sm80` (MoE prefill and reference/fallback paths); corrupted expert-score inputs also make MoE routing flip run to run.
- `qmv` (M * B < 8) and `qmm_naive` (no cp.async pipeline, barriers present) are not affected. `qmm_sm90` uses the CUTLASS collective pipeline and was not implicated on this hardware (not exercisable on sm_121).
