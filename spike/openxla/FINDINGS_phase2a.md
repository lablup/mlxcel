# Phase 2a findings: int4 dequant-in-graph lowering, fusion, and correctness

Issue #449 Phase 2a (the int4 comment on the issue). Scope: on the proven JAX
harness, add a dequant-in-graph int4 path and characterize the lowering
structure, whether XLA and IREE fuse the unpack/convert/scale into the matmul,
and whether greedy coherence holds. Frontend-independent, so the JAX harness is
the vehicle. The issue scoped 2a as no-perf; CUDA turned out to be available on
this box, so a GPU correctness run is included (perf stays Phase 2b).

## Verdict

The dequant-in-graph int4 route is correct and stays coherent. The central
question (does the compiler fuse the dequant into the matmul, or materialize a
full fp32 weight first?) resolves the same way on every target tried: the
dequantized fp32 weight is materialized, not fused into the matmul. So this naive
route buys an 8x smaller weight to load and upload, but no matmul compute or
bandwidth win on its own. A real int4 matmul win needs a recognized
quantized-matmul fusion (which this pattern did not trigger) or a `custom_call`
to an int4 GEMM. That is the Phase 2 finding the success bar depended on.

## Quantization used

Affine asymmetric int4, group size 64 along the contraction dim, packed 8 nibbles
per uint32 (per group: scale + min). Applied to the 7 per-layer linear weights
(q, k, v, o, gate, up, down); embedding and norms stay fp32. Representative of the
mlx-community affine 4-bit scheme; exact format alignment with mlxcel's loader is
a Phase 3 concern, not needed to characterize the route.

- Layer-0 relative Frobenius error per weight: 0.090 to 0.097 (typical for 4-bit
  round-to-nearest, no GPTQ).
- One weight as a memory check: `down` is 67.1 MB fp32 to 8.4 MB packed int4 plus
  per-group scales (about 1/64 of the fp32 size). The 8x storage shrink is the
  real, target-independent win.

## Correctness of the in-graph dequant

Two paths over the SAME quantized weights: (a) dequant on the host then run the
plain fp32 graph, (b) dequant inside the graph. Their prefill last-token logits
agree to max abs diff 7.4e-5 with identical argmax. The in-graph unpack
(shift + mask), convert, and scale reproduce the host dequant. This isolates "is
the in-graph dequant correct" from "does quantization change the model".

## Coherence

int4 greedy continuation for the fixed prompt:

```
Here are three short tips for staying focused while working:

1. **Set clear goals and deadlines**: Before starting your work, define what
needs to be done and by when. Break down large tasks into smaller, manageable
chunks, and set specific
```

Coherent and on-task. Versus the fp16 greedy run it agrees for the first 18
tokens then diverges (30/48 tokens match overall), which is expected: 4-bit RTN
changes the weights, so token-exactness with fp16 is not the bar; coherence is,
and it holds.

## StableHLO structure

The exported int4 `decode_step` keeps the dequant as explicit standard ops, no
custom op: `shift_right_logical` 112, `and` 112, `convert` 113 (= 16 layers x 7
linears), feeding 145 `dot_general`, `custom_call` absent. Packed weights enter
as `ui32` graph inputs. So the dequant is visible and separable in the graph,
exactly the shape a later `custom_call`-to-int4-GEMM swap would replace.

## The core question: fuse or materialize

Isolated one int4 linear (`y = x @ dequant(packed, scale, min).T`, K=N=2048) so
the optimized IR is readable. Findings per target:

- **XLA, CPU.** The whole dequant and the dot land in a single `kLoop` fusion
  (`%copy_dot_fusion` calling one `%fused_computation` that contains
  shift-right-logical, and, convert, broadcast, multiply, add, then ROOT `dot`).
  Fused at the HLO level. But inside the fusion the full `f32[2048,2048]` weight
  is reconstructed (`add.0`, then `transpose.0`, then `copy.1`, all
  `f32[2048,2048]`) before the dot consumes it. So the dot reads a full fp32
  weight; no int4 bandwidth advantage on the matmul. Fused as one HLO op, but the
  fp32 weight is materialized within it.

- **IREE, CPU.** Two separate dispatches: `main_dispatch_2_elementwise_transpose`
  carries `arith.shrui` + `arith.uitofp` + `arith.mulf` + `arith.addf` (the
  unpack, convert, and scale), and `main_dispatch_3_vecmat` is the matmul. The
  dequant dispatch writes a full f32 weight to memory; the vecmat reads it. Not
  fused.

- **IREE, CUDA (GB10).** Same split, because dispatch-region formation happens at
  the target-independent flow level. The dumped CUDA executables are a separate
  dequant kernel (`dispatch_2`: shrui, uitofp) and a separate `vecmat` kernel
  (`dispatch_3`). Not fused on the GPU either.

Net: out of the box, none of XLA-CPU, IREE-CPU, or IREE-CUDA produce a fused int4
matmul that reads packed weights directly. XLA fuses the ops into one loop fusion
but still rebuilds the fp32 weight inside it; IREE forms a distinct dequant
dispatch that materializes the fp32 weight. The matmul always reads fp32. To get
a real int4 matmul (bandwidth and memory on the GEMM itself) you need IREE's
quantized-matmul / data-tiling fusion to recognize the pattern (this naive affine
RTN pattern did not trigger it in 3.11) or a `custom_call` to a target int4 GEMM.
The graph here is already shaped for that swap: the dequant feeding each
`dot_general` is the exact subgraph a custom op would replace.

## GPU run (enabled by CUDA on this box)

CUDA is available here after all: GB10 is compute capability 12.1 (sm_121), and
IREE 3.11 ships both the `cuda` compiler target and the `cuda` runtime driver.
The exported int4 StableHLO compiles to CUDA and runs on the GB10:

- The full int4 loop (prefill + decode) runs on the GB10 and is **token-exact
  (48/48)** with the CPU int4 run, coherent. The acceleration path
  (exported StableHLO to IREE-CUDA to GB10) works end to end.
- Compile invocation: `--iree-hal-target-device=cuda` (default target; the
  driver JITs PTX to sm_121 at load). Passing `--iree-cuda-target=sm_121`
  explicitly is rejected by this build's device syntax with "missing GPU target";
  the default-target path is the one that works.
- Portability fix: prefill's range-slice cache write (`kcache.at[li, :Lp].set`)
  lowers to `iree_linalg_ext.scatter`, which the CUDA backend rejects ("expected
  indices to be equal to batch rank"). Rewriting the cache build as pad + stack
  (a `dynamic_update_slice`) lowers cleanly on both CPU and CUDA and is
  numerically identical. Decode's single-index write
  (`kcache.at[li, cache_len].set`) was already fine. Worth recording for the
  in-tree emitter: prefer dynamic-update-slice cache writes over range-slice
  scatters for GPU portability.
- No perf claim. The functional-call harness re-uploads all ~700 MB of weights as
  graph inputs every decode step and syncs logits to host per token, so its
  per-token time is upload-bound, not a throughput number. Real GPU numbers need
  weights resident on-device, batching, and on-device sampling, which is Phase 2b.

This also updates the Phase 1 "GPU blocked on aarch64+Blackwell" note: blocked for
a CUDA jaxlib (still true, none on PyPI), but NOT blocked for execution, because
IREE compiles the same StableHLO to CUDA and runs it on the GB10. Phase 2b can
proceed on this box through the IREE-CUDA path.

## Files

`quant.py` (affine int4 pack/unpack), `model_int4.py` (dequant-in-graph
prefill/decode), `run_phase2a.py` (quant error, in-graph vs host dequant,
coherence, StableHLO op check), `fusion_probe.py` (XLA optimized HLO + IREE
dispatch analysis on one linear), `gpu_iree.py` (compile to CUDA, run on GB10,
token-match check). Artifacts in `artifacts/`:
`decode_step_int4.stablehlo.mlir`, `qmatmul_xla_optimized.hlo`,
`qmatmul.flow.mlir` (IREE dispatches), `*.cuda.vmfb`, `results_int4*.json`.

## Recommendation

Record dequant-in-graph as correct and coherent but not self-fusing: the matmul
reads a materialized fp32 weight on every target tested, so int4 alone is a
storage and weight-transfer win, not a GEMM win. For the Phase 2 success bar the
route to a real int4 matmul is a `custom_call` to a target int4 GEMM (the graph
is already shaped for it) or getting IREE's quantized-matmul fusion to recognize
the pattern. Carry the dynamic-update-slice cache-write rule into whichever
authoring frontend wins (#451). Defer throughput to Phase 2b, which is now
runnable here via IREE-CUDA on the GB10.
