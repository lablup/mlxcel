# ModelOpt NVFP4 direct transcode benchmark (issue #693)

This note records the CUDA load and throughput comparison between the direct
ModelOpt NVFP4 transcode (issue #693) and the dense f16 requantize fallback it
replaces (issue #692) on NVIDIA GB10. The local `gemma-4-31b-it-nvfp4`
checkpoint is NVIDIA ModelOpt NVFP4 metadata with `quant_method=modelopt`,
`quant_algo=NVFP4`, and per-linear `weight/weight_scale/weight_scale_2`
triplets. The direct path reinterprets the packed FP4 U8 bytes into MLX native
NVFP4 U32 words, preserves the per-block E4M3 scales verbatim, and keeps
`weight_scale_2` as a per-linear global-scale sidecar. It never materializes a
dense f16 matrix, so it loads far faster than the dense fallback and is
bit-exact to the checkpoint. `MLXCEL_NVFP4_DENSE_REPACK=1` forces the dense
path for A/B comparison.

## Environment

| Item | Value |
|------|-------|
| Hardware | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| Backend | CUDA |
| Build | `cargo build --release --features cuda --bin mlxcel --bin mlxcel-bench-decode` |
| Harness | `target/release/mlxcel-bench-decode` |
| Test date | 2026-07-08 |
| Prompt | `Hello, how are you today?` (short) and a synthesized 2048-token prompt |
| Raw CSV | `benchmarks/cuda_gb10_issue693_nvfp4_direct_vs_dense_2026-07-08.csv` |

Each run is a separate process, so the reported cold-load wall time and MLX
peak are measured against a fresh MLX allocator. `mlxcel-bench-decode` now
resets the MLX high-water mark before the load and prints `[Load] wall` and the
peak after load completes.

## Results

| Run | Repack | Prompt tokens | Cold-load wall (s) | Load peak (GB) | Prefill tok/s | Decode tok/s |
|-----|--------|--------------:|-------------------:|---------------:|--------------:|-------------:|
| direct, short | direct triplet transcode | 20 | 58.77 | 36.60 | 76.13 | 4.84 |
| dense, short | dense f16 requantize | 20 | 190.72 | 36.60 | 79.51 | 5.26 |
| direct, 2048 | direct triplet transcode | 2048 | 58.87 | 36.60 | 443.85 | 5.01 |
| dense, 2048 | dense f16 requantize | 2048 | 190.69 | 36.60 | 395.20 | 5.38 |

## Acceptance

The acceptance gate is a 20% improvement in cold-load wall time or peak load
memory. Cold-load wall time drops from 190.72 s (dense) to 58.77 s (direct), a
69.2% reduction, so the gate passes on load time by a wide margin. The dense
path spends most of that time reconstructing a dense f16 matrix for each of the
180 NVFP4 MLP weight groups and re-quantizing it; the direct path only
reinterprets bytes and re-encodes the small per-block scale tensors.

Peak load memory is identical at 36.60 GB for both paths. The dense f16
transients are freed between weight groups, so they do not raise the MLX
high-water mark above the loaded model. The load-time win, not peak memory, is
the improvement here.

## Throughput

Prefill at 2048 tokens is 443.85 tok/s for the direct path versus 395.20 tok/s
for the dense path, a 12% gain, because the direct weights carry the
checkpoint's own block scales rather than the dense path's re-derived ones and
the op-at-a-time MLP schedules gate and up in parallel. Short-prompt prefill is
within noise (76.13 vs 79.51 tok/s on a 20-token prompt).

Decode is about 7% slower on the direct path (4.84 to 5.01 tok/s versus 5.26 to
5.38 tok/s). The global scale forces the Gemma 4 dense MLP onto the
op-at-a-time path plus a per-linear scalar multiply, and Gemma 4 decode runs
with CUDA graphs disabled, so the extra element-wise ops add dispatch overhead
per step. Folding the three global scales into the fused
`compiled_gelu_approx_mlp_forward` C++ kernel would recover this and is a
reasonable follow-up; the load-time and correctness wins are the point of this
change.

## Correctness

The direct transcode dequantizes bit-exactly to the ModelOpt reference
(`fp4 * e4m3_decode(weight_scale) * weight_scale_2`): the FP4 nibble order maps
onto MLX native NVFP4's eight-nibbles-per-u32 layout by a little-endian byte
reinterpret, the per-block E4M3 scales are preserved verbatim (the load-time
F8_E4M3 to f16 decode re-encodes losslessly), and `weight_scale_2` is applied
on the matmul output as an exact per-tensor scalar. The dense fallback re-derives
each block scale from the reconstructed f16 values, so it drifts by roughly one
E4M3 block-scale plus one FP4 rounding step (about 10% mean relative weight
error on this checkpoint). Greedy continuation therefore diverges between the
two paths, and the direct path is the faithful one. See the fixture
`nvfp4_direct_transcode_is_exact_and_bounds_dense_drift` in
`src/models/sanitize.rs` for the documented tolerance.

## Greedy parity spot-check

`mlxcel generate --temp 0 --max-tokens 48`, prompt "Explain in a few sentences
why the sky appears blue during the day." Both paths produce coherent,
semantically identical Rayleigh-scattering explanations:

- direct: "The sky appears blue due to a phenomenon called Rayleigh scattering.
  As sunlight enters Earth's atmosphere, it collides with gas molecules and
  scatters in all directions. Because blue light travels in shorter, smaller
  waves, it is scattered"
- dense: "The sky appears blue because of a phenomenon called Rayleigh
  scattering. As sunlight reaches Earth's atmosphere, it is scattered in all
  directions by the gases and particles in the air. Because blue light travels
  in shorter, smaller waves, it"

The token streams diverge early ("due to" vs "because of", "enters" vs
"reaches") because the dense fallback re-quantizes while the direct transcode
keeps the checkpoint's exact weights. Token-identical output between the two
paths is not expected; the meaningful bit-exactness is direct against the
checkpoint reference.

## Apple Silicon A/B runbook (issue #694)

This benchmark ran on NVIDIA GB10, so it only validates the direct transcode
under CUDA. Issue #694 asks whether Metal/non-CUDA builds should switch from
the affine 4-bit fallback to the same direct transcode, and that decision
needs an M-series host running the same checkpoint. The opt-in override lands
in the PR that references this issue: `MLXCEL_NVFP4_NATIVE_REPACK=1` makes a
non-CUDA build take the direct transcode without a code change, so the two
paths can be compared on the same binary. This section is the exact procedure
for that run; it has not been executed yet and the acceptance criteria in
issue #694 remain open until it is. Note that the direct-transcode leg is
itself unvalidated at runtime on Metal (the in-tree native NVFP4 qmm patches
are CUDA-side), so a load or inference failure on the
`MLXCEL_NVFP4_NATIVE_REPACK=1` leg is a possible and itself-informative
outcome of this A/B rather than a harness mistake.

### Build

```bash
cargo build --release --bin mlxcel --bin mlxcel-bench-decode
```

### Runs

Run each command twice on the same host and the same `gemma-4-31b-it-nvfp4`
checkpoint: once with no override (the current affine fallback default) and
once with `MLXCEL_NVFP4_NATIVE_REPACK=1` (the direct transcode).

Affine fallback (default):

```bash
./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" -n 100 --warmup-tokens 20
./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" --prompt-tokens 2048 -n 32 --warmup-tokens 20
```

Direct transcode (opt-in):

```bash
MLXCEL_NVFP4_NATIVE_REPACK=1 ./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" -n 100 --warmup-tokens 20
MLXCEL_NVFP4_NATIVE_REPACK=1 ./target/release/mlxcel-bench-decode -m models/gemma-4-31b-it-nvfp4 -p "Hello, how are you today?" --prompt-tokens 2048 -n 32 --warmup-tokens 20
```

### What to record

For each of the four runs, record:

- Cold-load wall time and MLX peak memory at load (`mlxcel-bench-decode`
  prints `[Load] wall` and the peak after load completes, as in the CUDA
  table above).
- Short-prompt prefill and decode tok/s.
- 2048-token prompt prefill and decode tok/s.
- A spot-check of at least 40 generated tokens at `--temp 0` comparing the
  affine fallback against the direct transcode, allowing only the expected
  FP8/FP4 rounding drift described in the Correctness section above (the two
  paths are not expected to be token-identical; the direct transcode is the
  one that should match the ModelOpt reference bit-exactly).

Record the raw numbers under `benchmarks/` (following the naming pattern of
the existing `benchmarks/metal_*.csv` files) and update this section, or add a
new `docs/benchmark_results/` note, with the results once the run completes.

### Decision gate

Per issue #694's acceptance criteria, enable the direct transcode as the
default on Metal/non-CUDA only if, on the same host, 2048-token prefill
improves by at least 20% and decode does not regress by more than 5% versus
the affine fallback. If the gate passes, flipping the non-CUDA default is a
separate follow-up change (the CUDA feature flag branch in
`nvfp4_repack_strategy`, `src/models/sanitize.rs`) once the M-series numbers
are in; this issue and this PR do not flip that default.

## Follow-up: NVFP4 global-scale fold into the fused MLP (issue #698, 2026-07-09)

The direct transcode keeps `weight_scale_2` as a per-linear `global_scale` sidecar, applied by `UnifiedLinear::forward` as `astype(multiply(qmm_out, s), qmm_out.dtype())` after each projection. Before this change, a sidecar-carrying gemma-4 MLP bypassed the fused C++ path entirely and ran three separate op-at-a-time projections. Issue #698 folds those scales back into the fused kernel at the mathematically correct points: the gate scale before the GeGLU activation (nonlinear), the up scale on the up product, and the down scale on the fused output. The C++ bridge itself compiles the scaled graph only for single-token calls and falls back to an eager fold otherwise; `src/models/gemma4.rs` now gates both fused call sites (`MLP::forward` and the per-layer-input gate) to single-token inputs directly, so a sidecar-carrying multi-token (prefill) call never reaches that eager fold and instead takes the same op-at-a-time projections non-sidecar prefill has always used. `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE` restores the op-at-a-time bypass for decode too.

All rows below are the same `mlxcel-bench-decode` release binary on the same GB10 host (raw CSV: `benchmarks/cuda_gb10_issue698_nvfp4_fused_scale_2026-07-09.csv`). `dense` is the `MLXCEL_NVFP4_DENSE_REPACK=1` baseline (the sidecar is folded into the E4M3 block scales at load, so decode runs no scale ops); `op-at-a-time` is the pre-#698 bypass (`MLXCEL_DISABLE_FUSED_GLOBAL_SCALE=1`); `fused fold` is the new default.

| Run | Scale path | Prompt tokens | Prefill tok/s | Decode tok/s |
|-----|-----------|--------------:|--------------:|-------------:|
| short | dense (folded into block scales) | 20 | 78.55 | 5.25 |
| short | op-at-a-time bypass | 20 | 74.62 | 4.86 |
| short | fused fold (default) | 20 | 74.63 | 4.90 |
| 2048 | dense (folded into block scales) | 2048 | 395.20 | 5.38* |
| 2048 | op-at-a-time bypass | 2048 | 449.90 | 5.01 |
| 2048 | fused fold (initial, pre-fix)** | 2048 | 412.48 | 5.07 |
| 2048 | fused fold (default, post-fix) | 2048 | 447.24 | 5.05 |

*Dense 2048 decode is the 2026-07-08 measurement; dense short reproduced yesterday's number exactly today (5.25 vs 5.26), so the 2048 dense row is carried over rather than re-running its ~190 s dense-requantize load.

**The initial fold gated only on sidecar presence and the kill switch, with no check on query length, so multi-token prefill also called into the C++ fused-scale function; that function's own decode-only compile gate then dropped it into the eager fold described above. The "Prefill regression and fix" section below covers this in full; the "post-fix" row is the re-measurement after the single-token gate landed, same host, same binary flags, 2026-07-09.

### Reading the numbers

The fold is throughput-neutral on GB10 decode: +0.8% short (4.86 to 4.90) and +0.8% at 2048 post-fix (5.01 to 5.05) versus the op-at-a-time bypass. This matches the documented behavior of the sibling GeGLU fusion (`MLXCEL_COMPILED_QGELU_MLP`): GB10 gemma-4 decode is weight-bandwidth-bound (about 94% GPU-busy streaming the NVFP4 MLP weights), so removing per-layer element-wise dispatches is throughput-neutral there.

The fold rules out dispatch count as the cause of the residual gap to the dense baseline: fused decode stays 6.7% below dense short and 6.1% below dense at 2048 post-fix, and folding the scale multiplies into one compiled graph does not close either gap. The dense-vs-direct decode difference itself is not conclusively isolated by this benchmark. `src/models/sanitize.rs` shows both routes produce identical MLX native NVFP4 layout (same mode, group_size, bits, weight dtype, and scale dtype); they differ only in scale values (dense re-derives its block scales with about 10% drift, direct is bit-exact against the ModelOpt checkpoint) plus a negligible f32 `[1]` sidecar on the direct route. The current comparison is also confounded: the dense short run generated 42 tokens against 26 for direct, and the dense 2048 row is a cross-day carry-over from 2026-07-08 rather than a same-run measurement. Closing that gap needs a controlled re-run with matched generation length on the same day, which is out of scope here.

The fold still removes three FFI crossings and three element-wise dispatches per MLP layer for single-token decode, which reduces CPU dispatch work and helps op-count-bound backends, so it ships default-on for decode with the kill switch retained; multi-token prefill takes the unaffected op-at-a-time path unconditionally (see below).

### Prefill regression and fix

The initial version of this fold checked only sidecar presence and the kill switch before calling into the C++ fused-scale function, with no check on the query length. That function compiles the scaled graph for single-token calls but falls back to an eager, uncompiled `multiply(gelu_tanh_approx(gate), up)` for everything else, mirroring the decode-only compile gate of the unscaled fused path. Multi-token prefill therefore fell into that eager fold instead of the compiled op-at-a-time activation (`compiled_geglu_approx_activation`) non-sidecar prefill already used, and the eager path is slower: on the 2048-token prompt, prefill dropped from 449.90 tok/s (op-at-a-time) to 412.48 tok/s under the initial fold, an 8.3% regression. The 20-token short prompt showed no measurable change (74.62 to 74.63 tok/s) because a matmul that small is too cheap for the eager-versus-compiled activation cost to clear the noise floor.

The fix gates both fused call sites in `src/models/gemma4.rs` (`MLP::forward`'s dense MLP branch and the per-layer-input gate in `DecoderLayer::forward_with_profile`) to single-token inputs, checking the same query-length dimension the surrounding decode-step code already reads. A sidecar-carrying multi-token call now bypasses the fused path unconditionally, regardless of the kill switch, and takes the identical op-at-a-time projections plus `compiled_geglu_approx_activation` that non-sidecar prefill has always used. Re-measured on the same host with the same release binary, 2048-token prefill recovers to 447.24 tok/s, within 0.6% of the op-at-a-time reference, while decode stays on the fused path at 5.05 tok/s (the "post-fix" row above). The C++ bridge itself is unchanged; its eager-fold branch remains correct and is exercised by the existing multi-token FFI test, just no longer reached from gemma-4 prefill.

### Parity

Greedy `--temp 0` decode is byte-identical between the fused fold and the op-at-a-time bypass on the 26-token short prompt (`Hello, how are you today?`). On a 120-token generation (`List five interesting facts about the planet Jupiter.`) the two paths agree for the first ~68 tokens and then diverge at a near-tie fact selection. That divergence is baseline CUDA nondeterminism, not the fold: two runs of the unmodified default path diverge from each other as well (at ~115 tokens, `so large that Earth could` vs `so large that it is`), so the NVFP4 qmm FP-reduction order is not run-to-run reproducible on this backend for either path. The fused output stays inside that envelope. The synthetic unit tests (`compiled_qgelu_mlp_global_scale_*` in `mlxcel-core`) confirm the fold matches the op-at-a-time `apply_global_scale` reference to 1e-5 on the compiled single-token branch, the eager multi-token fallback, and mixed sidecar sets.
