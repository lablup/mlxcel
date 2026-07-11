# Multirow qmv: amortizing small-M quantized matmul on GB10 (CUDA), 2026-07-11

Issue #725. On Blackwell sm_120/121 the CUDA quantized-matmul dispatch routes
`M*B < 8` to per-row `qmv`, which re-reads the full weight matrix once per
input row (one row per `grid.y`/`grid.z` block). That made batched decode
aggregate throughput flat in B and made the speculative-verify `[1, K]`
forward cost ~K classic forwards, the root cause of the GB10 MTP regression
measured in `speculative-pairing-gb10-2026-07-10.md`.

The fix is a multirow qmv variant in the `qmv.cu` overlay
(`src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmv.cu`): when the
weights are broadcast and `2 <= M*B <= 8`, one warp keeps per-row accumulator
arrays and applies each dequantized weight tile to every input row, so weight
traffic is O(1) in the row count instead of O(R). The per-row arithmetic
(dequant order, fma order, accumulator types, final float reduction) is kept
exactly equal to the stock kernel, so per-row outputs are bit-identical to the
per-row launches it replaces. `M*B == 1` (classic single-token decode) keeps
the stock kernel unchanged. Kill switch: `MLXCEL_QMV_MULTIROW=0`.

## Hardware and build

- Host: NVIDIA GB10 (Grace-Blackwell, sm_121), CUDA backend.
- Build: `cargo build --release --features cuda`.
- Greedy (`temperature = 0`), decode-only tok/s.

## Parity

- `qmv_multirow_matches_per_row_qmv_bitwise` (ffi test, `-p mlxcel-core
  --features cuda`): bf16/f16/f32, 4-bit and 8-bit affine, group sizes 32/64,
  rows 2/3/4/7, including a residue-k shape; every multirow row is
  byte-identical to the matching single-row qmv launch.
- End-to-end: on a quiet GPU the K=4 MTP runs with `MLXCEL_QMV_MULTIROW=0` and
  `=1` produce the identical token stream (85 generated, 41 rounds, acceptance
  0.358). Note the `[1, K]` verify path itself is load-sensitive run-to-run on
  CUDA (near-tie argmax flips under concurrent GPU work, the known FP-reduction
  jitter also documented for `MLXCEL_FUSED_QK_NORM`); the jitter pre-dates this
  change and reproduces with the kill switch on.

## Speculative decoding (MTP): the #638 matrix, re-run

Target `models/gemma-4-12b-it-4bit` (`gemma4_unified`), drafter
`models/gemma-4-12b-it-assistant-4bit`, 14-token `DEFAULT_PROMPT`,
`--max-tokens 128`, one process per row (same invocations as the 2026-07-10
matrix).

| Kind | K | tok/s | speedup vs no-drafter | 2026-07-10 (per-row qmv) | acceptance rate | mean accepted len |
|------|---|------:|----------------------:|-------------------------:|----------------:|------------------:|
| none (baseline) | — | 14.5 | 1.00x | 14.5 (1.00x) | — | — |
| mtp | 2 | 19.0 | 1.31x | 11.1 (0.77x) | 56.6% | 0.57 |
| mtp | 4 | 21.2 | 1.46x | 7.6 (0.52x) | 35.8% | 1.07 |
| mtp | 8 | 20.4 | 1.41x | 7.5 (0.52x) | 34.1% | 1.02 |
| mtp | 4 (`MLXCEL_QMV_MULTIROW=0`) | 7.7 | 0.53x | — | 35.8% | 1.07 |

The K=4 round cost drops from ~268 ms (per-row: verify M=4 ran as 4 narrow
GEMVs, ~3.9 classic forwards) to ~98 ms (verify amortizes toward one classic
forward plus the drafter and per-round overhead). B=1 MTP on GB10 flips from a
0.52-0.77x regression to a 1.31-1.46x win at the same acceptance, clearing the
1.4x target from issue #638 at K=4/K=8. K=8 still runs at an effective K=4
(the adaptive block controller holds the drafter's configured block size at
this acceptance).

Policy note (#736, resolved): the adaptive MTP policy now settles verdicts
from a measured comparison: profiling bursts run a couple of classic-step
probe rounds (drafterless `[1, 1]` verifies, each emitting one real greedy
token) and the estimator divides tokens-per-round by the measured
round-cost-to-classic-step ratio. With the multirow verify this pairing
profiles to about 1.5x and enables in serving without any manual override;
the pre-#736 `sqrt(K)` heuristic (which landed this profile at ~1.0, the
decline boundary) remains only as a fallback for windows with no probe
signal. Persisted pre-#736 verdicts re-profile once (hint format v3).

## Batched serving decode: aggregate scaling

`mlxcel serve -m models/llama-3.1-8b-4bit --n-parallel 4` +
`scripts/bench_serving_concurrency.py --concurrency 1,2,4 --prompt-tokens 512
--max-tokens 128`:

| clients | aggregate tok/s (`MLXCEL_QMV_MULTIROW=0`) | aggregate tok/s (multirow) | per-request decode tok/s (off -> on) |
|--------:|------------------------------------------:|---------------------------:|-------------------------------------:|
| 1 | 40.5 | 40.4 | 46.5 -> 46.5 |
| 2 | 49.8 | 68.5 | 25.1 -> 34.7 |
| 4 | 49.6 | 74.2 | 12.5 -> 18.8 |

Aggregate decode now scales with concurrency (1.84x from 1 to 4 clients)
instead of staying flat (1.22x before, the "throughput wash" documented in
`docs/CONTINUOUS_BATCHING.md`). Scaling is sublinear because only the
quantized-matmul weight reads amortize: attention KV reads still grow with B,
and the 4-client row interleaves prefill with decode. The remaining gap is the
small-M `qmm_sm80` tile territory tracked upstream (see below).

## Kernel evidence

`nsys profile --cuda-graph-trace=node` over a K=4 MTP run shows the verify-side
`qmv` work collapsing from per-row launches into single `qmv_multirow_kernel`
launches (see the PR for the capture summary). `ncu` counter capture on this
host requires `sudo` (driver `RmProfilingAdminOnly=1`); the nsys kernel-time
comparison plus the wall-clock A/B above stand in as the amortization
evidence.

## Scope and follow-ups

- `fp_qmv` (non-affine fp modes, sm120+) still maps one row per block for
  `vec_batch` up to 8; the same multirow treatment applies but is not needed
  for the affine checkpoints measured here.
- `gather_qmv` (MoE decode) keeps per-row launches: rows select different
  experts, so there is no shared weight tile to amortize in general.
- `M*B == 8` exactly still dispatches to `qmm_sm80` (the `< 8` gate is
  unchanged); the multirow kernel accepts up to 8 rows, so widening the gate is
  a measurement away if an 8-row shape shows up hot.
- Upstream MLX issue: the gap is upstream territory (`qmv.cu` is stock MLX);
  an issue draft with this evidence is attached to the PR for filing against
  ml-explore/mlx.
- #735 (pad the MTP verify past the qmv threshold) is superseded for K in
  [2,7]: the verify now amortizes without padding.
