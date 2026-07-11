# Speculative drafter pairing matrix on GB10 (CUDA), 2026-07-10

> **2026-07-11 update (#725):** the multirow qmv path landed and flips this
> matrix: the same pairing now measures 1.31x (K=2), 1.46x (K=4), 1.41x (K=8)
> against the same 14.5 tok/s baseline, at unchanged acceptance. The verify
> forward amortizes as the analysis below predicted once the `M*B < 8` per-row
> qmv fallback is removed. Current numbers:
> `qmv-multirow-gb10-2026-07-11.md`. The measurements below are kept as the
> pre-#725 record.

Measured B=1 MTP speculative-decoding numbers for the Gemma 4 pairing whose
target and drafter checkpoints are present on the GB10 CUDA host, filling the
CUDA column of the speculative pairing matrix (issue #638). The headline result:
B=1 MTP with the Gemma 4 assistant is a consistent regression on GB10 across
K, so the adaptive policy is tuned to decline it there. The regression is a
kernel-dispatch effect (the small-M qmv fallback tracked in #725), not a
hardware limit; see "Why MTP loses here".

## Hardware and build

- Host: NVIDIA GB10 (Grace-Blackwell), CUDA backend.
- Build: `cargo build --release --features cuda --bin speculative_bench`.
- Sampling: greedy (`temperature = 0`). Decode-only tok/s from
  `GenerationStats::decode_tok_per_sec` (excludes prefill), matching the
  methodology in `model_tests.md`.
- Prompt: the 14-token `DEFAULT_PROMPT`, `--max-tokens 128` (both runs reach a
  natural EOS at ~83 generated tokens).
- Warm-up: one 4-token generation before the timed run (also compiles the CUDA
  kernels on first use).

## Matrix: Gemma 4 Unified 12B + `gemma4_unified_assistant`

Target `models/gemma-4-12b-it-4bit` (`gemma4_unified`), drafter
`models/gemma-4-12b-it-assistant-4bit`.

| Kind | K (block_size) | tok/s | speedup vs no-drafter | acceptance rate | mean accepted len |
|------|----------------|------:|----------------------:|----------------:|------------------:|
| none (baseline) | — | 14.5 | 1.00× | — | — |
| mtp | 2 | 11.1 | 0.77× | 55.6% | 0.56 |
| mtp | 4 | 7.6 | 0.52× | 35.0% | 1.05 |
| mtp | 8 | 7.5 | 0.52× | 35.0% | 1.05 |

Invocation (one process per row; each reloads the target so a single run stays
well under the 5-minute wall-clock budget):

```bash
./target/release/speculative_bench --target models/gemma-4-12b-it-4bit \
    --kind none --max-tokens 128
./target/release/speculative_bench --target models/gemma-4-12b-it-4bit \
    --draft models/gemma-4-12b-it-assistant-4bit --kind mtp \
    --block-size {2,4,8} --max-tokens 128
# or the full sweep at K in {2,4,8}:
./target/release/speculative_bench --sweep --k-values 2,4,8 --max-tokens 128
```

## Why MTP loses here

The Gemma 4 assistant accepts too few drafts on this pairing to pay for the
extra forwards. At K=4 the drafter proposes 3 tokens per round and the target
accepts 1.05 on average (35%), so each round emits ~2 tokens but costs a K-wide
verify forward plus the drafter forward. On GB10 that round costs more than the
~2 classic forwards it replaces, so the net is 0.52×.

K=8 lands on the same acceptance (35%, mean 1.05) as K=4 because the drafter's
configured block size is 4: the adaptive block controller
(`effective_mtp_block_size`) holds the verify block at the configured 4 and only
expands toward the requested ceiling after recent rounds show the configured
prefix is usually fully accepted. That gate never trips at 35% acceptance, so
`--block-size 8` runs at an effective K=4. K=2 accepts a higher fraction (55.6%)
but proposes only one draft per round (mean accepted 0.56), so its best case is
still 0.77×.

The verify cost is linear in K, and the mechanism is the quantized-matmul
dispatch, not the GPU's compute budget. The MTP verify is a `[1, K]`
sequence-axis forward, so every quantized linear in it runs with M = K rows.
The CUDA dispatch
(`src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/quantized.cpp`) routes
`M*B < 8` to per-row `qmv`, and the sm90 CUTLASS GEMM is Hopper-gated, so on
sm_121 a K=4 verify executes as four narrow GEMVs that each re-read the full
weight matrix. This is the same root cause as the flat batched-decode
throughput tracked in #725.

The measured numbers fit this model exactly. The classic forward is 69 ms
(14.5 tok/s); the K=2 round is ~140 ms (2.0 forwards) and the K=4 round is
~271 ms (3.9 forwards), with the 258 MB drafter contributing only a few ms.
The measured speedups are tokens-per-round divided by round cost:
1.56 / 2.0 = 0.78x (measured 0.77x) and 2.05 / 3.9 = 0.53x (measured 0.52x).
An earlier revision of this note read the `--block-size 8` row as evidence of
sub-linear growth ("a 4x wider verify for ~1.9x the time"); that row runs at
an effective K=4 (see above), so the datum actually shows a 2x wider verify
for 1.94x the time, i.e. linear scaling.

## Contrast with Apple Silicon

The same pairing measures ~1.87x on M5 Max (`gemma4-mtp-speculative-decoding.md`).
On Metal the K-wide verify reads the target weights once, so verifying K
positions is nearly as cheap as decoding one token; the speculative path even
pads the verify up to a 32-token NA tile on M5+ to force the tiled `qmm_nax`
GEMM instead of GEMV (`speculative/mod.rs`). The CUDA path has no analogous
padding (#735), and with the `M*B < 8` qmv fallback the verify runs as K
narrow forwards instead.

This is a kernel gap, not a hardware limit. GB10 has more compute headroom per
byte than Apple Silicon (~273 GB/s of memory bandwidth alongside ~100 TFLOPS
dense bf16, versus ~614 GB/s on M5 Max with far less compute), so an
amortizing kernel would make the K-wide verify nearly free here too. Runtimes
with mature Blackwell kernels demonstrate it on the same hardware: LMSYS
measured SGLang batched decode scaling near-linearly on a DGX Spark
(llama-3.1-8b, 20.5 tok/s at B=1 to 368 tok/s at B=32) and EAGLE-3
speculative decoding reaching up to 2x end-to-end
(<https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/>), and NVIDIA ships
speculative-decoding recipes for the same box
(<https://build.nvidia.com/spark/speculative-decoding>). Once #725 lands an
amortizing small-M path, this pairing's arithmetic flips positive even at the
measured acceptance: ~2.05 tokens per round for ~1.2 forwards of round cost
is ~1.7x at K=4.

## Policy tuning applied (issue #638)

The adaptive MTP policy's speedup estimator assumed a bandwidth-bound verify
(`verify_cost_multiple == 1.0`), which is optimistic on GB10 and would wrongly
enable this regressing pairing. The estimator is now backend-conditional
(`mtp_policy::estimate_speedup_scaled`): on non-Apple-Silicon hosts (detected
as `AppleSiliconGen::Unknown`) the K-wide verify cost is scaled by `sqrt(K)`.
Apple Silicon keeps `verify_cost_multiple == 1.0`, so its estimates and
verdicts are byte-identical to before. With the de-rating, the profiler
settles to a decline for this pairing on GB10 after its bounded profiling
window, while a pairing with high accepted length still clears the enable
floor.

Known limitation (#736, resolved): the `sqrt(K)` constant was calibrated
against the "~1.9x growth from K=2 to K=8" datum, which the section above
corrects to effective K=4; the true verify scaling is linear, so `sqrt(K)`
under-costs a K=4 round by about 2x (sqrt(4) = 2 versus the measured ~3.9
classic-forward equivalents). The decline verdict for this pairing still held
because its accepted length was low, but a moderately better pairing could
have been wrongly enabled. #736 replaced the shape heuristic with an
estimator measured against classic-step probe rounds (hint format v3);
`sqrt(K)` remains only as a fallback for windows with no probe signal.

## Follow-ups

- #725: amortizing small-M quantized GEMM, the root-cause fix. Re-run this
  matrix once it lands.
- #735: pad the CUDA MTP verify past the qmv dispatch threshold, the
  verify-side consumer of #725.
- #736: measured-round-cost policy estimator replacing `sqrt(K)`. Done; see
  `qmv-multirow-gb10-2026-07-11.md`.
- #737: acceptance-rate cross-check on Apple Silicon. GB10's 35-56% is below
  the 70-87% third parties report for the Gemma 4 assistants, and the M5 Max
  run predates the bench's acceptance reporting.

## Deferred rows

- **Gemma 4 31B + MTP assistant** (`gemma-4-31b-it-4bit` +
  `gemma-4-31b-it-assistant-bf16`): the 31B checkpoint loads as the batch-capable
  `gemma4` text family, not `gemma4_unified`, so the current bench MTP driver
  (which targets the Unified decode path) declines it; the bf16 assistant is also
  large. Not measured on GB10 this session.
- **Qwen 3.5 DFlash**: no Qwen 3.5 target or DFlash drafter checkpoint is present
  on the GB10 host, so the DFlash numerator stays deferred (unchanged from the
  reference host).
