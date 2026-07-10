# Speculative drafter pairing matrix on GB10 (CUDA), 2026-07-10

Measured B=1 MTP speculative-decoding numbers for the Gemma 4 pairing whose
target and drafter checkpoints are present on the GB10 CUDA host, filling the
CUDA column of the speculative pairing matrix (issue #638). The headline result:
B=1 MTP with the Gemma 4 assistant is a consistent regression on GB10 across
K, so the adaptive policy is tuned to decline it on compute-bound hardware.

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

The per-round wall-clock grows sub-linearly with K: ~140 ms/round at K=2 versus
~271 ms/round at K=8 (a 4x wider verify for ~1.9x the time). The verify forward
is therefore not free on GB10, but it does not amortize to a single classic
forward the way the memory-bandwidth-bound Apple path assumes.

## Contrast with Apple Silicon

The same pairing measures ~1.87x on M5 Max (`gemma4-mtp-speculative-decoding.md`).
The economics differ because the M5 decode is memory-bandwidth-bound at B=1: the
K-wide verify reads the target weights once, so verifying K positions is nearly
as cheap as decoding one token. GB10 at B=1 is comparatively compute-bound, so a
K-wide verify runs closer to K narrow forwards and the drafter overhead is not
recovered at this acceptance rate. This is the same compute-vs-bandwidth split
noted for GB10 batched decode in the epic notes.

## Policy tuning applied (issue #638)

The adaptive MTP policy's speedup estimator assumed a bandwidth-bound verify
(`verify_cost_multiple == 1.0`), which is optimistic on GB10 and would wrongly
enable this regressing pairing. The estimator is now backend-conditional
(`mtp_policy::estimate_speedup_scaled`): on compute-bound hardware (detected as
non-Apple-Silicon, `AppleSiliconGen::Unknown`) the K-wide verify cost is scaled
by `sqrt(K)`, chosen to match the measured ~1.9x round-cost growth from K=2 to
K=8. Apple Silicon keeps `verify_cost_multiple == 1.0`, so its estimates and
verdicts are byte-identical to before. With the de-rating, the profiler settles
to a decline for this pairing on GB10 after its bounded profiling window, while
a genuinely favourable compute-bound pairing (high accepted length) still clears
the enable floor.

## Deferred rows

- **Gemma 4 31B + MTP assistant** (`gemma-4-31b-it-4bit` +
  `gemma-4-31b-it-assistant-bf16`): the 31B checkpoint loads as the batch-capable
  `gemma4` text family, not `gemma4_unified`, so the current bench MTP driver
  (which targets the Unified decode path) declines it; the bf16 assistant is also
  large. Not measured on GB10 this session.
- **Qwen 3.5 DFlash**: no Qwen 3.5 target or DFlash drafter checkpoint is present
  on the GB10 host, so the DFlash numerator stays deferred (unchanged from the
  reference host).
