# Gemma 4 MTP speculative decoding

How much each Gemma 4 variant benefits from MTP (multi-token prediction)
speculative decoding, and why the answer depends heavily on hardware and on
whether the target can batch. This consolidates external reference numbers, the
behavior mlxcel ships today, and a local Apple Silicon measurement.

## Summary

| Target (mlxcel)                        | Drafter                          | M5 Max B=1 MTP        | mlxcel default        |
| -------------------------------------- | -------------------------------- | --------------------- | --------------------- |
| Gemma 4 Unified 12B (`gemma4_unified`) | `gemma-4-12B-it-assistant-4bit`  | ~1.87x (measured)       | **on** (B=1, all hardware)  |
| Gemma 4 31B (`gemma4`)                 | `gemma-4-31B-it-assistant-bf16`  | ~1.2 to 1.4x (measured) | **on** (B=1, M5+ only)  |
| Gemma 4 26B-A4B MoE (`gemma4`)         | `gemma-4-26B-A4B-it-assistant`   | limited at B=1 (MoE)    | on (B=1, M5+ only)  |

Both the 12B Unified and the 31B see a real B=1 MTP speedup on M5 Max (1.87x and
~1.3x), and mlxcel now runs B=1 (single-request) MTP by default for every MTP
target. `MLXCEL_ENABLE_MTP_B1=0` opts out, for lower-bandwidth Apple Silicon
where the B=1 verify forward may not pay for itself. The batched B>1 path stays
off by default (see the 31B section).

## External reference statistics (datacenter GPU)

The published Gemma 4 MTP numbers are measured on datacenter GPUs with batched
serving runtimes (vLLM), not single-request Apple Silicon.

- **H100 80GB, vLLM, `k=8` draft tokens, temperature 0** (jarvislabs):
  - concurrency 1: 40.3 to 125.3 tok/s, **3.11x**; TPOT 24.4 ms to 8.0 ms.
  - concurrency 16: 375 to 953 tok/s, **2.54x**.
- A separate H100 run reports ~14 to ~27 tok/s, about **1.9x**.
- Google advertises "up to 3x" with no quality loss. The drafter is the ~0.5B
  `google/gemma-4-31B-it-assistant`, which shares the target KV cache and reads
  its final-layer activations, keeping acceptance (~80%) stable as context
  grows. Recommended draft tokens start at 4 and can rise to 8 while acceptance
  holds.

On the H100 the single-request case sees the *largest* gain, because the GPU is
underutilized at concurrency 1 and verifying 8 draft positions in one forward is
nearly free. That economics does not carry over to Apple Silicon (see below).

## mlxcel on Apple Silicon

### Gemma 4 Unified 12B (measured)

Apple M5 Max (128 GB), `mlx-community/gemma-4-12b-it-4bit` target +
`mlx-community/gemma-4-12B-it-assistant-4bit` drafter, block size 4,
`temperature 0`, 200 decode tokens:

| Path                        | decode tok/s | speedup |
| --------------------------- | -----------: | ------: |
| classic decode (no drafter) |         ~39  |  1.00x  |
| MTP (B=1)                   |         ~74  | ~1.87x  |

Output is byte-identical to classic decode. The Unified target cannot batch
(`supports_batching() == false`), so B=1 is its only decode path and the
scheduler runs B=1 MTP for it by default.

Acceptance rate for this pairing is measured in
`speculative-acceptance-m5max-2026-07-11.md` (issue #737): 35% at K=4 on the
short 14-token prompt, rising to 52-88% on longer, realistic prompts. Those
same-prompt M5 Max numbers match GB10 within noise, so the acceptance is a
prompt property rather than a host difference.

### Gemma 4 31B (measured on M5 Max)

The 31B target is batch-capable, so an earlier "B=1 is slower" calibration had
mlxcel decline its singleton MTP burst by default. That no longer holds on this
M5 Max: B=1 MTP with the bf16 assistant is measured **faster** than classic
decode and stays byte-identical at `temperature 0`, so mlxcel now runs B=1 MTP
for the 31B by default as well (`MLXCEL_ENABLE_MTP_B1=0` opts out).

`models/gemma-4-31b-it-4bit` target + `mlx-community/gemma-4-31B-it-assistant-bf16`
drafter (bf16, backbone 5376, 4 layers), block size 4, 160 decode tokens:

| Prompt                          | classic tok/s | MTP B=1 tok/s | speedup | identical |
| ------------------------------- | ------------: | ------------: | ------: | --------- |
| explain speculative decoding    |         26.5  |         36.8  |  1.39x  | yes       |
| Apple Silicon paragraph         |         26.1  |         33.5  |  1.29x  | yes       |
| five data structures            |         25.5  |         30.7  |  1.20x  | yes       |

The 31B gains roughly **1.2x to 1.4x** from B=1 MTP on M5 Max (an earlier
single run reached 1.50x). That is well below the ~3x reported on an idle H100,
because Apple Silicon decode is bandwidth-bound rather than compute-starved, but
it is consistently above 1.0x, which is why B=1 MTP is now on by default for
batch-capable targets too. Lower-bandwidth Apple Silicon, where the earlier
"slower" calibration may still hold, can opt out with `MLXCEL_ENABLE_MTP_B1=0`.

### Gemma 4 on M1 Ultra (measured, issue #165)

Apple M1 Ultra (128 GB), same pairings, block size 4, `temperature 0`,
160 decode tokens, decode tok/s = `(N - 1) / (T_N - T_1)` over streamed
requests (a 1-token request absorbs the prefill cost identically in both
modes; at measurement time the burst path emitted tokens in lumps, so
inter-token client timing was not usable there. Since issue #734 the B=1 MTP
arm is tick-cooperative and streams each round's accepted tokens as they
land, so inter-token timing is usable again, arriving in per-round groups of
`accepted + 1` tokens).

**31B + bf16 assistant: a consistent regression.** Four greedy prompts:

| Prompt                          | classic tok/s | MTP B=1 tok/s | speedup |
| ------------------------------- | ------------: | ------------: | ------: |
| explain speculative decoding    |         18.2  |         15.6  |  0.86x  |
| hash table                      |         18.2  |         17.4  |  0.96x  |
| Romeo and Juliet summary        |         18.1  |         13.7  |  0.75x  |
| TCP vs UDP                      |         18.0  |         16.3  |  0.91x  |

**12B Unified + 4-bit assistant: still profitable** (~1.1 to 1.4x on prompts
with stable prose; this checkpoint's degenerate-output prompts were excluded).

Two consequences, shipped with issue #165:

- The B=1 default is now **per hardware**: non-batchable targets (the 12B
  Unified family) keep B=1 MTP on everywhere; batch-capable targets (the 31B)
  default it on only on M5+ (Neural Accelerator generation) chips.
  `MLXCEL_ENABLE_MTP_B1` overrides in both directions. The discriminator is
  GPU compute generation rather than memory bandwidth: the M1 Ultra has
  datacenter-class bandwidth (~800 GB/s) yet the drafter and K-wide verify
  forwards do not pay for themselves on its older GPU cores.
- Temperature-0 outputs on M1 Ultra are NOT always byte-identical between
  classic and MTP: several prompts diverged at word-choice level
  (e.g. "the violence escalates" vs "the fragile peace is shattered").
  This is the evaluation-path fp jitter documented during issue #203: the
  MTP verify computes logits through a `[1, K]` forward while classic uses
  `[1, 1]` steps, and near-tie argmaxes flip more readily on this GPU class
  (the same runs were byte-identical on M5 Max). Output quality is
  unaffected; both streams are valid greedy chains under jitter.

#### B>1 (batched) stays off by default

Forcing the batched burst with `MLXCEL_ENABLE_MTP_BATCH=1`, 4 concurrent
requests, 160 tokens each, `temperature 0`:

| Case                                   | classic agg tok/s | B>1 MTP agg tok/s | speedup | parity |
| -------------------------------------- | ----------------: | ----------------: | ------: | ------ |
| same-length window (true B>1 burst)    |              31.8 |              33.7 |  1.06x  | identical |
| variable-length mix (realistic)        |              32.0 |              24.8 |  0.78x  | identical |

The batched burst only groups requests that share a prompt length. A true
same-length window is a marginal 1.06x. Once prompt lengths differ, the requests
serialize into per-request B=1 bursts that head-of-line-block each other, so the
realistic mixed case is **slower** (0.78x). Output stayed byte-identical to
classic decode in both cases on M5 Max, so the earlier greedy-parity concern did
not reproduce here, though it has not been exhaustively re-validated. Because the
throughput is at best marginal and negative under realistic load, B>1 MTP stays
off by default.

Variable-length ragged batching is now available behind `MLXCEL_ENABLE_MTP_BATCH_RAGGED=1` (requires `MLXCEL_ENABLE_MTP_BATCH=1`). With this flag, requests with different prompt lengths form a single B>1 burst via left-padding to `max_prompt_len`, and greedy parity is confirmed on the 31B: every row's output is byte-identical to classic decode at temperature 0. Throughput measured ~0.94–1.13x vs classic batched decode across runs, which recovers the prior 0.78x serialization penalty but is not a consistent win. The flag stays off by default for this reason.

### Gemma 4 26B-A4B (MoE)

For the MoE variant, expert-weight loading at batch 1 dominates, so the drafter
may not yield a speedup on platforms without strong parallelism. mlxcel keeps it
on the classic path by default.

## Why Apple Silicon differs from the H100 numbers

The datacenter gains assume a compute-rich GPU that is idle at low concurrency,
so the speculative verify is almost free and B=1 wins by ~3x. Apple Silicon
decode is bound by unified-memory bandwidth, so the verify forward over several
draft positions is not free and the gain is smaller: the 31B measures ~1.3x and
the 12B Unified ~1.87x here, versus ~3x on an idle H100. The gain is real on
both platforms; its size tracks how much spare compute the verify forward can
use. Higher concurrency narrows it further, which is why the H100 number drops
from 3.11x at one request to 2.54x at sixteen.

The 12B Unified pair has the extra advantage that it cannot batch, so B=1 is its
native path rather than a fallback, and its 4-bit assistant accepts enough tokens
per round that one verify forward replaces two to three target forwards.

## Reproduce

```bash
# Classic baseline (disable B=1 MTP).
MLXCEL_ENABLE_MTP_B1=0 mlxcel serve -m models/gemma-4-31b-it-4bit --port 8094

# B=1 MTP (on by default; pass an absolute --draft-model path or one already in
# the model store).
mlxcel serve -m models/gemma-4-31b-it-4bit \
  --draft-model ~/.cache/mlxcel/models/mlx-community/gemma-4-31B-it-assistant-bf16 \
  --draft-kind mtp --port 8095

# Send the same temperature-0 completion to each and compare decode tok/s; at
# temperature 0 the two outputs must be byte-identical.
```

See [Benchmarks, Speculative decoding (MTP)](../benchmarks.md) for the metadata
to record and the `speculative_bench` harness shape.

## Sources

- [Gemma 4 MTP vs DFlash benchmark, H100 (jarvislabs)](https://jarvislabs.ai/blog/gemma-4-mtp-vs-dflash-benchmark)
- [Speed-up Gemma 4 with Multi-Token Prediction (Google AI for Developers)](https://ai.google.dev/gemma/docs/mtp/overview)
- [Accelerating Gemma 4 with multi-token prediction drafters (Google blog)](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [google/gemma-4-31B-it-assistant (Hugging Face)](https://huggingface.co/google/gemma-4-31B-it-assistant)
