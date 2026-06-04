# Gemma 4 MTP speculative decoding

How much each Gemma 4 variant benefits from MTP (multi-token prediction)
speculative decoding, and why the answer depends heavily on hardware and on
whether the target can batch. This consolidates external reference numbers, the
behavior mlxcel ships today, and a local Apple Silicon measurement.

## Summary

| Target (mlxcel)                        | Drafter                          | M5 Max B=1 MTP        | mlxcel default        |
| -------------------------------------- | -------------------------------- | --------------------- | --------------------- |
| Gemma 4 Unified 12B (`gemma4_unified`) | `gemma-4-12B-it-assistant-4bit`  | ~1.87x (measured)       | **on** (B=1)  |
| Gemma 4 31B (`gemma4`)                 | `gemma-4-31B-it-assistant-bf16`  | ~1.2 to 1.4x (measured) | **on** (B=1)  |
| Gemma 4 26B-A4B MoE (`gemma4`)         | `gemma-4-26B-A4B-it-assistant`   | limited at B=1 (MoE)    | on (B=1)      |

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

**B>1 (batched)** stays declined regardless: the batched MTP burst is not
consistently faster than classic batched decode on the 31B and does not preserve
greedy parity there yet (`MLXCEL_ENABLE_MTP_BATCH=1` forces the experimental
path).

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
