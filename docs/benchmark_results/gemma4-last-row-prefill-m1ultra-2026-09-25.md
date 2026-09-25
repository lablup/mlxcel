# Gemma 4 prefill projects only the sampled row (issue #1972)

Gemma 4 prefill computed the full `[1, L, 262144]` logits and a
`final_logit_softcapping` copy, then kept one row, on every path the shipped
checkpoints take:

- `Gemma4Wrapper::forward_last_logits` kept the full-logits path up to 4096
  prompt tokens (#672) for byte-identical short-context greedy output.
- `Gemma4Wrapper` had no sequence-id overrides, so server prefill used the
  trait default at every length.
- The published checkpoints carry vision and audio weights and load as
  `Gemma4VLModel` or `Gemma4UnifiedModel`, which did not forward any
  `forward_last_logits*` entry point. Even the #672 long-prompt saving did not
  reach them.

All three entry points on the text wrapper and both VLM wrappers now slice the
final hidden state at the sampled position before the tied head and softcap,
at every prompt length, as the eleven families in #1955, #1969 and #1971 do.

- **Date:** 2026-09-25
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** main `2dfc33e5` binary against the change, ABBA order
- **Condition:** `mlxcel-bench-decode` with `bench_decode.sh`'s arguments
  (warmup 20, tg128, `--ignore-eos`), `--prompt-tokens` 512 and 2048
- **Machine load:** indexers suspended; load average 5.3 to 2.9

## Results (prefill tok/s, two runs per arm)

| Model | Prompt | main | last row | Change |
|---|---|---|---|---|
| Gemma 4 E4B 4-bit | 512 | 844.31, 844.96 | 975.26, 975.76 | +15.5% |
| Gemma 4 E4B 4-bit | 2048 | 828.61, 828.47 | 955.21, 955.31 | +15.3% |
| Gemma 4 12B 4-bit | 512 | 342.33, 342.32 | 372.57, 372.71 | +8.8% |
| Gemma 4 12B 4-bit | 2048 | 330.87, 330.95 | 359.39, 359.41 | +8.6% |

Decode is unchanged (E4B 74.6 to 79.0, 12B 38.8 to 39.2 tok/s in both arms).

## Correctness

- Greedy output byte-identical to main on E4B and 12B: a short prompt and a
  512-token prompt (100 tokens each) and an image prompt through the vision
  tower (80 tokens).
- Unit test: all three `Gemma4Wrapper` entry points match the sliced full
  forward at the last and an interior position, for sliding and full
  attention layers.

## Not measured

CUDA and ROCm; the Gemma 4 MTP target path, which keeps its own full forward;
prompts over 4096 tokens, where the text wrapper already sliced before this
change (the VLM wrappers did not).
