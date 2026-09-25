# Llama3-family prefill projects only the sampled row (issue #1968)

`Llama3Model`, which also serves Mistral and Qwen2/Qwen2.5, used the
`LanguageModel::forward_last_logits` default: a single-sequence prefill computed
`[1, L, vocab]` logits through the LM head and kept one row. The model now
slices the final-normed hidden state at the sampled position before the head,
as Cohere2 has since #1955. `VisionLanguageModel` now forwards the three
`forward_last_logits*` entry points to its text model, so the same saving
applies behind the vision wrapper.

- **Date:** 2026-09-24
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** one binary; the old full-logits slice reproduced through a
  measurement-only switch against the change, ABBA order
- **Condition:** `mlxcel-bench-decode` with `bench_decode.sh`'s arguments
  (warmup 20, tg128, `--ignore-eos`), `--prompt-tokens` 512 and 2048
- **Machine load:** indexers suspended; load average 5.5 to 3.8

## Results (tok/s, two runs per arm)

| Model | vocab | Prompt | full logits prefill | last row prefill | Change |
|---|---|---|---|---|---|
| Llama 3.1 8B Instruct 4-bit | 128256 | 512 | 771.26, 771.07 | 825.94, 826.47 | +7.1% |
| Llama 3.1 8B Instruct 4-bit | 128256 | 2048 | 764.60, 764.40 | 818.29, 818.86 | +7.1% |
| Qwen2.5 7B Instruct 4-bit | 152064 | 512 | 817.05, 816.68 | 881.09, 880.87 | +7.8% |
| Qwen2.5 7B Instruct 4-bit | 152064 | 2048 | 816.05, 816.12 | 880.48, 880.83 | +7.9% |

Decode is unchanged (109 to 115 tok/s in both arms). The gain is smaller than
Cohere2's +14% because the vocabulary is 128k to 152k rather than 256k.

## Correctness

- Greedy output byte-identical between the arms: Llama 3.1 8B and Qwen2.5 7B,
  three prompts each (two short, one 512-token), 150 tokens; LLaVA 1.6 Mistral
  7B 4-bit with an image, 80 tokens.
- Unit test: all three entry points match the sliced full forward at the last
  and an interior position, and the caches advance; shifting the slice by one
  row fails it.

## Not measured

CUDA and ROCm, and the families still on the trait default (Qwen3, Qwen3-MoE,
Phi-3, Mixtral, Gemma 2/3, DeepSeek V3, Nemotron-H, Jamba), which are the
follow-up.
