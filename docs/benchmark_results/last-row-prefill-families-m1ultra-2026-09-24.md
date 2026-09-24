# Last-row prefill for eight more families (issue #1970)

Qwen3, Qwen3-MoE, Phi-3, Mixtral, Gemma 2, Gemma 3, Jamba, and Nemotron-H used
the `LanguageModel::forward_last_logits` default: a single-sequence prefill
computed `[1, L, vocab]` logits through the LM head and kept one row. Each
model's forward is now split into a hidden-state path ending at the final norm
and a head (Gemma 2 keeps its final-logit softcap on the head side), and the
three `forward_last_logits*` entry points slice the hidden state at the sampled
position before the head, as Cohere2 (#1955) and Llama3 (#1969) already do.
Models with per-sequence state (the Gemma 3 wrapper, Jamba, Nemotron-H) use the
same state closure and mask rules as their existing `forward` and
`forward_with_sequence_id`. Models without an embeddings input route the
embeddings variant to the plain path, as the trait default did.

- **Date:** 2026-09-24
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** main `13b6d33a` binary against the change, ABBA order
- **Condition:** `mlxcel-bench-decode` with `bench_decode.sh`'s arguments
  (warmup 20, tg128, `--ignore-eos`), `--prompt-tokens` 512, prefill tok/s
- **Machine load:** indexers suspended; load average 3.7 to 4.3

## Results (prefill tok/s, two runs per arm)

| Model | vocab | main | last row | Change |
|---|---|---|---|---|
| Gemma 2 2B | 256k | 2106.1, 2103.6 | 2659.0, 2664.7 | +26.5% |
| Gemma 3 4B | 262k | 1025.6, 1025.6 | 1222.1, 1222.4 | +19.2% |
| Qwen3 1.7B | 152k | 2175.2, 2165.4 | 2566.7, 2582.4 | +18.6% |
| Qwen3-MoE 30B-A3B | 152k | 887.5, 886.1 | 947.4, 945.1 | +6.7% |
| Phi-3 mini | 32k | 1520.8, 1476.9 | 1555.9, 1557.4 | about +3.8% |
| Nemotron 3 Nano 30B | 131k | 354.4, 354.5 | 365.5, 355.1 | +0.2% to +3% |
| Jamba reasoning 3B | 65k | 217.1, 217.0 | 218.8, 218.6 | +0.8% |
| Mixtral 8x7B | 32k | 337.0, 336.9 | 339.3, 339.3 | +0.7% |

The gain tracks the head's share of prefill work, which is the vocabulary size
relative to the model size: small models with 150k to 262k vocabularies gain
the most, large MoE and hybrid models with small vocabularies are near noise.
The Phi-3 and Nemotron ranges are wide because one run in one arm moved.

## Correctness

- Greedy output byte-identical to main on all eight, 100 tokens each for a
  short prompt and a 512-token prompt.
- Unit tests (Qwen3, Gemma 2, alongside the existing Llama3 and Cohere2 tests):
  a shared helper, `test_support::last_logits::assert_last_logits_match_full_forward`,
  checks all three entry points against the sliced full forward at the last and
  an interior position. Shifting the slice by one row fails both new tests.

## Not measured

CUDA and ROCm; pp2048 for these families; DeepSeek V3, which also still uses
the trait default and has no local checkpoint. Jamba, Nemotron-H, Gemma 3,
Mixtral, Qwen3-MoE, and Phi-3 are covered by the greedy comparison but have no
dedicated unit test.
