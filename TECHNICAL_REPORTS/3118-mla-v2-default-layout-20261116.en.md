# Technical Report: PR #3118 - Keep DeepSeek V2's decompressed MLA cache as the default, by measurement

**Date**: 2026-11-16
**Status**: Completed (decision recorded; no model code changed)
**Risk Level**: Low

## Executive Summary

#3102 asked whether DeepSeek V2 should default to the absorbed (latent) MLA cache, which peaks about 6.7 GB lower at 28K tokens with 2048-token chunks. The default had to hold decode throughput within a few percent and show no material trace disagreement. The trace condition passed: 0 of 3,653 decided positions disagree across four shapes, and perplexity moves at most 0.03%. The throughput condition failed. Absorbed decode is 9.9% slower at pp512/tg128 (180.7 against 200.6 tok/s), 3.7% slower behind a 2048-token prompt, and 57% faster behind a 16K prompt. The decompressed layout therefore stays the default, `MLXCEL_MLA_ABSORBED=1` stays opt-in, and `docs/mla-absorbed-decode.md` records the numbers and what would change the decision.

## 3. Technical Decisions

- **Short-context decode decides.** That is the condition most deployments hit, and the issue set its bar there. The crossover is near 3K tokens of context. Absorbed decode carries a fixed per-step cost of about 0.55 ms, a lead that a later issue can chase: the dense f16 folded `W_UK`/`W_UV` contractions or the composed attention.
- **No other model is affected.** Only `deepseek_v2` reads the switch. V3 and V3.2 always absorb, and minicpm3 and dots1 have no absorbed path.
- **Harness fix.** `examples/logit_trace.rs` printed the MLA path counters on only one arm, so width-one decode traces could not show which path they took. Both arms now print them.

## 7. Change Summary

| measurement | decompressed (default) | absorbed |
|---|---|---|
| decode pp512/tg128, median | 200.6 tok/s | 180.7 tok/s |
| decode behind 16,384 tokens | 70.9 tok/s | 111.2 tok/s |
| peak at 28,705 tokens, 2048 chunks | 20.20 GB | 13.52 GB |
| peak, single pass | 48.55 GB | 41.02 GB |

Details: `docs/benchmark_results/mla-v2-default-layout-m5max-2026-11-15.md`.
