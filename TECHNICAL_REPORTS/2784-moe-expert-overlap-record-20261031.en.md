# Issue #2784: Record the MoE expert overlap and decline a grouped kernel

**Date**: 2026-10-31 (generated before merge)
**Status**: In review
**Languages**: Rust, Markdown
**Risk Level**: Low (one diagnostic line behind `MLXCEL_PROFILE_MOE_INNER`)

## Executive Summary

Gemma 4 26B's batched decode reads `top_k * B` expert matrices per gather. Reading each selected expert once would cut those reads by 30.4% at 8 clients and 12.5% at 4 on different prompts. Timing the live path bounds the gain at about 13% aggregate at 8 clients and about 5% at 4. No kernel was built. The issue allowed closing with the number.

## 1. Problem Statement

#2690's fold cannot reach `GatherQMM`, because each pair reads a different expert. On an RTX Spark N1X, aggregate decode stays flat at 34-40 tok/s from 4 to 8 clients. The issue asked for the overlap ratio first, since it sets the possible saving.

## 2. Technical Decisions

### 2.1 Measure on different prompts

`scripts/bench_serving_concurrency.py` sends one prompt to every client. With greedy decode every row then picks the same experts: 8 distinct of 64, an 87.5% read reduction that real traffic would not show. This measurement used 8 different prompts and a separate driver. The harness's behavior also explains an earlier wrong figure: the profile that put everything before the gather at 1.9 ms repeated one prompt, and with different prompts it is about 4.8 ms.

### 2.2 Bound the gain before writing a kernel

The ratio alone overstates the gain, since a gather is not only weight reads. A temporary patch timed `gate_proj` over only the first row of each distinct expert: about 1.95 ms against about 2.5 ms for 64 pairs, so 30% fewer reads saved about 22% of a gather. Gathers are about 54% of a layer. A perfect kernel would save about 12% of a step, which ignores its cost for the other rows.

### 2.3 Decline the kernel

The kernel would need a variable row count per expert in `GatherQMM`, used only on this MoE decode path. The bound is about twice this model's run-to-run spread (about 6% with identical kernels), and the default 4 slots would gain about 5%.

## 3. Change Summary

- `src/models/gemma4.rs`: `MLXCEL_PROFILE_MOE_INNER=1` also prints the distinct-expert count.
- `docs/benchmark_results/moe-distinct-experts-n1x-2026-10-31.md`: the measurement and the decision.
- `docs/benchmark_results/cuda-qmm-fold-batch-n1x-2026-10-31.md`: the corrected attention figure and a note on the harness's single prompt.
- `docs/benchmarks.md`: index entries for both records.

## 4. Results

| Clients | Prompts | Pairs | Distinct, median | Mean read reduction |
|---|---|---:|---:|---:|
| 8 | different | 64 | 45 | 30.4% |
| 4 | different | 32 | 28 | 12.5% |
| 8 | identical | 64 | 8 | 87.5% |

## 5. Trade-offs and Residuals

- **Scope:** one model on one host. Other MoE models and GB10 were not measured. gpt-oss 20B has E=32 and may overlap more, but its mxfp4 gather takes a different kernel.
- **Forced evals:** the profile inflates absolute times, so only ratios were used.
- **Still open:** Gemma 4 26B stays flat from 4 to 8 clients. About 36% of a layer is spent before the gather, attention included, and attention is split by row.

## 6. Learning Points

- A benchmark that repeats one prompt across clients can hide or invent effects that depend on content. Dense reads do not care, but routing does.
- A cheap timing of the best case can settle whether a kernel is worth writing, and takes one rebuild.
