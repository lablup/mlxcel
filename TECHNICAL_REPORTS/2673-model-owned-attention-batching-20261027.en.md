# Native decode for model-owned attention families

**PR**: #2673 · **Issue**: #2570 · **Epic**: #2574 · **Date**: 2026-10-27

## Outcome

AFMoE, DeepSeek-V4, IQuestLoopCoder and Inkling expose native decode over isolated sequence-owned state. Inkling VLM delegates text decode. Single-row routing retains scalar behavior; masked or multi-token inputs use sequence-aware row fallback. Float synthetic multistep tests cover ragged histories, fresh admissions, cache growth and DeepSeek pooling boundaries.

## Implementation and review

Shared embedding, normalization and projections operate on the batch; attention reads each row's own state. DeepSeek pooled caches retain monotonic positions, IQuest preserves both passes, and Inkling splits its convolution state. Invalid rank, row/cache cardinality and missing/duplicate identities are rejected before state mutation. Paged shadow storage follows real visible cache lengths, and multi-row concatenation avoids left-deep growth.

The initial AFMoE batch changed the single-row full-fused sparse expert kernel to a whole-batch expert graph. The measured Trinity perplexity rose 15.440%. Final code retains the complete fused expert path per row when that kernel accepts the configuration, while attention and surrounding operations remain batched. Unsupported or disabled fused kernels retain the whole-batch path. Independent review confirmed scalar graph ordering and capability checks; a float regression exercises the declined-fusion path.

The trace tool now offers `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1` with batch rows 1. Older default batched dispatch dropped sequence IDs after identified prefill, so nominal baseline B1 traces read the wrong state. The reference binary uses baseline model code plus this instrumentation-only patch; invalid initial traces are excluded.

## Measurements

Apple M3 Ultra, 512 GB, macOS 26.6.2; release `metal,accelerate`; baseline `a29a2692`. Final Trinity uses `4b6dbbff`; Inkling/DeepSeek use `4d076353`, whose model code is unchanged by subsequent AFMoE/instrumentation work. Serving uses 512 prompt tokens, at most 128 generated tokens and concurrency 1/2/4. Trace uses 256 teacher-forced positions and decided gap >=2.

| Checkpoint | Baseline aggregate tok/s C1/C2/C4 | Final aggregate tok/s C1/C2/C4 | B4 top-1 mismatch | Decided mismatch |
|---|---|---|---:|---:|
| Trinity Nano Preview 4-bit | 88.1 / 98.4 / 98.9 | 96.6 / 158.4 / 202.3 | 91/256 | 0/16 |
| Inkling 0.6B | 152.9 / 164.8 / 177.4 | 158.1 / 200.6 / 340.9 | 0/256 | 0/96 |
| DeepSeek-V4 Flash 4-bit | 6.8 / 7.3 / 6.4 | 7.9 / 11.5 / 15.4 | 5/256 | 0/129 |

All three have identical baseline/final greedy text and corrected B1 traces, with zero recorded B1 logit delta. Final Trinity B4 PPL is 69357.6777 versus B1 69864.6970 (-0.726%); its 91 top-1 disagreements remain substantial despite zero decided mismatches. DeepSeek PPL changes 10.9590→10.8562; Inkling matches exactly. The servers report native batched decode without a clamp.

Trinity already produces degenerate greedy text and high corpus perplexity on baseline; Inkling also has very high unchanged corpus perplexity. These measurements verify preservation and throughput, not general model quality. IQuestLoopCoder and Inkling's image path have synthetic/delegation coverage only. The local non-loop IQuest checkpoint does not validate the loop model.

## Validation and evidence

Focused native/isolation/multistep/delegation tests, malformed-input atomicity tests, paged accounting tests, formatting and targeted clippy passed. The final 12-test source coverage contract passed before merge. The integrated workspace gate is owned by epic finalization; it is not claimed here as already completed.

Machine-readable ladders, comparison output, source revisions and limitations: [`benchmarks/epic-2574/model-owned-attention.json`](../benchmarks/epic-2574/model-owned-attention.json). Reproduce with `scripts/bench_serving_concurrency.py --concurrency 1,2,4 --prompt-tokens 512 --max-tokens 128 --metrics`, and compare `MLXCEL_TRACE_BATCH_ROWS=1`/`=4` using `scripts/compare_logit_traces.py`. Baseline traces additionally enable the scalar-reference flag.
