# Issue #3385: Qwen3 8B decode on GB10

Measured on 2026-11-24. This compares the retained binaries from the earlier #3385 split experiment, not a fresh build of the latest main. Binary paths and SHA-256 hashes are in `provenance.json`; the full-change source branch is `fix/paged-attention-graph-budget`, commit `06b472ce`.

## Method

- NVIDIA GB10, CUDA, Qwen3 8B 4-bit (`models/qwen3-8b-4bit`).
- FP16 KV cache; fused paged v2 path confirmed in the server log.
- Approximately 8192 input tokens per row, four concurrent requests, 3072 output tokens per row, greedy decoding, prompt cache disabled, server context capacity 65536.
- Each experiment has three rotated rounds of main-a / variant / main-b, with two repetitions per arm: 18 measurements per experiment, 36 total.
- Entire run guarded by `gpu-lock`; no builds or tests launched by this session during measurement. Lock was free after completion.
- Decode rate is aggregate throughput within the interval where all four rows stream, using the existing benchmark estimator. It is not end-to-end throughput or TTFT.
- Full change: graph byte-budget exemption plus in-kernel Q widening and host-created scale. Budget-only: exemption with the prior Q/scale implementation restored.
- The two experiments ran sequentially; compare each variant against its own controls, not directly against the other experiment's absolute rate.

## Results

| Variant | Combined control median (tok/s) | Variant median (tok/s) | Change | Control range | Variant range |
|---|---:|---:|---:|---:|---:|
| Full change | 60.520 | 61.225 | +1.16% | 60.33–60.99 | 61.02–61.36 |
| Budget only | 60.825 | 60.900 | +0.12% | 60.51–61.37 | 59.81–61.39 |

Full-change per-round deltas: +0.86%, +1.24%, +1.12%. All six variant measurements exceeded the twelve control measurements. Control-arm medians differ by 0.03 tok/s.

Budget-only per-round deltas: -1.23%, +0.06%, +0.19%. Its pooled +0.12% does not demonstrate a consistent benefit; its values overlap the control range and the first round regressed. Control-arm medians differ by 0.07 tok/s. These descriptive ranges and null-arm differences are not confidence intervals.

All 144 requests in the measured cells produced 3072 tokens. Text hashes matched between variants and controls for every corresponding round/repetition/row. The legacy summary keys named `b1_text_*` also check this B=4 cell; the name does not mean B=1 was measured.

## Interpretation and limits

The full #3385 change shows a small, repeatable improvement on this larger model in this cell. Budget-only does not show a consistent improvement. This differs from the earlier Qwen3 0.6B findings and supports model-dependent tradeoffs, not an unconditional optimization claim.

The experiment does not isolate Q widening from scale creation, establish the causal GPU mechanism, or demonstrate benefits on all larger models. B=1, short inputs, other models, other GPUs/backends, TTFT and current-main integration were not measured here. Measurements preceded the landing decision. Raw data and the reused benchmark are retained beside this report; run.sh accepts BIN_DIR for the three retained binaries.

## Adoption decision (2026-11-25)

Adopt the full change for lower execution overhead and the repeatable Qwen3 8B improvement, not as a reduction in arithmetic operations or FLOPs. Q conversion moves inside the attention kernel; conversion work remains. Removing its standalone kernel avoids an intermediate f32 buffer, and host-created scale removes a GPU fill kernel. The CUDA budget exemption changes graph grouping rather than attention mathematics.

The owner accepted the model-dependent tradeoff: Qwen3 0.6B at 8192×4 measured -1.16% initially and -1.36% on the rerun. Budget-only gave +0.33% on that small model, within control variation, and +0.12% without a consistent benefit on 8B. Do not advertise a universal throughput improvement. Small-model raw data and the previous profiler summary are included here. The profiler's lower kernel/graph counts were measured on Qwen3 0.6B and the separate sliding-window branch, not on 8B.

The larger sliding-window PR #3334 remains stopped; this isolated optimization does not satisfy #3256, #3257 or epic #3259.
