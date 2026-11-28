# PR #3397: Inkling 2 Small on the real checkpoint

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3373). Narrow suites pass; validated on an M3 Ultra (512 GB) on `models/inkling-2-small` against mlx-vlm 466069f5

## Executive Summary

Inkling 2 Small loads and generates on a 512 GB host, and its output matches mlx-vlm: the 128 greedy tokens are identical, and teacher-forced traces have no disagreement at a decided position (0/409 text, 0/88 image). The first run exposed one Small-specific problem: the first token built a contiguous copy of most of the expert weights while the originals were still held, so the process peaked at 459 GiB and the host fell to 10% free memory. The loader now evaluates the weights one layer at a time, as mlx-vlm's README prescribes for Small. The peak drops to 367 GiB and the first prefill from 8.2 s to 0.34 s, with output unchanged. Decode runs at 19.5 tok/s (mlx-vlm 20.5), four concurrent server requests reach 36.8 tok/s in aggregate, and the bundled drafter gains 71% on a code prompt and nothing on prose.

## 1. Problem Statement

#3367 covered Small (`inkling2`, 48 layers, hidden 4608, 384 experts, no loop) only with tiny-model tests. Its target weights take 363 GiB of the host's 480 GiB wired limit, so mlxcel and the mlx-vlm reference can only run one at a time, and any transient copy during load counts against a small margin. Upstream loads Small with the MLX cache disabled and one `mx.eval` per layer. mlxcel had no such bound.

| Risk | Impact | Handling |
|---|---|---|
| Load does not fit or peaks far above the weights | Small unusable, or the host swaps | Measured; bounded per layer |
| Small-specific numerics (no loop, 384 experts) | Wrong text with no error | Greedy and trace parity against mlx-vlm |
| Loop-only kernels taken on Small | Departure from upstream | Gate logged at load, dispatches counted |

## 2. Design

### 2.1 Where the memory went

The eager materialization policy evaluates the checkpoint tensors at load: 1348 tensors, 389.40 GB. Sanitizing then splits each interleaved expert `w13` into gate and up with `contiguous`, about two thirds of the expert bytes, and those copies stayed lazy until the first token. That token built all 48 layers' copies in one graph. The sources stayed referenced until each copy finished, and the freed source buffers sat in MLX's buffer cache, whose sizes the new copies cannot reuse. The process footprint reached 458.9 GiB and the first prefill of 22 tokens took 8.2 s.

### 2.2 Per-layer materialization (`src/loading/vlm_inkling2.rs`)

`materialize_by_layer` runs between sanitize and model construction, only under the eager policy. It sets the cache limit to 0 with a guard that restores it, then evaluates and detaches the sanitized tensors in groups: one per decoder layer in order, then everything else. Each layer's copies are made and its `w13` sources freed before the next layer is read. Vision tower tensors are left out, because the policy keeps a modality tower lazy until a request needs it. The values are those the first token would have computed, so output does not change. The load `[TTFT]` line now reports 868 tensors materialized by the policy and 480 unreferenced: the consumed `w13` sources. `layer_groups` is a pure function, and its tests cover the order, the `layers.1` against `layers.10` prefix, an index past the config, and the excluded tower.

### 2.3 Kernel gate evidence (`src/models/inkling2/kernels.rs`)

Each kernel dispatch is counted per thread, and the first dispatch of each kernel is logged at debug level. `Inkling2Model::from_weights` logs the gate decision:

```
Inkling 2 kernels: short convolution on, attention head preparation off, grouped expert matvec off
Inkling 2 short convolution kernel dispatched (first use)
```

The count is per thread, so a test reads only its own dispatches. The test checks that a gate refusal is not counted.

### 2.4 Reference scripts

`mlx_vlm_image_logit_trace.py --bounded-load` loads the way upstream's README does. `mlx_lm_logit_trace.py --mlx-vlm` already evaluated parameters in slices and now also sets the cache limit to 0 for that loop.

## 3. Validation

All runs were serial, one runtime at a time, under the GPU lock.

| Check | Result |
|---|---|
| Load before / after | 54.7 s / 59.3 s; MLX peak 362.66 / 367.04 GiB; process footprint 458.9 / 368.5 GiB; lowest free memory 10% / 27% |
| `mlxcel inspect` | 364.82 GiB for weights, KV and activations, 0.6% under the measured peak; 437.78 GiB with the generic 1.2x headroom |
| Greedy, `Explain why the sky is blue.`, 128 tokens | text identical to mlx-vlm, before and after the fix |
| Text trace, 4 x 256 positions | 30/1024 top-1, 0/409 decided, largest gap 0.320, perplexity -0.68% |
| Image trace (1280 x 720) | 600 prompt ids identical, 4/126, 0/88 decided, largest gap 0.260 |
| Server | text, `image_url` and `get_weather` (`{"city":"Warsaw"}`), streamed equals non-streamed |
| Decode, ABAB kernels off / on | 18.91 / 19.45 tok/s; mlx-vlm 20.5 |
| Prefill, 2274 tokens | 361.0 / 362.5 tok/s |
| Server aggregate, C1 / C4 | 10.8 to 17.2 / 36.5 to 37.0 tok/s |
| Mini greedy | identical between the per-layer load and the unbounded lazy path |

DSpark: Small's 5-layer drafter binds without code changes. The exactness probe declines (503855 of 800232 logit bytes), as on Mini. With `MLXCEL_MTP_ALLOW_INEXACT=1`, 256-token decode goes from 19.3 to 19.3 tok/s on prose (acceptance 0.20) and from 19.3 to 33.0 tok/s on code (0.46). An 8-row verify costs about 107 ms against a 52 ms decode step, so a round must emit more than about 2.3 tokens to pay off.

Selectors: `--lib inkling` (172 passed), `--lib coverage_tests` (18). Clippy, fmt and the contract scripts pass.

## 4. Limitations and Follow-ups

- The drafter on Small breaks even on prose. Making the verify forward cheaper (the grouped expert kernel is gated on a looped model, as upstream) would move that.
- The mlx-vlm drafter was not run on Small. Its round-by-round agreement was checked on Mini in #3384.
- Server C1 wall times varied from 7.4 to 11.9 s for the same 128 tokens. They are recorded as measured.
- Small decodes about 5% behind mlx-vlm.
