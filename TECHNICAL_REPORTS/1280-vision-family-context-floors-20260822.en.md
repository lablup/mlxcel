# Technical Report: PR #1280 - Image context floors for LLaVA and Qwen2-VL

## Executive Summary

`xla_image_context_floor` predicts, from config alone, the largest number of prompt tokens one image can expand into, so the OpenXLA startup guard can reject a graph shape that could never admit an image. Only Molmo2 had a formula, leaving LLaVA and Qwen2-VL unguarded.

Both families now derive one. The interesting result is not the arithmetic but what Qwen2-VL's number exposed: real expansion on that family spans 64 to 16384 tokens depending on image shape, a 256x range, and the guard's existing message was giving operators advice they could not follow.

## 1. Problem Statement

`xla_image_context_floor` matched `Molmo2VLM` and returned `None` for everything else. `LlavaVLM` and `Qwen2VL` are both qualified for the OpenXLA image path, so they fell through and still hit the late-admission failure mode: the vision tower runs, then admission rejects the request for exceeding the static graph capacity.

## 2. Technical Decisions

### 2.1 Match the implementation, not the upstream family

The issue described LLaVA as needing an anyres grid walk for `llava_next` variants, and Qwen2-VL as reading `max_pixels` from `preprocessor_config.json`. Neither matches this codebase.

`load_llava_host_preprocessor` computes `mm_tokens_per_image.unwrap_or((image_size / patch_size)^2)` and applies no anyres grid, including for `llava_next` checkpoints, which `get_model_type` routes to `LlavaVLM` unless the text backbone is Granite. `llava_token_block_info` sets `use_boi_eoi: false` with empty prefix and suffix lists, so there are no framing tokens either.

`qwen_vl_processor` builds the processor with `Qwen2VLProcessor::new`, which sets `min_pixels` and `max_pixels` from constructor defaults and never reads `preprocessor_config.json`. A `max_pixels` key in a checkpoint therefore has no effect on what this path admits, and a floor derived from one would be a number the runtime does not honor.

The guard has to predict what the runtime will emit, so both derivations follow the code. This is recorded because the divergence is invisible from the issue text and will look like a bug to anyone comparing against HF behavior.

### 2.2 Share the pixel bounds rather than restate them

The Qwen2-VL floor depends on `max_pixels`, which lived as a literal in two constructors. Restating it in the derivation would create two numbers that must agree with no mechanism to keep them agreeing. `DEFAULT_MIN_PIXELS`, `DEFAULT_MAX_PIXELS` and `max_image_tokens` now live in the processor module and both call sites use them.

### 2.3 Rewrite the guard message, which the Qwen2-VL floor proved wrong

The message told the operator to set the capacity to the floor "to serve images". At a Molmo2 floor of 1834 that is followable. At a Qwen2-VL floor of 16384 it is not, given that capacity is the sequence length every decode step attends over and measured decode already falls from 3.18 to 1.41 tok/s between 256 and 2048.

It was also wrong on its own terms. A capacity below the floor serves every image whose expansion fits it. On this family that is most real images: a 768x1024 photograph expands to 999 tokens, so a 2048 capacity serves it comfortably. The old text implied such a configuration could serve no images at all.

The message now asks for the largest expansion the operator intends to serve, states that a smaller value accepts everything that fits and rejects the rest at admission after the vision tower has run, and keeps the explicit-pin escape. It carries no issue numbers: a GitHub reference is not useful to someone reading a server startup failure.

This edits `ensure_xla_image_context_capacity`, which the issue placed out of scope. The scope line was written before the 256x range was known, and shipping a correct number attached to impossible advice would have been worse than the gap it closes.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/multimodal/host_preprocessor.rs` | `llava_image_context_floor`, `qwen2_vl_image_context_floor`, `read_config_json`, new arms in `xla_image_context_floor`, rewritten guard message |
| `src/vision/processors/qwen2_vl.rs` | `DEFAULT_MIN_PIXELS`, `DEFAULT_MAX_PIXELS`, `max_image_tokens`, constructors use them |
| `src/multimodal/host_preprocessor_tests.rs` | Per-family pinned floors, tight-bound check against the real processor, missing and degenerate config cases, boundary tests, an ignored real-checkpoint test |

## 4. Review Findings

The Qwen2-VL magnitude was raised before the PR was opened rather than shipped quietly, because it changes what the guard does to a family that previously started. Three options were considered: ship the true worst case unchanged, ship it with a corrected message, or decline to derive a floor for Qwen2-VL on the grounds that an unreachable one is worse than none. The second was chosen: the number is correct and becomes the image bucket size once capacity bucketing lands, so the defect was in what the guard said about it, not in the number.

## 5. Validation

Unit tests: 26 passed in `host_preprocessor`, with the existing Qwen2-VL processor test still green.

Real checkpoints, floors derived from each checkpoint's own `config.json`:

| checkpoint | family | floor |
| --- | --- | --- |
| llava-1.5-7b-4bit | LlavaVLM | 576 |
| llava-interleave-qwen-0.5b-bf16 | LlavaVLM | 729 |
| llava-next-mistral-7b-4bit | LlavaVLM | 576 |
| qwen2-vl-2b | Qwen2VL | 16384 |
| qwen2-vl-2b-4bit | Qwen2VL | 16384 |

The Qwen2-VL validation runs the real `smart_resize` and `compute_grid_thw` over extreme shapes rather than re-evaluating the derivation, which would only have proved the formula agrees with itself:

```text
224x224   ->    64 tokens
768x1024  ->   999 tokens
4000x4000 -> 16384 tokens
8000x8000 -> 16384 tokens
20000x300 ->  7854 tokens
worst = 16384, floor = 16384
```

The worst observed count equals the floor exactly, so the bound is tight rather than a safe overestimate. That direction matters: a floor above the true maximum only costs a larger graph, while one below reinstates the failure the guard exists to prevent.

## 6. Related Work

Issue #1272 is closed by this PR, and issue #916 introduced the guard being extended. The Qwen2-VL range is the strongest argument yet for capacity bucketing, tracked separately: with one static shape the floor is a throughput tax on every text-only request, and with buckets it becomes the size of the image bucket and costs text requests nothing.

One forward-looking caveat is recorded in the code: if the LLaVA host preprocessor ever implements an anyres grid, the LLaVA floor stops being a fixed per-image count and has to be revisited. The derivation is correct for the loader as it exists, not for the family in general.
