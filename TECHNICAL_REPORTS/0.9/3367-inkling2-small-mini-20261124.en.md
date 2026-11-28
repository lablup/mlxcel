# PR #3367: Inkling 2 Small and Mini (`inkling2`, `inkling2_loop`)

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3228). Narrow unit suites pass; validated on an M3 Ultra (512 GB) against mlx-vlm 466069f5 on both Inkling 2 Mini checkpoints

## Executive Summary

Inkling 2 is a new architecture that shares little beyond a name and a few primitives with the Inkling port. This PR adds the text decoder, the looped Mini variant, the tile ViT, the native mixed-precision and Transformers-layout loaders, and the Inkling tool-call parser, and wires them into `mlxcel generate`, `mlxcel-server` and `mlxcel arch`. On the native Mini checkpoint the CLI reproduces mlx-vlm's 128 greedy tokens, and teacher-forced traces find no disagreement at any decided position for native text (0/490), native image (0/76) or the 4-bit export (0/494). Forcing the loop to one pass breaks the output, which shows the loop and its message controllers are wired in. Decode runs at about 32 tok/s on both checkpoints. The Metal kernels (#3359), the DSpark drafter and audio are out of scope, and Small has not been run on a real checkpoint.

## 1. Problem Statement

#3228 asked for both released forms: Small (`inkling2`, 48 layers, 384 experts, about 393 GB) and Mini (`inkling2_loop`, 24 physical layers with the middle 12 run three times, 128 experts, about 20 GB). The mlxcel Inkling port could not load either one. The architecture adds block attention residuals, query-dependent attention sinks, dynamic short convolutions, separate q/k/v input norms, sigmoid routing, softcapped SwiGLU experts, a looped core with cross-pass message controllers, and a RoPE ViT in place of the HMLP tower. The server could not return tool calls for this family either. `<|content_invoke_tool_json|>` was not handled, so a call came back as content text with the function name in front.

| Risk | Impact | Likelihood |
|---|---|---|
| Inkling 2 checkpoints fail to load | High: the family is unusable | Certain without this PR |
| Tool calls returned as text | High for agent use on the server | Certain without the parser |

## 2. Design

### 2.1 Module layout

| Area | Files |
|---|---|
| Text decoder | `src/models/inkling2.rs` (model, loop, logits), `inkling2/{config,quant,sanitize,conv,attention,moe,residual,layer,runtime}.rs` |
| Vision | `src/vision/processors/inkling2.rs`, `src/vision/encoders/inkling2_vit.rs`, `src/vision/inkling2_vl.rs` |
| Loading and wiring | `src/loading/vlm_inkling2.rs`, `ModelType::Inkling2VLM`, `LoadedModel::Inkling2VLM`, detection, registry, metadata, `vlm_runtime` |
| Tool calls | `src/server/tool_calls/inkling.rs`, `parser.rs`, `stream_filter.rs` |

The Inkling 1 files are unchanged except for three sanitize helpers and `repeat_kv_heads`, which became crate-visible so Inkling 2 can reuse them. The Inkling `--lib inkling` tests (129) pass.

### 2.2 Layer visits and caches

`layer_indices` lists the physical layer for every visit: `[0..core_start) + [core_start..core_end) * passes + [core_end..n)`. For Mini that is 6 + 36 + 6 = 48 visits. Each visit owns its own `InklingLayerCache`, which holds the KV cache and four f32 convolution histories, so `num_layers()` and the model-owned sequence state both count visits. The message controllers blend the previous pass's sublayer inputs into the current pass. The tiny looped-model test checks this by showing that zeroed controllers change the output.

### 2.3 Attention sinks without a rotating cache (upstream f2956e68)

Upstream uses an `AttentionSinkKVCache` for every visit of a loop model. It returns the keys from `max(0, end - max_size + 1)`, and trimming it is limited to `end - max_size`. This port keeps a chronological `KVCache` for each visit. A sliding visit keeps at most `L + W - 1 + 64` keys, where `SLIDING_KV_MARGIN` is 64, and attends over the newest `L + W - 1` of them. A zero K/V row is concatenated on every call, and the sinks column is appended to the banded mask. This sees exactly the keys the upstream cache returns.

The 64-key margin and a 16-row convolution margin (`CONV_HISTORY_MARGIN`) let a tail trim of up to 16 tokens rewind both the KV cache and the convolution state exactly. Inkling 1 instead clears its convolution state on trim. `trim_caches` checks every visit before it changes anything, so a trim that is too deep fails and leaves the state untouched. The decode-after-prefill, chunked-prefill and trim tests cover this layout. A trim of 10 succeeds and matches a fresh forward, and a trim of 40 is refused without mutation.

### 2.4 Checkpoint layouts

| Layout | Dense | Experts |
|---|---|---|
| Native (`model.llm.*`, `model.visual.*`) | BF16/FP32 cast groups | NVFP4 (layers 0-19) and MXFP8 (layers 20-23) from `hf_quant_config.json`; the u8 codes are viewed as u32 and the global scale is `2688 * amax / denominator` |
| mlx-community 4-bit (`model.language_model.*`, `model.vision_tower.*`, `lm_head.*`) | affine 4-bit, group 64, including the vision tables | Same native planes. `gate_up_proj` is bit-identical to native `w13` (interleaved), and `global_scale` is already MLX's native scale |
| mlx-vlm conversion | passes through unchanged | passes through unchanged |

Expert formats are inferred from tensor shapes and cross-checked against the hf table when one is present. The bf16-to-f16 load-time cast is skipped for the family, as it is for Inkling, because the native export declares its quantization only in the sidecar. The vision tower loads its position, class and layer-scale tables through `load_dense_param`, so an affine-quantized table is dequantized at load, and a malformed sidecar is a load error that names the tensor.

### 2.5 Tool calls

Inkling frames every segment as `<|message_model|>[recipient]<|content_KIND|>body<|end_message|>`, and a call is the `invoke_tool_json` kind addressed to the function. Thinking and calls close with the same `<|end_message|>`, so `parse_tool_calls` extracts Inkling calls from the raw output before the thinking pass, which would otherwise pair a thinking open with a later call's close. The markers are special tokens of the Inkling tokenizer, and Inkling 1's template emits the same invoke kind, so the format is claimed by marker presence, as Harmony and Kimi K3 are.

The stream filter holds a `<|message_model|>` followed by recipient-shaped text (at most 256 bytes, no `<|`) until its kind marker arrives. An invoke marker turns the whole header into one `EnterToolCall`. The first server run showed the recipient `get_weather` leaking into `delta.content`. The fixed `<|message_model|>` Strip row matched at the header's own start position, so it was consumed before the header resolved. The fix stops any delimiter at or after a pending Inkling header. The per-character parity tests fail without it.

The issue text said a malformed payload should leave the message in content. This PR follows the repository rule instead (#1743: tool markup is never content). A malformed invoke message yields no call and is dropped from the visible text. The stream already suppresses it, and the parity tests require the streamed and non-streamed content to agree.

## 3. Validation

All runs are on an M3 Ultra (512 GB), serialized under the GPU lock, against mlx-vlm 466069f5. mlx-vlm 466069f5 reads only the native layout, so the 4-bit reference loads the export through a rename-only remap. That remap renames tensors, splits `gate_up` with mlx-vlm's own `_split_gate_up`, sets `weight_scale_2 = global_scale / 2688`, and declares per-expert modes.

| Check | Result |
|---|---|
| Native text, `Explain why the sky is blue.`, 128 greedy tokens, default effort | identical to mlx-vlm (128/128) |
| Native text trace, 4 x 256 positions | 27/1024 top-1 differences, 0/490 on decided positions, largest gap at a disagreement 0.331, perplexity -0.21% |
| 4-bit text trace, same corpus | 30/1024, 0/494 decided, largest gap 0.185, perplexity -0.17% |
| 4-bit greedy text | agrees for about 45 tokens, then an undecided position picks another word |
| Native image (1280 x 720, 18 x 32 tiles) | 600 prompt ids identical to mlx-vlm; teacher-forced 128-token continuation 2/128, 0/76 decided; greedy agrees for the first 440 characters |
| Server `image_url` | byte-identical to the CLI answer |
| One-pass ablation (`MLXCEL_INKLING2_LOOP_PASSES=1`) | output degenerates into repeated sentences on both checkpoints |
| Server `get_weather` tool call | `tool_calls` entry `{"city":"Warsaw"}`, streaming and non-streaming, empty `delta.content`; the follow-up turn with the tool result answers "Warsaw is currently 4°C with light rain." |
| Decode | 32.1 tok/s native, 32.0 tok/s 4-bit (mlx-vlm: 32.8 and 38.5); peak 17.6 GB and 15.5 GB |

Unit selectors, run serially with `RUST_MIN_STACK=16777216` and `--profile test-fast --features metal,accelerate`:

| Selector | Passed |
|---|---|
| `inkling2` | 26 |
| `server::tool_calls` | 506 |
| `inkling` | 129 |
| `coverage_tests` | 18 |
| `models::detection` | 97 |
| `models::registry` | 21 |
| `model_metadata`, `model_type`, `loaded_model`, `memory_estimate` | 8, 86, 12, 70 |

## 4. Limitations and Follow-ups

- Batched decode runs each row as a separate single-sequence forward (`PerRowFallback`), and padded prefill is declined.
- The Metal kernels for the short convolution and grouped experts are #3359. Each sits behind one function (`conv::short_convolution`, `moe::expert_projections`).
- Not ported: the DSpark/MTP drafter (`mtp.safetensors`, refused by detection when a directory holds only that), audio, and `is_varloop`.
- Small has not been run on a real checkpoint.
- The 4-bit image path was checked for coherence, but no teacher-forced trace was run, because the mlx-vlm image trace script has no remap hook.
- 4-bit decode is about 17% slower than mlx-vlm (32.0 vs 38.5 tok/s). The ops path runs NVFP4/MXFP8 expert matvecs without the upstream grouped kernels, which is the gap #3359 targets.
