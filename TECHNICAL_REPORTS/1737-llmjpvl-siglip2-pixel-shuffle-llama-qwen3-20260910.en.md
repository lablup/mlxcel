# Technical Report: PR #1737 - feat(llmjpvl): port LLM-jp-4 VL and Jagle-VL

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation and validation cycle
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium (new model family reusing an existing runtime; four silent-failure divergences from that runtime; validated against both released checkpoints and their own reference implementation)

---

## Executive Summary

PR #1737 adds `model_type: "llmjpvl"`, the LLM-jp lab's Japanese VLM family: one architecture over two released checkpoints whose only real difference is the decoder named by `llm_config.model_type`, Llama for `llm-jp-4-vl-9B-beta` and Qwen3 for `Jagle-VL-2.2B-Jagle-FineVision`. A SigLIP2-so400m tower runs behind InternVL-style dynamic tiling, `pixel_shuffle(0.5)` folds each 512 px tile into 256 vectors of width 4608, and the `mlp1` connector projects them into the decoder width where they replace the `<|image_pad|>` positions. The InternVL runtime is reused as the issue asked, and four things around it differ in ways that fail silently: the decoder config sits under `llm_config` rather than `text_config`; the SigLIP tower has no CLS token, so InternViT's `[:, 1:, :]` slice would drop a real patch row; the `mlp1` LayerNorm takes torch's default 1e-5 rather than the vision config's 1e-6; and the runtime has to carry either decoder. Two prompt defects that no synthetic test would have caught were found and fixed: InternVL's "splice after the first token" fallback puts the image block inside the *system* turn for this template, and the server was handing a template that never inspects typed media a content *list* for `{{ message['content'] }}`. A tiling tie-break inherited from the shared InternVL processor also disagreed with the upstream rule both families read, returning 1 tile for a 768x768 image where the reference returns 5. Validation is against the checkpoints' own `modeling_llmjpvl.py`: both backbones reproduce the reference's prompt token count and its greedy output exactly on the CLI, and on the server Jagle-VL is byte-identical while the 9B's one-token difference is attributed to the batched decode path (`--max-batch-size 1` matches). 32 new tests, plus 269 passing in the regression scopes of the shared files touched.

---

## 1. Problem Statement

### 1.1 Background

Issue #1362 asked for `model_type: "llmjpvl"`, the LLM-jp lab's Japanese VLM family. Two released checkpoints declare it and share one architecture, differing only in the decoder that `llm_config.model_type` names: `llm-jp/llm-jp-4-vl-9B-beta` (`llama`, hidden 4096, 32 layers, 196608-token vocab, untied LM head) and `llm-jp/Jagle-VL-2.2B-Jagle-FineVision` (`qwen3`, hidden 2048, 28 layers, tied embeddings, per-head q/k RMSNorm). Both put a SigLIP2-so400m tower (512 px, patch 16, 1024 patches per tile, 27 layers of width 1152, `gelu_pytorch_tanh`) behind InternVL-style dynamic tiling, fold 2x2 patch groups with `pixel_shuffle(0.5)` into `[tiles, 256, 4608]`, and project through a `LayerNorm / Linear / GELU / Linear` `mlp1` into the decoder width.

The issue named the InternVL runtime as the thing to reuse, and that is what this port does: `InternVLConnector`, `InternVLProcessor`, `insert_internvl_image_tokens` and the `SigLipVisionModel` tower are all consumed unchanged. What is new is the arrangement around them.

### 1.2 What actually differs from InternVL

Four things, and each of them is silent when got wrong:

- **The decoder config key.** These checkpoints carry `llm_config` and no `text_config`. The InternVL loader reads `text_config` and would fail outright, which is the benign failure; a converted checkpoint that renamed the key is accepted as a fallback.
- **No CLS token.** `InternVLChatVLM::get_input_embeddings` slices `[:, 1:, :]` off the tower output because InternViT prepends a CLS token. SigLIP does not. Copying that slice would drop a real patch row and shift every remaining one, which produces fluent but ungrounded output rather than an error.
- **The connector epsilon.** Upstream builds the `mlp1` LayerNorm as a bare `nn.LayerNorm(vit_hidden_size * 4)` and never passes `eps`, so it normalizes at torch's default 1e-5. `vision_config.layer_norm_eps` is 1e-6, which is what the InternVL loader passes at the equivalent call site.
- **Two decoders behind one `model_type`.** The runtime has to carry either a Llama or a Qwen3 graph, chosen at load from a config string.

### 1.3 Two hazards this repository has been burned by before

- **Synthetic parity is not checkpoint evidence.** The two checkpoints disagree on every image and stop id (14 / 15 / 16 and stops 2, 11 for the 9B; 151655 / 151669 / 151670 and stops 151675, 151645, 151672 for Jagle-VL), so any constant would serve at most one of them.
- **A VLM can pass every CLI test while the server runs zero decoder layers.** The CLI builds its own caches, so an unoverridden `sequence_state_layout()` plus `supports_batching() == false` hands the scheduler an empty cache vector and the model silently does nothing (the Falcon-OCR precedent, PR #1075).

---

## 2. Technical Review

### 2.1 Routing and the two-decoder runtime

`src/models/detection.rs` routes `"llmjpvl"` to `ModelType::LlmJpVLM`, pinned by a test that writes a temporary `config.json` for each released shape (Llama 9B and Qwen3 2.2B) and asserts both reach the same model type. The detector deliberately does not look at `llm_config`: one `model_type` covers both checkpoints, and the decoder choice belongs to the loader, not to routing.

`src/vision/llmjp_vl_text.rs` holds `LlmJpTextModel`, a two-arm enum over `Llama3Model` and `Qwen3Model` that implements `LanguageModel` by delegation and adds `get_embed_tokens` (which the trait does not carry and the merge needs). It exists because the two backbones are distinct Rust types with no shared supertrait beyond `LanguageModel`. `qwen2` is accepted alongside `llama` because mlxcel already serves that graph with the same Llama backbone, so a conversion that relabelled the decoder still loads. Anything else is a load error naming the value, raised **before** the weight load rather than after: the 9B checkpoint is 18 GB and an operator pointing at an unsupported backbone should learn that in a second.

### 2.2 The tower, and the CLS slice that is not there

`src/vision/llmjp_vl.rs` builds the tower with `SigLipVisionModel::from_weights_with_quant_and_gelu(&weights, &vision_cfg, "vision_backbone.vision_model", ...)` and consumes its `post_layernorm` output whole. The InternVL runtime slices `[:, 1:, :]` at the same point; doing that here would drop patch row 0 and shift the remaining 1023, which the model would answer through rather than fail on.

The conv layout needs no special handling: `VisionEmbeddings::from_weights` already sanitizes `[O, I, kH, kW]` into `[O, kH, kW, I]` by a shape test that both released bf16 originals (`[1152, 3, 16, 16]`) and any already-converted checkpoint satisfy. The attention-pooling `head.*` tensors are dropped from the weight map at load; nothing reads them, and they are roughly 20 MB of bf16.

The decoder tensors are **moved** out of the weight map rather than copied, because the 9B decoder is 18 GB and neither the tower nor the connector reads it.

### 2.3 The connector epsilon

`InternVLConnector::from_weights` is called with `LLMJP_MLP1_LAYER_NORM_EPS` (1e-5), a named constant rather than the vision config's 1e-6, and a unit test both pins the constant against `vision_config.layer_norm_eps` and shows the two produce different embeddings on the same weights, so the choice is not incidental.

### 2.4 Prompt construction, which is where the placement risk lives

Upstream's `LLMjpVLProcessor` rewrites the templated text, replacing each `<image>` with `<|image_start|> + <|image_pad|> * (256 * tiles) + <|image_end|>`. That placeholder sits immediately after the Harmony user-turn opener, because `apply_chat_template` joins the structured content parts before the template renders `<|start|>user<|message|>{content}<|end|>`.

mlxcel renders the template first and tokenizes it, so `src/multimodal/llmjp_vl_prompt.rs` reproduces the placement on the token stream. The InternVL fallback (splice after the prompt's first token) would be wrong here: the first token of a rendered LLM-jp-VL prompt is the **system** turn's `<|start|>`, so the image block would land inside the system message. The module instead locates the last `<|start|>user<|message|>` in the rendered text, re-encodes only that prefix, and verifies it really is a prefix of the token vector the caller already produced before splicing at that offset. A mismatch degrades to prepending rather than splicing at a wrong offset, and the request keeps its original tokenization either way. A prompt that already carries bare `<|image_pad|>` placeholders takes the shared InternVL in-place expansion instead.

The per-request tile budget is upstream's, ported with floor division so the negative intermediates a long prompt produces behave the way Python's `//` does: `max_num = ((model_max_length - text_tokens) // images - 2) // image_seq_length - 1`, clamped into `[1, max_dynamic_patch]`.

`ensure_image_token_feature_cardinality` guards the scatter: `merge_llava` writes one feature row per `<|image_pad|>` position, so a placement bug that lost or duplicated a block fails loudly instead of producing garbled text.

### 2.5 The generation prompt, on both front ends

The shipped `chat_template.jinja` ends an `add_generation_prompt` render at `<|start|>assistant`. The checkpoint's own processor then appends `<|channel|>final<|message|>`, so a bare template render is an incomplete prompt and the model is left to pick a Harmony channel.

Both front ends build their prompt through `ChatTemplateProcessor::from_model_path`, so the completion is installed there as a `GenerationPromptSuffix` rule keyed on `ModelType::LlmJpVLM`. Keying on the model family rather than on the trailing text matters: other Harmony templates (gpt-oss) also end a generation render at `<|start|>assistant` and must not be forced onto the final channel. The rule fires only when `add_generation_prompt` is true and the render really does end at the opener, so the prompt cache's history-boundary render is left alone.

One further difference between the front ends had to be closed. The CLI hands the template a string `content`; an OpenAI request carries a content *list* with an `image_url` part, and `prepare_chat_request` routed any request with typed media parts to the raw-JSON render. This template never inspects typed media items (it never says `image` at all), so minijinja was handed a list for `{{ message['content'] }}` and would have printed the list itself into the prompt. The raw-JSON route is now taken only when the template actually reads typed media (or the request carries audio, where the ordered sentinels are emitted); otherwise the content flattens to text, which is both what the template expects and what the token-level expansion assumes. A test asserts byte equality between the server render and the CLI render, not a prompt-token count: two different prompts can share a count, which is how an earlier VLM port in this tree shipped a wrong prompt with matching numbers.

### 2.6 The server cache layout

`sequence_state_layout()`, `supports_batching()`, `supports_batched_prefill()`, `supports_maskless_padded_prefill()` and `supports_paged_decode_backend()` all delegate to the text backbone rather than falling to the trait default, and a test asserts the layout carries one entry per decoder layer and that `make_caches()` does not hand back an empty vector, for both backbones. The trait default would happen to be right here, but the Falcon-OCR precedent is that being right by accident is how a VLM ships running zero decoder layers on the server while every CLI test passes.

The three framing ids are returned from `output_suppressed_token_ids()` so neither the CLI nor the scheduler can sample one into the text stream; a test also asserts the stop ids stay emittable.

---

## 3. Technical Decisions

### 3.1 Verify the split point instead of trusting it

The image block has to go where upstream's `<image>` was, which is a *text* position, while mlxcel works on a token vector the caller already produced. Re-tokenizing the whole prompt around the insertion would risk a boundary difference; splicing at a guessed token index would risk landing in the wrong turn. The module does neither: it encodes only the prefix up to and including `<|start|>user<|message|>`, checks that those ids really are the head of the caller's token vector, and splices at that length. The original tokenization is preserved exactly, and a tokenizer that disagrees produces a documented fallback rather than a silently shifted block. Both `add_special_tokens` settings are tried, because which one the caller used is not visible here.

### 3.2 Complete the generation prompt in the processor, not in each front end

The suffix could have been appended in the CLI's prompt builder and again in the server's, which is two places to keep in sync and the shape that let an earlier port ship divergent prompts. Installing it in `ChatTemplateProcessor::from_model_path` puts it on the one seam both front ends already share, and makes "the two agree" a property of construction rather than of two matching edits. The rule is keyed on the model family because the trailing text alone (`<|start|>assistant`) is shared with a Harmony family that must not be completed this way.

### 3.3 Fix the typed-content route rather than special-casing this family

The content-list-into-a-string-template mismatch is not specific to LLM-jp-VL: it applies to every family whose template carries the image through token ids rather than through template text. Gating the raw-JSON route on whether the template actually reads typed media is the smaller and more honest change than teaching one loader to work around a render it should never have been given. Templates that do read typed media are unaffected, and the audio path keeps its ordered sentinels.

### 3.4 Reject `select_layer != -1` at load

Upstream reads `hidden_states[select_layer]` for any other value. Serving `last_hidden_state` instead would be features the checkpoint was not trained against, and the model would answer through it. The loader fails with the value named.

---

## 4. Validation

Host: Linux aarch64, NVIDIA GB10, CUDA sm_121. Binaries built with `--profile test-fast --features cuda`. Both checkpoints are the released bf16 originals, not MLX conversions, so the HF conv layout branch (`[1152, 3, 16, 16]`) is exercised on real weights.

### 4.1 The reference oracle

The checkpoints' own `modeling_llmjpvl.py` and `processing_llmjpvl.py` were run under `torch 2.14.0+cpu` (bf16, CPU) and `transformers 4.57.1`, greedy, `max_new_tokens=32`:

| | Jagle-VL-2.2B | llm-jp-4-vl-9B-beta |
|---|---|---|
| prompt tokens | 301 | 297 |
| tiles / image pads | 1 / 256 | 1 / 256 |
| output | `赤色です。` | `この画像は、単色の赤色で塗りつぶされた正方形の図形です。` |

Reference tile counts, both checkpoints agreeing: 224x224 and 512x512 give 1 tile; 768x768 and 1000x1000 give 5 (2x2 plus a thumbnail); 1024x768 gives 13 (4x3 plus a thumbnail, 3328 image tokens).

### 4.2 CLI, both backbones

```
$ ./target/test-fast/mlxcel generate -m models/mlx/jagle-vl-2.2b-jagle-finevision \
    --image solid_red.png -p "この画像の色は何色ですか。" -n 32 --temp 0 --profile
LLM-jp-VL: inserted 1 image block(s), 1 tile(s) under a budget of 12 (256 total image tokens)
赤色です。
  Prompt tokens:    301
  Generated tokens: 4
```

```
$ ./target/test-fast/mlxcel generate -m models/mlx/llm-jp-4-vl-9b-beta \
    --image solid_red.png -p "この画像について説明してください。" -n 32 --temp 0 --profile
LLM-jp-VL: inserted 1 image block(s), 1 tile(s) under a budget of 12 (256 total image tokens)
この画像は、単色の赤色で塗りつぶされた正方形の図形です。
  Prompt tokens:    297
  Generated tokens: 15
```

Both prompt-token counts and both generated texts are identical to the reference. The 9B answer repeated identically across three consecutive runs. Tile counts on Jagle-VL: 224x224 gives 1 tile and 256 image tokens, 1024x768 gives 13 tiles and 3328 image tokens, both matching the reference.

### 4.3 Server, both backbones

`mlxcel-server` with the same image as a `data:image/png;base64` content part and `temperature: 0`:

| | prompt_tokens | completion_tokens | content |
|---|---|---|---|
| Jagle-VL | 301 | 4 | `赤色です。` |
| llm-jp-4-vl-9B | 297 | 15 | `この画像は、単色の赤色で塗りつぶされた正方形のキャンバスです。` |

Jagle-VL is byte-identical to the CLI and the reference. The 9B differs at one token (`キャンバス` where the CLI and the reference have `図形`) and reproduced that stably across three requests, so it is a path difference rather than run-to-run noise. It is attributable: re-running the same server with `--max-batch-size 1`, which processes requests sequentially with no batch scheduler, returns `図形` and matches the CLI and the reference exactly. The prompt is not the variable, and this is pinned rather than argued: `one_image_prompts_are_token_exact_against_the_reference_processor` compares every id of the assembled prompt against the reference processor's own `input_ids` for both checkpoints, and `the_server_image_request_renders_the_same_prompt_as_the_cli` asserts byte equality between the two front ends' renders. What is left is the batched decode path's reduction order flipping one near-tie token, which is the drift class this repository records rather than gates (issue #932).

### 4.4 Gates

- `cargo clippy --lib --tests --features cuda -- -D warnings`: clean.
- `cargo fmt --all`: applied; no residual diff.
- `cargo check --lib --tests --features cuda`: clean.
- New unit tests, `--profile test-fast --features cuda`: `vision::llmjp_vl` 7/7, `multimodal::llmjp_vl_prompt` (including the parity gate) 8/8, `loading::vlm::llmjp_vl` 7/7, `server::llmjp_chat_template_tests` 5/5, the detection test 1/1, `models::metadata_tests` 4/4.
- Regression scopes for the shared files this PR touches: `vision::processors::internvl` 5/5 (including the new tie-break gate), `server::chat_request` 104/104, `server::chat_template` 154/154, `vision::internvl` 2/2, `multimodal::internvl_prompt` 4/4.

---

## 5. What Was Not Verified

- **`cargo test --workspace --profile test-fast --features metal,accelerate` was not run.** The issue lists it as an acceptance criterion, but this host is Linux aarch64 with CUDA: there is no Metal and no Accelerate here. The CUDA equivalents above are what was actually run, scoped to the modules this change touches; the full workspace suite is left to the merge gate.
- **The batched-decode divergence in 4.3 was attributed, not fixed.** Whether the batch scheduler's decode path should agree bit-for-bit with the single-sequence path is a repo-wide question, not an LLM-jp-VL one, and nothing here changes that path.
- **Only single-image requests were exercised on real checkpoints.** The multi-image path is implemented and unit-tested (per-image tile counts size each block, blocks concatenate in image order, and the feature-cardinality guard covers the scatter), but no real checkpoint was run with two images.
- **Photographic images were not compared against the oracle.** mlxcel's bicubic resize is `image`'s Catmull-Rom and upstream's is PIL's, so tile pixels differ slightly on a photo and greedy token-exactness is not expected there. The solid-color image used above is filter-independent, which is why it is the token-exact comparison.
- **No quantized `llmjpvl` conversion was loaded.** The loader inherits a top-level `quantization` block into the decoder config and passes the group size and bit width to the tower and connector (unit-tested), but no 4-bit checkpoint of this family exists to run.
- **`select_layer != -1` is rejected, not implemented**, and video input is out of scope.

---

## 6. Lessons

- **The checkpoint's own processor is the reference, and it is cheap to run for the prompt alone.** Loading `processing_llmjpvl.py` under `transformers` and asking it for `input_ids` costs seconds and no GPU, and it produced the exact id sequence this port is now pinned against. A synthetic fixture that agrees with itself would have proved nothing about either checkpoint, and the two disagree on every image and stop id.
- **A matching prompt-token count is not evidence that two front ends rendered the same prompt.** The server and CLI paths reach the template through different content shapes, and the count survives a wrong render. Byte equality is the assertion worth writing.
- **A reused runtime carries its neighbour's assumptions.** The InternVL CLS slice and the InternVL "splice after the first token" fallback are both correct for InternVL and both wrong here, and neither fails loudly. Reuse is worth it, but every borrowed line has to be re-derived against the new checkpoint's own code rather than assumed.
- **The trait default being right is not the same as the layout being stated.** `sequence_state_layout()` would have resolved correctly here by delegation, but the Falcon-OCR CRITICAL came from exactly that kind of accidental correctness one wrapper earlier.
