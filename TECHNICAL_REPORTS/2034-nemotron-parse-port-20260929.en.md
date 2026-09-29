# Technical Report: PR #2034 - Port Nemotron-Parse on the seq2seq worker

**Date**: 2026-09-29

**Status**: Implemented and validated on real checkpoints; pending merge.

**Languages**: Rust (model, loader, CLI, server), Python (oracle and fixture generator, not shipped)

**Risk Level**: Medium (one change reaches every `tokenizer.json` model)

## Executive Summary

Nemotron-Parse (`nvidia/NVIDIA-Nemotron-Parse-2.0`, `model_type: nemotron_parse`) is a document OCR / layout model that writes markdown annotated with page-normalized `<x_..><y_..>` box tokens and `<class_..>` tags. It is encoder-decoder: a C-RADIOv2-H ViT-H/16 tower and a compression neck turn a white-padded 2048x1664 page into 3329 encoder states, and a 10-layer pre-norm mBART decoder, seeded with the tokenized task prompt, decodes greedily against them. mlxcel already serves one seq2seq family (Florence-2), and this port reuses its attention sublayers, dual self/cross KV cache and causal-mask helper, its CLI early-exit pattern and its single-stream server worker pattern.

Two findings reached beyond the model. The checkpoint's `tokenizer.json` serializes `padding: Fixed(9000)`, which the `tokenizers` crate applies on every encode, so mlxcel now drops serialized padding and truncation on load, as `transformers` does. And bf16 decoder activations flip two near-ties on the 8-bit export, so the decoder runs its activations in f32.

The result matches a CPU fp32 run of the checkpoint's own `transformers` code for all 79 model-chosen greedy tokens, with and without a 1.1 repetition penalty, on both the Hub checkpoint and the 8-bit export.

## 1. Problem Statement

mlxcel had no `nemotron_parse` arm. The model cannot run on the decoder-only generation loop (its decoder needs cross-attention K/V against a per-page encoder pass), and two parts of it have no counterpart in the tree: the C-RADIO tower with its CPE positional grid and teacher CLS tokens, and a decoder that is pre-norm, has no positional table, scales embeddings by `sqrt(d_model)`, and is seeded with a multi-token prompt rather than a single start token.

## 2. Change Summary

- `src/models/nemotron_parse/`: `config.rs` (the `encoder` / `decoder` sub-configs; ViT geometry derived from `args.model`, CLS and register counts and summary indices from the teacher list), `checkpoint.rs` (Hub to MLX key canonicalization and shape-gated conv rewrites), `encoder.rs` (patchify, CPE positional crop or align-corners resample, 32 pre-norm blocks through the fused SDPA kernel), `neck.rs` (1x1 conv as a Linear, the `(1, 4)` stride-4 conv as a reshape plus one matmul, summary row), `decoder.rs` (pre-norm mBART over the Florence-2 attention), `model.rs` (greedy loop, seeding checks, repetition penalty), `processor.rs` (resize, white pad, CLIP normalization, seed construction), `runtime.rs` (the `LoadedModel` unit).
- Integration: detection, `ModelType`, registries, `model_metadata`, loader, `LoadedModel`, `mlxcel generate` early exit, `mlxcel run` refusal, `mlxcel arch` catalog entry, server worker selection at both worker sites, single-stream queue admission, warmup skip, built-in chat template.
- `src/server/nemotron_parse_worker.rs`: batch-1 loop with request-boundary validation.
- `src/tokenizer/mod.rs`: `clear_serialized_padding_and_truncation` on the `tokenizer.json` load path.
- Tests: 22 model unit tests, 7 worker tests, detection, chat-template and tokenizer tests, and `tests/nemotron_parse_real_model.rs` with a committed fixture page and its generator script.
- `docs/supported-models.md`: family entry with the task-prompt tokens.

## 3. Technical Decisions

### Reuse the Florence-2 seq2seq pieces, not a shared trait

`Florence2Attention` already has the biased `q/k/v/out` projections and the one-shot cross K/V cache the mBART decoder needs, and `Florence2LayerCache`, `additive_causal_mask` and `layer_norm` (eps 1e-5, the decoder's value) are `pub(crate)`. The server worker is a sibling of `florence2_worker.rs` rather than a generalization of it: the two differ in prompt parsing, validation and output shape (Florence-2 parses task markers and returns structured coordinates; Nemotron-Parse passes the prompt through and returns text), so a shared trait would have been an abstraction over two call sites with little in common.

### Seeding follows `transformers` generate, not the issue's rule

The issue proposed tokenizing with special tokens and stripping the wrapper. The reference processor call uses `add_special_tokens=False`, and `generate` prepends `decoder_start_token_id` when the first id differs. The oracle confirmed it: `<predict_bbox><predict_classes><output_markdown>` became `[2, 50004, 50008, 50001]`. The port implements that; the default prompt yields `[2, 0, 50004, 50008, 50001, 50010]` either way.

### Repetition penalty covers the seed

`RepetitionPenaltyLogitsProcessor` sees the whole decoder sequence, including the seed's `</s>` and control tokens. The port penalizes the same set so a 1.1 penalty reproduces the reference; this was checked token for token.

### The neck conv is a matmul

A `(1, 4)` kernel with stride `(1, 4)` reads disjoint runs of four columns, so the conv is exactly `reshape [B, hp, wp/4, 4*C] @ W_flat`, with the kernel flattened in `(kw, c_in)` order from the MLX `[out, 1, kw, in]` layout. A direct-convolution unit test pins the index order, and the Hub checkpoint (torch layout, transposed at load) matching the reference is the end-to-end check.

### Decoder activations in f32

The first 8-bit run diverged at generated step 2 (`# Hello` instead of `**Hello`) and step 60 (one coordinate bin). Running the reference on the 8-bit export's weights dequantized to fp32 gave the unquantized sequence, so the quantization was not the cause. Widening either the decoder activations or the tower activations to f32 restored parity; the decoder is 10 layers of width 1024 over a short sequence, so it is the cheap choice. The quantized projections keep their bf16 scales. The reference's top-2 margins at those steps are 0.04 to 0.07.

### Serialized padding and truncation are cleared for every model

`transformers` fast tokenizers never apply `tokenizer.json` padding or truncation on a plain encode; the `tokenizers` crate applies them on every encode. Scoping the fix to this family would have left the generic prompt tokenization in `generate` and the server dispatch thread producing 9000 ids for the same checkpoint, and matching `transformers` is the correct behavior for any model. The tokenizer, loading and chat-template test modules pass with the change.

### The built-in template overrides the checkpoint's

The Hub checkpoint ships `{% for message in messages %}{{ message['content'] }}{% endfor %}`. For the image request this family requires, `content` is a typed list, so that template would render the list itself, base64 page included, as the decoder seed. mlxcel's rule is that a built-in never shadows a shipped template; this family is the one exception, and the text-only built-in is equivalent to the shipped template for string content.

## 4. Validation

- Oracle: the checkpoint's own `transformers` code with the `nvidia/C-RADIOv2-H` tower, fp32 on CPU (torch 2.14.0, transformers 5.17.0), default prompt, greedy, 80 new tokens, on a 1240x1754 page that needs no resampling.
- Hub checkpoint (f32) and 8-bit export: all 79 model-chosen tokens equal at penalty 1.0 and 1.1. The reference's 80th token is `</s>` forced by `forced_eos_token_id` at the budget (its logit margin is infinite), which mlxcel does not emulate.
- 4-bit export: reads the page (`**Hello Nemotron**` with the same boxes).
- Server: `/v1/chat/completions` on the Hub checkpoint returns the reference text; empty prompt, two images, zero images, control characters and streaming behave as specified.
- Gate: fmt, `clippy --release -p mlxcel --lib --tests -D warnings`, the three contract tests, and the unit modules listed in the PR.

## 5. Limitations and Follow-up

- Pages larger than 1664x2048 are shrunk with `image`'s bilinear filter, which differs slightly from PIL's antialiased BILINEAR taps; output for such pages is close to, not bit-identical with, the reference.
- The untied `lm_head` path for v1.x is implemented and unit-tested on a synthetic model but not run on a real v1.x checkpoint.
- The model card's table-insertion and repetition-stop logits processors and its post-processing scripts are out of scope, as is batch size above one.
- A decode that stops at the 9000-position bound reports `finish_reason: "stop"`; only an exhausted `max_tokens` reports `"length"`.
