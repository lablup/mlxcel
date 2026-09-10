# Technical Report: PR #1744 - feat(got): port GOT-OCR 2.0 (SAM ViT-B + Qwen2-0.5B)

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation and validation cycle
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Medium (new model family reusing an existing tower and decoder; one shared-tokenizer change that touches another family; validated greedy token-exact against the checkpoint's own reference on both released key layouts)

---

## Executive Summary

PR #1744 adds `model_type: "GOT"`, the 0.58B document-OCR VLM released as `stepfun-ai/GOT-OCR2_0` and converted as `mlx-community/GOT-OCR2_0-{bf16,8bit,4bit}`. Architecturally there is very little new: the SAM-style ViT-B tower is the DeepSeek-OCR one reused unchanged, the projector is a single `Linear(1024, 1024)`, and the decoder is Qwen2-0.5B on the Llama-family backbone. What the port actually consists of is five things around that, and four of them fail silently when got wrong. The tiktoken loader hardcoded the HunYuan special-token table, under which `<|im_end|>` and `<imgpad>` resolve to no id at all, so the stop token and the image placeholder both vanish. The two released key layouts disagree on every prefix and on the tower neck's names. The stop set has to carry 151645, which no config file in the checkpoint names, or generation runs to the token cap instead of terminating. The checkpoint ships no chat template, so the fixed conversation `modeling_GOT.py::chat` builds has to live in mlxcel on both front ends. And the runtime has to state its sequence-state layout rather than inherit the trait default, which would hand the server scheduler an empty cache vector.

Validation is greedy token-exact against a reference assembled from the checkpoint's own files. On a committed rendered page, the bf16 conversion and the original `stepfun-ai` layout both reproduce the reference's 287 prompt tokens and all 11 generated ids and then stop on 151645. Termination is demonstrated by a differential build, not inferred. 36 new tests, plus 517 passing in the regression scopes of the shared files touched.

---

## 1. Problem Statement

### 1.1 Background

Issue #1359 asked for GOT-OCR 2.0, `model_type: "GOT"`, `architectures: ["GOTQwenForCausalLM"]`. The stack is short. A SAM-style ViT-B tower (1024x1024 input, window attention 14, global blocks 2/5/8/11, decomposed relative-position bias, a two-conv neck, two stride-2 compressor convs) turns a page into a 16x16 grid of width 1024, so exactly 256 feature rows. A single `Linear(1024, 1024)` projects them. A Qwen2-0.5B decoder (24 layers, width 1024, q/k/v biases, tied embeddings) reads them.

mlxcel already shipped every one of those pieces. `SamEncoder` in `src/vision/encoders/deepseekocr_sam.rs` is the identical tower: `got_vision_b.py` and mlx-vlm's `sam.py` agree on the geometry, on the `1e-6` neck LayerNorm epsilon, and on the bias-free `net_2` / `net_3` compressor, so `SamConfig::default()` describes GOT exactly. `src/models/qwen2.rs` re-exports the Llama-family backbone, which honours attention bias and tied embeddings. `merge_llava` is the fusion. The issue's framing was therefore right: reuse the named pieces, and implement what is genuinely new.

### 1.2 What is genuinely new, and why each piece is dangerous

- **The QWen tiktoken special table.** `TiktokenTokenizer::from_file` built exactly one table, HunYuan's: `<|endoftext|>, <|startoftext|>, <|bos|>, <|eos|>, <|pad|>, <|extra_0..204|>`, numbered from `encoder.len()`, then overridden from `tokenizer_config.json`'s `added_tokens_decoder`. GOT ships `qwen.tiktoken` with an **empty** `added_tokens_decoder`, so there is nothing to override with. Under the HunYuan table `<|im_end|>` and `<imgpad>` are not in the vocabulary at all, which means the stop token cannot be sampled and the 256-token image block tokenizes as literal text.
- **Two key layouts.** The original ships `model.vision_tower_high.*` with the tower neck at its `nn.Sequential` indices `neck.0..3`, `model.mm_projector_vary.*`, `model.*` for the decoder, and a tied `lm_head.weight` copy. The MLX conversions ship `vision_tower.*` with the neck renamed to `conv1 / norm1 / conv2 / norm2`, `multi_modal_projector.*`, `language_model.model.*`, and no `lm_head`.
- **The stop set.** `config.json` declares `eos_token_id: 151643` and nothing else. An OCR answer never emits `<|endoftext|>`. Upstream stops on the `<|im_end|>` turn separator through a `KeywordsStoppingCriteria`, which is a generation-loop construct with no representation in the config at all.
- **The prompt.** No chat template ships in any of the three sources mlxcel reads. The conversation is a constant built inside `modeling_GOT.py::chat`.
- **The server path.** A VLM can pass every CLI test while the server runs zero decoder layers, because the CLI builds its own caches (the Falcon-OCR precedent, PR #1075).

### 1.3 The trap the issue warned about, checked rather than trusted

The issue's own text gave a special-token table with `<ref>` at 151851 through `<imgpad>` at 151859. Those numbers were not taken on faith. They were re-derived from the checkpoint's `qwen.tiktoken` (151643 ranks) and `tokenization_qwen.py`'s ordering (`SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND) + EXTRAS`, then `IMAGE_ST`), numbered from `len(mergeable_ranks)`. That gives 151643 + 3 + 205 + 9 = 151860, which is the `vocab_size` `config.json` declares, and 151857 / 151858 / 151859 for `<img>` / `</img>` / `<imgpad>`, which are the `im_start_token` / `im_end_token` / `im_patch_token` the same file declares. Three independent statements agree, which is what makes the table trustworthy rather than the issue asserting it.

---

## 2. Technical Review

### 2.1 The tokenizer table, and keeping HunYuan intact

`src/tokenizer/tiktoken.rs` now picks its table from the checkpoint. `declares_qwen_tokenizer_class` accepts either `tokenizer_class == "QWenTokenizer"` or a `auto_map.AutoTokenizer` entry naming it, because a conversion that keeps only the `auto_map` still has to reach the QWen table; GOT sets both. `build_qwen_special_token_list` reproduces `tokenization_qwen.py` order exactly. The `added_tokens_decoder` override runs after either table, unchanged.

This is the one shared-file change with reach beyond the new family, so the HunYuan path is pinned by its own gate. `hunyuan_table_unchanged_without_qwen_class` asserts the five named specials and `<|extra_0|>` land where they always did and that the QWen-only spellings resolve to nothing, on a synthetic three-rank vocabulary where the offsets are checkable by eye. The QWen gate is the mirror: `<|im_end|>` at 5 and `<imgpad>` at 219 on the same vocabulary, the same offsets that put them at 151645 and 151859 on GOT's.

### 2.2 The canonicalizer

`canonicalize_got_keys` in `src/loading/vlm_got_ocr.rs` folds both layouts onto one naming. It targets the **original** spelling for the tower neck, mapping the conversions' `conv1 / norm1 / conv2 / norm2` back to `neck.0..3`, because `SamEncoder::from_weights` already reads those indices for DeepSeek-OCR. Renaming four keys in the loader is smaller than threading a neck-key parameter through a shared encoder that another family depends on.

Two properties are worth stating because both are load-bearing. The function is idempotent: the converted layout's tower and projector prefixes **are** the canonical ones, so a second pass must leave them alone rather than, for instance, re-prefixing `language_model.model.*` into `language_model.language_model.model.*`. And the per-block `norm1` / `norm2` inside `blocks.N.*` must survive the neck rename untouched, which is why the rename is anchored to the tower prefix rather than applied by substring. Both are tested on the real key lists, read out of the two checkpoints' safetensors headers.

The tied `lm_head.weight` copy the original ships (151860 x 1024 bf16, roughly 311 MB) is dropped when `tie_word_embeddings` is set. The backbone ties the head to the embedding itself, so the copy is never read; keeping it would cost resident memory and let a divergent copy silently win. The drop is conditional on the config flag rather than assumed, and tested both ways.

Conv layout needs no new code: the shared `conv_channels_last` shape gate in `SamEncoder` already absorbs the difference. All four conv cases pass it correctly, `[256,768,1,1]` and `[256,256,3,3]` transposing while `[256,1,1,768]` and `[256,3,3,256]` stay.

### 2.3 The decoder config

There is no `text_config` and no `language_config`. The text config **is** the flat root. `got_text_config` clones it, rewrites `model_type` from `"GOT"` to `"qwen2"`, and removes `vision_config` and `quantization_config`. The root `quantization` block is deliberately left in place so a converted checkpoint reaches the backbone with its group size and bit width, and a test parses the result as `llama3::ModelArgs` and asserts `group_size() == 64` and `bits() == 4` rather than trusting that serde saw the block.

`attention_bias` is not forced, and that is not an oversight. GOT's `config.json` never declares it, but `FusedQKVLinear::from_weights_separate` detects q/k/v linear biases from the weight map rather than from the flag, which is how every other Qwen2 checkpoint in the tree already loads.

### 2.4 The stop set

`got_eos_token_ids` reads `eos_token_id` from `generation_config.json` and `config.json`, accepting a scalar or a list, and then adds 151645 unconditionally, de-duplicated. The constant carries the reason in its doc comment, because a future reader looking only at the checkpoint would see no justification for it anywhere.

### 2.5 The prompt, on both front ends

The conversation is fixed:

```
<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img><imgpad>x256</img>
OCR: <|im_end|><|im_start|>assistant
```

Three details are load-bearing and easy to lose. The system line carries eight literal spaces, which are the Python source indentation of a triple-quoted literal inside a method body. `<|im_end|>` closes the system turn with no newline after it. And the image block travels as ordinary text, because `<img>` / `<imgpad>` / `</img>` are real vocabulary entries once the QWen table is selected.

That conversation lives in two places, and they have to agree. `GOT_OCR_CHAT_TEMPLATE` in `src/server/chat_template.rs` is the builtin the server (and, since it is the same seam, the CLI) renders through; `build_got_prompt` in `src/multimodal/got_ocr_prompt.rs` splices the image block and wraps text that is not already framed. The builtin deliberately does not render the block: its 256 placeholders have to match the tower's feature rows, so it is spliced on the token-adjacent path like every other family here.

Placement has three rules, in order. An explicit `<image>` marker is replaced where it sits. Otherwise, in already-framed text the block goes immediately after the first `<|im_start|>user\n`, which is where upstream puts it; blindly prepending would place it ahead of the **system** turn. Otherwise the text is a bare instruction and gets the block, a newline, then the instruction, all wrapped. Already-framed text is never re-wrapped, which is what makes the server render and a bare CLI instruction converge on one prompt instead of nesting two system turns.

The builtin also drops a client-supplied `system` message and emits the fixed one. That is not politeness. The vision features are scattered into a prompt whose prefix the model saw on every training example; substituting a caller's system text there degrades OCR accuracy with nothing to point at.

### 2.6 The server path

`GotOcrVlModel` delegates `sequence_state_layout`, `supports_batching`, `supports_batched_prefill`, `supports_maskless_padded_prefill` and `supports_paged_decode_backend` to the decoder rather than leaving them to the trait default. The default reports no per-layer sequence state, which hands the server scheduler an empty cache vector; the model then runs zero decoder layers and returns fluent-looking nonsense with no error anywhere, and the CLI does not catch it because it builds its own caches from `make_caches`.

`output_suppressed_token_ids` returns the three framing ids. Sampling `<imgpad>` would also desynchronize a follow-up turn's scatter, since the count of placeholders is what pairs the prompt with the tower's rows.

`ensure_image_token_feature_cardinality` guards the scatter in the runtime arm. If the tokenizer ever resolved the tags to something other than the checkpoint's ids, the placeholder count would collapse to zero and this is where that surfaces, rather than as a model that quietly ignores the page.

---

## 3. The Reference

The oracle is assembled from the checkpoint's own files rather than from `modeling_GOT.py`, which targets transformers 4.37 and does not import under 5.18:

- `got_vision_b.py::build_GOT_vit_b()`, loaded straight out of the checkpoint directory. It is plain torch with no transformers dependency, so it runs as shipped.
- `nn.Linear(1024, 1024)` loaded from `model.mm_projector_vary.{weight,bias}`.
- A transformers `Qwen2Model` built from GOT's own config fields, with `lm_head` tied. The oracle asserts `lm_head.weight` and `model.embed_tokens.weight` are actually equal before relying on either.
- `GOTImageEvalProcessor`'s three transforms, reproduced with torchvision: `Resize((1024, 1024), BICUBIC)`, `ToTensor()`, `Normalize(CLIP mean, CLIP std)`.
- The `tokenization_qwen.py` contract: `tiktoken.Encoding` over the 151643 `qwen.tiktoken` ranks with `SPECIAL_TOKENS + IMAGE_ST` numbered from `len(mergeable_ranks)`.
- The splice from `modeling_GOT.py::forward`: `cnn_feature.flatten(2).permute(0, 2, 1)`, then `cat(embeds[:pos+1], features, embeds[pos+num_patches+1:])` at the `<img>` position, with the assertion that `</img>` really follows the run.

It ran fp32 on CPU under torch 2.14.0, so the claim under test is greedy token-exactness, not bitwise agreement.

One thing this construction proves in passing: mlxcel's channels-last reshape of the tower's `(1, 16, 16, 1024)` grid to `(1, 256, 1024)` produces the same row order as the reference's NCHW `flatten(2).permute(0, 2, 1)`, both ordering by `h * 16 + w`. Getting that wrong would transpose the page under the decoder, which reads as plausible-but-wrong text rather than as an error, so it has its own gate.

---

## 4. Validation

Linux aarch64, NVIDIA GB10, CUDA sm_121, `--profile test-fast --features cuda`. Checkpoints: `models/mlx/got-ocr2_0-bf16`, `models/mlx/got-ocr2_0-4bit`, `models/mlx/got-ocr2_0-original`.

### 4.1 Special-token ids against the real files

Re-derived from `qwen.tiktoken` (151643 ranks, max rank 151642) and `tokenization_qwen.py`'s ordering: `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645, `<|extra_0..204|>` 151646..151850, `<ref>` 151851 through `<imgpad>` 151859, totalling 151860 which is the declared `vocab_size`. `tokenizer_config.json` declares `tokenizer_class: "QWenTokenizer"` and an empty `added_tokens_decoder` on all three checkpoints. `tests/got_ocr_prompt_parity.rs` asserts mlxcel's loaded tokenizer resolves all six framing spellings to those ids on the real checkpoint.

### 4.1b The other family on the tiktoken path

Only three checkpoint directories on this host reach the tiktoken loader at all besides GOT's: `hunyuan-13b` and `hunyuan-a13b-instruct-4bit` declare `HYTokenizer`, and `phi-3-small-8k-instruct-aq4_64` declares `Phi3SmallTokenizer`. None of them can enter the new branch. `hunyuan-13b` was run end to end after the change and answers `The capital of France is Paris.` as before, which is the shared-file regression check that the unit gate on the HunYuan table cannot make.

### 4.2 Greedy parity, both key layouts

`tests/got_ocr_real_model.rs` on the committed fixture page:

| | prompt tokens | `<imgpad>` | generated ids | terminator |
|---|---|---|---|---|
| reference (fp32 CPU) | 287 | 256 | 38 1793 80577 1378 1459 7168 715 54159 419 2150 198 | 151645 |
| `got-ocr2_0-bf16` | 287 | 256 | identical | 151645 |
| `got-ocr2_0-original` | 287 | 256 | identical | 151645 |

Both decode to `"GOT OCR two point zero \nrenders this page"`. The test also asserts the run terminated strictly inside the 64-token cap, never emitted `<|endoftext|>` mid-stream, and that `make_caches` returned a non-empty per-layer cache.

The prompt is pinned by ids rather than by count: `got_prompt_ids_match_the_reference_tokenizer` compares the 22-token head, the 9-token tail, the length, and that the run between `<img>` and `</img>` is exactly 256 uniform placeholders, against vectors produced by the reference tokenizer.

### 4.3 CLI, all three checkpoints

`OCR: ` on the rendered page, `-n 1024`:

| | prompt tokens | generated | output |
|---|---|---|---|
| `got-ocr2_0-bf16` | 287 | 7 | `GOT OCR two point zero` |
| `got-ocr2_0-4bit` | 287 | 7 | `GOT OCR two point zero` |
| `got-ocr2_0-original` | 287 | 7 | `GOT OCR two point zero` |

All three transcribe verbatim and stop at 7 tokens against a 1024 budget. A three-line page transcribes as `Hello world\nThe quick brown fox\njumps over the lazy dog` in 15 tokens, matching the reference exactly. `OCR:` without the reference's trailing space also works, at 286 prompt tokens; `OCR with format: ` works at 289.

`--no-chat-template`, which exercises the bare-instruction wrap instead of the builtin render, produces the identical 287-token prompt and the identical 11-token output on the fixture. The two branches converge on the real checkpoint, not only in the unit test.

### 4.4 Termination, proven

The issue warned that without 151645 generation never terminates. That was demonstrated rather than inferred. A differential build with the `push(&mut ids, IM_END_STOP_TOKEN_ID)` line removed, run on the same page with `-n 64`:

```
OCR: GOT OCR two point zero
<|im_end|><|im_end|><|im_end|>... (57 more)

[Generated 64 tokens]
```

The model emits `<|im_end|>` at exactly the position where the shipped build stops, then repeats it until the cap. `<|endoftext|>` is never reached. The probe was reverted and the shipped build re-confirmed at 7 tokens.

### 4.5 Server

`mlxcel-server` on the bf16 conversion, `/v1/chat/completions` with the page as a `data:image/png;base64` content part and `temperature: 0`:

| | prompt_tokens | completion_tokens | finish_reason | content |
|---|---|---|---|---|
| chat completions | 287 | 7 | `stop` | `GOT OCR two point zero` |

Identical to the CLI on every field. Two concurrent requests on the three-line page, one `OCR: ` and one `OCR with format: `, both returned the full transcription (287 and 289 prompt tokens, 15 and 14 completion tokens), which exercises the batched decode path.

The 4-bit conversion was served the same way and returned 287 prompt tokens, 11 completion tokens, `finish_reason: "stop"` and `GOT OCR two point zero\nrenders this page` on the fixture page. That is the same token count as bf16 and one token of content different (bf16 keeps a space before the newline), which is the quantization drift this repository records rather than gates.

That the two front ends render the same prompt is pinned rather than argued. `server_render_and_cli_instruction_tokenize_identically` renders through the real `ChatTemplateProcessor` and asserts byte equality of the assembled prompt and equality of every id, because a matching `prompt_tokens` count survives a wrong render.

### 4.6 Gates

- `cargo clippy --lib --tests --features cuda -- -D warnings`: clean.
- `cargo fmt --all`: applied; no residual diff.
- `cargo check --lib --tests --features cuda`: clean.
- New tests, `--profile test-fast --features cuda`: `multimodal::got_ocr_prompt` 12/12, `loading::vlm::got_ocr` 7/7, `vision::got_ocr` 2/2, `vision::processors::got_ocr` 5/5, `server::got_chat_template_tests` 5/5, `tokenizer::tiktoken` 4/4, the detection test 1/1.
- Real-checkpoint gates, `-- --ignored`: `tests/got_ocr_real_model.rs` 1/1, `tests/got_ocr_prompt_parity.rs` 3/3.
- Regression scopes for the shared files touched: `server::chat_template` 154/154, `loading::vlm` 216/216, `multimodal::vlm_runtime` 50/50, `models::detection_tests` 61/61, `execution::memory_estimate` 43/43, `models::registry` 9/9, `model_metadata` 8/8, `vision::merge` 4/4, `cli_help_consistency` 27/27.

---

## 5. What Was Not Verified

- **`cargo test --workspace --profile test-fast --features metal,accelerate` was not run.** The issue lists it as an acceptance criterion, but this host is Linux aarch64 with CUDA: there is no Metal and no Accelerate here. The CUDA equivalents above are what was actually run, scoped to the modules this change touches; the full workspace suite is left to the merge gate.
- **The unscoped `cargo test --lib --features cuda` is not usable as a gate on this host, and was not treated as one.** It aborts the whole test process with `terminate called ... cudaStreamEndCapture(stream, &handle_) failed`, a C++ abort rather than a test failure. Two consecutive runs on this branch died at unrelated points, one after 2184 passing tests in `loading::vlm`, the other after 1366 in `audio::phi4mm`, which is the parallel CUDA graph-capture signature this host has shown before on branches that have nothing to do with graph capture. The scoped suites listed in 4.6 are what was actually gated on. An orchestrator running the full suite afterward should expect the same abort and control-run the base branch before attributing it here.
- **The 4-bit conversion was not part of the greedy-parity gate.** It transcribes the page correctly from both the CLI and the server and its quantization block is unit-tested through `got_text_config`, but the reference oracle runs the original fp32 weights, so quantized greedy agreement is not a meaningful comparison and was not claimed; its one-token difference from bf16 on the fixture is recorded above rather than gated. The 8-bit conversion was not downloaded and was not run at all.
- **Photographic images were not compared against the oracle.** mlxcel's bicubic resize is `image`'s Catmull-Rom and upstream's is PIL's, so a photo's resampled pixels differ slightly and greedy token-exactness is not expected there. The rendered high-contrast page used above is close to filter-independent, which is why it is the token-exact comparison.
- **The fine-grained region modes were not compared against the reference.** `[x1,y1,x2,y2] OCR with format: ` and `[red] OCR with format: ` are passed through verbatim as instructions and were not exercised on a page with a known region answer.
- **Text-only requests are not a supported mode and were not made to work.** With no image the runtime arm never runs, and the model produces junk, which is inherent to an OCR-only checkpoint whose reference `chat()` always passes an image. Behaviour was left alone rather than papered over.
- **Multi-image is refused, not implemented.** The decoder was trained with a single 256-token block and upstream's `forward` asserts one `</img>` per `<img>`, so a second image is rejected at prompt build with the count named.
- **Multi-crop OCR and the model card's rendered-HTML helpers are out of scope**, as the issue stated.

---

## 6. Lessons

- **A number in an issue body is a hypothesis, not a fact.** The issue's special-token table happened to be right, but it was only trustworthy after `qwen.tiktoken`'s rank count, `tokenization_qwen.py`'s ordering, and `config.json`'s `vocab_size` and three `im_*_token` fields were made to agree with it independently. That cross-check costs a minute and is the difference between a verified id table and a repeated one.
- **A shared tokenizer's default is a silent contract with every family that uses it.** The HunYuan table was not wrong for HunYuan; it was wrong as a universal. Adding a second table needed a gate on the first, on a synthetic vocabulary small enough that the offsets are checkable by eye, or the change would have been unfalsifiable without a HunYuan checkpoint on disk.
- **Some behaviour is only in the reference's generation loop, not in its config.** The `<|im_end|>` stop lives in a `KeywordsStoppingCriteria` that no config file mentions. Reading `config.json` and `generation_config.json` alone would have produced a model that never terminates, and the failure would have looked like a decode bug rather than a missing stop id.
- **Removing a line is a better proof than reading one.** Asserting "generation stops on 151645" from the code is circular. Rebuilding without it and watching the model emit `<|im_end|>` at exactly the position where it previously stopped is not, and it cost two builds.
- **When a checkpoint ships no template, the two front ends will diverge unless one artifact makes them converge.** The builtin template plus a builder that recognizes already-framed text is what keeps a CLI `-p` and a server chat request on the same bytes; a byte-equality test between them is what keeps it that way.
