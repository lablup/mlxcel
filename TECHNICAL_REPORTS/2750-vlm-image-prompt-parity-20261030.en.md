# PR #2750: Align InternVL, MiniCPM-V 4.6 and Gemma 3 image prompts with the reference processors

**Date**: 2026-10-30 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Python
**Risk Level**: Medium (every image request on InternVL, MiniCPM-V 4.6 and Gemma 3 now reads a different token sequence, and template-less MiniCPM-V 4.6 / MiniCPM-o checkpoints render a different server prompt, text-only requests included)

## Executive Summary

Issue #2547 found that the image-conditioned prompt mlxcel builds for InternVL, MiniCPM-V 4.6 and Gemma 3 disagreed with the reference processors before the vision tower ran, so cross-runtime logit comparisons for those families measured the prompt rather than the model. This PR attributes each difference at the token level and fixes it in `prepare_vlm_embeddings`, the path shared by `mlxcel generate`, `mlxcel-server` and the logit trace: InternVL's image block moves from the system turn to the head of the last user turn, MiniCPM-V 4.6 gets its own `MiniCPMV4_6Processor` layout plus a built-in ChatML template for checkpoints that ship none, and Gemma 3's newline wrapper is fused the way the reference tokenizer fuses it. Prompts now equal transformers for InternVL (292) and Gemma 3 (272), and the MiniCPM-V 4.6 server prompt equals the CLI prompt (82). With identical prompts, InternVL and MiniCPM-V 4.6 drop to zero decided-position disagreement against the float32 mlx-vlm reference; Gemma 3 stays at 5 of 43, which places its remaining gap in the forward pass or the reference's mask.

## 1. Problem Statement

### The measured mismatch

While measuring the GELU question in #2423, the image logit trace (`MLXCEL_TRACE_IMAGE`) and the float32 mlx-vlm 0.7.8 reference (`scripts/mlx_vlm_image_logit_trace.py`) disagreed on prompt length for three of five checkpoints, with image `tests/fixtures/test_image_shapes.png` (448x448) and prompt `Describe the image.`: internvl3-1b 262 vs 268, minicpm-v-4.6-bf16 78 vs 76, gemma-3-4b-it-4bit 273 vs 274. `docs/benchmarks.md` treats a length mismatch as the processors disagreeing before the tower runs. The perplexity gaps it left were large (InternVL 1.79 vs 1.39; MiniCPM-V 4.6 4.66 vs 1.22 with 2 of 39 decided positions disagreeing), and the issue did not know whether the cause was the template, the image token count, or BOS.

A length match against mlx-vlm alone would not have been the right target, because mlx-vlm itself departs from the checkpoint and from transformers in several places (section 2.6). The PR therefore dumped three id sequences per checkpoint (mlxcel, mlx-vlm, transformers) and attributed each difference.

### InternVL: the block sat inside the system turn

`internvl3-1b` ships Qwen2's ChatML template with no image branch, so the rendered prompt has no `<IMG_CONTEXT>` placeholder and mlxcel splices the image block itself. The old rule spliced it after the first token, which in a ChatML prompt is the `<|im_start|>` that opens the system turn: the model read `<|im_start|>` + image + `system\nYou are Qwen...`. InternVL's `model.chat` (`'<image>\n' + question`), the transformers InternVL template (`<IMG_CONTEXT>\n`) and mlx-vlm (`<image>\n`) all put the block at the head of the user content, followed by a newline.

The 262 in the issue also hid a second defect, in the measurement harness. The trace handed a typed `[image, text]` content list to a template that has no image branch; the render failed and fell back to the bare question with no chat turn (4 text tokens plus the 258-token block). `mlxcel generate` did render the template, so the production prompt was the 33 template tokens plus the misplaced block, not 262 tokens.

### MiniCPM-V 4.6: the MiniCPM-o rule was the wrong layout

MiniCPM-V 4.6 was routed through the MiniCPM-o expansion, which builds `<image><unk>...<unk></image>`. `MiniCPMV4_6Processor` builds `<image_id>{i}</image_id><image>` + `<|image_pad|>` x N + `</image>` (`use_image_id: true` in the checkpoint's `preprocessor_config.json`), and the checkpoint's template joins the image item and the question with `\n`. The old rule therefore dropped the image id prefix and the newline, used the wrong filler, and with the upstream template would have left the template's own `<|image_pad|>` marker in the prompt as text.

The server had an additional framing bug on template-less checkpoints (the model store's `minicpm-v-4.6-bf16` copy has no `chat_template.jinja`). The server rendered the generic `User: ...\n\nAssistant: ` fallback, and the image expansion, finding no ChatML turn, wrapped that again in `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`. The server prompt carried both framings (88 tokens) while the CLI carried one.

### Gemma 3: one newline token too many

transformers' `Gemma3Processor` rewrites the rendered text, replacing each `<start_of_image>` with `\n\n<start_of_image>` + soft tokens + `<end_of_image>\n\n`, and only then tokenizes. Gemma's vocabulary has one token per newline run (up to 31), so the template's `user\n` plus the wrapper's `\n\n` becomes the single token `\n\n\n` (109). mlxcel expands on the token stream (`apply_image_token_blocks`, wrapper token 108 = `\n\n`), which leaves `\n` (107) and `\n\n` (108) as two tokens: 273 instead of 272.

## 2. Technical Decisions

### 2.1 InternVL: locate the user turn in text, verify it in tokens

`insert_internvl_image_tokens` now takes the rendered prompt and the encoder. It finds the last `<|im_start|>user\n` in the text (`rfind`), encodes the text up to and including that opener, and checks that the encoding is a token prefix of the caller's prompt tokens. If it is (`UserTurnSplit::Aligned`), the blocks, each followed by the encoded `\n`, are spliced at that offset and the rest of the request keeps its original tokenization. Placeholder expansion (a template that emits `<IMG_CONTEXT>`) still runs first and wins.

Re-encoding the whole prompt with the block inserted as text would also have matched, but it would re-tokenize text that is already correct and would need the block's special tokens to round-trip through the tokenizer. Searching for the opener in tokens alone would miss the case below.

When the question starts with a newline, Qwen2's pre-tokenizer fuses the opener's `\n` with it (for example `\n\n\n` as one token), so the opener is not a token boundary at all. `user_turn_split` detects this (every head token except the final `\n` is a prefix) and returns `UserTurnSplit::Fused`; the caller then keeps the head tokens, appends the blocks without the last newline, and encodes `"\n" + rest` behind them, which is the text the reference tokenizes there (291 tokens for `\n\nLeading newlines`, equal to the reference). This path re-tokenizes the remainder of the prompt; it is reached only for the fused case.

Two degradations are deliberate. A prompt with no user-turn opener (`--no-chat-template`, a raw prompt) gets the blocks prepended, which is `model.chat`'s `'<image>\n' + question` without a template. A head encoding that does not line up with the caller's tokens either way also prepends, rather than inserting a block mid-token. Because the caller's `add_special_tokens` choice is not visible here, the plain encoding is tried first and the special-token encoding only when the plain one does not align (commit 4305d138 made the common path a single encode).

The placeholder-only half was split out as `expand_internvl_image_placeholders`. LLM-jp-VL, which already located its own Harmony user turn and only used the InternVL helper for in-place placeholder expansion, now calls that function directly; its behavior is unchanged.

### 2.2 MiniCPM-V 4.6: a separate module, not a flag on MiniCPM-o

`src/multimodal/minicpmv46_prompt.rs` reuses MiniCPM-o's `<image>`/`</image>` bound computation but owns the 4.6 layout. MiniCPM-o keeps its `<unk>` filler and no image id, so parameterising the MiniCPM-o rule would have mixed two processors' rules in one function for no shared logic beyond the bounds. The expansion works on text, then encodes with special tokens, finds each `<image>`...`</image>` span, and checks the span length against the image's feature count, returning an error on a mismatch instead of feeding misaligned embeddings.

Three cases, in order: both the template marker `<|image_pad|>` and the MiniCPM-V inline marker `<image>./</image>` (what mlx-vlm and older clients write) are replaced in order of appearance, one image each, with a count check; without a marker, the blocks and a `\n` open the last `<|im_start|>user\n`; with neither marker nor user turn, the turn is wrapped in ChatML user and assistant openers.

Both marker kinds are split in one pass so the `<|image_pad|>` filler of a block already inserted is never re-read as a marker. The first version re-searched the remaining prompt for both kinds after every marker, which is quadratic when one kind is absent; security review measured 24 s for 100k typed markers (1.3 MB). This matters because the prompt is user-controlled, the split runs on the batch scheduler before the marker-count error, VLM requests skip the context-length check, and the JSON body limit is sized for many large images. Each kind now caches the offset of its next occurrence and is searched again only after the cursor passes it (commit 6cece708), with a regression test that took 24 s before.

### 2.3 A built-in ChatML template for template-less MiniCPM checkpoints

`builtin_chat_template` now returns `MINICPM_CHATML_TEMPLATE` for `ModelType::MiniCPMV46VLM` and `ModelType::MiniCPMOVLM`. That function is reached only when the checkpoint ships no template, so a shipped template still wins. With the server rendering a ChatML turn, the expansion finds `<|im_start|>user\n` and opens it instead of wrapping a second time, and the server and the CLI build the same 82-token prompt.

The alternative was to teach the expansion to recognize the generic `User:`/`Assistant:` framing and strip it. That would couple a model-specific prompt module to the server's generic fallback text. Fixing the render instead makes the server produce the turn the family's own template would, and the same template covers MiniCPM-o, whose expansion has the same wrap-if-no-turn rule. The template renders the text parts of a typed content list and drops image items, since images travel as tokens spliced in after rendering.

This changes text-only server requests on template-less MiniCPM-V 4.6 and MiniCPM-o checkpoints too: they now get ChatML rather than `User:`/`Assistant:`.

### 2.4 Gemma 3: fuse newline runs after the token-level expansion

Rather than switching Gemma 3 to transformers' text-level expansion, which would change how every other `ImageTokenBlockInfo` family is prepared, `gemma3_vlm_prompt::merge_image_block_newline_runs` post-processes the token stream. It probes the tokenizer for the single-token ids of newline runs (`newline_run_token_ids`, stopping at the first run length that is not one token; Gemma 3 covers 1 to 31), marks the wrapper tokens next to each `<start_of_image>` and `<end_of_image>`, and re-encodes every maximal stretch of newline-run tokens that contains a wrapper token as the reference would encode that many newlines (longest run first, so 33 newlines become the 31-run plus the 2-run). Stretches without a wrapper token came from the tokenizer and are left alone.

The `Standard` arm of `vlm_runtime` calls it only when the layout has a non-empty block prefix or suffix, and the function returns immediately if no wrapper token is a newline run. In the current loaders only Gemma 3 (`block_prefix_tokens: vec![108]`) has a wrapper, so the other families are untouched and pay nothing. Gemma 3 pays the probe, about 32 single-string encodes per image request.

### 2.5 The trace harness scores the same prompt on both arms

The image trace now renders the way `mlxcel generate` does (text-only render when the template has no image branch, typed content otherwise) and prints `# prompt_ids` next to `# prompt_tokens`. `scripts/mlx_vlm_image_logit_trace.py --mlxcel-prompt` reads those ids and scores them with mlx-vlm's own preprocessed pixels, so where the PR deliberately diverges from mlx-vlm the two arms differ only in the forward pass. It refuses a prompt whose image-token count differs from mlx-vlm's (the pixel features could not be shared), recomputes MiniCPM-V's `image_bound` from the substituted ids, and prints mlx-vlm's own length as `# mlx_vlm_prompt_tokens`.

The reference also runs Gemma 3 and PaliGemma causally: their mlx-vlm `__call__` builds a 4D mask from the all-ones 2D attention mask, which would let every teacher-forced row see the whole continuation. The script merges the image embeddings and calls the language model with `mask=None`.

### 2.6 Deliberate divergences from mlx-vlm

Each follows transformers or the checkpoint and is recorded in `docs/benchmarks.md`:

- InternVL template: mlx-vlm replaces the checkpoint's template with its own `User: ...\nAssistant:\n`, which is neither the checkpoint's ChatML template nor InternVL's conversation template. mlxcel keeps the checkpoint's template, Qwen system turn included, as `tokenizer.apply_chat_template` does (292 vs mlx-vlm's 268).
- MiniCPM-V 4.6 without a template: mlx-vlm sends the bare `(<image>./</image>)\nQUESTION`; mlxcel wraps it in the family's ChatML turn (82 vs 76). With the upstream template, mlxcel is 84, mlx-vlm 84 and transformers 86; the remaining differences are the marker's parentheses, an empty think block, and mlxcel's server-wide `enable_thinking=true` default.
- Gemma 3: mlx-vlm tokenizes a render that already starts with `<bos>` with special tokens on and carries two BOS tokens; transformers and mlxcel carry one. mlx-vlm also places the image after the question; mlxcel keeps the image first, the order of the model card and of mlxcel's other image paths (#797).

## 3. Change Summary

16 files, +1645/-147, five commits (452f90d2, fff22a28, f4063ebd, 6cece708, 4305d138) plus a merge of origin/main (b769c15d).

| Checkpoint | mlxcel before | mlxcel after | mlx-vlm | transformers |
|---|---|---|---|---|
| internvl3-1b | 262 (trace) | 292 | 268 | 292 |
| minicpm-v-4.6-bf16, no template | 78 CLI / 88 server | 82 / 82 | 76 | n/a |
| gemma-3-4b-it-4bit | 273 | 272 | 274 | 272 |
| paligemma2-3b-6bit | 1030 | 1030 | 1030 | |
| gemma-4-e2b-it-4bit | 278 | 278 | 271 | |

- `src/multimodal/internvl_prompt.rs`: user-turn placement with the aligned and fused splits; `expand_internvl_image_placeholders` split out. Tests moved to `internvl_prompt_tests.rs`.
- `src/multimodal/minicpmv46_prompt.rs` (new): 4.6 block layout, linear two-marker split, span-size check.
- `src/multimodal/gemma3_vlm_prompt.rs` (new): newline-run table and wrapper fusion.
- `src/multimodal/vlm_runtime.rs`: the MiniCPM-V 4.6 arm calls `prepare_minicpmv46_prompt_tokens`; the InternVL arm passes prompt and encoder; the `Standard` arm runs the newline merge after `apply_image_token_blocks`.
- `src/multimodal/llmjp_vl_prompt.rs`: calls `expand_internvl_image_placeholders`.
- `src/server/chat_template.rs`: `MINICPM_CHATML_TEMPLATE` for `MiniCPMV46VLM` and `MiniCPMOVLM`.
- `examples/logit_trace.rs`, `examples/logit_trace/image.rs`: render like `generate`, print `# prompt_ids`.
- `scripts/mlx_vlm_image_logit_trace.py`: `--mlxcel-prompt`, `# prompt_ids`, causal Gemma 3 / PaliGemma.
- `docs/benchmarks.md`: the parity table, the fixes, the divergences and the trace pairs.
- Tests (all checkpoint-free; the reference pins use real token ids): `internvl3_1b_one_tile_prompt_matches_reference_ids`, `internvl3_1b_question_with_leading_newline_keeps_block_in_user_turn`, `fused_opener_newline_with_two_images_keeps_blocks_in_user_turn`, `caller_tokens_encoded_with_special_tokens_still_align`, `tokenizer_disagreement_degrades_to_prepending`; `minicpmv46_untemplated_prompt_matches_reference_ids`, `many_typed_markers_are_split_in_linear_time`, `marker_count_must_match_images`, `span_size_mismatch_is_an_error`; `gemma3_one_image_prompt_matches_transformers_ids`, `a_run_past_the_longest_token_splits_like_the_reference`, `non_gemma_layouts_are_untouched`; `server::minicpm_chat_template_tests` (4, including `a_shipped_minicpm_template_is_not_shadowed_by_the_builtin`).

## 4. Trace Pairs

Teacher-forced over 60 to 64 reply positions against the float32 mlx-vlm reference on CPU, decided positions at a logit gap of 2. Before is the pre-fix build against mlx-vlm's own prompt; after is the fixed build against the reference scoring mlxcel's prompt (`--mlxcel-prompt`).

| Checkpoint | Before: top-1, decided, ppl mlxcel vs ref | After |
|---|---|---|
| internvl3-1b | 8/64, 0/38, 1.7881 vs 1.3916 | 0/64, 0/39, 1.5853 vs 1.5869 |
| minicpm-v-4.6-bf16 | 17/60, 2/39, 4.6649 vs 1.2230 | 2/60, 0/47, 4.2442 vs 4.2325 |
| gemma-3-4b-it-4bit | 21/64, 5/40, 4.2819 vs 3.9110 | 20/64, 5/43, 4.8500 vs 5.1620 |

For InternVL and MiniCPM-V 4.6 the earlier gaps were the prompt: with identical tokens, decided disagreement is zero and perplexity agrees within 0.3%. The reference's own perplexity also moves, because the reply was generated under mlx-vlm's prompt. Gemma 3 still disagrees on 5 of 43 decided positions with identical prompts, so its gap is not the prompt.

## 5. Trade-offs and Residuals

- **Gemma 3 forward-pass gap.** The prompt is now exact, but 5 of 43 decided positions still disagree. The reference runs Gemma 3 causally from merged embeddings, while transformers lets image tokens attend to each other within their block; that mask is the first suspect. The PR marks this as a follow-up but names no issue number.
- **OpenXLA Gemma 3 path.** The host preprocessor (`host_preprocessor.rs`, `GEMMA3_VLM_NEWLINE_WRAPPER_TOKEN_ID`) keeps its own token-level wrapper and does not run the new merge, so the XLA backend still builds the 273-token prompt.
- **MiniCPM-o built-in template is unsmoked.** No MiniCPM-o checkpoint was available; the template is covered by `template_less_minicpmo_server_prompt_matches_the_cli_prompt` only.
- **Multi-turn image placement.** For InternVL without placeholders and MiniCPM-V 4.6 without markers, every image goes to the head of the last user turn. A text-only render carries no per-turn image position, so an image attached to an earlier turn is moved to the last one. The old code had the same limitation (it used the first token); a template with an image branch would place each image in its own turn.
- **Cache and output changes.** Prompt ids change for every image request on the three families and for every request on template-less MiniCPM-V 4.6 / MiniCPM-o servers, so prefix-cache entries from earlier builds will not match and outputs will differ from previous releases.
- **gemma-4-e2b-it-4bit** remains 278 vs mlx-vlm's 271, from a system `<|think|>` turn under the thinking default; not touched here.

## 6. Validation

- Failing first for every new pin: InternVL 292 and leading-newline 291, Gemma 3 272, MiniCPM-V 4.6 82, the server MiniCPM fallback (3 of 4 tests failed before), and the linear split (24 s before, milliseconds after).
- `cargo test --lib _prompt` 450 passed, `multimodal::` 318 passed, `server::minicpm_chat_template_tests::` 4 passed; fmt and clippy (`--lib --tests`, the `logit_trace` example) clean.
- GB10 smoke: `mlxcel generate --image` and one `mlxcel-server` request per family, server `prompt_tokens` InternVL 292 (291 for a leading-newline question), Gemma 3 272, MiniCPM-V 4.6 82, coherent output.
- paligemma2-3b-6bit unchanged at 1030; gemma-4-e2b-it-4bit 278 before and after.

Not verified: the OpenXLA Gemma 3 path, a MiniCPM-o checkpoint, MiniCPM-V 4.6 with the upstream template through the server, and multi-turn image conversations on any of the three families.

## 7. Learning Points

- Compare against more than one reference. mlx-vlm was the issue's yardstick, but it departs from the checkpoint and from transformers for all three families; dumping mlxcel, mlx-vlm and transformers ids side by side separated real bugs from deliberate divergences.
- Text-level and token-level expansion are not interchangeable when the vocabulary merges runs. A processor that rewrites text before tokenizing lets the inserted characters fuse with their neighbors; a token-level splice must reproduce that fusion explicitly (Gemma 3 newlines) or locate boundaries that may not exist as token boundaries (Qwen2's fused opener newline).
- When splicing into an already-tokenized prompt, find the position in the rendered text and verify it against the tokens, then degrade to a safe placement when they disagree, instead of re-tokenizing the whole request.
- Check the measuring tool before the measured system. Part of the InternVL gap was the trace harness's own render fallback, not the production path.
- String scanning on user-controlled prompts runs before most limits. A per-marker rescan that looked harmless was quadratic and took 24 s on a 1.3 MB body.
