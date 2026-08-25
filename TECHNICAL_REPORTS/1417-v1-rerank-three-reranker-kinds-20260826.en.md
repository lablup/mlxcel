# Technical Report: PR #1417 - `/v1/rerank` with sequence-classifier, Qwen3 generative and Qwen3-VL multimodal rerankers

**Date**: 2026-08-26
**Author**: mlxcel maintainers
**Status**: Completed; `BAAI/bge-reranker-v2-m3` matches its model card to 2.6e-5, the other four checkpoints have no published reference and are gated on ordering and margins
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1417 implements issue #1356, the last unit of embedding epic #1348. It adds `POST /v1/rerank` (Cohere and Jina compatible), the offline `mlxcel rerank` command, and a dedicated single-thread rerank worker beside the embedding one.

Three checkpoint shapes are served and all three end at a probability in `[0, 1]`. One-label cross-encoders on BERT, XLM-RoBERTa and ModernBERT reuse the classification heads #1321 and #1332 already merged, unchanged; the new code is only the pair tokenization, the truncation strategy and `sigmoid(logit)`. The Qwen3 generative reranker asks the model a yes/no question and reads `sigmoid(logit("yes") - logit("no"))` at the last prompt position. The Qwen3-VL multimodal reranker performs the same read over a prompt whose query and documents may carry images.

The change with the widest blast radius is a detection change: a `ForSequenceClassification` export on one of the three encoder families now resolves to a new `ModelType::SequenceClassifier` with a new `ModelKind::Reranker`, where before it fell through to the generation dispatch and errored. That is what makes `-m <cross-encoder>` serve `/v1/rerank` with no second flag, and it required updating three merged tests that asserted the old error.

Two smaller pieces of shared infrastructure moved: `EncodedBatch` gained a `PaddingSide` option (the generative rerankers need left padding, every encoder family keeps right padding), and `BertSequenceClassifier` gained a `num_labels()` read off its projection tensor rather than off `config.json`.

---

## 1. Problem Statement

### 1.1 Background

Retrieval with embeddings ranks by cosine over two independently computed vectors. A reranker instead runs the query and the document through one forward pass together, which cannot be indexed but discriminates far better. The usual shape is to retrieve with `/v1/embeddings` and reorder the top candidates with `/v1/rerank`.

The five checkpoints the issue names are three different things wearing two different config shapes.

`BAAI/bge-reranker-v2-m3` declares `XLMRobertaForSequenceClassification` with `id2label` of length one; `cross-encoder/ms-marco-MiniLM-L6-v2` declares `BertForSequenceClassification`, also one label, with `max_position_embeddings: 512`; `Alibaba-NLP/gte-reranker-modernbert-base` declares `ModernBertForSequenceClassification` with `classifier_pooling: "mean"`. All three are encoder trunks with a small head, and all three trunks were already in the tree from the embedding epic.

`mlx-community/Qwen3-Reranker-0.6B-4bit` declares `Qwen3ForCausalLM`, `model_type: qwen3`, a 4-bit `quantization` block and tied embeddings. Nothing in that config says "reranker". The relevance signal is the model's own next-token distribution over `yes` and `no` at the end of a fixed chat prompt.

`Qwen/Qwen3-VL-Reranker-2B` declares `Qwen3VLForConditionalGeneration`, `model_type: qwen3_vl`. It ships two side files a generator does not: `additional_chat_templates/reranker.jinja`, which owns the prompt and reads `role: query` and `role: document` messages rather than `user`, and `1_LogitScore/config.json`, which names the two answer tokens (`true_token_id: 9693`, `false_token_id: 2152`). Its `modules.json` lists a `LogitScore` module rather than a `Pooling` one.

### 1.2 Existing obstacles

- **Detection refused every `ForSequenceClassification` checkpoint outright.** `is_embedding_checkpoint` returned `Ok(None)` for them, correctly refusing to call a reranker an embedder, but nothing downstream claimed them, so `get_model_type` fell through to the `model_type` dispatch, which has no `bert` / `xlm-roberta` / `modernbert` arm, and reported `Unsupported model type: bert`. Three merged tests asserted exactly that message.
- **The tokenizer infrastructure only right-padded.** `EncodedBatch::from_rows` writes padding after the real tokens, which is what every encoder family needs. A generative reranker read at one column needs the opposite.
- **`num_labels` came from `config.json` on the BERT path.** `ModernBertSequenceClassifier` already derived it from `classifier.weight` rows and warned on a mismatch; `BertSequenceClassifier` exposed only `args().num_labels`, so a "must be exactly one label" check on that path would have been checking the config's claim rather than the head's real width.
- **`Qwen3VLModel`'s `lm_head` is a private field.** The only public way to logits was `forward_for_sequence`, which produces `[B, L, vocab]`. On the 2B checkpoint with a 400-token image prompt that is roughly 240 MB of f32 per row, all but one column of which is discarded.
- **`-m` is required by `mlxcel-server`.** A generative reranker is unreachable from `-m` (its config is a chat model's), so serving one alone means passing the same directory to both `-m` and `--reranker-model`, and the chat worker would happily load a second full copy of the weights.
- **Qwen3-VL image prefill is single-row.** `compute_rope_index` flattens `[B, L]` and reads row 0, and the DeepStack state is per request. The Qwen3-VL-Embedding family already documents this.

### 1.3 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| The detection change reroutes a checkpoint that is not a reranker | High: an unrelated family stops loading | Low, the arm is restricted to three `model_type` values |
| The Qwen3 prompt drifts from the reference recipe by one byte | High: scores look plausible and are wrong | Medium |
| Right padding leaks into the generative path | High: the score is read off a pad token for every row but the longest | Medium |
| A multi-label classifier is served as a reranker | Medium: `sigmoid` of one class logit is meaningless | Low, but silent |
| Truncation eats the assistant header | High: the scored position stops being the answer slot | Medium |
| The rerank-only shape doubles resident memory | Medium: a 2B reranker costs 8 GB instead of 4 | High, without a guard |
| Image placeholder count disagrees with the vision feature count | High: the merge scatters the wrong number of vectors | Medium |

---

## 2. Technical Decisions

### 2.1 Detect the cross-encoder, and only the cross-encoder

`is_sequence_classifier_checkpoint` runs before `is_embedding_checkpoint` in `get_model_type` and returns `ModelType::SequenceClassifier` when `architectures[0]` ends with `ForSequenceClassification` **and** `model_type` is one of `bert`, `xlm-roberta`, `xlm_roberta`, `modernbert`. Everything else returns `None` and keeps the routing it had.

The restriction is the point. A `DebertaV2ForSequenceClassification` checkpoint has no head port here, and claiming it would turn a clear `Unsupported model type: deberta-v2` into a load failure deeper in the stack. A `Qwen3ForSequenceClassification` checkpoint is a different thing again and keeps detecting as `Qwen3`, which a merged test already asserts.

The generative rerankers are deliberately not detectable. Their configs are byte-for-byte what a chat export looks like, so any rule that caught them would also catch chat models. They reach the reranker worker through `--reranker-model` only, and `detect_reranker_kind` keys on `model_type` once the operator has already said "this is a reranker".

### 2.2 Derive `num_labels` from the projection tensor

`BertSequenceClassifier` now reads the row count of `classifier.weight` (BERT) or `classifier.out_proj.weight` (XLM-RoBERTa) and logs a warning when `config.json` disagrees, mirroring what `classifier_rows` already did for ModernBERT. `require_single_label` checks that number, not the config's.

Quantization packs the input axis and never the output axis, so the row count is `num_labels` on both the dense and the quantized paths, which is why it is a safe source of truth. The config is not: a re-export that forgets to update `id2label` would otherwise make `num_labels()` lie about the shape `logits` actually produces, and the reranker would then compute `sigmoid` of one class logit out of several without noticing. A unit test builds one-label weights, sets `num_labels = 7` in the args, and asserts the head reports 1 and emits `[1, 1]`.

### 2.3 Let the tokenizer truncate the cross-encoder pair

`with_longest_first_truncation` installs `TruncationParams { max_length, strategy: LongestFirst }` on the HuggingFace tokenizer after the shared `strip_padding_and_truncation` has cleared whatever `tokenizer.json` baked in.

The alternative was to trim ids after encoding, which is what the embedding engine does for single texts. It would have been wrong here in a way that is hard to see: `tokenizers::Tokenizer::post_process` truncates **before** the post-processor adds its special tokens, and subtracts `get_n_added_tokens(is_pair)` from the budget first. Reproducing that ordering by hand means reimplementing the special-token accounting and the alternating drop, and any divergence shows up as a slightly different score on long inputs rather than as a failure. Delegating gives the reference `tokenizer(query, document, truncation=True, max_length=...)` behaviour exactly.

### 2.4 Assemble the Qwen3 prompt from token ids, not from one formatted string

`PROMPT_PREFIX`, the rendered `CONTENT`, and `PROMPT_SUFFIX` are encoded separately (each with `add_special_tokens: false`) and concatenated, and only the middle piece is truncated.

Formatting one string and truncating the result would have cut the assistant header off exactly the inputs that need the model most, and the header is the position the score is read from. Building from ids makes "the suffix is always kept" a structural property rather than a bound to be respected. A unit test asserts the encoded row starts with the prefix ids and ends with the suffix ids even when the document is 3400 words long, and that a scaffold larger than the limit is rejected at load with `leaves no room` rather than silently scoring an empty pair.

The three strings are asserted byte for byte against the model card's recipe in `prompt_bytes_match_reference_recipe`, including the escaped quotes around `"yes"` and `"no"` and the empty think block in the suffix.

### 2.5 Left padding as a new option on the shared batch builder

`EncodedBatch::from_rows_with_padding(rows, pad_id, pad_to, side)` is the new entry point; `from_rows` delegates to it with `PaddingSide::Right`, so no embedding family changed.

Left padding is what puts every row's last real token at column `L - 1`. With right padding the score would be read off a pad token for every row shorter than the longest, which produces finite, plausible, wrong numbers. The mask is `create_causal_padding_mask`, whose doc comment already covered the left-padded case including the rescue that keeps a leading-padding row's softmax finite.

One consequence is worth recording because it looks like a bug and is not. Absolute positions still run `0..L` across the padded row, so a document's score depends slightly on the batch it was scored in. That is exactly what the reference does: `Qwen3ForCausalLM.forward(input_ids, attention_mask)` derives `position_ids` from the cache position when none is passed, and the official Qwen3-Reranker example calls it with `padding_side='left'` and no `position_ids`. Matching the reference was chosen over batch invariance. The unit test that recomputes a score by hand therefore scores that document alone, and says why.

### 2.6 Read the score from one column

Both generative paths call `forward_hidden` (already `pub(crate)` on Qwen3, newly reachable on Qwen3-VL through a five-line `lm_head_forward`), slice `[:, L-1, :]`, and apply the head to that single column. The two answer logits are then gathered with `take` on the vocabulary axis.

Applying the head to the whole sequence first would allocate `[B, L, vocab]`. For the VL reranker at batch 2 with 227-token image rows that is about 276 MB of transient per call, discarded immediately. The alternative of gathering the two head rows and doing a `[B, 1, H] x [H, 2]` product was rejected because it does not work uniformly for a quantized head, where the rows are packed.

`lm_head_forward` is a deliberate minimum: it exposes the head as a function, not the field, so the two paths cannot drift the way public `layers` and `norm` would let them.

### 2.7 Render the Qwen3-VL prompt with the checkpoint's own template

`Qwen3VlReranker` loads `additional_chat_templates/reranker.jinja` into a `ChatTemplateProcessor` and renders a message list verbatim, rather than reproducing the prompt in Rust.

That template does three unusual things: it reads `role: query` and `role: document` (not `user`), it supplies its own default instruction when no `system` message is present, and it emits `<|vision_start|><|image_pad|><|vision_end|>` per image content item. Reproducing them would pin today's checkpoint's prompt into the binary. `rerank_messages` therefore omits the `system` turn entirely when the request carries no `instruction`, so the template's own default wins; supplying one replaces it. Both behaviours are asserted against the real template file, which needs the checkpoint directory but no weights, and soft-skip when it is absent.

The answer token ids come from `1_LogitScore/config.json` for the same reason, and a checkpoint without a readable module is rejected at load rather than falling back to hard-coded 9693 / 2152.

### 2.8 Score image rows one at a time, batch the text rows

`Qwen3VlReranker::score` partitions the encoded rows: any row carrying an image is scored on its own, text-only rows go through the left-padded batched path, and the results are written back into index slots.

The constraint is the port's, not the request's: `compute_rope_index` reads row 0 of a flattened `[B, L]`, and `set_deepstack_state` holds one request's visual mask and features. The merged Qwen3-VL-Embedding family documents the same thing and embeds images one at a time. Partitioning rather than falling back to batch 1 for the whole request means a mixed request pays the per-row cost only for its image documents; a text-only request through the VL reranker keeps the batched path.

This is a deviation from the issue's suggestion that a batch of two image documents exercises left padding with images. Left padding with images is not reachable while the M-RoPE index is single-row. The corresponding gate instead asserts that two image documents in one request both produce finite probabilities, and the manual validation adds the stronger check that the ranking flips when the query changes.

### 2.9 Truncate the Qwen3-VL pair before the template renders it

`longest_first_keep(query_tokens, document_tokens, budget)` computes the same fixed point the tokenizer's `longest_first` converges to: the shorter side survives whole whenever it fits in its half, otherwise both are cut to half the budget. The budget is `max_length` minus the scaffold (measured once at load by rendering the template with empty texts) minus the visual tokens the images will contribute.

Truncating after rendering was not an option: right-truncating a rendered prompt can cut into an image token run, and the merge then scatters a different number of vectors than the row has placeholders, which is a hard failure rather than a quality loss. Truncating the texts first keeps the template's output structurally intact. The trade is one decode-and-re-encode round trip on the rare over-long pair.

The function is unit-tested against the same expectations as the tokenizer path, so the two truncation strategies in this PR are pinned to each other.

### 2.10 A chat worker that never loads a model, for the rerank-only shape

When `config.reranker_model_path` equals `startup.model_path`, `ModelProvider::new_with_server_config_and_prompt_cache` returns `new_without_chat_model`, which builds the provider around a thread that logs why chat is unavailable and exits, dropping the request receiver.

Two alternatives were rejected. Rejecting the combination outright would make a generative-reranker-only server impossible, because `-m` is required and cannot name one on its own. Threading a `skip_model_load` flag through `new_with_full_config_and_speculative_dispatch` (already 31 positional parameters) and its two wrappers would have touched the hottest constructor in the server for one boolean.

The state this produces is the one a failed chat load already produces (`-m <embedding checkpoint>` reaches it too): `/v1/chat/completions` errors, `/health` reports `loading model`. No route needed a new case, which is the argument for it. The inherited rough edge is that the chat error text is `sending on a closed channel` rather than something that names the reason; the log line does name it, and improving the HTTP text would change behaviour for the embedding case too, so it is left for a follow-up.

### 2.11 Share the embedding worker's queue and timeout flags

`--rerank-batch-size` is the only new tuning knob; queue depth and per-request timeout come from `--embedding-queue-depth` and `--embedding-request-timeout-secs`. Both single-thread workers shed load the same way and there is no reason for an operator to tune them apart. The default batch size is `0`, meaning "the kind's own default": 8 for a text reranker, 2 for the multimodal one, whose rows each carry a full image's worth of visual tokens.

---

## 3. Implementation Details

### 3.1 The cross-encoder path

```
rows  = [encode_pair_row(tokenizer, query, doc, opts) for doc in chunk]   # [CLS] q [SEP] d [SEP]
batch = EncodedBatch::from_rows(rows, pad_id, None)                       # right padded
logits = head.logits(ids, mask, type_ids?)                                # [B, 1]
scores = sigmoid(astype(logits, f32))
```

`token_type_ids` are requested only for the BERT dialect (`BertSequenceClassifier::needs_token_type_ids`); XLM-RoBERTa has a single-row segment table and ModernBERT has none. `max_length` comes from the shared `derive_max_length`, with `is_absolute_position` true only for the BERT trunk, and is then lowered by `BertArgs::max_sequence_length()` so `bge-reranker-v2-m3`'s 8194 position rows cap at 8192 real tokens. Observed: 8192 for bge and gte, 512 for ms-marco.

The two head types sit behind a private `ClassifierBackbone` enum with four methods (`num_labels`, `needs_token_type_ids`, `weight_max_length`, `logits`), so the batching loop is written once.

### 3.2 The Qwen3 path

```
ids   = prefix_ids ++ truncate(encode(CONTENT), max_length - |prefix| - |suffix|) ++ suffix_ids
batch = EncodedBatch::from_rows_with_padding(rows, pad_id, None, Left)
mask  = create_causal_padding_mask(batch.attention_mask, 0)               # [B, 1, L, L]
h     = model.forward_hidden(ids, None, fresh caches, Some(mask))         # [B, L, H]
last  = slice_axis(h, 1, L - 1, L)                                        # [B, 1, H]
lg    = lm_head.forward(last) or embed_tokens.as_linear(last)             # [B, 1, vocab]
pick  = take(lg, [yes_id, no_id], axis 2)                                 # [B, 1, 2]
score = sigmoid(pick[..0] - pick[..1])
```

On `mlx-community/Qwen3-Reranker-0.6B-4bit` the scaffold measures 39 prefix tokens and 9 suffix tokens, the answer ids resolve to 9693 and 2152, and `pad_token_id` resolves to `<|endoftext|>` (151643) through the shared `resolve_pad_token_id`. The head is tied on this checkpoint, so `lm_head` is `None` and `embed_tokens.as_linear` runs.

`max_length` is `derive_max_length(model_dir, false, override).min(8192)`. `max_position_embeddings` is deliberately not read: the position table is RoPE, so 40960 is not a cap worth honouring, and the 8192 ceiling is what keeps one pair's prefill bounded.

### 3.3 The Qwen3-VL path

```
images  = query.image? ++ document.image?                                  # template order
counts  = processor.compute_grid_thw(images) -> t * (h/merge) * (w/merge)
(q, d)  = truncate_texts(query, document, sum(counts))
prompt  = reranker.jinja(rerank_messages(instruction, q, has_q_img, d, has_d_img))
ids     = expand_image_placeholders(encode(prompt), image_token_id, counts)
```

then, for an image row (batch 1):

```
pixels, grid = processor.preprocess_with_grid(images)
merged       = vlm.get_input_embeddings(ids, pixels, grid)                 # sets M-RoPE + DeepStack
h            = text_model.forward_hidden(ids, Some(merged), fresh caches, Some(causal))
```

and for text rows the same left-padded batched shape as the Qwen3 path. The M-RoPE and DeepStack slots are cleared before and after every call so a later text row cannot inherit an image row's state, which is the same discipline `Qwen3VLEmbeddingModel` uses.

`expand_image_placeholders` and `apply_pixel_bounds` are reused from `src/models/qwen3_vl_embedding.rs` (the latter promoted to `pub(crate)`), so the pixel ceiling comes from the checkpoint's `preprocessor_config.json` (`max_pixels: 1310720`) rather than the Qwen2-VL default, and the count the prompt expands with is the count the forward pass merges.

### 3.4 Detection and registration

- `ModelType::SequenceClassifier` in `src/models/mod.rs`, in `ALL_MODEL_TYPES`, with metadata `("Cross-encoder sequence classifier (BERT / XLM-RoBERTa / ModernBERT)", "Reranker")`.
- `"Reranker"` added to `FAMILY_ORDER` in `src/main.rs`, right after `"Embedding"`, so `mlxcel arch` groups it deterministically.
- `ModelKind::Reranker` in `src/model_metadata.rs` with `is_reranker_model_type`, and the registration row that makes adapters unsupported with an explanatory message.
- `load_model` bails for the reranker kind with the same shape of message it uses for embedding checkpoints and Whisper, so `mlxcel generate -m <cross-encoder>` names `/v1/rerank` instead of failing on a missing tensor.
- `fallback_architecture` in the tensor-parallel dispatch table gets a `"reranker"` placeholder, keeping that match total.

### 3.5 The HTTP surface

`RerankInput` is an untagged enum over a bare string and an object with `text`, `image` and `image_url`. `RerankImage` is untagged over a bare URL string and `{"url": ...}`, so a client that already builds OpenAI content parts does not need to special-case this endpoint. `image` wins over `image_url` when both are present, matching the Jina schema.

Validation order in `create_rerank`: body parse, provider presence (`501`), model id, `top_n >= 1`, `instruction` against `RerankerKind::accepts_instruction`, then every item (empty, image against `supports_images`). Images are fetched and decoded through the same `current_image_input_limits` / `try_read_image_url_with_limits` / `decode_request_images_with_limits` path `/v1/embeddings` uses, so the payload, dimension and decode-allocation bounds are shared.

`sort_and_truncate` sorts descending by score with ties broken by ascending index, then cuts to `top_n`. The tie rule is not cosmetic: equal scores are common once a batch saturates the sigmoid, and a stable sort alone would leave their order dependent on the worker's completion order.

### 3.6 Server wiring

`AppState.rerank_model: Option<Arc<dyn RerankModelProvider>>` with `with_rerank_model`, mirroring the embedding slot. `resolve_rerank_source` returns `Explicit` / `Primary` / `None`, with the same-path case resolving to `Primary`. A failing explicit `--reranker-model` is a startup error; a failing `-m` reranker is logged and leaves the route answering `501`.

`/v1/models` lists the reranker unless its id already appears, which covers both the `-m` case (ids coincide) and the rerank-only case (one entry).

---

## 4. Test Strategy

### 4.1 Without a checkpoint

- `src/rerank/mod_tests.rs` (9 tests): detection for all four `model_type` spellings, both generative kinds staying on their chat routing, `deberta-v2` and a bare `BertModel` rejected, the single-label guard, `num_labels` precedence, and a `1_LogitScore` `modules.json` not being read as a Pooling layout.
- `src/rerank/qwen3_generative_tests.rs` (6 tests) on a synthetic 2-layer Qwen3 and a word-level Qwen-shaped tokenizer: the byte-exact prompt strings, truncation keeping both ends, left padding placing the last token at `L - 1` (and right padding still doing the opposite), the yes/no single-token guard (forced by declaring an added token that splits `yes`), and a hand-recomputed `sigmoid(yes - no)` matching `score()` to 1e-5.
- `src/rerank/sequence_classifier_tests.rs` (5 tests): pair segment ids, the longest-first split in three regimes, the installed truncation params, and the VL split function pinned to the same fixed point.
- `src/rerank/qwen3_vl_generative_tests.rs` (6 tests): the message list with and without an instruction, image-only sides, the `1_LogitScore` reader, and the truncation split.
- `src/server/routes/rerank_tests.rs` (17 tests) over a stub reranker through the real router: sorting with ties, `top_n`, `return_documents` echoing both item forms, every `400` case, the `501`, the `/rerank` alias, `/v1/models`, and the provider-error-to-status mapping.
- `src/server/rerank_worker_tests.rs` (7 tests): info reporting, round trip, family-error mapping, loader failure, panic recovery followed by a successful request, the reply timeout, and bounded-queue shedding.
- `src/models/bert_heads_tests.rs`: `num_labels` from the tensor against a lying config, for both dialects.
- `src/models/detection_tests.rs`: the four cross-encoder spellings routing to `SequenceClassifier` even with a Pooling `modules.json`, and `deberta-v2` keeping its generator routing.

All MLX-touching tests take the shared `mlx_test_guard`, and the gate numbers were recorded with `--test-threads=1`.

### 4.2 With the checkpoint present, soft-skipping otherwise

`src/rerank/real_checkpoint_tests.rs` (6 tests) uses the same `local_checkpoint` lookup as the embedding gates. `bge-reranker-v2-m3` is compared against its model card; the other four are gated on ordering plus margins, and the VL one draws its two test images in code rather than committing fixtures. A sixth test walks all five checkpoints and asserts each resolves to the kind the issue's table assigns.

### 4.3 Merged tests that had to change

Three assertions in already-merged code encoded the old behaviour and were updated with a comment naming this issue: `src/embeddings/real_checkpoint_tests.rs` (three `Err("Unsupported model type")` rows became `Ok(ModelType::SequenceClassifier)`), `src/models/modernbert_real_checkpoint_tests.rs`, and `src/models/modernbert_tests.rs`. In each case the intent ("a reranker is never an embedder") is preserved; only the way detection expresses it changed.

---

## 5. Real-Checkpoint Results

All numbers below were produced on the Linux/CUDA validation host with the `test-fast` profile. Every run was repeated three times and every repeat was byte-identical unless stated.

### `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa cross-encoder, `max_length` 8192)

`mlxcel rerank -q "what is panda?" -d "hi" -d "<the panda passage>"`, three runs:

| Pair | mlxcel | Model card | Delta |
|------|--------|------------|-------|
| unrelated | 0.00027900 | 0.00027803 | 9.7e-7 |
| relevant | 0.99486631 | 0.99484038 | 2.6e-5 |

This is the only checkpoint of the five with published reference scores, and it is inside the issue's 2e-2 tolerance by three orders of magnitude.

Through `mlxcel-server -m <this checkpoint>` with no second flag, the Beijing request scores `[0.9999681, 1.5995e-5, 4.4010e-5]`, ranking `[0, 2, 1]`, identical across three runs.

### `cross-encoder/ms-marco-MiniLM-L6-v2` (BERT cross-encoder, `max_length` 512)

`[0.9999217, 1.59e-5, 3.35e-5]`, ranking `[0, 2, 1]`, three identical runs. No published reference; the gate is the ordering. `max_length` resolving to 512 confirms the absolute-position cap is being read.

### `Alibaba-NLP/gte-reranker-modernbert-base` (ModernBERT cross-encoder, `max_length` 8192)

`[0.9800415, 0.1246974, 0.7244105]`, ranking `[0, 2, 1]`, three identical runs. The in-process library gate recorded `0.9800463` for the same input, a cross-process spread of 4.8e-6; that is CUDA reduction-order noise, not a behaviour difference. This checkpoint scores the Berlin distractor high (0.72), which is a property of the checkpoint's calibration rather than of the port; the ordering is still correct.

### `mlx-community/Qwen3-Reranker-0.6B-4bit` (generative, 4-bit)

`[0.9883127, 8.94e-6, 1.47e-5]`, ranking `[0, 2, 1]`, three identical runs through both the CLI and `POST /v1/rerank`. The issue's gate is `results[0] > 0.9` with the other two below 0.2; the margins are four orders of magnitude clear of it.

Combined process: `mlxcel-server -m models/mlx/qwen3-0.6b-4bit --reranker-model <this checkpoint>` answers `/v1/chat/completions` from the 0.6B chat model and `/v1/rerank` from the reranker in the same process, and `/v1/models` lists both ids. A per-request `instruction` changes the scores as expected (0.2451 / 3.76e-5 for a shortened query with the default instruction spelled out).

### `Qwen/Qwen3-VL-Reranker-2B` (multimodal generative)

Text-only pairs through the same prompt: `[0.8807971, 0.0953494, 0.2337064]`, ranking `[0, 2, 1]`, three identical runs.

Image documents, two drawn PNGs (a bar chart titled "Quarterly revenue" and a textured scene with an animal-like silhouette), submitted as data URIs through `POST /v1/rerank`, three identical runs:

| Query | chart.png | cat.png | Ranking |
|-------|-----------|---------|---------|
| "a chart of quarterly revenue" | 0.46879062 | 0.36296920 | chart first |
| "a photo of an animal with two ears and eyes" | 0.1919 | 0.4378 | animal first |

The ranking flipping with the query is the load-bearing evidence: it shows the image content is reaching the model, not just that the arithmetic produces finite numbers. The chart margin (0.106) is below the issue's suggested 0.3, which assumed real photographs; with the query swapped the margin is 0.246. Both images are synthetic and crude, so the absolute values say more about the drawings than about the port.

A mixed request (two text documents plus the two images) orders all four sensibly: revenue text 0.5622, chart image 0.4688, animal image 0.3630, cat text 0.1192. That exercises the partition between the batched text path and the per-row image path in one call.

Rerank-only shape: `mlxcel-server -m <this checkpoint> --reranker-model <same path>` logs `Chat generation is disabled: ... the chat worker did not load a second copy of its weights`, serves `/v1/rerank` normally, lists one model id, and returns an error on `/v1/chat/completions`.

---

## 6. Validation Summary

| Check | Command | Result |
|-------|---------|--------|
| Rerank unit and real-checkpoint gates | `cargo test --profile test-fast --features cuda --lib rerank:: -- --test-threads=1` | 47 passed |
| Rerank worker | `cargo test ... --lib server::rerank_worker -- --test-threads=1` | 7 passed |
| Detection | `cargo test ... --lib models::detection_tests -- --test-threads=1` | 42 passed |
| BERT heads | `cargo test ... --lib models::bert -- --test-threads=1` | 28 passed |
| ModernBERT | `cargo test ... --lib models::modernbert -- --test-threads=1` | 19 passed |
| Embedding subsystem (regression) | `cargo test ... --lib embeddings:: -- --test-threads=1` | 69 passed |
| CLI registry and `mlxcel arch` | `cargo test ... --bin mlxcel -- --test-threads=1` | 197 passed |
| Lint | `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | clean |
| Format | `cargo fmt --all -- --check` | clean |

The workspace-wide `--all-targets` clippy and the `metal,accelerate` test run named in the issue are macOS-side gates and were not runnable on this host; the CUDA equivalents above cover the same code.

---

## 7. Change Summary

52 files, +5693 / -40.

**New subsystem** (`src/rerank/`, 2923 lines including tests): `mod.rs` (trait, kinds, items, detection, single-label guard, sigmoid readback), `loader.rs`, `sequence_classifier.rs`, `qwen3_generative.rs`, `qwen3_vl_generative.rs`, `stub.rs`, and five test modules.

**New server layer**: `src/server/rerank_model.rs` (78), `src/server/rerank_worker.rs` (380) plus 291 lines of tests, `src/server/routes/rerank.rs` (256) plus 464 lines of tests, `src/server/types/rerank.rs` (175).

**New CLI**: `src/commands/rerank.rs` (260) and the `Commands::Rerank` arm.

**Shared infrastructure touched**: `src/embeddings/tokenize.rs` (+48, `PaddingSide` and `from_rows_with_padding`), `src/models/bert_heads.rs` (+53, tensor-derived `num_labels`, `max_sequence_length`), `src/models/qwen3_vl.rs` (+10, `lm_head_forward`), `src/models/qwen3_vl_embedding.rs` (+2, `apply_pixel_bounds` visibility), `src/models/detection.rs` (+48), `src/models/mod.rs` (+17), `src/model_metadata.rs` (+15), `src/loading/mod.rs` (+12), `src/distributed/tensor_parallel/inference.rs` (+5), `src/server/model_provider.rs` (+64).

**Flags and config**: `src/bin/mlx_server.rs` (+45), `src/main.rs` (+43), `src/commands/serve.rs` (+18), `src/server/cli_input.rs` (+29), `src/server/config.rs` (+19), `src/server/startup.rs` (+141), `src/server/state.rs` (+16).

**Docs**: `docs/embeddings.md` (+141, the Reranking section plus source-map, detection and family-note corrections), `docs/supported-models.md` (+13, the Reranker models table).

---

## 8. What Remains Unverified

- **Four of the five checkpoints have no published reference scores.** `ms-marco-MiniLM-L6-v2`, `gte-reranker-modernbert-base`, `Qwen3-Reranker-0.6B-4bit` and `Qwen3-VL-Reranker-2B` are gated on ordering and margins only. No PyTorch or `transformers` install exists on the validation host, so no parity number was computed for them and none is claimed.
- **The Qwen3-VL image gate uses drawn images, not photographs.** The absolute scores (0.47 / 0.36) are a property of two crude synthetic PNGs. The query-swap flip is the meaningful signal; a photograph-based margin against the issue's 0.3 suggestion was not measured.
- **Left padding with images is not exercised**, because the Qwen3-VL port scores image rows one at a time. If the M-RoPE index ever becomes batch-aware, this path needs a new gate.
- **Batch-position dependence of the generative scores is documented but not bounded.** A document's score shifts slightly with the batch it lands in, matching the reference; how large that shift gets on long, heterogeneous batches was not measured.
- **No performance numbers.** The epic runs its performance pass once at the end on a quiet machine, so nothing here was benchmarked.
- **macOS was not exercised.** All validation was on Linux/CUDA.
- **The `metal,accelerate` workspace test run** named in the issue's acceptance criteria was not runnable here.

---

## 9. Follow-up Actions

1. Give the rerank-only shape a better chat error than `sending on a closed channel`. The fix belongs with the shared failed-load path, so it also improves `-m <embedding checkpoint>`.
2. Consider `--rerank-max-length` as a server flag. The load option exists and `mlxcel rerank --max-length` uses it, but the issue's flag list did not include a server-side one, so it was left out rather than added silently.
3. Revisit image batching if Qwen3-VL's `compute_rope_index` becomes batch-aware; `Qwen3VlReranker::score` already partitions, so only the image branch would change.
4. Measure the Qwen3-VL image margin against real photographs, and record it next to the drawn-image numbers rather than replacing them.

---

## References

- Issue #1356: the specification this PR implements
- Epic #1348: the embedding and reranking epic
- PR #1408 / issue #1353: the embedding foundation this reuses (tokenizer, limits, worker shape)
- PR #1411 / issue #1321: `BertSequenceClassifier`
- PR #1412 / issue #1332: `ModernBertSequenceClassifier`
- PR #1416 / issue #1345: Qwen3-VL-Embedding, whose `forward_hidden` split and image-placeholder expansion this reuses
- `docs/embeddings.md`, section "Reranking (`/v1/rerank`) and `mlxcel rerank`"
- `docs/supported-models.md`, section "Reranker models"
