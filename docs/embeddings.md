# Embeddings API (`/v1/embeddings`) and `mlxcel embed`

mlxcel serves embedding checkpoints through the OpenAI-compatible `POST /v1/embeddings` endpoint and the offline `mlxcel embed` command. This page covers the shared foundation every embedding family builds on: detection, pooling, normalization, length limits, the request and response schema, the server flags, and the CLI. Per-family details (prompt prefixes, instruction formats, image support) are filled in by each family as it lands; the family table lives in [supported-models.md](supported-models.md#embedding-models).

**Current status.** The route, the worker thread, batching, pooling and the CLI are in place. Detection recognizes every family listed below and the loader dispatches on it, but the forward passes land one family at a time (epic #1348): loading a checkpoint whose family has not landed reports `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings`. The [Embedding models table](supported-models.md#embedding-models) records which families are served today.

## Implemented endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/embeddings` | Embed a string, a list of strings, a token-id array, a list of token-id arrays, or a list of typed parts. |
| POST | `/embeddings` | Alias without the `/v1` prefix. |
| GET | `/v1/models` | Lists the embedding model next to the chat model when `--embedding-model` loads a second checkpoint. |

## Implementation source map

| Module | Responsibility |
|--------|----------------|
| `src/embeddings/model.rs` | `EmbeddingModel` trait, `EmbeddingBatch`, `EmbeddingOutput`, `ImageInput`. |
| `src/embeddings/pooling.rs` | `PoolingMode`, `1_Pooling/config.json` reader, `pool`, `normalize_l2`, `truncate_dimensions`, the `MLXCEL_EMBEDDING_POOLING` override. |
| `src/embeddings/limits.rs` | `max_length` derivation, pad token and vocabulary size. |
| `src/embeddings/tokenize.rs` | Right-padded batch tokenization, trailing-special truncation, pair encoding. |
| `src/embeddings/loader.rs` | `load_embedding_model`: family dispatch, subfolder weights, bf16 rule. |
| `src/embeddings/engine.rs` | Length-sorted micro-batching, normalization, `dimensions`, readback. |
| `src/embeddings/maxsim.rs` | `maxsim` / `maxsim_mlx`: the late-interaction score multi-vector families are ranked with. |
| `src/lib/mlxcel-core/src/utils.rs` | `create_bidirectional_padding_mask`, `create_causal_padding_mask`, `create_bidirectional_window_mask`. |
| `src/lib/mlxcel-core/src/weights.rs` | `load_weights_from_dir_with_subfolders` (`2_Dense/...` tensors prefixed `2_Dense.`). |
| `src/models/detection.rs` | `is_embedding_checkpoint`: the detection rules below. |
| `src/models/bert.rs` | BERT / XLM-RoBERTa encoder trunk: weight sanitization, position ids, blocks. |
| `src/models/bert_config.rs` | BERT / XLM-RoBERTa config resolution, re-exported through `bert`. |
| `src/models/bert_heads.rs` | `BertEmbeddingModel` and `BertSequenceClassifier` on that trunk. |
| `src/server/embedding_model.rs` | `EmbeddingModelProvider` trait and `EmbeddingError`. |
| `src/server/embedding_worker.rs` | `EmbeddingWorker`: the dedicated MLX-owning thread, bounded queue, timeout, panic boundary. |
| `src/server/routes/embeddings.rs` | HTTP handler, validation order, error mapping. |
| `src/server/types/embeddings.rs` | Request and response types, base64 encoding. |
| `src/commands/embed.rs` | `mlxcel embed`. |
| `src/models/embedding_sanitize.rs` | Weight-key normalization shared by the decoder-backbone families: `{N}_Dense.linear.*` folding, head dropping, `model.` prefixing. |
| `src/models/gemma3_embedding.rs` | EmbeddingGemma: bidirectional Gemma 3, mean pooling, two `Dense` projections. |
| `src/models/qwen3_embedding.rs` | Qwen3-Embedding: causal Qwen3, last-token pooling. |
| `src/models/col_late_interaction.rs` | Shared by the two late-interaction families: `embedding_dim`, the `1_Dense` projection override, the LoRA-only rejection, per-token projection and normalization, the query format. |
| `src/models/colidefics3.rs` | ColIdefics3: SmolVLM / Idefics3 without a head plus a 128-dim projection. |
| `src/models/colqwen2_5.rs` | ColQwen2.5: Qwen2.5-VL without a head plus a 128-dim projection. |
| `src/models/llama_bidirec.rs` | Bidirectional Llama (LLM2Vec): Llama 3 layers under a padding-only mask, mean pooling. |
| `src/models/ministral3_embedding.rs` | Nemotron-3-Embed: bidirectional Ministral 3, Llama 4 attention scaling at offset 0, mean pooling. |
| `src/models/lfm2_embedding.rs` | LFM2.5-Embedding: bidirectional LFM2 (non-causal short conv), CLS pooling. |

## Detection

A checkpoint is served as an embedding model, before the ordinary `model_type` dispatch runs, when any of these hold:

- `config.json` `model_type` is an encoder-only family: `bert`, `xlm-roberta`, `modernbert`, `siglip`. These never generate text; a `BertForMaskedLM` or `ModernBertForMaskedLM` export loads as an embedder with its MLM head dropped.
- `config.json` `architectures[0]` is an embedding architecture: `BertModel`, `XLMRobertaModel`, `ModernBertModel`, `SiglipModel`, `SiglipTextModel`, `Gemma3TextModel` with `use_bidirectional_attention: true`, `LlamaBidirectionalModel`, `LlamaNemotronVLModel`, `Lfm2BidirectionalModel`, `Ministral3Model` with `is_causal: false`, `ColIdefics3`, `ColQwen2_5`, `ColQwen2ForRetrieval`.
- `modules.json` exists and lists a module whose `type` ends with `.Pooling` (the sentence-transformers layout). A `modules.json` whose only extra module is `1_LogitScore` (Qwen3-VL-Reranker) does not qualify.
- `1_Pooling/config.json` exists.

A checkpoint whose `architectures[0]` ends with `ForSequenceClassification` is a reranker, never an embedder, whatever else its layout says.

The resolved family is keyed on `model_type`:

| `model_type` | `ModelType` |
|--------------|-------------|
| `bert` | `Bert` |
| `xlm-roberta` | `XlmRoberta` |
| `modernbert` | `ModernBert` |
| `siglip` | `SiglipText` |
| `gemma3_text`, `gemma3` | `Gemma3Embedding` |
| `qwen3` | `Qwen3Embedding` |
| `qwen3_vl` | `Qwen3VLEmbedding` |
| `lfm2` | `Lfm2Embedding` |
| `ministral3` | `Ministral3Embedding` |
| `llama`, `llama_bidirec` | `LlamaBidirec` |
| `llama_nemotron_vl` | `LlamaNemotronVLEmbedding` |
| `idefics3` | `ColIdefics3` |
| `qwen2_5_vl`, `colqwen2` | `ColQwen25` |

A pooling layout on any other `model_type` is reported as an error naming the `model_type`, rather than misrouted to a causal generator. `Qwen3ForCausalLM` without a pooling layout still detects as the `Qwen3` generator.

`mlxcel arch` lists the embedding variants under the `Embedding` family. The generation loader (`load_model`, used by `mlxcel generate`, `mlxcel run` and the chat worker) rejects every embedding variant with a message pointing at `/v1/embeddings` and `mlxcel embed`, so `mlxcel-server -m <embedding checkpoint>` serves embeddings and leaves chat unloaded (the same way a Whisper checkpoint leaves chat unloaded), and `/v1/chat/completions` returns the existing "model is not loaded" error.

## Pooling

Given hidden states `[B, L, D]` (cast to f32) and the attention mask `[B, L]` (int32, `1` = real token, `0` = padding):

| Mode | Definition |
|------|------------|
| `cls` | `hidden[b, first_real_index[b], :]` with `first_real_index = argmax(mask, axis=1)`; index 0 for right padding, the first `1` for left padding. |
| `mean` | `sum(hidden * mask[..., None], axis=1) / max(sum(mask, axis=1), 1e-9)`. |
| `max` | `max(where(mask == 0, -inf, hidden), axis=1)`. |
| `lasttoken` | `hidden[b, last_real_index[b], :]`; an all-padding row uses `L - 1`. Correct for left and right padding. |

Unsupported modes (`weightedmean`, `mean_sqrt_len_tokens`, `include_prompt: false`, more than one legacy flag set) are a load error naming the mode, not a silent fallback.

Resolution order:

1. `<model_dir>/1_Pooling/config.json` when present. Both key styles are accepted: the new-style `"pooling_mode": "cls" | "mean" | "max" | "lasttoken"` and the legacy booleans `pooling_mode_cls_token`, `pooling_mode_mean_tokens`, `pooling_mode_max_tokens`, `pooling_mode_lasttoken`, `pooling_mode_weightedmean_tokens`, `pooling_mode_mean_sqrt_len_tokens`. Exactly one legacy flag may be true; none true means `mean`.
2. Otherwise the family default (`EmbeddingModel::default_pooling()`).
3. `MLXCEL_EMBEDDING_POOLING=cls|mean|max|lasttoken` overrides both, for debugging, and is logged at startup.

Normalization is L2 per vector, `v / max(||v||_2, 1e-9)`, applied when `EmbeddingModel::normalize()` is true (the default; `config.json` `normalize: false` turns it off for families that declare it). The request field `dimensions` keeps the first `dimensions` components and re-normalizes when normalization is on; `1 <= dimensions <= D` is required, any other value is a `400`.

## Length limits and padding

`max_length` is the smallest of: `sentence_bert_config.json` `max_seq_length` when present; `tokenizer_config.json` `model_max_length` when `0 < value < 1_000_000`; `config.json` `max_position_embeddings` for absolute-position encoders (BERT, XLM-RoBERTa, SigLIP); the hard cap `8192`; and `--embedding-max-length` when set. Inputs longer than `max_length` are truncated from the right after special tokens are added, keeping the trailing special token when the tokenizer appends one, so a BERT input keeps its `[SEP]` and a Qwen3-Embedding input keeps its `<|endoftext|>`. Token-id inputs are truncated from the right verbatim.

For `sentence-transformers/all-MiniLM-L6-v2` this resolves to `256` (from `sentence_bert_config.json`); for `Qwen/Qwen3-Embedding-0.6B` to the cap of `8192` (no `sentence_bert_config.json`, `model_max_length` 131072, rotary positions so `max_position_embeddings` is not consulted).

Micro-batches are right-padded to their longest member (or to a fixed width when the family requires one, as SigLIP text does) with the pad token id: `tokenizer_config.json` `pad_token`, falling back to `eos_token`, then `0`. Any padding or truncation baked into `tokenizer.json` (MiniLM pads to a fixed 128) is stripped at load, because the engine pads per micro-batch and truncates per checkpoint limit.

## Family notes

Prompt prefixes are caller-side everywhere: nothing is injected server-side, so an input embeds exactly as it is sent. The `instruction` request field (and `mlxcel embed --instruction`) reaches `EmbeddingModel::format_text`, and only a family that documents a format below does anything with it.

### BERT and XLM-RoBERTa

One encoder covers both (`src/models/bert.rs`): a post-LayerNorm block stack over absolute position embeddings, selected by a `BertVariant` switch. `model_type: bert` builds position ids as `0..L` and carries a real `token_type_ids` axis; `model_type: xlm-roberta` builds them as `cumsum(input_ids != pad_token_id) * mask + pad_token_id`, so the first real token sits at `pad_token_id + 1` and padding stays at `pad_token_id`. `intfloat/multilingual-e5-small` is a `BertModel` despite shipping the XLM-RoBERTa tokenizer, and follows the BERT rule.

| Checkpoint | Pooling | Prefix the checkpoint expects |
|------------|---------|-------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2` | mean | none |
| `intfloat/multilingual-e5-small` (and the other `multilingual-e5-*` sizes) | mean | `query: ` on a search query, `passage: ` on an indexed document, on every input including symmetric tasks |
| `BAAI/bge-m3` | cls | none |

Send them verbatim, for example `{"input": ["query: how much protein should a female eat", "passage: As a general guideline, the average daily protein intake is 46 grams."]}`. Omitting the e5 prefixes silently degrades retrieval quality rather than failing.

Length is capped by the absolute position table. BERT indexes it from 0, so `max_position_embeddings` is the token cap directly. XLM-RoBERTa starts at `pad_token_id + 1`, so it holds `pad_token_id + 1` fewer tokens: `bge-m3`'s 8194 rows cap at 8192 real tokens, which the family reports through `EmbeddingModel::max_sequence_length()` and the loader folds into the derived `max_length`. A batch longer than that is a load-time truncation, never an out-of-bounds gather.

Non-quantized f32 checkpoints run in f32. `all-MiniLM-L6-v2` ships f32 with `layer_norm_eps: 1e-12`, which underflows in f16, so the shared text bf16-to-f16 rule applies only to bf16 exports. Quantized checkpoints (`config.quantization`) load every projection and the word-embedding table through `UnifiedLinear` / `UnifiedEmbedding`.

`BertForSequenceClassification` and `XLMRobertaForSequenceClassification` checkpoints (`BAAI/bge-reranker-v2-m3`) are rerankers: detection refuses to call them embedders, and `BertSequenceClassifier::load(dir)` in `src/models/bert_heads.rs` loads one by path. It reuses the same trunk, keeps the `pooler.` tensors that the embedder path drops, and returns `[B, num_labels]` logits from `tanh(dense(h[:, 0, :]))` followed by the label projection (`pooler.dense` plus `classifier` for BERT, `classifier.dense` plus `classifier.out_proj` for XLM-RoBERTa). The `/v1/rerank` wiring is a separate port.

### ModernBERT (`modernbert`)

ModernBERT is an 8192-context bidirectional encoder with RoPE instead of an absolute position table, so `max_position_embeddings` does not cap the input length; `max_length` comes from `sentence_bert_config.json` and `tokenizer_config.json` and resolves to `8192` for both published checkpoints. Two out of every three layers use a sliding window of `local_attention / 2` keys on each side (64 with the published `local_attention: 128`) and rotate with `local_rope_theta` (10000); the remaining layers see the whole sequence and rotate with `global_rope_theta` (160000). Layer 0 legitimately ships no `attn_norm` (upstream makes it `nn.Identity()`); a missing `attn_norm` on any other layer is a load error. `Wqkv` and `Wi` are fused tensors and are split after the projection, never by slicing the weight, so a quantized export loads unchanged. `ModernBertModel` and `ModernBertForMaskedLM` both load as embedders, the MLM `head.*` / `decoder.*` tensors being dropped.

`nomic-ai/modernbert-embed-base` is asymmetric and expects an explicit task prefix in the text: `search_query: ` before a query and `search_document: ` before a passage. The prefixes are part of the input string, not a server-side wrapper, so pass them yourself in `input` (or in `mlxcel embed -p`); omitting them costs retrieval quality. It also supports Matryoshka truncation, so `dimensions` down to 256 stays meaningful (the engine re-normalizes after truncating).

```sh
mlxcel embed -m nomic-ai/modernbert-embed-base \
  -p "search_query: What is TSNE?" \
  -p "search_document: t-SNE is a dimensionality reduction technique used for visualizing high-dimensional data." \
  -p "search_document: The Eiffel Tower is a wrought-iron lattice tower in Paris."

mlxcel-server -m nomic-ai/modernbert-embed-base --port 8080
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["search_query: What is TSNE?", "search_document: t-SNE is a dimensionality reduction technique used for visualizing high-dimensional data."]}'
```

`Alibaba-NLP/gte-reranker-modernbert-base` declares `ModernBertForSequenceClassification`, which detection deliberately refuses to route to any embedding variant (a reranker is not an embedder), so `-m` on that directory is a "not an embedding checkpoint" error today. Its head is implemented as `crate::models::modernbert_heads::ModernBertSequenceClassifier`, which loads by directory and returns `[B, num_labels]` logits from `classifier(norm(gelu(dense(pooled))))` with `classifier_pooling` (`cls` or `mean`) deciding the pooling; `/v1/rerank` wiring is #1356.

### SigLIP text (`siglip`, `SiglipModel` / `SiglipTextModel`)

`src/models/siglip_text.rs` serves the text tower of a SigLIP checkpoint; the vision tower is not served on `/v1/embeddings` yet, so `image_url` items are rejected. The tower is the token plus learned-position embedding, the same pre-norm encoder block the VLM vision towers use (`src/vision/encoders/siglip.rs`), a final LayerNorm and a linear projection `head`.

- The tokenizer normalizes before it splits: it lower-cases, strips ASCII punctuation and collapses runs of whitespace, so `"A photo of a Cat!"` and `"a photo of a cat"` produce the same vector. This is the checkpoint's own `tokenizer.json`, not an mlxcel choice.
- Inputs are capped at 64 tokens (`tokenizer_config.json` `model_max_length`, and the tower's `max_position_embeddings`): 63 tokens plus the trailing `</s>` the post-processor appends. Every row is then right-padded to exactly 64, which the family requests through `EmbeddingModel::pad_to_max_length`.
- `pad_token` is `</s>` (id 1), the same id the tokenizer appends, so position 63 always holds `</s>`: the "sticky EOS" the reference pools.
- No attention mask is applied; every position attends to all 64. That is the training-time recipe, and the checkpoint agrees: its `tokenizer_config.json` declares `model_input_names: ["input_ids"]`, so the reference processor emits no mask either.
- Pooling is fixed at position 63 followed by the projection `head` and L2 normalization. `1_Pooling/config.json` is not consulted and `MLXCEL_EMBEDDING_POOLING` does not apply; `EmbeddingModel::default_pooling` reports `lasttoken` for the startup log only.
- `text_config` keys are all optional and fall back to the reference `SiglipTextConfig` defaults (`vocab_size` 32000, `hidden_size` 768, `intermediate_size` 3072, 12 heads, 12 layers, `max_position_embeddings` 64, `layer_norm_eps` 1e-6, `projection_size` = `hidden_size`, `hidden_act` `gelu_pytorch_tanh`). `google/siglip-base-patch16-224` declares only five of them.
- `vision_model.*`, `logit_scale`, `logit_bias` and any `position_ids` buffer are dropped at load; the remaining keys are used as they are.

One property to know before using these vectors for text-to-text retrieval: SigLIP's text tower is trained contrastively against images, never against other texts, so nothing in training pushes two unrelated captions apart and its text-only cosine similarities sit on a high anisotropic floor. Measured on `google/siglip-base-patch16-224` over six sentences spanning animals, machinery, finance, food and physics, the fourteen unrelated pairs scored between 0.519 and 0.725 (mean 0.653), while the one related pair (`a photo of a cat` against `a photo of a kitten`) reached 0.966. Rank candidates by margin rather than thresholding the absolute score, which is the opposite of what a sentence-transformers encoder invites.

### EmbeddingGemma (`Gemma3Embedding`)

The Gemma 3 text backbone run bidirectionally: the full-attention layers get a padding-only mask, the sliding layers get the same padding mask intersected with a symmetric `|q - k| < sliding_window` band. Up to `sliding_window` tokens (512 on the published checkpoints) the two masks are identical, so the window only starts to matter on longer inputs. The final norm output is mean pooled, then projected through two bias-free `Dense` modules (`768 -> 3072 -> 768`, no activation) before the engine's L2 normalization.

Both published layouts load. The mlx conversions fold the projections into the main shards as `dense.0.*` and `dense.1.*` and keep the `model.` prefix; the sentence-transformers original stores them in `2_Dense/` and `3_Dense/` module folders, which arrive as `2_Dense.linear.weight` and `3_Dense.linear.weight` and are ranked into `dense.0` and `dense.1`. The `Dense` widths are checked to chain from the backbone hidden size, so a swapped pair is a load error instead of a quietly wrong vector.

The alternation period comes from `config.json` `layer_types` when present: transformers 4.57 renamed the scalar to `_sliding_window_pattern`, which the Gemma 3 args type does not parse, and a period that is off by one still loads and still runs.

Prompt prefixes (from `config_sentence_transformers.json`), applied by the caller:

| Task | Prefix |
|------|--------|
| Retrieval query, reranking, bitext mining | `task: search result \| query: ` |
| Retrieval document | `title: none \| text: ` |
| Semantic similarity (STS) | `task: sentence similarity \| query: ` |
| Classification | `task: classification \| query: ` |
| Clustering | `task: clustering \| query: ` |
| Code retrieval query | `task: code retrieval \| query: ` |
| Summarization | `task: summarization \| query: ` |

`max_length` is 2048 (`sentence_bert_config.json`). The trained Matryoshka widths are 768, 512, 256 and 128, so `dimensions` at those values keeps the ranking; any other value in `1..=768` is accepted and re-normalized but is not a trained width.

```sh
mlxcel embed -m mlx-community/embeddinggemma-300m-4bit \
  -p "task: search result | query: Which planet is known as the Red Planet?" \
  -p "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet."
```

### Qwen3-Embedding (`Qwen3Embedding`)

The causal Qwen3 backbone with a causal-plus-padding mask and last-token pooling: the sentence embedding is the hidden state at the `<|endoftext|>` the tokenizer appends. Batches are right-padded, which leaves the causal mask over the real tokens identical to the unpadded single-row case. The tied `lm_head` is dropped at load, and the backbone stops at the final norm, so no `[B, L, vocab_size]` logit tensor is ever built.

Queries use `Instruct: {task}\nQuery: {query}` and documents are raw text. Pass the task through `instruction` (or `--instruction`) and the family wraps the text; send the fully formatted string and leave `instruction` unset to do it yourself. `instruction` applies to every input of the request, so a mixed query-and-document batch either splits into two requests or formats its queries in the text itself. `max_length` is the hard cap of 8192. The checkpoint supports `dimensions` from 32 to 1024.

```sh
mlxcel embed -m Qwen/Qwen3-Embedding-0.6B \
  --instruction "Given a web search query, retrieve relevant passages that answer the query" \
  -p "What is the capital of China?"
```

### ColIdefics3 (`idefics3`, `ColIdefics3`)

A late-interaction visual document retriever on the SmolVLM / Idefics3 stack: the SigLIP tower, the `pixel_shuffle(scale_factor)` connector and the SmolLM2 decoder are the ones `mlxcel generate` already runs. Retrieval changes only the ends. The decoder stops at its final norm (the checkpoint ships no `lm_head` at all), one `Linear` (`linear.{weight,bias}`, `[128, 576]`) projects every token's hidden state to 128 dimensions, each token vector is L2-normalized and padding rows are zeroed.

Prompts are built by the family, not by the caller, because the two item kinds use different formats. An image document renders as `<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:`, and the engine then replaces the single `<image>` with the processor's framed tile runs (`<fake_token_around_image><row_r_col_c>` plus 64 image tokens per tile, one row per line, then the global thumbnail block). A query renders as `Query: {text}` followed by ten `<end_of_utterance>` augmentation tokens, which is what gives a short query enough vectors for MaxSim to discriminate. The `instruction` field is not used by this family.

`vidore/colSmol-256M` is a LoRA adapter on `vidore/ColSmolVLM-Instruct-256M-base` and carries the trained projection in `1_Dense/`; mlxcel does not merge adapters, so a directory holding `adapter_model.safetensors` with no base shard is rejected with a message asking for the merged checkpoint. Merge it with the checkpoint's own tooling and keep `1_Dense/model.safetensors` next to `config.json`: when that folder is present its `linear.{weight,bias}` replace the ones in the main shard, which is what sentence-transformers does. The base repository loads on its own and produces finite, correctly shaped output, which validates the weight layout, but its projection is untrained and its ranking is not meaningful. `mask_non_image_embeddings: true` is rejected at load rather than silently ignored.

```sh
mlxcel embed -m models/colSmol-256M-merged \
  -p "What was the total revenue in 2023?" \
  --image revenue-table.png --image unrelated-page.png
```

### ColQwen2.5 (`qwen2_5_vl` / `colqwen2`, `ColQwen25`)

The same recipe on the Qwen2.5-VL stack: the windowed vision tower and the patch merger feed the Qwen2 decoder with M-RoPE, `Qwen2VLModel::forward_hidden` stops at the final norm, and `custom_text_proj` (`[128, 2048]` plus bias) projects every token. Position ids follow the input: a text-only micro-batch uses the sequential `[3, B, L]` positions the backbone builds for a text prefill, an image input keeps the real M-RoPE grid, and the per-request M-RoPE slot is cleared before a text batch so a previous image request cannot leak its spatial positions into it.

An image document renders as `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>` with the `<|image_pad|>` run expanded to one token per merged patch, and a query as `Query: {text}` plus ten `<|endoftext|>` tokens. The document turn is closed with `<|endoftext|>` rather than an assistant header, which is what the reference processor and the checkpoint's own `additional_chat_templates/sentence_transformers.jinja` both emit; nothing is ever generated after the page. `preprocessor_config.json` caps the image at `768 * 28 * 28` pixels, which bounds one page at 768 visual tokens.

Three key layouts load. `vidore/colqwen2.5-base` and its merged descendants store `model.*`, `visual.*` and `custom_text_proj.*`; the native `transformers` retrieval export nests everything under `vlm.` and names the projection `embedding_proj_layer.*`; an mlx conversion already stores `model.*` and `vision_tower.*`. The sanitizer strips the `vlm.` wrapper first, then maps `embedding_proj_layer.` to `custom_text_proj.`, `model.language_model.` and `language_model.` to `model.`, and `visual.` / `model.visual.` to `vision_tower.`. The tied `lm_head` is dropped at load. A raw HuggingFace export also stores the vision patch filter in `Conv3d`'s native `[out, in, kT, kH, kW]` layout while the encoder expects the channels-last mlx conversion, so that tensor is permuted at load; without it the tower reads scrambled filters and every page embeds to nearly the same vectors. As for ColIdefics3, `vidore/colqwen2.5-v0.2` is a LoRA adapter and has to be merged into the base first, and the base repository loads on its own with an untrained projection, which validates the layout but not the ranking.

```sh
mlxcel embed -m models/colqwen2.5-v0.2-merged \
  -p "What was the total revenue in 2023?" \
  --image revenue-table.png --image unrelated-page.png
```

## Multi-vector (late-interaction) output

A family whose `EmbeddingModel::multi_vector()` is `true` returns one vector per token instead of one per input. `EmbeddingModel::embed` produces `[B, L, D]` with the rows of padding tokens zeroed; the engine normalizes, applies `dimensions`, and trims each item to its own real token count, so the response carries `[num_real_tokens, D]`. For an image item the real token count is the count after the image placeholder has been expanded, which is what `EmbeddingModel::expand_image_tokens` computes before the batch is padded, so `usage.prompt_tokens` and the number of returned rows always describe the same sequence.

Candidates are ranked with MaxSim, not cosine:

```
maxsim(q, d) = sum over query rows i of ( max over document rows j of dot(q_i, d_j) )
```

`crate::embeddings::maxsim` implements it over read-back rows and `maxsim_mlx` over device arrays. Because every row is L2-normalized, each inner product is a cosine and the score is bounded by the query's row count; it is reported raw rather than averaged, since comparing two documents for one query is what the number is read for. The score is asymmetric: the outer sum always runs over the first argument.

`mlxcel embed` prints this matrix, labelled `MaxSim similarity`, in place of the cosine matrix whenever the loaded model is multi-vector.

### Bidirectional Llama, LLM2Vec (`LlamaBidirec`)

The LLM2Vec recipe: an ordinary Llama 3.2 1B decoder converted to an encoder by dropping the causal mask. `nvidia/llama-nemotron-embed-1b-v2` exports it as `model_type: llama_bidirec`, `architectures: ["LlamaBidirectionalModel"]`, `use_bidirectional_attention: true` and a `1_Pooling` module declaring mean pooling. Nothing about the layers changes: the same `rope_scaling` `llama3` schedule at factor 32, the same norms, the same GQA 32/8 with `head_dim` 64. Only the mask and the missing head differ, so the port builds `llama3::TransformerBlock` layers straight from the weight map rather than through `Llama3Model`, whose `lm_head` field is not optional and would otherwise materialize a tied `128256 x 2048` projection this path never applies.

The published export saves the inner `LlamaBidirectionalModel`, so the roots arrive bare (`embed_tokens.weight`, `layers.0.…`, `norm.weight`) and are prefixed with `model.` at load. A `language_model.` wrapper prefix is stripped first, and the derived `rotary_emb.inv_freq` and `position_ids` buffers are dropped, since this tree rebuilds the frequency table from `rope_theta` and `rope_scaling`.

LLM2Vec checkpoints that ship as PEFT adapters are not loadable. A directory carrying `adapter_model.safetensors` and no full shard is rejected with a message saying to merge the adapter into its base model and export a complete `LlamaBidirectionalModel` checkpoint first; mlxcel does not merge adapters.

`max_length` is 8192 (`sentence_bert_config.json`, which happens to equal the hard cap). Prompt prefixes from `config_sentence_transformers.json`, applied by the caller: `query: ` before a query, `passage: ` before a document.

```sh
mlxcel embed -m nvidia/llama-nemotron-embed-1b-v2 \
  -p "query: how do solar panels generate electricity" \
  -p "passage: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect."
```

### Nemotron-3-Embed (`Ministral3Embedding`)

The Ministral 3 backbone run bidirectionally. `nvidia/Nemotron-3-Embed-1B-BF16` and the `mlx-community` 8-bit conversion declare `model_type: ministral3`, `architectures: ["Ministral3Model"]` and `is_causal: false`, which is the flag detection keys on; the pooling module says mean.

Two details carry the port. The per-position Llama 4 attention scale (`1 + beta * ln(1 + floor(pos / original_max_position_embeddings))`, `beta` 0.1, window 16384) is computed at offset 0, because the embedder runs one prefill and never decodes, which is the same value the generator's fresh cache reports for the same length. And `tie_word_embeddings` is true on both checkpoints, so `lm_head` is absent and the forward pass stops at the final norm. Both published checkpoints carry `sliding_window: null` and no `layer_types`, so every layer is full attention; a checkpoint that did declare `sliding_attention` layers would get `create_bidirectional_window_mask` for those.

`max_length` is the hard cap of 8192, below the checkpoint's declared 32768. Prompt prefixes, applied by the caller: `query: ` and `passage: `.

```sh
mlxcel embed -m nvidia/Nemotron-3-Embed-1B-BF16 \
  -p "query: how do solar panels generate electricity" \
  -p "passage: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect."
```

### LFM2.5-Embedding (`Lfm2Embedding`)

The hybrid LFM2 backbone with both of its mixers made bidirectional. `LiquidAI/LFM2.5-Embedding-350M` is 16 layers alternating 10 gated short convolutions with 6 full-attention layers (`conv_L_cache` 3, GQA 16/8 with per-head Q/K RMSNorm, `embedding_norm` as the final norm), exported as `Lfm2BidirectionalModel` with a `1_Pooling` module declaring CLS.

The attention layers take the usual padding-only mask. The short conv needs two changes, and they are the only ones that reach into the generator's file:

- `ModelArgs` grows a `conv_causal` flag, default `true`, that the embedding loader sets to `false`. The mixer then splits its `L_cache - 1` zero padding across both sides instead of prepending all of it, so position `t` mixes `t - 1`, `t` and `t + 1` and the output length stays `L`. It also stops writing a conv state, which a one-shot bidirectional pass has no use for. Generation keeps the default and is byte-identical.
- The conv input is zeroed at padding positions through a `[B, L, 1]` multiplier, the same thing the reference's `apply_mask_to_padding_states` does. A convolution has no key axis for an attention mask to act on, so without this the pad-token embeddings mix into the real positions next to the boundary and the attention layers above spread that across the whole row. Measured on the published checkpoint before the fix, changing only the masked tail of a padded row moved the pooled vector by cosine 0.94.

Pooling is CLS: the tokenizer's post-processor prepends `<|startoftext|>` and the sentence vector is the hidden state there. Right padding puts it at index 0 in every row, though the pooling finds it by first-real-token argmax rather than assuming that.

The LFM2 and LFM2.5 ColBERT late-interaction checkpoints share this `model_type` and this architecture but project every token through a `1_Dense` module and emit one vector per token. They are rejected at load rather than served as a single pooled vector.

`max_length` is 512 (`sentence_bert_config.json`), well under the hard cap. Prompt prefixes, applied by the caller: `query: ` and `document: `.

```sh
mlxcel embed -m LiquidAI/LFM2.5-Embedding-350M \
  -p "query: how do solar panels generate electricity" \
  -p "document: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect."
```

### Batch geometry and bf16

One property is worth knowing before comparing vectors across runs. On a bf16 checkpoint the attention kernels pick their accumulation shape from the batch geometry, so the same text embedded alone and embedded as a row of a larger batch agrees to roughly cosine 0.9999 rather than exactly. This is arithmetic, not padding: embedding one text as several unpadded copies of a single batch shows the same spread, and it is visible on every bf16 decoder-backbone family here. Padding itself changes nothing, which each family gates exactly, at zero tolerance, by holding the batch shape fixed and changing only the token ids underneath the mask.

## Attention masks

Families build additive f32 masks (`0.0` = attend, `-inf` = blocked) from the `[B, L]` attention mask with three builders in `mlxcel_core::utils`, next to `create_causal_mask`:

- `create_bidirectional_padding_mask(mask) -> [B, 1, 1, L]`: key column `k` is blocked iff `mask[b, k] == 0`. Padding query rows still see every real key, so no row is fully blocked.
- `create_causal_padding_mask(mask, offset) -> [B, 1, L, L + offset]`: `k <= q + offset` and `mask[b, k] == 1`. A fully padded query row keeps its diagonal column, the same rescue `create_causal_mask_with_left_padding` applies.
- `create_bidirectional_window_mask(mask, window) -> [B, 1, L, L]`: blocked iff `mask[b, k] == 0` or `|q - k| >= window`.

A mask is never built as `(1 - m) * C` with a finite `C` in the activation dtype: in f16, `-1e9` overflows to `-inf` and `0 * -inf` is NaN. The builders produce `0 / -inf` in f32, which survives the cast `fast_scaled_dot_product_attention` applies for f16 and bf16 activations.

## Request and response

```json
{
  "model": "optional; must equal the served embedding model id when given",
  "input": "string | [string] | [int] | [[int]] | [{\"type\": \"text\", \"text\": ...} | {\"type\": \"image_url\", \"image_url\": {\"url\": ...}}]",
  "encoding_format": "float | base64",
  "dimensions": 256,
  "instruction": "optional; forwarded to EmbeddingModel::format_text",
  "user": "ignored"
}
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.01, 0.02]}],
  "model": "all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 17, "total_tokens": 17}
}
```

- Token-id inputs are used verbatim (no special tokens added) and every id must be `< vocab_size`.
- `encoding_format: base64` encodes each vector as little-endian f32 bytes, standard base64 with padding. For multi-vector (late-interaction) models `embedding` is a list of vectors, `[num_real_tokens][D]`, in float mode; in base64 mode it is the row-major bytes plus a sibling `"shape": [num_real_tokens, D]`.
- `usage.prompt_tokens` counts the real (non-padding) tokens across all inputs, special tokens included; `total_tokens` equals it.
- `image_url` items accept a `data:image/...;base64,` URI, a `file://` path, an `http(s)://` URL or a local path, bounded by the shared image limits. Images are embedded one at a time; text items are sorted by token length, cut into micro-batches of `--embedding-batch-size`, and written back in request order.

### Errors

| Status | Type | When |
|--------|------|------|
| 400 | `invalid_request_error` | Malformed body, empty `input`, an empty string item, an empty token list, a token id `>= vocab_size`, an image item for a model with `supports_images() == false`, `dimensions` out of range, an unsupported `encoding_format`, or `model` not matching the served id. |
| 501 | `not_implemented` | No embedding model loaded: `No embedding model loaded; start with -m <embedding checkpoint> or --embedding-model <path>`. |
| 503 | `server_busy` | The bounded worker queue is full (`--embedding-queue-depth`). |
| 504 | `server_timeout` | The worker did not reply within `--embedding-request-timeout-secs`. |
| 500 | `server_error` | The forward pass failed. |

## Server flags

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `-m <embedding checkpoint>` | `LLAMA_ARG_MODEL` | | Serves the checkpoint on `/v1/embeddings`; chat stays unloaded. |
| `--embedding-model <path or repo-id>` | `LLAMA_ARG_EMBEDDING_MODEL`, `MLXCEL_EMBEDDING_MODEL` | unset | A second checkpoint served on `/v1/embeddings` next to the chat model in `-m`. Resolved and auto-downloaded like `-m`. Combining it with an embedding checkpoint in `-m` is a startup error ("two embedding models"). A load failure of an explicit `--embedding-model` is a startup error; a load failure of an `-m` embedding checkpoint is logged and the route answers `501`. |
| `--embedding-batch-size N` | `MLXCEL_EMBEDDING_BATCH_SIZE` | `16` | Texts per forward pass. |
| `--embedding-max-length N` | `MLXCEL_EMBEDDING_MAX_LENGTH` | derived | Lowers the derived `max_length`. |
| `--embedding-queue-depth N` | `MLXCEL_EMBEDDING_QUEUE_DEPTH` | `8` | Bound on the worker command queue; a full queue returns `503`. |
| `--embedding-request-timeout-secs N` | `MLXCEL_EMBEDDING_REQUEST_TIMEOUT_SECS` | `120` | Per-request reply timeout; `0` falls back to the default. |

Both `mlxcel serve` and `mlxcel-server` accept every flag. The embedding model runs on its own dedicated MLX thread (the model, tokenizer and every array live there), so chat and embeddings never share a stream.

```sh
# Embeddings only
mlxcel-server -m sentence-transformers/all-MiniLM-L6-v2 --port 8080

# Chat plus embeddings
mlxcel-server -m mlx-community/Qwen3-4B-4bit --embedding-model Qwen/Qwen3-Embedding-0.6B

curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["The weather is lovely today.", "It is so sunny outside!", "He drove to the stadium."]}'
```

## `mlxcel embed`

```text
mlxcel embed -m <path or repo-id> -p "text" [-p "text2" ...] [--image <file> ...]
             [--instruction "..."] [--dimensions N] [--max-length N] [--batch-size N] [--json]
```

Prints one vector per input (`[v1, v2, ...]`, one line each; a list of rows for multi-vector models) and, with two or more inputs, the similarity matrix: cosine for single-vector models, MaxSim for multi-vector ones (the header names which one). `--json` prints one object with `embeddings`, `shapes`, `prompt_tokens` and `similarity` instead. This is the offline validation tool for every family: the same loader, pooling, normalization and batching as the server, without a listener.

## Adding a family

1. Add the family module under `src/models/` with an `EmbeddingModel` implementation. Read weights with `crate::embeddings::loader::load_embedding_weights` (module subfolders included, text bf16 rule applied) and resolve the pooling mode with `crate::embeddings::resolve_pooling_mode(model_dir, family_default)`. Build attention masks from `EmbeddingBatch::attention_mask` with the builders above. Return pooled `[B, D]` vectors (`[B, L, D]` with padding rows zeroed for multi-vector families); the engine normalizes and truncates. A vision-language family whose prompt carries an image placeholder also implements `EmbeddingModel::expand_image_tokens`, which the engine calls before padding so the reported token count matches the sequence the forward pass sees.
2. Replace the family's `not yet supported` arm in `build_family_model` (`src/embeddings/loader.rs`) with the constructor.
3. Quantized checkpoints (`config.quantization = {group_size, bits}`) go through `UnifiedLinear::from_weights` / `UnifiedEmbedding::from_weights`, which accept a tensor with or without `.scales`; `quantization_params(config)` reads the block.
4. Validate with `mlxcel embed` against a real checkpoint and add the family row to [supported-models.md](supported-models.md#embedding-models).

Detection (`src/models/detection.rs`), the `ModelType` variant, the `mlxcel arch` entry and the `ModelKind::Embedding` registration in `src/model_metadata.rs` already exist for every family in the table above.
