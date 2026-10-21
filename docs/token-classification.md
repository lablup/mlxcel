# Token classification and span extraction

mlxcel runs two kinds of per-token output model offline, each behind its own subcommand:

| Shape | Subcommand | Labels | Output |
|---|---|---|---|
| Token classification | `mlxcel tag` | fixed by the checkpoint (`id2label`, BIO or BIOES) | tagged character spans and a redacted copy of the text |
| Span extraction | `mlxcel extract` | chosen per request (`--labels`, `--labels-json`, `--classify`) | label-keyed span lists, or one class per task |

The two shapes are separate traits (`TokenClassifierModel` and `SpanExtractorModel` in `src/token_classification/`) because they take different inputs: a fixed-label model handed request labels would have to ignore them silently. Passing a checkpoint to the other shape's subcommand fails with a message naming the right one.

mlxcel does not serve either shape over HTTP, by decision. mlx-vlm, the reference implementation, defines no route, so there is no upstream request or response schema to match, and OpenAI-compatible clients have no endpoint for per-token tags or request-time spans. A route would be designed in its own issue if a client needs one; until then the CLI is the only surface.

## Supported families

| Family | Detection | Kind | Decode | Validated checkpoint |
|---|---|---|---|---|
| BERT / XLM-RoBERTa token classifier | `model_type` `bert`, `xlm-roberta` or `xlm_roberta`, `architectures[0]` ending in `ForTokenClassification`, non-empty `id2label` | `bert_token_classifier` | per-token argmax | [`nativ-community/rampart-mlx-fp16`](https://huggingface.co/nativ-community/rampart-mlx-fp16), [`nativ-community/rampart-mlx-4bit`](https://huggingface.co/nativ-community/rampart-mlx-4bit) |
| OpenAI Privacy Filter | `model_type` `openai_privacy_filter`, non-empty `id2label` | `openai_privacy_filter` | constrained BIOES Viterbi (default), per-token argmax with `--argmax` | [`mlx-community/openai-privacy-filter-bf16`](https://huggingface.co/mlx-community/openai-privacy-filter-bf16), [`mlx-community/openai-privacy-filter-8bit`](https://huggingface.co/mlx-community/openai-privacy-filter-8bit) |
| GLiNER2.5 span extractor | `model_type` `gliner2` or `gliner2_5`, or `architectures` naming `Gliner2ForSchemaExtraction` or `BoundaryExtractor`; `architecture` `boundary` | `gliner2_5` | sigmoid span scores over a shared candidate pool, overlaps resolved per label | [`fastino/gliner2.5-small-v1`](https://huggingface.co/fastino/gliner2.5-small-v1) |

GLiNER2.5 is the first span-extraction family; see [GLiNER2.5](#gliner25) under `mlxcel extract`.

Detection for these commands reads `config.json` only and is separate from the generation, `/v1/embeddings` and `/v1/rerank` dispatch, which is unchanged. In particular `mlxcel embed` still loads a `BertForTokenClassification` checkpoint as a BERT embedder (its `model_type` is encoder-only), exactly as before this feature existed.

## `mlxcel tag`

```sh
mlxcel tag -m nativ-community/rampart-mlx-fp16 --keep CITY,STATE,ZIP_CODE \
  "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net"
```

```json
{
  "text": "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net",
  "spans": [
    {"label": "GIVEN_NAME", "start": 4, "end": 9, "text": "Sarah", "placeholder": "<GIVEN_NAME>"},
    ...
  ],
  "redacted_text": "I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>"
}
```

The flags mirror `python -m mlx_vlm.token_classification`:

| Flag | Meaning |
|---|---|
| `-m, --model` | checkpoint directory or HuggingFace repo-id (auto-downloaded like `mlxcel generate -m`) |
| `TEXT` | the text to tag; `-` reads it from stdin |
| `--argmax` | force per-token decoding on a model whose default decoder is Viterbi |
| `--replacement STR` | the string every redacted span becomes, instead of its placeholder |
| `--keep L1,L2` | entity labels reported in `spans` but left in `redacted_text`; unknown labels are ignored |
| `--operating-point NAME` | calibration operating point of a Viterbi-decoded model (OpenAI Privacy Filter; default `default`); refused before any weight loads for a model that decodes with argmax only |
| `--models-dir` | model-store root for repo-id resolution |

The JSON goes to stdout; a one-line summary (span count, kind, `model_type`) goes to stderr.

Only `redacted_text` is redacted: `text` and every `spans[].text` still hold the original values, so pass on `redacted_text` alone when the output leaves a trusted context.

### Pipeline

1. Tokenize without special tokens. The tokenizer must report character offsets, which every `tokenizer.json` tokenizer does; a SentencePiece or tiktoken fallback is refused with `Token classification requires a tokenizer with offsets` before any inference. Padding and truncation baked into `tokenizer.json` are removed, so long text is never cut.
2. Split the ids into windows of `context_size - prefix - suffix` tokens and wrap each in the special tokens (the tokenizer's `[CLS]` / `[SEP]`, or `<s>` / `</s>` for XLM-RoBERTa, when both exist). For BERT the context is `max_position_embeddings`; for XLM-RoBERTa it is the number of positions the table actually addresses (`max_position_embeddings - pad_token_id - 1`).
3. Score each window, take a log-softmax in f32, strip the special positions, and concatenate the windows.
4. Decode a label path (argmax unless the model defaults to a constrained decoder, as Privacy Filter's Viterbi does).
5. Group sub-tokens into words when the model asks for it (BERT does): a word takes the tag of its first non-`O` piece and the rest become that entity's `I` state, so a word split across a window boundary still forms one span.
6. Build spans from the BIO / BIOES runs: `I` after a different entity starts a new span, and an `E` with no matching open span becomes a one-token span. Character ranges come from the offsets with whitespace trimmed at both ends; empty spans are dropped. Spans are sorted by `(start, -length, label)`, a span starting before the end of the previous kept one is dropped, and (with word grouping) a same-label span starting exactly where the previous one ended is merged into it, which is how `617-555-0142` stays one phone number.
7. Redact right to left with `--replacement` or the placeholder, skipping `--keep` labels. The placeholder is `<` + the label upper-cased with every run of non-alphanumerics replaced by `_` and trimmed of `_` (`REDACTED` when nothing is left) + `>`.

Offsets are Unicode scalar indices into `text`, which is what Python string indexing uses, so the output matches mlx-vlm field for field even with multi-byte characters before a span. Empty text returns no spans and the text unchanged.

### Load-time errors

| Condition | Message |
|---|---|
| a label that is neither `O` nor `B-`, `I-`, `E-` or `S-` followed by a name | `Invalid token label '<l>'; expected BIO or BIOES labels` |
| no `O` label | `Token labels must include the background label 'O'` |
| `id2label` missing or empty | `Token-classification checkpoints must define id2label` |
| `id2label` size differs from the `classifier` tensor | `id2label has <n> entries but the classifier has <m> outputs` |
| a span-extraction checkpoint | names `mlxcel extract` |
| Privacy Filter entity without all of `B`, `I`, `E`, `S` | `Privacy Filter labels for '<entity>' are missing ["E", "S"]` |
| `--operating-point` not in `viterbi_calibration.json` | `Unknown Privacy Filter operating point '<name>'` |
| an operating point missing one of the six biases | `Viterbi calibration is missing biases: [...]` |
| `--operating-point` on an argmax-only model | `--operating-point '<name>' applies only to Viterbi-decoded checkpoints (openai_privacy_filter)` |

### OpenAI Privacy Filter

[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) is a 1.4B-parameter bidirectional sparse-MoE encoder (8 layers, 128 experts, top-4) that tags PII with BIOES labels: `O` plus `B-`, `I-`, `E-`, `S-` for `account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url` and `secret`. mlxcel runs it through the same `mlxcel tag` pipeline with the upstream `PrivacyFilter` settings: windows of `default_n_ctx` (128000) raw ids, no special tokens, no word grouping.

```sh
mlxcel tag -m mlx-community/openai-privacy-filter-bf16 "Alice Smith can be reached at alice@example.com."
```

```json
{
  "text": "Alice Smith can be reached at alice@example.com.",
  "spans": [
    {"label": "private_person", "start": 0, "end": 11, "text": "Alice Smith", "placeholder": "<PRIVATE_PERSON>"},
    {"label": "private_email", "start": 30, "end": 47, "text": "alice@example.com", "placeholder": "<PRIVATE_EMAIL>"}
  ],
  "redacted_text": "<PRIVATE_PERSON> can be reached at <PRIVATE_EMAIL>."
}
```

The encoder is `src/models/openai_privacy_filter.rs`, built from the GPT-OSS parts (`src/models/gpt_oss_common.rs`: YaRN frequencies, clipped SwiGLU, biased expert projections): biased GQA attention with per-head sinks over a symmetric 128-token band, interleaved YaRN RoPE (`factor` 32, `truncate: false`) with the `1 + 0.1 ln(factor)` attention factor, `head_dim^-0.25` on both q and k, and router and experts in f32 (routed rows are sorted by expert before the gathers once a chunk has 64 of them, on dense and quantized planes alike; the "allocation-history-dependent" drift that kept dense planes unsorted was the CUDA overlay reading the stride of a size-1 axis as the sorted `gather_mm` leading dimension, fixed in #2462). Attention runs in query chunks of `attention_chunk_size` (4096) against the keys within the band, so no `T x T` mask is built, and routing in chunks of `moe_chunk_size` (2048). Besides the bf16 and 8-bit builds, the other `mlx-community/openai-privacy-filter-*` quantizations (`-4bit`, `-5bit`, `-6bit`, `-mxfp4`, `-mxfp8`, `-nvfp4`) load through the same path but were not validated; the fused Hugging Face expert layout (`gate_up_proj`, `down_proj`) of `openai/privacy-filter` is split at load as upstream `sanitize` does.

**Decoding.** The default is the constrained Viterbi decoder of `src/token_classification/viterbi.rs`, a port of `mlx_vlm.privacy_filter.ViterbiDecoder`: a path may start with `O`, any `B` or any `S` and must end with `O`, any `E` or any `S`; `O` follows `O` or a closed `E` / `S`; `B` and `S` follow `O` or a closed state; an entity's `I` and `E` follow that entity's `B` or `I`. Ties take the first option listed, and among closed or end states the first candidate in upstream's order (every `E` before every `S`, `O` first among end states), not label output order. `--argmax` switches to independent per-token decoding, which can emit paths the span builder then repairs (an `I` after `O` becomes a new span).

**Operating points.** `viterbi_calibration.json` in the checkpoint carries `operating_points.<name>.biases`, six additive log-domain biases on the lattice transitions: `transition_bias_background_stay`, `transition_bias_background_to_start`, `transition_bias_inside_to_continue`, `transition_bias_inside_to_end`, `transition_bias_end_to_background`, `transition_bias_end_to_start`. `--operating-point NAME` selects one (default `default`); a checkpoint without the file decodes with all zeros, an unknown name or a missing key fails before any weight is read, and the flag is refused for models without a Viterbi decoder. The published `openai/privacy-filter` and mlx-community checkpoints ship a single `default` operating point with every bias at 0.

Parity with mlx-vlm 0.7.8 (`tests/fixtures/token_classification/privacy_filter_mlx_vlm.json`, produced by `generate_privacy_filter_fixture.py`): per-token label probabilities of one sentence within 2e-2 (tail log-probabilities below -18 nats move by several nats between a bf16 and an f32 forward of the same weights, so the log domain is only bounded on labels above -10 nats), and identical spans for six sentences (CJK and umlaut text included) plus `--replacement` and `--keep` under both Viterbi and argmax, on the bf16 build; the 8-bit build prints the same spans. Chunked attention and routing past 4096 / 2048 tokens are checked against an unchunked forward of the same weights, since on long repeated text the model itself sits at 50/50 on many tokens and no oracle span comparison is meaningful there. `mlxcel generate -m <privacy-filter>` fails with `openai_privacy_filter is a token classifier; use \`mlxcel tag\``, and `mlxcel embed` does not claim the checkpoint.

## `mlxcel extract`

```sh
mlxcel extract -m fastino/gliner2.5-small-v1 --labels company,person,location --include-confidence --include-spans "Apple hired Sam in New York."
mlxcel extract -m fastino/gliner2.5-small-v1 --labels-json '{"person": "a human name", "company": null}' "Ada Lovelace worked with Charles Babbage."
mlxcel extract -m fastino/gliner2.5-small-v1 --classify '{"sentiment": ["positive", "neutral", "negative"], "topics": {"labels": ["tech", "sports"], "multi_label": true, "threshold": 0.3}}' "The launch was excellent."
```

```json
{
  "entities": {
    "company": [{"text": "Apple", "confidence": 0.9993, "start": 0, "end": 5}],
    "person": [{"text": "Sam", "confidence": 0.9905, "start": 12, "end": 15}],
    "location": [{"text": "New York", "confidence": 0.9995, "start": 19, "end": 27}]
  }
}
```

Exactly one of `--labels`, `--labels-json` and `--classify` is required. `--threshold` (default 0.5, between 0 and 1) is the minimum entity score and the default multi-label threshold. `--include-confidence` and `--include-spans` apply to entity extraction only. `--max-len <N>` and `--word-splitter whitespace|char` are GLiNER2.5 only: the first overrides the checkpoint's `max_len` (an integer in 1..=32768; default the checkpoint's value, 4096 on `fastino/gliner2.5-small-v1`), the second cuts the text into words one per non-space character (`char`, for scripts without word spacing such as CJK) instead of the default whitespace run.

Entity output follows mlx-vlm's GLiNER2.5 `extract_entities`: `{"entities": {"<label>": [...]}}` with one key per requested label, in request order. Each item is the matched string, or `{"text", "confidence"?, "start"?, "end"?}` when `--include-confidence` or `--include-spans` is set. `--classify` takes `{"<task>": [labels] | {"labels", "prompt"?, "label_descriptions"?, "multi_label"?, "threshold"?}}` and prints `{"<task>": "<label>"}`, or a label list for a multi-label task.

A token-classification checkpoint passed to `mlxcel extract` fails before its weights load, naming `mlxcel tag`.

### GLiNER2.5

[GLiNER2.5](https://huggingface.co/fastino/gliner2.5-small-v1) (Fastino) is a DeBERTa-v3 encoder with a *boundary head*: the labels of a request are encoded together with the text, and a span is scored by its start and end boundaries, its content and the label query, without generation. `src/token_classification/gliner2_5.rs` ports mlx-vlm's `mlx_vlm/models/gliner2_5` (`gliner2_5.py`, `boundary.py`); the encoder is the reusable `src/models/deberta_v2.rs`.

**Detection.** The live `fastino/gliner2.5-small-v1` `config.json` declares `model_type: gliner2` and `architectures: ["Gliner2ForSchemaExtraction"]`, which mlx-vlm's loader does not map (it keys on `BoundaryExtractor`); mlxcel accepts `model_type` `gliner2` or `gliner2_5` and either architecture. `architecture` must be `boundary` (it may be absent only on a `BoundaryExtractor` export); any other head fails with `GLiNER2 architecture '<a>' is not supported; only the boundary extractor is`. The encoder config is read inline from `encoder_config`, else from `encoder_config/config.json`, else the load fails naming that file. As upstream, a `candidate_pool` other than `shared`, `candidate_attention_layers` or `query_attention_layers` above 0 are refused at load; so are `boundary_head.pool_boundary_top_k` above 1024 (the candidate grid is its square), `max_len` above 32768, and an encoder config without `share_att_key: true` (Hugging Face's default is `false`, the separate-projection layout this port does not implement). The `boundary_proposer`, `pair_scorer`, `candidate_encoder`, `record_decoder` and `relation_scorer` tensors (relation and record extraction) are dropped. Weights load in the checkpoint's own dtype (`gliner2.5-small-v1` ships float32 tensors and runs in float32, as it does in mlx-vlm).

**Input.** The sequence is `( [P] <prompt> ( [E] label ... ) ) [SEP_TEXT]` followed by the sub-tokens of each word, where the prompt is `entities` for extraction (the task name for classification, `"<task>: <prompt>"` when the task gives one) plus ` [DESCRIPTION] <label>: <description>` for every described label; classification uses `[L]` markers. The tokenizer must already define `[SEP_TEXT]`, `[P]`, `[E]`, `[L]` and `[DESCRIPTION]`, else the load fails with `checkpoint tokenizer is missing GLiNER2.5 special tokens`. Words come from upstream's regex (URLs, e-mail addresses and `@handles` stay whole, `\w+` runs may be joined by `-` or `_`, any other non-space character is a word), lower-cased; a word with no sub-token becomes `[UNK]`. Words are added until the next one would exceed `max_len` (4096 by default); a schema that alone reaches `max_len` fails with `schema alone exceeds max_len`, and a text none of whose words fit with `no text tokens fit within max_len`. A text not ending in `.`, `!` or `?` gets a `.` appended before offsets are reported. Each call encodes one sequence. The `char` word splitter of the Python model (runs of ASCII letters, digits and `@._-+` are words and every other non-space character is its own word, for scripts without spacing) is available through `--word-splitter char`; `max_len` comes from the config and can be overridden with `--max-len` (both GLiNER2.5 only, an integer in 1..=32768). Word characters follow Python's `\w` (letters, numbers and `_`, no combining marks or joiners), and the e-mail branch of the word regex is bounded to the RFC 5321 lengths so a long dotted run is not rescanned per word; only addresses that are invalid anyway split differently from upstream. Words past `max_len` are dropped with a `tracing` warning (visible with `RUST_LOG=warn`). A checkpoint converted by mlx-vlm (`encoder.layers.`, `.self_attn.`, `.layer_norm.` names, quantized or not) loads through the same path as the Hugging Face layout.

**Decoding.** Span logits go through a sigmoid; a label whose abstention logit exceeds 0.5 after the sigmoid returns nothing. Candidates at or above `--threshold` with a valid word range are resolved per label by upstream's weighted-interval dynamic program (`_resolve_flat_overlaps`), keeping the non-overlapping set with the largest total score, listed by descending score, then start, then end. Character offsets are Unicode scalar indices into the (possibly `.`-terminated) text. `--classify` runs one sequence per task, takes the sigmoid of the classifier over the label markers and returns the best label, or every label at or above the task threshold when `multi_label` is set.

**Parity.** `tests/fixtures/token_classification/gliner2_5_mlx_vlm.json` (written by `generate_gliner2_5_fixture.py` with mlx-vlm 0.7.8 at git `73d366ec`) holds the README extraction example with and without label descriptions, an e-mail / URL text without terminal punctuation, a CJK text under the `char` splitter, and two classification tasks; mlxcel reproduces the input ids, word and marker positions, spans and offsets exactly and the confidences within 1e-3. Every fixture case is under 128 tokens because mlx-vlm's DeBERTa leaves relative positions unbucketed when `max_relative_positions` is -1 and clips them, where Hugging Face log-buckets them; mlxcel follows Hugging Face, and `src/models/deberta_v2_tests.rs` checks a 306-token input against `transformers` `DebertaV2Model` (fixture `deberta_v2_hf_long.json`), where mlx-vlm's hidden states differ from `transformers` by up to 7e-2. The same test asserts that an encode and its replay agree bit for bit, and `real_encoder_first_run_equals_replay` does so in fresh processes: on CUDA the first run of a fresh process used to return wrong states in about 40% of the processes on GB10, a cuBLASLt GEMM overlapping unrelated kernel nodes of the same CUDA graph, which the CUDA device overlay now prevents by making captured child graphs barriers (#2462, `MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER`). `mlxcel generate -m <gliner>` fails with `GLiNER2.5 is a span extractor; use \`mlxcel extract\``.

## Adding a family

1. Add a `TokenModelKind` variant with its `as_str`, `description` and `output_shape` arms in `src/token_classification/loader.rs`.
2. Add a detector function to `DETECTORS` there (config only, returning `Ok(None)` for configs it does not claim).
3. Implement `TokenClassifierModel` (fixed labels; `log_probs` gets one wrapped window and returns one row per id, and `decode_modes()[0]` is the default) or `SpanExtractorModel`, and add the loader arm in `load_token_classifier` or `load_span_extractor`.
4. Add a real-checkpoint test against a fixture produced by the reference implementation, as `src/token_classification/real_checkpoint_tests.rs` does for Rampart with `tests/fixtures/token_classification/generate_rampart_spans.py`, `src/token_classification/privacy_filter_checkpoint_tests.rs` does for Privacy Filter with `generate_privacy_filter_fixture.py`, and `src/token_classification/gliner2_5_checkpoint_tests.rs` does for GLiNER2.5 with `generate_gliner2_5_fixture.py`.
