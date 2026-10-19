# Token classification and span extraction

mlxcel runs two kinds of per-token output model offline, each behind its own subcommand:

| Shape | Subcommand | Labels | Output |
|---|---|---|---|
| Token classification | `mlxcel tag` | fixed by the checkpoint (`id2label`, BIO or BIOES) | tagged character spans and a redacted copy of the text |
| Span extraction | `mlxcel extract` | chosen per request (`--labels`, `--labels-json`, `--classify`) | label-keyed span lists, or one class per task |

The two shapes are separate traits (`TokenClassifierModel` and `SpanExtractorModel` in `src/token_classification/`) because they take different inputs: a fixed-label model handed request labels would have to ignore them silently. Passing a checkpoint to the other shape's subcommand fails with a message naming the right one.

There is no HTTP route for either shape yet. mlx-vlm, the reference implementation, defines none either, so serving these over HTTP is left for a later issue.

## Supported families

| Family | Detection | Kind | Decode | Validated checkpoint |
|---|---|---|---|---|
| BERT / XLM-RoBERTa token classifier | `model_type` `bert`, `xlm-roberta` or `xlm_roberta`, `architectures[0]` ending in `ForTokenClassification`, non-empty `id2label` | `bert_token_classifier` | per-token argmax | [`nativ-community/rampart-mlx-fp16`](https://huggingface.co/nativ-community/rampart-mlx-fp16), [`nativ-community/rampart-mlx-4bit`](https://huggingface.co/nativ-community/rampart-mlx-4bit) |

No span-extraction family is implemented yet; `mlxcel extract` currently refuses every checkpoint, and GLiNER2.5 is the planned first one.

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
| `--models-dir` | model-store root for repo-id resolution |

The JSON goes to stdout; a one-line summary (span count, kind, `model_type`) goes to stderr.

### Pipeline

1. Tokenize without special tokens. The tokenizer must report character offsets, which every `tokenizer.json` tokenizer does; a SentencePiece or tiktoken fallback is refused with `Token classification requires a tokenizer with offsets` before any inference. Padding and truncation baked into `tokenizer.json` are removed, so long text is never cut.
2. Split the ids into windows of `context_size - prefix - suffix` tokens and wrap each in the special tokens (the tokenizer's `[CLS]` / `[SEP]`, or `<s>` / `</s>` for XLM-RoBERTa, when both exist). For BERT the context is `max_position_embeddings`; for XLM-RoBERTa it is the number of positions the table actually addresses (`max_position_embeddings - pad_token_id - 1`).
3. Score each window, take a log-softmax in f32, strip the special positions, and concatenate the windows.
4. Decode a label path (argmax unless the model defaults to a constrained decoder).
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

## `mlxcel extract`

```sh
mlxcel extract -m <MODEL> --labels person,company "Ada Lovelace worked with Charles Babbage."
mlxcel extract -m <MODEL> --labels-json '{"person": "a human name", "company": null}' --include-confidence --include-spans "..."
mlxcel extract -m <MODEL> --classify '{"sentiment": ["positive", "negative"], "topics": {"labels": ["tech", "sports"], "multi_label": true, "threshold": 0.3}}' "..."
```

Exactly one of `--labels`, `--labels-json` and `--classify` is required. `--threshold` (default 0.5, between 0 and 1) is the minimum entity score and the default multi-label threshold. `--include-confidence` and `--include-spans` apply to entity extraction only.

Entity output follows mlx-vlm's GLiNER2.5 `extract_entities`: `{"entities": {"<label>": [...]}}` with one key per requested label, in request order. Each item is the matched string, or `{"text", "confidence"?, "start"?, "end"?}` when `--include-confidence` or `--include-spans` is set. `--classify` takes `{"<task>": [labels] | {"labels", "prompt"?, "label_descriptions"?, "multi_label"?, "threshold"?}}` and prints `{"<task>": "<label>"}`, or a label list for a multi-label task.

A token-classification checkpoint passed to `mlxcel extract` fails before its weights load, naming `mlxcel tag`.

## Adding a family

1. Add a `TokenModelKind` variant with its `as_str`, `description` and `output_shape` arms in `src/token_classification/loader.rs`.
2. Add a detector function to `DETECTORS` there (config only, returning `Ok(None)` for configs it does not claim).
3. Implement `TokenClassifierModel` (fixed labels; `log_probs` gets one wrapped window and returns one row per id, and `decode_modes()[0]` is the default) or `SpanExtractorModel`, and add the loader arm in `load_token_classifier` or `load_span_extractor`.
4. Add a real-checkpoint test against a fixture produced by the reference implementation, as `src/token_classification/real_checkpoint_tests.rs` does for Rampart with `tests/fixtures/token_classification/generate_rampart_spans.py`.
