# Typed decisions API and CLI

A decision model answers named questions about a state with calibrated probabilities instead of generated text: route a ticket to `billing`, rate a screenshot from `low` to `high`, say whether a transcript asks for a refund. mlxcel serves them through `POST /v1/decisions`, the `--decision-model` server flag, and the offline `mlxcel decide` command. The contract is mlx-vlm's shared decision API ([`decision.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/decision.py), [`decide.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/decide.py), [`server/decisions.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/decisions.py)), including its validation messages, so a client written against mlx-vlm sees the same responses.

**Current status.** One decision family is ported: Decider-2b (`decider2`, below). Clef (#2365) and the LiquidAI d1 models (#2367) each add a loader arm and a scorer. Any other checkpoint is refused by `--decision-model` and `mlxcel decide` with `<path> is not a decision checkpoint`.

## Decision models

| Family | `model_type` | Types | Media | Checkpoint |
|--------|--------------|-------|-------|------------|
| Decider-2b | `decider2` | `choice`, `score`, `bool` (`noul`) | none | [`nativ-community/decider-2b`](https://huggingface.co/nativ-community/decider-2b) |

**Decider-2b** (`src/decision/decider2.rs`) is a Qwen3.5-2B finetune that reads label-token logits at an open answer slot; it never generates, and `mlxcel generate` refuses it with a pointer to `mlxcel decide`. Each question is its own prompt: `Context:\n<state>` (a string state as is, any other JSON value serialized like Python's `json.dumps(..., ensure_ascii=False)` after lists of 8 or more items get an `_index` per item), truncated to `decision_config.max_state_tokens` (32768) tokens, then the question, one `(A) option` line per option and `Answer: (`. Options are labelled `A` to `Z`, then `AA` onwards (255 labels); past 10 options each label token id is spliced in directly. Prompts are batched while `rows * longest <= 8192` tokens, right-padded to a multiple of 64, and the post-norm hidden at each prompt's last token is multiplied by the label tokens' embedding rows. Logits are divided by `decision_config.temperature_by_type` (`choice` 1.164, `score` 1.124, `noul` 1.624) before an f32 softmax. A request is refused with a 400 when one prompt is longer than the backbone's `max_position_embeddings` (262144) or when its prompts together exceed 1048576 tokens; each question (and each isolated score level) repeats the state, so this bounds how much prefill one request can queue on the decision worker. With `isolated_levels` (the published default), a `score` question is asked once per level as a yes/no "Does the proposed answer fit?" row, and level probabilities are the per-level `p(yes)` normalized by their sum.

Model limits (400): `choice needs 2 to 255 options`, `score needs 2 to 10 levels`, `Question needs instructions` (`instructions`, falling back to `question`, must be non-empty). A state longer than `max_state_tokens` is truncated, not refused. Metadata: `confidence` (argmax probability) and `certainty` (`max(0, 1 - entropy / ln(n))`) on `choice` and `score`; `legend`, `level_fit` (per-level `p(yes)`) and `fit_mass` (their sum) on `score`. Probabilities are rounded to 4 places and the score `value` to 2. `usage.input_tokens` counts the prefix every prompt shares once. Upstream's packed mode (`independent=False`) and per-call `isolated` override are Python keyword arguments only and are not ported.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/decisions` | Answer typed questions about a state and optional media. |
| POST | `/decisions` | Alias without the `/v1` prefix. |
| GET | `/v1/models` | Lists the decision model with the `decision` capability next to the chat model. |

Without `--decision-model`, both routes answer `501 Not Implemented` with a message naming the flag. A decision model is always a side model: it is never served through `-m`, as upstream's `--decision-model` is never the chat model either.

## Request

```json
{
  "model": "optional id",
  "state": "string, any JSON value, or null",
  "questions": {
    "topic": {"type": "choice", "instructions": "Route the ticket", "criteria": ["billing", "technical", "sales"]},
    "urgent": {"type": "bool", "criteria": {"false": "can wait", "true": "needs a reply today"}},
    "severity": {"type": "score", "criteria": ["minor", "major", "critical"]},
    "tags": {"type": "multi_label", "criteria": {"refund": null, "outage": "service down"}, "threshold": 0.4}
  },
  "images": ["URL, path or data URI"],
  "videos": ["URL or path"],
  "audio": "URL, path or base64 data URI"
}
```

- `model` defaults to the served decision model's id (its directory name). Naming anything else is a 400.
- `images` are fetched and decoded with the same URL, private-address, `--media-path` and payload limits as chat images.
- `videos` go through the same resolver as chat video input, including `MLXCEL_VIDEO_DIR_ALLOWLIST` for local paths. A host without `ffmpeg` on `PATH` gets a 400 naming it before any clip is downloaded, as in chat. Chat sets no per-request video count cap, so none is set here. This field is an mlxcel addition: upstream's route accepts only `images` and `audio`, while its Python `predict` also takes `videos`, which Clef declares.
- `audio` is one clip in any form a chat `input_audio.data` accepts (wav, mp3 or flac bytes), decoded to 16 kHz mono on the blocking pool. A clip that fails to decode is a 400 carrying the decoder's message. A clip longer than 300 seconds once decoded (`MAX_DECISION_AUDIO_SECONDS`, the longest per-clip limit a chat audio family declares) is a 400 reading `audio: audio is <n> seconds long; decision audio is limited to 300 seconds`. The limit is a fixed constant, not an environment variable, and is checked on the native samples before resampling.

### Validation

The rules run in this order. Each failure is `400 invalid_request_error` on the route and a non-zero exit with the same message on the CLI.

| # | Rule | Message |
|---|------|---------|
| 1 | `state` may be `null` or absent only when `images`, `videos` or `audio` is present. | `state is required unless images or audio are given` |
| 2 | The model declares at least one question type. | `This model does not support decision prediction` |
| 3 | Every present medium is in the model's declared media. | `This model does not support image input` (or `video`, `audio`) |
| 4 | `questions` is a non-empty map of name to object. | `At least one named question is required`, `Questions must map names to question dictionaries` |
| 5 | `type` defaults to `choice`; it is one of `choice`, `score`, `bool`, `multi_label` (`noul` is an alias of `bool`) and is declared by the model. | `This model does not support '<type>' decisions` |
| 6 | `choice` and `score` need at least 2 `criteria`, `multi_label` at least 1, as a list or a `label -> description` map (a description may be `null`). Labels are non-empty and unique. `score` criteria are an ordered list. | `<type> needs at least <n> criteria`, `Decision labels must be nonempty strings`, `Decision labels must be unique`, `score criteria must be an ordered list` |
| 7 | `multi_label` `threshold` defaults to 0.5 and is a number in [0, 1]; booleans are refused. | `threshold must be a number between zero and one` |
| 8 | `bool` criteria are absent or a map with optional `false` and `true` descriptions. | `bool criteria must map false and true to descriptions` |

Media is validated before it is fetched, so a refused medium costs no download. Limits a model sets for itself (for example Decider's 2 to 255 choice options) are checked by the model and also surface as 400.

## Response

```json
{
  "model": "decider-2b",
  "answers": {
    "topic": {"type": "choice", "value": "billing", "probabilities": {"billing": 0.91, "technical": 0.06, "sales": 0.03}, "metadata": {"confidence": 0.91}},
    "urgent": {"type": "bool", "value": true, "probability": 0.83},
    "severity": {"type": "score", "value": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "metadata": {"legend": {"0": "minor", "1": "major", "2": "critical"}}},
    "tags": {"type": "multi_label", "value": ["refund"], "scores": {"refund": 0.72, "outage": 0.08}}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0}
}
```

- `choice`: `value` is the argmax label, `probabilities` are keyed by label in request order, and `metadata.confidence` is the argmax probability.
- `score`: `value` is the expected level index, `probabilities` are keyed `"0"` to `"n-1"`, and `metadata.legend` maps each level to its description.
- `bool` (also the answer to a `noul` question): `value` is `probability >= 0.5`, where `probability` is the probability of true.
- `multi_label`: `value` lists the labels at or above `threshold`, and `scores` are independent, so they need not sum to one.
- Answers appear in question order. Model-specific metrics (certainty, level fit, action probability) appear only in `metadata` and are not comparable across models.
- `usage.output_tokens` is always 0: no tokens are generated.

### Errors

| Status | When |
|--------|------|
| 400 | A validation rule, a model-specific limit, a `model` naming another checkpoint, media that cannot be fetched or decoded. |
| 501 | No decision model is loaded. The message names `--decision-model`. |
| 503 | The decision worker queue is full. |
| 504 | The worker did not answer within the request timeout. The in-flight work is not cancelled. |
| 500 | The model failed, or returned a result that does not answer the asked questions in order, with the wrong answer type for a question kind (a `bool` answer for a `choice` question, say), or with a non-finite number. The detail is `Decision prediction failed`; the cause is in the server log. |

## Server flag

| Flag | Environment | Default | Effect |
|------|-------------|---------|--------|
| `--decision-model <path or repo-id>` | `MLXCEL_DECISION_MODEL` | unset | A decision checkpoint served on `/v1/decisions` next to the chat model in `-m`, resolved and auto-downloaded like `-m`. The flag wins over the variable. A load failure, including a checkpoint that is not a decision model, is a startup error. |

The worker shares the `--embedding-queue-depth` (default 8) and `--embedding-request-timeout-secs` (default 120) bounds with the embedding and rerank workers. Both `mlxcel-server` and `mlxcel serve` take the flag. Image-only serving (`--image-model` without `-m`) refuses it.

```bash
mlxcel-server -m models/qwen3.5-4b-4bit --decision-model <decision checkpoint> --port 8080
curl -s localhost:8080/v1/decisions -H 'content-type: application/json' -d '{
  "state": "My invoice was charged twice",
  "questions": {"topic": {"type": "choice", "criteria": ["billing", "technical", "sales"]}}
}'
```

## `mlxcel decide`

```bash
mlxcel decide -m <MODEL_OR_REPO_ID> [--state TEXT | --state-file PATH] \
  [--image PATH_OR_URL]... [--video PATH]... [--audio PATH_OR_URL] \
  (--questions JSON | --questions-file PATH)
```

The command prints the `POST /v1/decisions` response JSON, pretty-printed with non-ASCII text kept as is. `--state` and `--state-file` are mutually exclusive, as are `--questions` and `--questions-file`. The questions are parsed and checked against rule 4, and a call with neither a state nor media is refused, before the model is resolved or loaded. Once the model is loaded, the full validation runs before any medium is read, so media the model does not declare is refused without being read or fetched. Local media paths are read directly; URLs and data URIs go through the server's fetchers and limits. `--models-dir` (or `MLXCEL_MODELS_DIR`) sets the store a repo id resolves into.

## Implementation source map

| Module | Responsibility |
|--------|----------------|
| `src/decision/mod.rs` | `QuestionKind`, `MediaKind`, `Question`, `Criteria`, `DecisionMedia`, `DecisionResult`, `DecisionError`, the `DecisionModel` trait and `predict`. |
| `src/decision/validate.rs` | `validate_questions` (rules 1 to 8) and `validate_question_structure` (rule 4). |
| `src/decision/answer.rs` | `Answer` constructors and the answer JSON. |
| `src/decision/loader.rs` | `DecisionKind`, `detect_decision_kind`, `load_decision_model`. |
| `src/decision/decider2.rs`, `decider2_prompt.rs`, `decider2_rows.rs` | Decider-2b: loading, batched scoring over `Qwen35Model::forward_prefill_hidden`, question rendering and prompt ids, row planning and answers. |
| `src/decision/cli.rs`, `src/decision/media.rs` | `mlxcel decide` input preparation, audio decoding, CLI media loading. |
| `src/server/decision_model.rs` | `DecisionModelProvider` and `DecisionProviderError`. |
| `src/server/decision_worker.rs` | `DecisionWorkerProvider` (bounded queue, timeout, panic boundary) and startup loading. |
| `src/server/routes/decisions.rs` | `create_decisions`, mounted at `/v1/decisions` and `/decisions`. |
| `src/commands/decide.rs` | The `mlxcel decide` subcommand. |

## Adding a decision model

A model port adds three things and touches nothing else; the worker, route, CLI and `/v1/models` listing pick it up unchanged.

1. A `DecisionKind` variant in `src/decision/loader.rs`, with its `as_str` arm.
2. A detection arm in `detect_decision_kind` that recognizes the checkpoint from `config.json`, and a constructor arm in `construct` that builds the scorer from the checkpoint directory.
3. A scorer implementing `DecisionModel`:
   - `decision_types` lists the question kinds it answers (`noul` requests arrive as `QuestionKind::Bool`).
   - `decision_media` lists the media it reads. A request carrying any other medium is refused before `predict` runs, so a model that reads media through extra arguments still declares them here.
   - `predict` receives only questions that passed the shared validation, in request order, and must answer each one in that order. It checks its own limits (option counts, level counts, a rule such as d1-omni's images-or-audio exclusivity) and reports them as `DecisionError::InvalidInput`, which surfaces as 400.
   - Build answers with `Answer::choice`, `Answer::score`, `Answer::bool` and `Answer::multi_label`, which fill `confidence`, `legend` and the bool `value` the same way for every model. Put model-specific metrics in the `metadata` map passed to them. Decider-2b is the exception: its scorer builds `Answer` values directly, because upstream rounds the score value to 2 places, which `Answer::score` does not do.

Tests for a port run the scorer through `decision::predict` with a real checkpoint against a fixture generated with mlx-vlm (`src/decision/decider2_parity_tests.rs` and `tests/fixtures/generate_decider2_fixture.py` are the template), and the port adds a row to the table under Decision models.
