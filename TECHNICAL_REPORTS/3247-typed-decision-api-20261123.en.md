# PR #3247: Typed decision-model API (/v1/decisions, --decision-model, mlxcel decide)

**Date**: 2026-11-23
**Status**: Draft PR, not merged (closes #2343, part of #2375); verified with unit tests and on an M3 Ultra without a real decision checkpoint

## Executive Summary

mlx-vlm added a family of decision models (Decider-2b, Laya, Clef, LiquidAI d1-3B and d1-omni-600M) that answer named questions about a state with calibrated probabilities instead of generated text. They share one API: a CLI, an HTTP route, a server flag, and a per-model declaration of the question types and media each model reads. mlxcel had no surface that returns structured, typed answers, so none of these checkpoints could be served even once their weights load.

This PR builds that surface once. `src/decision` holds the question and answer types, the `DecisionModel` trait, the shared validation (rules 1 to 8 with upstream's messages verbatim), the answer constructors, a kind dispatcher and CLI input preparation. The server gains `POST /v1/decisions` and `/decisions` on a dedicated worker, the `--decision-model` / `MLXCEL_DECISION_MODEL` side-model flag on `mlxcel-server` and `mlxcel serve`, and `mlxcel decide` prints the same JSON offline. The only `DecisionKind` today is a `#[cfg(test)]` stub, so every production load reports `<path> is not a decision checkpoint` until the first model port (#2359, #2365 or #2367) adds a variant, a detection arm and a scorer.

## 1. Change Summary

| Area | Change |
|---|---|
| `src/decision/mod.rs` | `QuestionKind` (`noul` deserializes to `Bool`), `MediaKind`, `Question`, `Criteria`, `DecisionMedia`, `DecisionResult`, `DecisionError`, the `DecisionModel` trait, `predict`, and `check_result`. |
| `src/decision/validate.rs` | `validate_questions` (rules 1 to 8) and `validate_question_structure` (rule 4 alone, used by the CLI before loading). |
| `src/decision/answer.rs` | `Answer::choice`, `score`, `bool`, `multi_label`, which fill `confidence`, `legend` and the bool `value` the same way for every model, plus the answer JSON. |
| `src/decision/loader.rs` | `DecisionKind`, `detect_decision_kind`, `load_decision_model` with HF repo id resolution like `rerank`. |
| `src/decision/stub.rs` | `#[cfg(test)]` deterministic model declaring `choice`, `score`, `bool` and images only. |
| `src/decision/cli.rs`, `media.rs` | `mlxcel decide` input preparation, CLI media loading, 16 kHz mono audio decode with a 300 s duration cap. |
| `src/server/decision_model.rs`, `decision_worker.rs` | `DecisionModelProvider` and `DecisionWorkerProvider`: dedicated MLX thread, bounded queue (503), reply timeout (504), panic boundary with a generic 500. |
| `src/server/routes/decisions.rs` | `create_decisions` at `/v1/decisions` and `/decisions`; 501 naming `--decision-model` when no model is loaded. |
| Server wiring | `cli_input.rs`, `startup.rs`, `state.rs`, `app.rs`, `routes/models.rs` (`decision` capability), `chat_unavailable_message`, `bin/mlx_server.rs`, `commands/serve.rs`. A load failure is fatal at startup. |
| `src/commands/decide.rs`, `src/main.rs` | The `mlxcel decide` subcommand. |
| Docs | New `docs/decisions.md` (contract, errors, flag, CLI, source map, "Adding a decision model"), plus `docs/README.md`, `README.md`, `docs/environment-variables.md`. |

36 files, +4175 / -3 lines against `origin/main`.

## 2. Technical Decisions

### 2.1 A separate subsystem instead of extending rerank

Reranking was the closest existing surface: a side model on a dedicated worker, loaded from a flag, served on its own route. But a reranker returns one `f32` per document, and a decision model returns a named map of typed answers, each with its own shape (an argmax label with per-label probabilities, an expected level index with a legend, a probability of true, a thresholded label set with independent scores). Adding a `RerankerKind` variant would have forced either a score vector that cannot express those answers or a second response type inside the rerank route, and the validation (eight ordered rules with fixed messages, media gating per model) has no counterpart in rerank.

The PR therefore mirrors the rerank layering (core module, loader, provider trait, worker, route, CLI) in a new `decision` subsystem and reuses rerank's mechanisms rather than its types: the worker structure, the 501 and error envelope helpers, and the side-model startup path. Media acquisition reuses the chat fetchers in `src/server/media.rs`, so the URL, private-address, `--media-path` and payload limits match chat requests.

### 2.2 Answers as an ordered `Vec` of pairs

`DecisionResult::answers` is `Vec<(String, Answer)>`. Upstream returns answers in question order and `choice` probabilities in label order. `serde_json` already has `preserve_order` enabled, so incoming maps keep request order, and a `Vec` of pairs keeps that order on the way out without adding an `indexmap` dependency.

### 2.3 Models declare media; the shared gate enforces it

Each model lists the media it reads in `decision_media`. Upstream moved to this in commit be0dbae9 after its gate, which inspected `predict`'s signature, refused images for Clef, which forwards media through `**kwargs`. A declaration list is explicit, so the shared validation can refuse an undeclared medium (rule 3) before any byte is fetched. Model-specific rules, such as d1-omni's images-or-audio exclusivity or Decider's 2 to 255 choice options, stay in the scorer and surface as `DecisionError::InvalidInput`, which is a 400.

### 2.4 `videos` holds `VideoSource`, not `PathBuf`

The issue specified `DecisionMedia::videos: Vec<PathBuf>`. The PR uses `VideoSource` so the server passes the fd-backed handle that the video allowlist resolver opened with `O_NOFOLLOW`, instead of a path that could be reopened and swapped between the check and the read. `videos` itself is an mlxcel addition to the wire contract: upstream's route accepts only `images` and `audio`, while its Python `predict` also takes `videos`, which Clef declares.

### 2.5 Production build has no decision family yet

`DecisionKind` contains only the `#[cfg(test)]` `Stub` variant. The non-test build compiles the dispatcher with no reachable arm and emits no dead-code or unreachable warnings, and `--decision-model` on any real checkpoint fails at startup with `<path> is not a decision checkpoint`. This keeps the API surface reviewable on its own while making the absence of a model explicit rather than silent.

## 3. Wire Contract and Validation Order

Request (`POST /v1/decisions`; `mlxcel decide` builds the same value):

```json
{"model": "optional id", "state": "string | any JSON value | null", "questions": {"<name>": {"type": "choice", "instructions": "...", "criteria": ["a", "b"]}}, "images": ["url, path or data URI"], "videos": ["url or path"], "audio": "url, path or base64 data URI"}
```

The rules run in this order, each failure being `400 invalid_request_error` on the route and a non-zero exit with the same message on the CLI:

1. `state` may be `null` or absent only when media is present: `state is required unless images or audio are given`.
2. The model declares at least one question type: `This model does not support decision prediction`.
3. Every present medium is declared: `This model does not support image input` (or `video`, `audio`).
4. `questions` is a non-empty map of name to object.
5. `type` defaults to `choice`, is one of `choice`, `score`, `bool`, `multi_label` (`noul` aliases `bool`), and is declared by the model.
6. Criteria counts (2 for `choice` and `score`, 1 for `multi_label`), non-empty unique labels, `score` criteria as an ordered list.
7. `multi_label` `threshold` is a number in [0, 1], booleans refused.
8. `bool` criteria are absent or a `false` / `true` description map.

On the route, the request is parsed, the provider presence (501) and `model` id are checked, and `validate_questions` runs against the worker's declared types and media with only the presence of each medium known. Media is acquired only after validation passes: images under `validate_image_count` and the chat image limits, then the ffmpeg check, then video resolution through the allowlist, then audio fetch and decode on the blocking pool. The worker runs `predict`, and `check_result` verifies the answer names, order, kinds and finiteness before the response is built. A failure there is a logged internal error returned as `500 Decision prediction failed`.

Response: `{"model", "answers": {name: answer}, "usage": {"input_tokens", "output_tokens": 0}}`, with the per-kind answer shapes in `docs/decisions.md`.

## 4. Extension Points for the Model Ports

Decider-2b (#2359), Clef (#2365) and the LiquidAI d1 models (#2367) each add three things and touch nothing else:

1. A `DecisionKind` variant in `src/decision/loader.rs` with its `as_str` arm.
2. A detection arm in `detect_decision_kind` that recognizes the checkpoint from `config.json`, and a constructor arm in `construct`.
3. A scorer implementing `DecisionModel`: `decision_types`, `decision_media`, and `predict`, which receives only validated questions in request order, checks its own limits, and builds answers with the `Answer` constructors, putting model-specific metrics in `metadata`.

```rust
pub trait DecisionModel: Send {
    fn decision_types(&self) -> &[QuestionKind];
    fn decision_media(&self) -> &[MediaKind];
    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError>;
}
```

The worker, route, CLI and `/v1/models` listing pick up a new variant unchanged. `check_result` is the safety net those ports rely on: a scorer that skips a question, reorders answers, returns a `bool` answer for a `choice` question or produces a NaN yields a 500 instead of a malformed response.

## 5. Hardening Added During Review

The implementation review and the security and performance review found no CRITICAL or HIGH items. Two follow-up commits addressed the MEDIUM and selected LOW findings.

| Commit | Finding | Fix |
|---|---|---|
| `0ce2c1c6` | MEDIUM: `mlxcel decide` read and decoded every `--image`, `--video` and `--audio` before validation ran inside `predict`, so undeclared media was read or fetched, and an undecodable clip reported a decode error instead of `This model does not support audio input`. | The CLI runs `validate_questions` against the loaded model's declarations, using argument presence, before `load_cli_media`. This matches the route and upstream `decide.py`, which hands paths to `predict` unread. |
| `0ce2c1c6` | LOW: the `side_model_id_for_path` doc comment did not list `--decision-model`. | Doc comment updated; the CLI ordering is stated in `docs/decisions.md`. |
| `3c62601e` | MEDIUM: route audio was bounded only by the 500 MiB encoded cap, so a long low-bitrate clip could expand to hours of samples before the bounded queue. | `MAX_DECISION_AUDIO_SECONDS = 300` (the longest per-clip limit a chat audio family declares, `AudioFamilyPolicy::inkling`), checked on the native samples before resampling. The 400 reads `audio: audio is <n> seconds long; decision audio is limited to 300 seconds`. Fixed constant, no new environment variable. |
| `3c62601e` | LOW: videos were downloaded before an `ffmpeg` availability check. | `check_video_support` runs before any clip is resolved, with the chat message. Chat has no per-request video count cap, so none was added. |
| `3c62601e` | LOW: `check_result` checked answer names and finiteness but not answer kind. | Each answer's kind is compared with its question's kind (`bool` for `noul` and `bool`, and so on). |

## 6. Verification

Unit tests (`cargo test --profile test-fast --features metal,accelerate`):

- `--lib decision`: 45 passed at review time, covering one test per validation rule with its exact message, answer construction, the loader's refusal of non-decision checkpoints, and the worker queue (503) and timeout (504). `3c62601e` added tests for the audio duration cap (at and over the limit), the ffmpeg gate, and the answer-kind check.
- Route tests with the stub provider: 200 with `model`, `answers` and `usage`; every 400; 501 naming the flag; 503; 504; 500; label order in `choice` probabilities; `noul` returned as `bool`; the media gate (audio refused, images reaching `predict`); over-long audio.
- `--bin mlxcel decide`: 7 passed (malformed questions, no state or media, flag conflicts).
- `coverage_tests`, `server::cli_input`, `server::routes::models`, `server::routes::availability`.
- `cargo clippy --lib --tests` and `--bins` with `-D warnings`, `cargo fmt --check`, and the `scripts/ci` contract checks.

Real hardware, M3 Ultra, with `models/qwen3.5-4b-4bit` as the chat model:

| Check | Result |
|---|---|
| Server without `--decision-model`, `POST /v1/decisions` and `POST /decisions` | 501, message names `--decision-model` |
| `--decision-model models/qwen3.5-4b-4bit` (a chat checkpoint) | Startup exits 1 with `is not a decision checkpoint` |
| `mlxcel decide` with malformed `--questions` | Fails before the model is resolved or loaded |
| `mlxcel decide` with neither state nor media | Fails before the model is resolved or loaded |
| `mlxcel decide -m models/qwen3.5-4b-4bit` with valid input | Reports `is not a decision checkpoint` |

A 200 response from a real checkpoint was not exercised, because no decision family is ported yet.

## 7. Unverified and Open Items

- **No real decision checkpoint.** The success path is tested only against the stub. The first model sub-issue to land (#2359, #2365 or #2367) must validate `POST /v1/decisions` and `mlxcel decide` end to end with real weights, which is also the last unchecked acceptance criterion of #2343.
- **Audio decode still happens before the queue.** The 300 s cap rejects a long clip, but it is checked after the native decode, so the native sample buffer of an oversized clip is still allocated once, and concurrent requests decode concurrently on the blocking pool. The first audio decision model (#2367 / #2368) should decode under a decision `AudioFamilyPolicy` or on the worker thread.
- **Remaining LOW items from the security review**, none reachable today because no shipped model declares audio or video:
  - Media acquisition (image fetch and decode, video download up to 1 GiB each, audio) happens before the queue check, so a full queue does not bound acquisition work. This matches `/v1/rerank` and chat.
  - `videos` has no per-request count cap, matching chat.
  - Image decode in `fetch_labeled_image` runs on a Tokio worker, as in rerank and embeddings.
  - The number of questions and labels is bounded only by the JSON body limit; validation is linear, matching rerank's document list.
- **Remaining LOW item from the implementation review.** `images: []` sent to a model without image support passes, while upstream `predict` refuses a present non-null list. Upstream's route also drops an empty list, so the wire behavior matches.
- **Docs not updated.** `CLAUDE.md` is gitignored and absent from the repository, and the mkdocs `docs/en` / `docs/ko` trees do not exist, so there was no reference row, nav entry or Korean mirror to add.

## 8. Learning Points

- When a new output shape differs from an existing subsystem's, copy the layering and reuse the mechanisms (worker, envelopes, fetchers), not the types. The layering is what keeps operations uniform; forcing a new shape into an old type spreads special cases through it.
- Validate against declarations before acquiring inputs. Knowing only which media are present is enough to refuse an undeclared medium, so a refused request costs no download or decode. The CLI bug fixed in `0ce2c1c6` came from running the same validation after input loading instead.
- A size cap on encoded bytes does not bound decoded size. Compressed audio needs a duration cap on decoded samples.
- A shared result check (`check_result`) at the boundary between model code and response code turns scorer bugs into logged 500s rather than malformed responses, which matters most when several model ports will be written against the same trait.

## Related Issues

- #2343: this issue. #2375: the parent epic.
- #2359 (Decider-2b), #2365 (Clef), #2367 / #2368 (LiquidAI d1): model ports that build on this API. #2363 and #3218 are also blocked by #2343.
- Upstream: Blaizzy/mlx-vlm#2396 (shared API and Decider), Blaizzy/mlx-vlm#2402 (CLI and HTTP route), commit be0dbae9 (media declaration).
