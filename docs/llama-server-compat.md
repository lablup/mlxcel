# llama-server compatibility boundary (b10621)

`mlxcel-server` accepts many `llama-server` flags, `LLAMA_ARG_*` / `LLAMA_API_KEY` environment variables, routes, and native request fields, but "accepted" is not "identical behavior". The verified compatibility boundary against one frozen upstream reference is maintained as a machine-readable manifest in this repository:

```
compat/llama-server/b10621/
```

The manifest is the authority. This page explains how to read it, how it is enforced, and how to regenerate it. Epic #1431 tracks the work of widening the boundary; issue #1443 introduced the manifest.

## Frozen reference

| Item | Value |
|---|---|
| Release | [`b10621`](https://github.com/ggml-org/llama.cpp/releases/tag/b10621) (published 2026-08-25) |
| Commit | [`c1d0e7a004015f23bc0233470b747b596f29b264`](https://github.com/ggml-org/llama.cpp/tree/c1d0e7a004015f23bc0233470b747b596f29b264) |
| Archive | `llama-b10621-bin-macos-arm64.tar.gz` |
| SHA-256 | `429c8270608600188035e5e92f7d78dffb7900904fe7dd7e6a84f48068cd13cf` |

The distributed binary exposes 249 help entries carrying 323 distinct long-option spellings and 134 `LLAMA_*` environment variables (plus `HF_TOKEN` and `MTMD_BACKEND_DEVICE`), 53 registered HTTP method/path pairs, and 74 native `/completion` request fields. Those counts are pinned in `pin.json` and re-verified by CI on every change.

## Reading the manifest

`pin.json` records the frozen reference, the inventory counts, and the shard map (`shards[name].owners`, the set of implementation issue numbers allowed to own entries in that shard; see [Sharding](#sharding)). Every other `*.json` file is an area shard holding entries of three kinds: `option` (one per help entry, with every accepted spelling, the environment binding, and the default), `route` (method/path), and `native_request_field` (native `/completion` schema fields with aliases). `pin.json` and every shard carry `schema_version: 3`.

Every entry carries exactly one compatibility-policy state, per the policy defined in epic #1431:

| State | Meaning |
|---|---|
| `supported` | Spelling, value domain, default, precedence, and observable behavior match, and the entry's `divergence` list is empty. |
| `aliased` | A different mlxcel spelling, route, or request field provides equivalent behavior, with a tested translation. The claim names the mlxcel identity, which must differ from the b10621 one; when mlxcel answers the b10621 name itself the entry is `supported`, not `aliased`. |
| `not_applicable` | No MLX/CUDA equivalent; mlxcel rejects the option with an actionable diagnostic or accepts only semantically inert forms, with a test and documentation. |
| `deferred` | Not yet true; a linked implementation issue owns it. Accepted-but-ignored flags are `deferred`, never `supported`. |

Every non-`supported` entry carries a linked issue, plus `notes` and a test id where they apply. A flag mlxcel parses but does not act on is `deferred` with its acceptance pinned in the `mlxcel` claim block, never `supported`.

### `divergence`

Every entry carries a `divergence` list: short strings, each naming one externally observable way mlxcel differs from b10621 for that entry. **A non-empty `divergence` forbids `supported`**, and the validator makes that a hard error naming the three honest alternatives (`aliased`, `not_applicable`, `deferred`) and asking for the owning issue.

The field exists because prose could not be the gate. `notes` had been carrying divergences like the semantic collisions on `--timeout`, `--models-dir`, `--cache-type-k/v` and `POST /completions`, the inverted DRY disable sentinel on `--dry-penalty-last-n`, and the penalty-window drift on `--repeat-penalty` / `--frequency-penalty` / `--presence-penalty`. That works while the state is already honest, but it cannot stop the opposite case: twenty-two entries once read `supported` while diverging observably, two of them under a `notes` field that opened with "BEHAVIOR DRIFT", and the CI gate locked the false claim in. Every remaining sub-issue of epic #1431 will flip more entries into `supported`, so the rule has to be machine-readable. `notes` keeps the context and the reasoning; `divergence` carries the specific observable difference and its owning issue.

Both fields are entry-level, alongside `state`, `issue`, `test` and `mlxcel`. An entry carries exactly its kind's fact keys plus those six policy keys, in that order, and nothing else: a misspelled `divergance` (or a `divergence` misfiled inside the `mlxcel` block) fails the gate instead of quietly recording nothing.

### The `mlxcel` claim block

Entries may also carry an `mlxcel` claim block describing what the current binaries already accept: `accepted_spellings`, `accepted_on_one_binary_only`, `env`, `env_binding` (`clap` or a tested `runtime` bridge, with `env_test`), `defaults`, `hidden`, `route`, `field`. That is a closed key set; the validator rejects any other key in the block, so a typo'd key fails the gate instead of silently recording a claim nothing checks. Claims are verified by CI regardless of state, so partial support cannot silently regress. A `deferred` entry keeps its claim: mlxcel does mount `POST /rerank` and does accept `--metrics`, and CI holds both to that; the entry is `deferred` because the behavior behind the accepted surface still differs.

## Migrating: what changed to reach b10621

Widening the boundary moves behavior that mlxcel already had. Each item below is a change an existing deployment can notice.

### `--timeout` is the HTTP socket timeout (#1432)

`--timeout` / `LLAMA_ARG_TIMEOUT` is the HTTP socket read/write timeout, default 3600 seconds, as it is in b10621. It used to be a 600-second decode watchdog. That watchdog kept its behavior and its default under the mlxcel-native `--decode-timeout` / `MLXCEL_DECODE_TIMEOUT`, and setting `--timeout` without it logs a migration warning naming both.

The default CORS response follows b10621 too: `--cors-origins *` with credentials enabled echoes the request `Origin` rather than emitting a literal `*`, the method and header lists are sent on `OPTIONS` preflights only, `Access-Control-Expose-Headers` is no longer sent, and an `OPTIONS` to any path is answered before authentication and before routing. The default SSE ping interval moved from 15 to 30 seconds, b10621's `--sse-ping-interval` default. A Unix domain socket is selected by a `--host` ending in `.sock`; `--port 0` with an ordinary host now binds an ephemeral TCP port, and the old `--port 0` socket spelling still works with a deprecation warning.

### `/completions` and `/embeddings` are native routes, not OpenAI aliases (#1441)

b10621 sends `/completion` and `/completions` to one handler and `/v1/completions` to a different one, and does the same for `/embedding` and `/embeddings` against `/v1/embeddings`. mlxcel answered the OpenAI shape on all of them, so a `llama-server` client reading the native schema got an object it could not parse.

| Path | Before | Now |
|---|---|---|
| `POST /completion` | native shape, partial | native shape, b10621 key set |
| `POST /completions` | **OpenAI** shape | **native** shape |
| `POST /v1/completions` | OpenAI shape | OpenAI shape (unchanged) |
| `POST /embedding` | not mounted | **native** shape |
| `POST /embeddings` | **OpenAI** shape | **native** shape |
| `POST /v1/embeddings` | OpenAI shape | OpenAI shape (unchanged) |

The native completion object carries `index`, `content`, `tokens`, `id_slot`, `stop`, `model`, `tokens_predicted`, `tokens_evaluated`, `generation_settings`, `prompt`, `has_new_line`, `truncated`, `stop_type`, `stopping_word`, `tokens_cached` and `timings` (whose `cache_n` leads). Its streaming form ends with that same object instead of a `[DONE]` sentinel, because b10621's native stream has none. The native embedding response is a bare JSON array of `{index, embedding}` whose `embedding` is an array of arrays.

**A client that was pointing at `/completions` or `/embeddings` and parsing the OpenAI shape must move to `/v1/completions` or `/v1/embeddings`.** Those two paths are unchanged and stay OpenAI compatible.

Native request fields mlxcel has no equivalent for are now refused with a 400 naming the field and the alternative, instead of being accepted and ignored: `n_cmpl` (and its `n` alias) above 1, `n_indent`, `t_max_predict_ms`, `return_progress`, `verbose` and `return_tokens`. Each is still accepted at its inert value, so a client that sends the whole schema at its defaults is not turned away.

Four more native fields moved from ignored to honored on the same routes, each checked against the pinned binary rather than against the schema's descriptions, which are wrong about two of them:

- `response_fields` projects the finished response down to the listed paths. A path containing `/` is walked through the object and stored under the **whole path string**, so `generation_settings/n_predict` comes back as a root key literally named `generation_settings/n_predict`; the schema description says the value is "unnested" to the root, and the binary does not do that. A missing path, a path that walks into a non-object, and an empty path are all omitted without an error, and a value that is not an array of strings returns the whole object rather than failing the request. The projection applies to the non-streaming body and to the final streaming frame, never to the per-token frames.
- `stream_options` and its `stream_options.include_usage` subfield are declared and type-checked. On the native route the value is inert on both sides, because the native final frame always carries the counts and the timing block; a non-boolean `include_usage` is refused with upstream's own wording, and a non-object `stream_options` is tolerated and ignored.
- `timings_per_token` frames now carry the real prefill figures. `cache_n`, `prompt_n`, `prompt_ms` and `tokens_evaluated` used to be zero on every frame until the last one; they now report the same values the final frame reports, from the first frame onward.
- `n_predict` accepts upstream's whole value domain instead of serde's. `-1` means "as many as the context allows" and is clamped to the effective context window instead of being refused with a 422, `0` evaluates the prompt (both implementations then emit exactly one token, which is what the binary does despite its description saying none), and a value outside `[-1, 2147483647]` is refused with upstream's wording.

Two derivations inside `timings` were corrected against the binary at the same time, so a client that graphs them sees different numbers than before: `prompt_n` is now `tokens_evaluated - cache_n` rather than the whole prompt, and `predicted_per_token_ms` / `predicted_per_second` divide by `predicted_n - 1` rather than `predicted_n`, because `predicted_ms` is measured from the first token.

### `stop` truncates the output instead of being ignored (#1466)

`stop` is honored on the shipping MLX serving path, on every completion route and in both the streaming and non-streaming forms. Generation ends at the first match, the matched string is excluded from the emitted text, and the native response reports `stop_type: "word"` with the match in `stopping_word`. It was previously parsed and carried into the scheduler and then never read, so a request that asked to stop on a string ran to `n_predict` with the string in its output.

The match runs on decoded text as it is produced, so a stop string that straddles token boundaries is caught, and a decoded piece whose tail could still become one is withheld from the stream until the next piece resolves it. The concatenation of the streamed chunks is therefore always exactly the non-streaming `content`: a client cannot observe text that a later token turns into a match. On a tie, the earliest match in the text wins, and among equal positions the first entry in the request's `stop` list.

The match is against the raw generated stream, as upstream does, not against a reasoning-stripped view of it. On a reasoning model a stop string that appears inside a `<think>` block therefore ends the request there. `/v1/messages` reports that as Anthropic's `stop_reason: "stop_sequence"` with the matched string in `stop_sequence`.

### Tokenization, templating and infill utilities (#1442)

Six b10621 routes that inspect a prompt rather than generate from one are now served, and `/tokenize` and `/detokenize` grew the rest of their schema.

`/tokenize` honors all four b10621 switches. `content` accepts the mixed shape, a string or an array whose elements are strings or already-tokenized ids, and may be absent, in which case the answer is an empty token list rather than an error. `add_special` defaults to `false`. `parse_special` defaults to `true` and, set to `false`, stops a special-token spelling written into the text from being recognized as that token. `with_pieces` switches the response from a flat id array to one `{id, piece}` object per token, where `piece` is a JSON string when the token's bytes are valid UTF-8 and an array of byte values when they are not. That second form is not an edge case: a byte-level BPE token routinely carries part of a multi-byte character, and the array is how a client reassembles text across it. Concatenating every piece's bytes reproduces the input exactly.

`/detokenize` renders special tokens rather than skipping them, as upstream's `tokens_to_str` does, and answers `{"content": ""}` for an absent or empty `tokens` instead of failing. Bytes that cannot form a character come back as U+FFFD, which is also what upstream emits, because its JSON writer runs with the replacing error handler.

`/apply-template` renders a chat-completions body through the loaded chat template and answers `{"prompt": "..."}` without generating. `/chat/completions/input_tokens`, `/v1/chat/completions/input_tokens`, `/responses/input_tokens` and `/v1/responses/input_tokens` answer `{"input_tokens": N}`, the key `/v1/messages/count_tokens` already used. All five run the same render and the same encode the generation path runs, so the number is the tokens that request would really have been prefilled with rather than a separate estimate that can drift from it. The Responses pair goes through the same translator `/v1/responses` uses, so `previous_response_id` and `conversation` rehydration are counted too.

`/infill` is served on models whose vocabulary declares fill-in-the-middle markers, and refused with upstream's own wording naming the missing ones otherwise. The marker spellings are the list `llama-vocab.cpp` scans for, so Qwen, StarCoder, Granite, DeepSeek-Coder and CodeLlama checkpoints resolve without configuration. `--spm-infill` moves the suffix block ahead of the prefix block while the middle marker stays last, which is the swap upstream makes; a model trained on one ordering answers a prompt in the other fluently and wrongly, so the flag is a correctness switch and not a preference.

Two `/infill` differences remain, and its manifest entry stays `deferred` for them rather than claiming support:

- Text in `input_prefix`, `input_suffix`, `input_extra[].text`, `input_extra[].filename` or `prompt` that contains one of the model's own FIM marker spellings is **refused** with a diagnostic naming the field and the marker. b10621 tokenizes those fields with special-token parsing off, so the marker stays literal text there. mlxcel's generation entry point takes a prompt string that is re-tokenized with parsing on, and a string cannot express "these characters are not that token", so carrying the text through would let a file the client did not write restructure the FIM prompt and return a fluent wrong completion. Failing loudly is the smaller divergence.
- The FIM context is not truncated to the prefill batch. b10621 keeps only the last `3*(n_batch/4)` prefix tokens and the first `(n_batch/4) - (2 + prompt)` suffix tokens; mlxcel serves the whole prefix and suffix, consistent with its policy everywhere else of clamping an over-long request rather than dropping prompt text.

### Embedding and reranking mode, pooling and normalization (#1452)

b10621's `--embedding` / `--embeddings`, `--rerank` / `--reranking`, `--pooling` and `--embd-normalize` are accepted on both server binaries, with `LLAMA_ARG_EMBEDDINGS`, `LLAMA_ARG_RERANKING` and `LLAMA_ARG_POOLING` bound. The two mode flags do the same two things upstream's do, and one more that upstream never needs.

They **restrict**: generation routes answer `501` naming the flag, so a client sees the same "this server does not generate" it would see from `llama-server --embeddings`. They **select**: a mlxcel server runs a dedicated embedding or reranking worker rather than reusing one set of weights, so `--embeddings` requires an embedding checkpoint to resolve from `-m` or `--embedding-model`, and `--reranking` requires a reranker from `-m` or `--reranker-model`. A command line that asks for a mode nothing can serve fails at startup naming the flag, instead of booting a server whose only route answers 501 forever. That startup refusal is the one behavior upstream does not have, because its `--embeddings` embeds with whatever `-m` happens to be and mlxcel has no path that produces embeddings from a causal chat checkpoint.

`--pooling` maps `mean`, `cls` and `last` onto mlxcel's own kernels, and the flag outranks both the checkpoint's `1_Pooling/config.json` and the `MLXCEL_EMBEDDING_POOLING` variable. The other two values are not pooling kernels at all: `rank` is how b10621 puts a model on its reranking path, so it is accepted as a synonym for `--reranking`, and `none` asks for one vector per token, which mlxcel cannot produce because every embedding family pools inside its own forward pass before the engine sees the output. `none` is refused at startup with that reason. mlxcel's own `max` pooling has no b10621 spelling and stays reachable through `MLXCEL_EMBEDDING_POOLING` and the checkpoint config.

`--embd-normalize` implements the whole b10621 domain with upstream's arithmetic: `-1` none, `0` the max-absolute rescale into the signed int16 range, `1` taxicab, `2` euclidean, and any value above 2 that p-norm. A zero vector normalizes to zeros rather than NaN, which is upstream's `norm = sum > 0 ? 1/sum : 0` rule and matters here because the embedding route refuses a non-finite response with a 500. `2` delegates to the L2 kernel every mlxcel embedding response already used, so the default path produces exactly the numbers it produced before. The value is also readable per request as `embd_normalize` on `/embedding`, `/embeddings` and `/v1/embeddings`, as upstream reads it, and `mlxcel embed` takes the same `--pooling` and `--embd-normalize` so the offline and server surfaces resolve identically.

One default differs: with the flag unset mlxcel follows the checkpoint's own `normalize` flag from `config.json`, which is euclidean for every checkpoint that does not set it and none for one that sets it to `false`. b10621 always defaults to `2`, because GGUF carries no such metadata.

#### Rerank envelope

The four rerank routes now answer b10621's shapes rather than only Cohere's.

| Request spells the document list | Response |
|---|---|
| `documents` (Jina / Cohere) | `{"model", "object": "list", "usage", "results": [{"index", "relevance_score"}]}`, with `document` echoed under `return_documents` |
| `texts` (b10621 / TEI) | a bare array of `{"index", "score"}`, with `text` echoed under `return_text` |

`object` was previously missing from the Jina envelope and the TEI shape was not served at all; both were recorded divergences on all four routes. A body carrying both list spellings stays on the Jina envelope, and the three shape errors (`"query" must be provided`, `"query" must be a string`, `"documents" must be a non-empty string array`) use upstream's wording. The `/embedding` family answers `"input" or "content" must be provided` for the same reason.

#### Readiness in a restricted mode

A server started with `--embeddings` or `--reranking` has no chat worker to be "loaded", and `/health` answered `503 {"status": "loading model"}` for the life of the process, which would make the mode unusable behind any container probe. In those two modes `/health` now reports on the worker the mode selected instead. The change is deliberately scoped to the flags: a server whose `-m` is an embedding checkpoint but that was started without them still reports on the chat provider, so nothing that exists today changes behavior. That wider case, along with the queue-full and loading-body drift, is recorded on `GET /health`'s manifest entry and belongs to #1440.

#### Resolved capability on `/props` and `/v1/models`

`/props` gains a `capabilities` block reporting what the server resolved rather than what was passed: whether generation is on, which mode flag restricted it, and, for each loaded side model, its id plus the pooling and `embd_normalize` really in force. A `--pooling` value the checkpoint overrode is therefore visible. `/v1/models` labels each entry with a `capabilities` array (`completion`, `embedding`, `rerank`), which is what a client needs to tell which id to send where; the field is omitted when empty, so a plain generation deployment keeps exactly the OpenAI object shape.

### Speculative decoding surface (#1433)

Every b10621 speculative option parses on both server binaries. `--spec-draft-model` (with `-md`, `--model-draft`, and the mlxcel `--draft-model` spelling) selects the MTP / DFlash draft checkpoint, and `--spec-draft-n-max` (with the removed `--draft` / `--draft-n` / `--draft-max` spellings kept alive as aliases) sets the draft-token cap; the cap's canonical variable is `LLAMA_ARG_SPEC_DRAFT_N_MAX`, with the removed `LLAMA_ARG_DRAFT_MAX` honored as a fallback. `--spec-type` translates what mlxcel can run: `none` disables speculation exactly as b10621 does, `draft-mtp` / `draft-dflash` map onto `--draft-kind`, and every other subsystem (the n-gram modes included) fails startup with a diagnostic. The remaining `--spec-draft-*` family is a hidden compatibility surface classified inert-or-reject like the GGML group: inert values (upstream defaults, full-offload `--spec-draft-ngl` spellings, `f16` draft cache types, any n-gram tuning value while no n-gram selector can be chosen) are accepted, and every value that would change behavior mlxcel cannot reproduce fails startup before the model load. The resolved configuration is reported in `/props` under `speculative` (draft model basename only, kind override, `n_max`). The native `/completion` `speculative.*` dotted fields are accepted and inert, matching b10621, whose schema block for them is compiled out.

### Control vectors are rejected, not ignored (#1449)

b10621's `--control-vector` and `--control-vector-scaled` load activation-steering vectors and add them to the model's residual stream. mlxcel has no control-vector application path, and accepting a vector without applying it would silently change what a deployment believes it is serving, so both flags parse (hidden) and any configured vector fails startup before the model load with a diagnostic naming logit-level steering (`--lang-bias`, per-request `logit_bias`) as the nearest mlxcel feature; a zero scale is still a configured vector set and is rejected as a whole rather than partially applied. `--control-vector-layer-range START END` alone is accepted as inert, because without vectors it configures nothing, exactly as upstream. Since only the no-vector configuration can ever serve, prompt/KV cache reuse across control-vector configurations is impossible by construction.

## Sharding

The manifest is sharded by area so that the concurrent implementation chains of epic #1431 edit disjoint files. Ownership is machine-readable, not prose: `pin.json`'s `shards` map records, per shard, the set of implementation issue numbers allowed to own entries in it (`shards["authentication"].owners == [1437]`, for example), and `scripts/ci/check_llama_compat_manifest.py` fails an entry whose `issue` is not a member of its own shard's owner set. That is what stops two concurrent chains from editing the same file: the file, not just the reviewer, rejects the second chain's entry.

| Shard | Chain | Owning issues |
|---|---|---|
| `transport-tls-cors.json` | A | #1432 |
| `authentication.json` | A | #1437 |
| `routes.json` | A | #1438, #1440, #1441, #1442, #1452, #1466, #1477 |
| `embeddings-and-rerank.json` | A | #1452 |
| `observability-and-slots.json` | A | #1440 |
| `router-models.json` | A | #1438 |
| `lora-adapters.json` | A | #1439 |
| `model-source.json` | B | #1434, #1438 |
| `ggml-runtime.json` | B | #1445 |
| `chat-templates.json` | B | #1447, #1470 |
| `multimodal-and-audio.json` | B | #1451, #1446 |
| `ui-tools-mcp-gcp.json` | B | #1435, #1456 |
| `streams-and-realtime.json` | B | #1444 |
| `runtime-and-context.json` | C | #1449, #1450, #1453, #1472, #1473 |
| `sampling-and-grammar.json` | C | #1436, #1377, #1466, #1485 |
| `speculative.json` | C | #1433 |
| `logging-and-presets.json` | C | #1448 |

Routes and native request fields live in the shard of the issue that owns them (for example the audio-transcription routes sit in `multimodal-and-audio.json` and the native sampling fields in `sampling-and-grammar.json`), so a chain never has to edit another chain's file.

## Enforcement

Three gates hold the manifest and the binaries together; all three run in CI and in `make verify`:

1. `scripts/ci/check_llama_compat_manifest.py` (the `llama-compat manifest` CI job, `make verify-llama-compat`): offline structural validation of counts, states, issue links, shard ownership, the entry key allowlist, the `mlxcel` claim-key allowlist, the `divergence` rule, test ids, and canonical serialization. Every entry's `issue` must belong to its own shard's `pin.json` owner set, an `mlxcel` block may only use its nine recognized keys, and a `supported` entry may not record a divergence. The CI job adds `--check-issues-open`, so a `deferred` entry pointing at a closed issue fails. `scripts/ci/check_llama_compat_manifest_test.sh` is that validator's own negative coverage: it mutates a throwaway copy of the manifest (passed via `--manifest-dir`) and asserts each rule actually rejects it, so a rule cannot degrade into one that only ever passes.
2. `tests/llama_compat_manifest.rs`: verifies every option claim against the real clap surfaces of both `mlxcel serve` and `mlxcel-server`, hidden compatibility arguments included, via the hidden `--dump-flag-surface` machine interface, and the other direction too: an option entry carrying no claim must be accepted by neither binary, so adding a b10621 flag without flipping its entry fails. It also asserts that the sentinel itself never renders in `--help`. It contains an archive-gated full-inventory conformance test as well: set `MLXCEL_LLAMA_B10621_DIR` to the extracted official archive directory to re-derive the option inventory from the real `llama-server --help` and compare it exactly (CI skips this; it never downloads the archive).
3. `src/server/llama_compat_tests.rs`: verifies route claims against the real router and native-field claims against `NativeCompletionRequest`, in both directions, and restates the `divergence` rule so a `cargo test` run cannot pass a manifest `make verify` would reject. Mounting a b10621 route or accepting a b10621 field without flipping its manifest entry fails, which is what turns silent drift into a reviewable diff. An `aliased` claim is checked both ways as well: the mlxcel identity must resolve and the b10621 identity must not, so an alias cannot be mislabelled as full support.

## Model sources

`mlxcel-server` and `mlxcel serve` accept b10621's model-source vocabulary, but mlxcel is an MLX runtime: it loads a SafeTensors checkpoint directory, not a GGUF file. Every flag whose value can only name a GGUF artifact is refused at startup, before a single weight is read, with a diagnostic naming the value, the reason, and a concrete replacement. The classifications below are the `model-source.json` shard.

| Flag | Environment | State | Behavior |
|---|---|---|---|
| `-m` / `--model` | `LLAMA_ARG_MODEL` | `aliased` | Takes an MLX checkpoint directory, a HuggingFace `owner/name` repo id, or a bare name expanded against `MLXCEL_DEFAULT_ORG`. A `.gguf` path, a `-NNNNN-of-NNNNN.gguf` split shard, and any URL are refused. |
| `--hf-repo` | `LLAMA_ARG_HF_REPO` | `aliased` | Resolved exactly like the same value passed to `-m`. Wins over `-m` when both are given, matching llama-server; the superseded value is logged. A `:<quant>` suffix is refused. |
| `--hf-token` | `HF_TOKEN` | `aliased` | Authenticates the snapshot download. Outranks the environment; never rendered in `--help`, logged, or written to disk. |
| `--offline` | `LLAMA_ARG_OFFLINE` | `supported` | Forces use of the caches and forbids every download, process-wide as in b10621: `-m`, `--hf-repo`, `--embedding-model`, `--reranker-model`, `mlxcel download`, the moondream starmie tokenizer fetch, and request-path media URLs all refuse. |
| `-a` / `--alias` | `LLAMA_ARG_ALIAS` | `aliased` | Comma-separated list, as in b10621, which holds the aliases in a `std::set` and serves its first element. mlxcel sorts identically, so `--alias zebra,apple` serves `apple` on both. The rest are recorded (`/v1/models` does not yet report them, tracked by #1438). |
| `--warmup` / `--no-warmup` | — | `supported` | `--no-warmup` genuinely skips the startup warmup pass. |
| `--hf-file` | `LLAMA_ARG_HF_FILE` | `not_applicable` | Refused: selects one GGUF file inside a repository; MLX loads a whole snapshot. |
| `--model-url` | `LLAMA_ARG_MODEL_URL` | `not_applicable` | Refused: mlxcel resolves by repository identifier, not by URL. A HuggingFace URL is translated into the `--hf-repo` value to use. |
| `--docker-repo` | `LLAMA_ARG_DOCKER_REPO` | `not_applicable` | Refused: Docker Hub model repositories distribute GGUF. |

The three always-refused flags are hidden from `--help`, so the operator-facing surface never implies a GGUF backend; they still parse, so a llama-server command line reaches the diagnostic instead of a clap "unexpected argument" error.

llama.cpp writes these options with a single dash and several letters (`-hf`, `-hfr`, `-hff`, `-hft`, `-mu`, `-dr`). clap reads a single dash as a cluster of one-letter shorts, so `-hf` would parse as `-h -f` and render `--help` with exit status 0: a command line upstream honours that neither runs nor reports an error. An argv pre-pass (`src/cli/llama_short_flags.rs`) rewrites those exact tokens to their long spellings before clap sees them. It consults the built clap command for which options consume the following argument and stops at a `--` terminator, so a *value* that happens to spell one of them is never rewritten.

The format gate lives in the shared `-m` resolver, not in the server, so `mlxcel generate`, `mlxcel chat`, `mlxcel serve` and `mlxcel-server` all refuse a GGUF reference identically. Issue #1434 owns this shard.

## GGML runtime, placement, and memory options

Everything in `ggml-runtime.json` describes the **GGML** backend: which CPU cores its thread pool runs on, how many layers to copy into VRAM, how to split a model across GPUs, whether to `mmap` or `mlock` the GGUF file, which RPC servers to farm work out to, and which GGML quantizer stores the KV cache. mlxcel runs every tensor operation through MLX on one Metal or CUDA device and has no GGML backend, so almost none of it has a counterpart.

Each option is accepted as a hidden compatibility argument and its **value** is classified at startup, before any weight is read:

- A value whose b10621 meaning mlxcel already satisfies, or that asks for nothing, is inert and is accepted silently: `--split-mode none`, `--threads` with any value `<= 0` (upstream replaces those with the hardware concurrency), `--cpu-strict 0`, `--poll 50`, `--prio 0`, `--flash-attn on|auto`, `--gpu-layers all`, `--main-gpu 0`, `--load-mode auto|mmap`, `--mmap`, `--no-direct-io`, `--repack` and `--no-repack`, `--kv-offload`, `--op-offload`, `--no-perf`, `--n-cpu-moe 0`, `--fit off`, `--backend-sampling`, and every value of `--defrag-thold`, whose upstream handler is a deprecation warning and nothing else.
- Anything else stops startup with a diagnostic naming the option, the value, the platform limitation, and the mlxcel alternative where one exists: `--split-mode row` and `--tensor-split` point at `--tp-size` / `--pp-size` and [`distributed.md`](distributed.md), `--rpc` at `--node-role` / `--cluster-peers`, `--mlock` and `--load-mode mlock` at `MLXCEL_WIRED_LIMIT`, `--device` at `MLXCEL_DEVICE`, `--perf` at `--metrics`, `--fit on` at `--estimate-memory`, `--no-kv-offload` at `--cache-type-k` / `--cache-type-v`, and `--check-tensors` / `--list-devices` at the `mlxcel` CLI's `inspect` subcommand.

`--load-mode`, `--mlock`, `--mmap` and `--direct-io` all write one field upstream, applied in command-line order, and b10621 deprecates the last three in favour of the first while warning that combining them is last-wins. mlxcel makes `--load-mode` the authority when it is present and does not classify the deprecated three at all, so `--no-mmap --load-mode mmap` starts here exactly as it does upstream. A deprecated flag written *after* `--load-mode` would win upstream and is ignored here; that residue is recorded on all four entries.

llama.cpp writes most of these with a single dash and several letters (`-ngl`, `-fa`, `-sm`, `-ctk`, ...), which clap reads as a cluster of one-letter shorts. The argv pre-pass in `src/cli/llama_short_flags.rs` rewrites those exact tokens before clap sees them, so `-ngl 99` reaches `--gpu-layers` instead of failing as an unknown argument. The pre-pass consults the built clap command for which options consume the following entry, so a value that happens to spell one of the tokens is never rewritten, and its table may not shadow an mlxcel short.

`--gpu-layers` is the one option whose classification needs the checkpoint: only its `num_hidden_layers` separates a full offload (inert, because mlxcel always runs every layer on the accelerator) from a partial one. It is therefore checked after the model reference resolves; every other option is checked before, so `--numa distribute` is reported immediately rather than after a multi-gigabyte download.

All of these are hidden from `--help`. They are compatibility surfaces, not mlxcel features, and rendering them would imply a GGML backend that does not exist. mlxcel's own Metal, Accelerate, CUDA, neural-accelerator, and TurboQuant options are unaffected and stay visible.

### KV cache types

`--cache-type-k` and `--cache-type-v` are the only entries in this shard that alias rather than reject, and they alias exactly one value. b10621 accepts `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`; `f16` names the same thing on both sides, unquantized half-precision KV storage, so it maps onto mlxcel's `KVCacheMode::Fp16`.

The six quantized names are **different quantizers, not different names for the same one**. `q8_0` is block-wise with a per-block scale; mlxcel's `int8` is per-token absmax; the Turbo modes are PolarQuant with a Walsh-Hadamard rotation. Issue #1445 deliberately does not map any of them onto a TurboQuant mode, because no numerical or storage equivalence has been demonstrated and naming similarity is not evidence. `f32` and `bf16` are refused for a different reason and get their own sentence: they are unquantized GGML float types, and mlxcel's KV cache stores f16 or one of its own quantized modes, with no f32 or bf16 storage to select. Each rejection names the value, why the vocabularies do not carry over, and the mlxcel modes that do exist. Demonstrating an equivalence, if anyone wants to, means the teacher-forced logit-trace procedure in [`benchmarks.md`](benchmarks.md) and a report of disagreement at decided positions.

mlxcel's own `int8`, `fp16+turbo4`, `fp16+turbo3`, `turbo4` and `turbo4-delegated` remain available on the same flag; see [`turbo-kv-cache.md`](turbo-kv-cache.md).

### Environment bindings

Value-taking options bind their `LLAMA_ARG_*` variable through clap, which passes the string through unchanged. Value-less flags and `--x` / `--no-x` pairs do not: b10621 fires a value-less option from the environment only for `on`, `enabled`, `true` or `1` (`common_arg_utils::is_truthy`, compared case-sensitively), and reads a bool pair through `parse_bool_value` plus a `LLAMA_ARG_NO_*` alias that means false. clap's own boolish parser accepts a wider vocabulary and *errors* outside it, so those nine are resolved at runtime against b10621's rules instead. `LLAMA_ARG_CPU_MOE=0` therefore does not enable `--cpu-moe`, and `LLAMA_ARG_PERF=sometimes` stops startup exactly as upstream throws.

## Chat templates, reasoning, and output parsing

`chat-templates.json` is the shard where matching names were least likely to mean matching behavior: mlxcel already had a Jinja engine, `reasoning_content`, a thinking budget and a `reasoning_effort` kwarg, but `--reasoning-format`, `--reasoning`, `--skip-chat-parsing` and `--prefill-assistant` did not parse at all, so a llama-server command line asking for any of them failed outright and a deployment could not move its thoughts out of `reasoning_content`.

### Where the thoughts go

`--reasoning-format` (`LLAMA_ARG_THINK`) chooses the placement, and mlxcel honors all four values on both chat paths:

| Value | `message.content` | `message.reasoning_content` |
|---|---|---|
| `none` | the thoughts, tags and all, left unparsed | absent |
| `deepseek` | the answer only | the thoughts |
| `deepseek-legacy` | the answer with the `<think>` tags | the thoughts |
| `auto` (default) | as `deepseek` here | the thoughts |

mlxcel's pre-#1447 behavior was `deepseek` unconditionally. The same placement applies on the disaggregated router front, which serves the same route. The entry stays `deferred` against #1470 for three reasons recorded on it: `auto` resolves to `deepseek` rather than being detected from the template, because mlxcel's reasoning split uses one marker set for every family it supports; in a *streamed* response under `none` or `deepseek-legacy` the thinking text reaches `delta.content` but its literal delimiters do not, because the streaming filter drops them as it matches them; and whether a streamed `deepseek-legacy` response should carry the thoughts in one field or both is unverified, because upstream's `reasoning_in_content` consumer sits outside the pinned source set.

The thoughts-preserving content form is rebuilt from the parser's own content plus the extracted reasoning, not from the raw text. Rebuilding matters twice over: the raw text still carries the tool-call syntax the parser removed, which would report the same call as `content` and as `tool_calls`, and the raw-text cleaning pass strips the Gemma 4 `<|channel>` delimiters that `none` exists to keep.

`--skip-chat-parsing` supersedes all of it, as it does upstream: everything the model emitted goes to `content` verbatim, reasoning and tool-call syntax included, and no `reasoning_content` or `tool_calls` is produced. Tool-call parsing is gated off with it on every path, so a call is never reported twice.

### The flags that are template kwargs

`--reasoning`, `--reasoning-effort` and `--reasoning-preserve` are not response shaping upstream: b10621 implements all three by writing `enable_thinking`, `reasoning_effort` and `preserve_reasoning` into `params.default_template_kwargs`, the same map `--chat-template-kwargs` fills. mlxcel writes the same keys into the same place, so the merge rule, the per-request override, and the template's freedom to ignore a key are shared with upstream rather than reimplemented.

clap gives no command-line order, so when `--chat-template-kwargs` names one of those keys too, the dedicated flag wins here and the collision is logged; b10621 applies whichever handler ran last. Upstream itself deprecates setting `enable_thinking` through the kwargs in favour of `--reasoning`, so the two agree whenever the flag came last. Same shape for `--chat-template` and `--chat-template-file`, which write one field upstream: the inline template wins here.

### `--chat-template` takes template text, not a name

b10621 accepts either Jinja template text or one of 54 built-in identifiers (`chatml`, `llama3`, `deepseek3`, ...). mlxcel has no built-in template library: an MLX checkpoint carries its own template in `tokenizer_config.json`, which is what mlxcel renders by default. A bare built-in name would become the template itself and every prompt would render to the literal string, so the name set is recognised and refused before the model resolves.

### Still deferred

Four entries are `deferred` against #1470 rather than claimed. `--reasoning-format` is the one whose behavior is otherwise complete; its streamed delimiter loss and the unverified `reasoning_in_content` question keep it there. `--prefill-assistant` is b10621's default and mlxcel diverges from it *with no flag passed*: a trailing assistant message is answered with a fresh turn here and continued upstream. `--reasoning-budget-message` parses and warns at startup but is not yet injected before the end-of-thinking tag. The native `echo` field is a plain completion feature mlxcel lacks. Every other native field in this shard is `not_applicable`: mlxcel's `POST /completion` is a raw-prompt endpoint with no chat template and no chat parsing for them to configure.

## Multimodal projectors and request media

b10621's multimodal support is a separate artifact. `libmtmd` loads a projector file next to the language model (`--mmproj`, `--mmproj-url`), places it on a device (`--mmproj-device`, `--mmproj-offload`), and can be told not to load one at all (`--no-mmproj`). mlxcel loads an integrated MLX VLM checkpoint: the vision tower, the audio tower and the multimodal projector are tensors inside the same SafeTensors snapshot as the language model, resolved by the same `-m` reference. That difference decides the whole `multimodal-and-audio.json` shard (#1451).

| Flag | Environment | State | Behavior |
|---|---|---|---|
| `--mmproj` / `-mm` | `LLAMA_ARG_MMPROJ` | `not_applicable` | Refused at startup. There is no separate GGUF projector to attach to an MLX checkpoint; the diagnostic names an MLX VLM checkpoint to point `-m` at instead. |
| `--mmproj-url` / `-mmu` | `LLAMA_ARG_MMPROJ_URL` | `not_applicable` | Refused for the same reason. |
| `--mmproj-auto` / `--no-mmproj` / `--no-mmproj-auto` | `LLAMA_ARG_MMPROJ_AUTO` | `aliased` | Honored. The default admits media; `--no-mmproj` refuses every image, audio and video part with b10621's own `<kind> input is not supported` clause. |
| `--mmproj-offload` / `--no-mmproj-offload` | `LLAMA_ARG_MMPROJ_OFFLOAD` | `not_applicable` | The default is inert; `--no-mmproj-offload` is refused, because mlxcel has no host projector path. |
| `--mmproj-device` / `-mmdev` | `MTMD_BACKEND_DEVICE` | `not_applicable` | Refused, pointing at `MLXCEL_DEVICE`, which selects the MLX device for the whole checkpoint. |
| `--image-min-tokens` | `LLAMA_ARG_IMAGE_MIN_TOKENS` | `aliased` | Honored on the dynamic-resolution preprocessing path; see below. |
| `--image-max-tokens` | `LLAMA_ARG_IMAGE_MAX_TOKENS` | `aliased` | Honored on the same path. |
| `--mtmd-batch-max-tokens` | `LLAMA_ARG_MTMD_BATCH_MAX_TOKENS` | `not_applicable` | Inert at b10621's own default of 1024; any other value is refused, because mlxcel encodes each image in one vision-tower forward and has no image-token batch to bound. |
| `--media-path` | - | `aliased` | Implemented, with a confined root; see below. |

Everything except `--media-path` is hidden from `--help`. Rendering the projector family would imply that a GGUF `mmproj` file can be attached to an MLX checkpoint, which is precisely what the classification denies. `--media-path` is a real mlxcel feature and is visible, under a `Multimodal Options` heading.

### Media parts a checkpoint cannot consume are refused

A text-only checkpoint used to accept an `image_url` content block at the HTTP boundary, drop it inside `prepare_request_vlm_embeddings`, and answer from the prompt alone. The reply was fluent, and a caller could not tell an ignored picture from a described one. The chat-completions, responses and Anthropic-messages routes now refuse the request before any referenced URL or file is read, with b10621's own leading clause and its `not_supported_error` type at HTTP 501:

```text
image input is not supported - hint: the loaded checkpoint 'qwen3-0.6b-4bit' has no image tower; ...
```

Capability comes from the same `config.json`-only model-type probe that already decided video support, so a newly ported VLM family is admitted with no list to update. The worker keeps its own copy of the refusal, so a path that reaches it without passing a route gate fails rather than degrading to text. The hint half of upstream's sentence is replaced on purpose: telling an operator to provide an `mmproj` has no meaning for an integrated checkpoint.

### `--image-min-tokens` / `--image-max-tokens` move real pixel bounds

Upstream converts a token budget into pixel bounds in `clip_hparams::set_limit_image_tokens`:

```text
patch_area       = patch_size^2 * n_merge^2
image_min_pixels = image_min_tokens * patch_area
image_max_pixels = image_max_tokens * patch_area
```

mlxcel's dynamic-resolution processors express the same two bounds as `min_pixels` / `max_pixels`, so the translation is the identity: the same multiplication against the same patch area, applied in `src/vision/image_token_overrides.rs` and consumed by `vision::processors::qwen2_vl::smart_resize`. Upstream's value domain carries over too. Only a positive value is a custom bound, `0` and a negative number mean "read it from the model", and a maximum below the minimum is refused exactly as upstream throws on `image_max_pixels < image_min_pixels`.

A checkpoint whose preprocessor resizes to a fixed geometry has no such bound to move. Rather than a hand-written list of honoring architectures, every bound-consuming processor increments an applications counter and the model worker refuses to serve when a budget was requested and the counter is still zero after the checkpoint loads, naming the checkpoint and the flag. That is the mechanism [`rope_overrides`](../src/models/rope_overrides.rs) already uses for `--rope-freq-base`, and for the same reason: a list goes stale the moment a family is ported.

### `--media-path` confines local files to one root

Without `--media-path`, a `file://` media URL in a request is refused with b10621's own sentence, `file:// URLs are not allowed unless --media-path is specified`. With it, the request's path is resolved against the configured directory and may not leave it.

mlxcel reproduces upstream's rules and then adds what a pure string check cannot do:

- **Concatenation, not join.** b10621 evaluates `media_path + file_path`, so `file:///etc/passwd` lands under the root instead of replacing it. mlxcel strips the leading separators before joining, because a Rust `Path::join` with an absolute component would discard the root and turn a compatibility feature into an arbitrary-file read.
- **b10621's whole name validation.** `fs_validate_filename(path, allow_subdirs=true)`: no `..` anywhere, no control characters, none of `: * ? " < > |`, no leading or trailing space, no trailing `.`, at most 255 bytes, and the Unicode separator look-alikes (`U+FF0E`, `U+2215`, `U+2216`, `U+FFFD`, `U+FEFF`) refused.
- **No percent decoding, and no percent-encoded traversal.** Upstream never decodes, so `%2e%2e` is a literal filename there. mlxcel does not decode either, and additionally refuses a path carrying a percent escape for `.`, `/`, `\` or NUL, so the property is checked rather than inherited from an absent call.
- **Canonicalize and contain.** The joined path is canonicalized and must stay inside the canonical root. A symlink whose target leaves the root is refused; one that stays inside still resolves, so organising the media root with links keeps working.
- **`O_NOFOLLOW` on the open.** The resolve-to-open window cannot be won by swapping the last component for a symlink; the open fails with `ELOOP` instead. This is the same primitive the video resolver already used.
- **Regular files only.** A directory, FIFO, socket or device node under the root is refused, so a FIFO cannot block a request task waiting for a writer.

The last three are stricter than b10621, whose check is a pure string test that reads whatever the path names, and are recorded as divergences on the entry. A bare relative path is governed by the same root, because that is what an operator who configured `--media-path` naturally writes.

### Remote media URLs have a network-address policy

A request may name an `http(s)` image, audio or video URL and the server fetches it, which makes the server a fetch proxy an unauthenticated caller steers. b10621 fetches with `common_remote_get_content` and applies no address policy at all. mlxcel refuses non-public addresses in three places, which is recorded as a divergence:

1. **Before the request.** The scheme must be `http` or `https`, the URL may not carry credentials, an IP-literal host is checked directly, and a host written as a name is resolved with every resulting address checked.
2. **On every redirect.** The redirect policy sees the next URL before it is followed, so a public origin cannot bounce the fetch onto `http://169.254.169.254/`. The chain stays bounded at five hops.
3. **After the connection.** The peer address the request actually reached is re-checked before a single body byte is read, which closes the DNS-rebinding window the first two leave open.

Refused: loopback, the unspecified address, multicast and broadcast, the IPv4 private and link-local ranges (which is what puts the cloud metadata service at `169.254.169.254` out of reach), carrier-grade NAT, the IETF protocol-assignment, benchmarking and documentation prefixes, `240.0.0.0/4`, IPv6 unique-local, link-local and documentation, and any IPv4-mapped or IPv4-compatible IPv6 spelling of a refused IPv4 address. A deployment that genuinely serves its media from an internal object store sets `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`, which turns every address check into a pass. It is off by default because a fetch proxy that reaches the private network has to be an explicit operator decision.

The existing byte and decode limits are unchanged and still apply: `--max-image-payload-size`, `--max-images`, the decoder's width, height and allocation caps, the audio per-clip and per-request ceilings, and the 10-second fetch timeout.

## Audio transcription

`POST /v1/audio/transcriptions` and its `/audio/transcriptions` alias are the only audio routes b10621 mounts, and the only ones this section governs. `/v1/audio/speech` and `/v1/audio/translations` are mlxcel's own and are documented in [`audio-api.md`](audio-api.md).

### b10621 has no speech-to-text model

Upstream's route is a translation layer over `/v1/chat/completions`. It refuses unless the loaded chat model takes audio, converts the multipart form into a chat request whose single user message carries an ASR prompt plus the uploaded clip, and renders the completion as a transcript event. There is no separate ASR model anywhere in it.

mlxcel's route was served by a dedicated Whisper worker, and that worker is populated only when `-m` names a Whisper checkpoint, which leaves the chat worker unloaded. The two were therefore mutually exclusive server shapes, measured on this tree:

| Command line | `/v1/chat/completions` | `/v1/audio/transcriptions` (before #1446) |
|---|---|---|
| `mlxcel-server -m models/mlx/whisper-tiny` | 503, no chat model | transcribes |
| `mlxcel-server -m models/mlx/gemma3n-e2b-4bit` | answers | **501 `audio model kind not loaded: stt`** |

The second row is the only shape b10621 can express, and it was the one that did not work. A `llama-server` deployment that transcribes through its loaded model had no mlxcel equivalent, however closely the multipart field set matched: route-name overlap was not compatibility. That is the design decision #1446 asked for, and it was settled by measurement rather than preference.

**The shared route now dispatches through the loaded chat model, exactly as upstream does.** Posting the clip to `gemma3n-e2b-4bit` returns its transcript. The Whisper worker stays the implementation when no chat model can take audio, which is the Whisper-server shape b10621 cannot express at all, so it adds no divergence in any configuration upstream can reach.

### What a client sees

The response is **not** OpenAI's classic `{"text": ...}` object. b10621 emits the transcript-event shape, and so does mlxcel now:

```json
{"type":"transcript.text.done","text":"The quick brown fox jumps over the lazy dog.","usage":{"type":"tokens","input_tokens":206,"output_tokens":10,"total_tokens":216,"input_tokens_details":{"cached_tokens":0}}}
```

**A client that was parsing `{"text": ...}` off `/v1/audio/transcriptions` must read `text` out of that object instead, or move to `/v1/audio/translations`, which is unchanged.** The streamed form is `data: {"type":"transcript.text.delta","delta":"..."}` frames followed by the `done` frame and `data: [DONE]`.

Aligned with upstream, each checked against the pinned source rather than the schema's prose:

- The capability refusal is `The current model does not support audio input.` as a 501 `not_supported_error`, and it precedes every field check, so a request that is wrong in two ways gets the 501.
- A form with no `file` part is `No input file found for transcription` (400).
- `response_format` defaults to `json` and anything else is `Only 'json' response_format is supported for transcription` (400). mlxcel's own `text` and `verbose_json` moved to `/v1/audio/translations`, which b10621 does not mount, so the shared route's default is upstream's.
- The prompt is upstream's ASR preset, `Transcribe audio to text`, overridden by a `prompt` field, with a non-empty `language` appended in upstream's own `" (language: xx)"` form rather than passed as a decoder parameter.
- `temperature` and `max_tokens` arrive as strings and are retyped, with a 400 on a value that does not parse, because upstream's `std::stof` / `std::stoul` throw `std::invalid_argument` and its wrapper maps that to 400.
- `stream` is compared against the literal `"true"`, not a boolean vocabulary, because the form carries strings.
- The multipart duplicate rules are upstream's: a repeated text field collapses into an array, which upstream's `json_value` then rejects on type and replaces with the default, so a duplicated `prompt` is ignored and a duplicated `response_format` falls back to `json`; a repeated `file` part keeps the last one.
- Unknown fields are carried and ignored.

### Limits

Every bound is applied before a decoder sees the clip: at most 32 multipart parts, 25 MiB per part (matching the route's body limit), and a WAV geometry read from the header rather than from a decode: at most 192 kHz, 8 channels and 600 seconds. A `data` chunk that declares more audio than the file carries is clamped to what is present, so an amplifying header costs a header parse. A malformed or truncated file is a 400 naming the structural problem.

### Still deferred

Both route entries stay `deferred` against #1446 for two divergences a b10621 client can observe:

- **Container support.** Only RIFF/WAVE is accepted. b10621's mtmd front-end decodes mp3, flac and the rest, so a non-WAV clip is a 400 here and a transcript there.
- **Stream granularity.** A streamed response arrives as one delta carrying the whole transcript. The frame shapes and the terminator are upstream's; the incremental delivery is not.

A third residue is recorded but cannot be reached from a b10621 command line: on the Whisper-server shape the `usage` counts are zeros, because the STT worker reports no token counts, and `prompt` steers nothing there.

## Regeneration

```bash
python3 scripts/compat/extract_b10621_manifest.py \
    --llama-server /path/to/llama-b10621/llama-server \
    --source-dir /path/to/llama.cpp-c1d0e7a \
    --archive /path/to/llama-b10621-bin-macos-arm64.tar.gz
```

The extractor consumes the official binary (`--help` is the authority for options) and the pinned sources (`server.cpp`, `server-http.cpp`, `server-schema.cpp` for routes and native fields). `--archive` verifies the download against the pinned SHA-256 and then requires the `--llama-server` binary to be byte-identical to a member of that verified archive, both before the binary is executed. That second half matters because the extractor runs the binary with `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` pointed at its own directory: hashing the tarball while executing an unrelated path would be assurance about a file that is never used. The archive is compared through `tarfile` in read mode and is never extracted. Omitting `--archive` runs the binary unverified and the extractor warns on stderr. Regeneration is deterministic: facts are rewritten wholesale, policy fields (`state`, `issue`, `test`, `notes`, `divergence`, `mlxcel`) are preserved by entry id and re-emitted in that canonical key order (a policy key an older schema never wrote is backfilled from the skeleton), `pin.json`'s `shards[name].owners` map is preserved by shard name exactly like `mlxcel_baseline` (a brand-new shard starts with an empty owner set, which needs a human before CI accepts entries in it), and running the extractor twice leaves the worktree clean. Entries that are new upstream land in `_unclassified.json`, which the validator rejects until a human classifies them, so bumping the pin to a newer nightly produces a merge-blocking, reviewable diff instead of silent drift.

## RoPE and YaRN runtime overrides

`--rope-scaling`, `--rope-scale`, `--rope-freq-base` and `--rope-freq-scale` are accepted on both server binaries with b10621's spellings, value domains and `LLAMA_ARG_ROPE_*` bindings, and they change the rotation for real. They rewrite the checkpoint's `rope_scaling` block and `rope_theta` before the model is constructed, which is before any KV cache exists, matching where `llama_model_load` applies them upstream.

| Flag | Environment | Effect |
|---|---|---|
| `--rope-scaling {none,linear,yarn}` | `LLAMA_ARG_ROPE_SCALING_TYPE` | `none` drops the checkpoint's block and rotates with the plain `base^(2i/d)` table. `linear` divides positions by `--rope-scale`, or by the checkpoint's own factor when that flag is absent. `yarn` is refused; see below. |
| `--rope-scale N` | `LLAMA_ARG_ROPE_SCALE` | Expands context by a factor of N (`rope_freq_scale = 1/N`). |
| `--rope-freq-scale N` | `LLAMA_ARG_ROPE_FREQ_SCALE` | The reciprocal spelling of the same setting. Passing both with values that are not reciprocals is a startup error rather than a silent precedence choice. |
| `--rope-freq-base N` | `LLAMA_ARG_ROPE_FREQ_BASE` | Replaces `rope_theta`. On Gemma 3 it reaches the global-attention layers only; the sliding layers keep `rope_local_base_freq`, which is llama.cpp's separate `rope_freq_base_train_swa`. |

Two requests are refused instead of being approximated. `--rope-scaling yarn` has no representation: mlxcel's shared RoPE path builds `default`, `linear` and `llama3` frequency tables only, and serving a YaRN request as one of those would change the rotation without saying so. A bare `--rope-scale` or `--rope-freq-scale` on a checkpoint that declares a banded scheme (`llama3`) is refused for the same reason: llama.cpp multiplies its own `rope_freq_scale` into that rotation, mlxcel's banded table has no such multiplier, and dropping either half silently changes the result. Name the scheme (`--rope-scaling linear` or `--rope-scaling none`) to say which rotation you want. Checkpoints whose own `config.json` declares YaRN (DeepSeek V2 / V3.2 / V4, gpt-oss, Mellum, TeleChat3) build it from that config and are unaffected by any of this.

The five `--yarn-*` flags are accepted at b10621's sentinel defaults only (`-1.0`, and `0` for `--yarn-orig-ctx`), which upstream defines as "use the values the model was trained with", so a deployment script that spells out the upstream defaults keeps working. Any other value is a startup error naming every non-sentinel flag at once.

### Which architectures honor the override

The override reaches the six families that resolve their rotation through `src/models/rope_utils.rs`: Llama 3.x (and through it Qwen2 / Qwen2.5, Helium, the `mllama` text decoder, and every VLM whose text backbone is one of those), Qwen3, Qwen3-MoE, Apertus, Gemma 3, and InternLM2 / InternLM3. Other families compute their frequencies inline and would ignore it.

They are not listed anywhere in the code, because a hand-maintained list of honoring architectures goes stale the moment a family is ported. Each seam that consumes the override increments a counter instead, and the model worker refuses to serve when an override was requested and the counter is still zero after the checkpoint loads, naming the checkpoint and the flag. A family that wires itself into `rope_utils` starts being accepted with no list to update; one that does not is reported rather than silently ignored.

### Validating a change to this path

Short prompts cannot see a `rope_scaling` defect: the banded and plain tables agree closely at low positions. Use the teacher-forced logit trace from [`benchmarks.md`](benchmarks.md), which reads `LLAMA_ARG_ROPE_*` from the environment so each arm is one process, and trace positions that are actually long:

```bash
cargo build --release --features metal,accelerate --example logit_trace
./target/release/examples/logit_trace MODEL CORPUS.txt 2048 4 8 8192 > ref.tsv
LLAMA_ARG_ROPE_SCALING_TYPE=none \
./target/release/examples/logit_trace MODEL CORPUS.txt 2048 4 8 8192 > none.tsv
python3 scripts/compare_logit_traces.py ref.tsv none.tsv
```

The sixth argument is the context established before each traced forward, and the traced positions of chunk `c` start at `c * CHUNK_TOKENS`, so a narrow chunk cannot reach a long position however large that argument is. The command above traces positions 0 to 8191 across four chunks. Setting `--rope-freq-base` to the checkpoint's own `rope_theta` is the control: it must come back byte-identical, which is what separates "the override moved the rotation" from "the override plumbing perturbs the model".

## Prompt caching and continuous batching

`--cache-prompt` / `--no-cache-prompt`, `--cache-reuse`, `--cache-ram` and `--cont-batching` / `--no-cont-batching` are accepted on both server binaries with b10621's environment bindings. Three of the four act; the fourth refuses.

| Flag | Environment | Effect |
|---|---|---|
| `--cache-prompt` / `--no-cache-prompt` | `LLAMA_ARG_CACHE_PROMPT` | Turns the prompt-prefix KV cache on and off. On by default, as upstream. Between this and mlxcel's own `--no-prompt-cache`, either disable wins: two explicit operator statements in opposite directions resolve to the one that caches nothing. |
| `--cache-ram N` | `LLAMA_ARG_CACHE_RAM` | The prompt cache's byte budget, stated in MiB with upstream's `-1` (no limit) and `0` (disable) sentinels. The same setting as `--prompt-cache-capacity-bytes`, which wins when both are given. mlxcel's unset default is 2048 MiB against upstream's 8192. |
| `--cont-batching` / `--no-cont-batching` | `LLAMA_ARG_CONT_BATCHING` | Disabling pins the decode width to one sequence (`--max-batch-size 1`). It is deliberately not mlxcel's `--no-batch`, which replaces the batch scheduler with a sequential worker and takes the prompt cache and speculative decoding out with it; upstream's `-nocb` keeps the slots and stops them interleaving. |
| `--cache-reuse N` | `LLAMA_ARG_CACHE_REUSE` | `0`, upstream's default, is accepted and inert. Any positive value refuses to start. |

Per request, the chat-shaped routes honor b10621's `cache_prompt`. Sending `false` withholds that request's cache context, which is the single handle the scheduler reaches the store through for both the prefix lookup and the donate-back, so the request is prefilled cold and leaves every entry another request might reuse untouched. It arrives at the request root, through the flattened OpenAI-SDK `extra_body`, or nested inside `extra_body`. Native `/completion` neither declares the field nor caches prompts.

### Why `--cache-reuse` refuses instead of accepting the number

Upstream reuses a cached chunk that is not a prefix of the incoming prompt by deleting the span between the divergence and the resumption point and shifting the rotary positions of everything after it back down. mlxcel's prompt cache reuses a strict token prefix: it adopts a cached KV set whose tokens are a prefix of the request and prefills the remainder. The gap is not a missing switch, it is a missing operation. `KVCache::trim_front_keep_sink` drops the oldest tokens by advancing `live_start` and deliberately leaves `offset` alone, and `gather_positions` / `gather_within_tail` compact surviving slots without touching the rotation already baked into each cached key. Nothing rewrites a cached key's RoPE rotation, so there is no way to express the shift, and a server that accepted the number would behave exactly as it does at `--cache-reuse 0` while the operator believed otherwise.

### What is not accepted, and why not

`--slot-prompt-similarity`, `--kv-unified`, `--cache-idle-slots`, `--ctx-checkpoints` and `--checkpoint-min-step` are not accepted at all. Each tunes per-slot retained prompts or context checkpoints, and mlxcel has neither: `/slots` is synthesized from the active count and the queue depth rather than reported from slot objects, reuse goes through a process-wide radix trie over token prefixes that is consulted for every request regardless of which sequence last held them, and at most one history-boundary snapshot is captured per sequence rather than a ring of checkpoints. `--slot-prompt-similarity` makes the point sharpest: its upstream default is `0.10`, not `0`, so accepting it inert would mean honoring a script that passes the upstream default while no slot-selection policy exists to honor. Their manifest entries carry the divergence instead.

### Validating a change to the prompt cache

`tests/prompt_cache_compat_e2e.rs` is the differential gate. It runs a real server and compares per-position tokens and logprobs between a cold evaluation, a prefix-cache hit, a per-request `cache_prompt: false`, and three concurrent requests of different prompt lengths against their own solo runs. What a prompt cache has to be is not "good" but "the same as not having one", which is a statement about two runs that no single-run perplexity can express.

Two properties of that comparison decide whether it means anything. The same-width arms (cache hit, per-request disable) are gated on exact agreement, because they run at the batch width the reference did. The concurrent arm is gated on the **first** divergence and the reference's own top-two gap there, not on the total disagreement: a decode at batch width three does not run the same kernels as one at width one (issue #203), near-ties can land the other way, and once a single token flips every position after it is conditioned on different text, so counting the cascade measures the flip's echo rather than the change. On `qwen3-0.6b-4bit` the concurrent arms part at reference gaps of 0.25 and 0.125 against a decided-position threshold of 2.0, which is that jitter class and not a cache defect; the same-width arms agree on every one of 32 positions.
