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

`pin.json` records the frozen reference, the inventory counts, and the shard map (`shards[name].owners`, the set of implementation issue numbers allowed to own entries in that shard; see [Sharding](#sharding)). Every other `*.json` file is an area shard holding entries of three kinds: `option` (one per help entry, with every accepted spelling, the environment binding, and the default), `route` (method/path), and `native_request_field` (native `/completion` schema fields with aliases). `pin.json` and every shard carry `schema_version: 4`.

Every entry carries exactly one compatibility-policy state, per the policy defined in epic #1431:

| State | Meaning |
|---|---|
| `supported` | Spelling, value domain, default, precedence, and observable behavior match, and the entry's `divergence` list is empty. |
| `aliased` | A different mlxcel spelling, route, or request field provides equivalent behavior, with a tested translation. The claim names the mlxcel identity, which must differ from the b10621 one; when mlxcel answers the b10621 name itself the entry is `supported`, not `aliased`. |
| `not_applicable` | No MLX/CUDA equivalent; mlxcel rejects the option with an actionable diagnostic or accepts only semantically inert forms, with a test and documentation. |
| `deferred` | Not yet true; a linked implementation issue owns it. Accepted-but-ignored flags are `deferred`, never `supported`. |
| `by_design` | Implemented and served, differing from b10621 in a recorded, tested way, permanently. Not `supported` (behavior differs), not `aliased` (no equivalent mlxcel identity), not `not_applicable` (the concept applies and is served), not `deferred` (no one will close it). The permanence is argued per entry in a `rationale` object and pinned by the entry's `test`. |

Every `aliased`, `not_applicable` or `deferred` entry carries a linked issue, plus `notes` and a test id where they apply. A flag mlxcel parses but does not act on is `deferred` with its acceptance pinned in the `mlxcel` claim block, never `supported`. A `by_design` entry deliberately needs **no open issue**: being closed to further work is the state's entire meaning, so its entries never feed the `--check-issues-open` gate; it may keep an `issue` number as provenance for the chain that established the behavior, and that number must still belong to the shard's owner set.

### `divergence`

Every entry carries a `divergence` list: short strings, each naming one externally observable way mlxcel differs from b10621 for that entry. **A non-empty `divergence` forbids `supported`**, and the validator makes that a hard error naming the four honest alternatives (`aliased`, `not_applicable`, `deferred`, `by_design`) and asking for the owning issue.

The field exists because prose could not be the gate. `notes` had been carrying divergences like the semantic collisions on `--timeout`, `--models-dir`, `--cache-type-k/v` and `POST /completions`, the inverted DRY disable sentinel on `--dry-penalty-last-n`, and the penalty-window drift on `--repeat-penalty` / `--frequency-penalty` / `--presence-penalty`. That works while the state is already honest, but it cannot stop the opposite case: twenty-two entries once read `supported` while diverging observably, two of them under a `notes` field that opened with "BEHAVIOR DRIFT", and the CI gate locked the false claim in. Every remaining sub-issue of epic #1431 will flip more entries into `supported`, so the rule has to be machine-readable. `notes` keeps the context and the reasoning; `divergence` carries the specific observable difference and its owning issue.

Both fields are entry-level, alongside `state`, `issue`, `test`, `rationale` and `mlxcel`. An entry carries exactly its kind's fact keys plus those seven policy keys, in that order, and nothing else: a misspelled `divergance` (or a `divergence` misfiled inside the `mlxcel` block) fails the gate instead of quietly recording nothing.

### `rationale`

Every entry carries a `rationale` key; it is `null` everywhere except on a `by_design` entry, where it is the machine-checked argument that the recorded divergence is permanent (the same pattern `divergence` set at schema 3: prose in `notes` cannot be the gate). A non-null `rationale` on any other state is a hard error. The object has a closed key set:

- `kind`: exactly `"architectural"` or `"policy"`. `architectural` means mlxcel structurally cannot produce b10621's behavior (for example, there is no unpooled hidden state left after an embedding family pools inside its own forward pass). `policy` means mlxcel can produce it and deliberately does not.
- `reason`: a non-empty string naming why the divergence is permanent, not merely restating it.
- `revisit_if`: required non-empty exactly when `kind` is `policy`, stating what would have to change for the decision to be revisited; it must be `null` when `kind` is `architectural`. This is what keeps an architectural impossibility and a policy choice mechanically distinguishable.

A `by_design` entry additionally must carry a non-empty `divergence` (an entry with none is `supported`), non-empty `notes`, and a `test` pointer that resolves and pins the permanent behavior itself. The obligations are deliberately harder to satisfy than `deferred`'s single issue number: a fifth state is also a way to stop trying, so "permanent" has to be argued per entry with a test behind it rather than asserted to make a checkbox go green. An entry that cannot meet all of them stays `deferred`. One structural limit is worth knowing: manifest granularity is the whole option, so a per-value divergence (say, one refused value among honored ones) is recorded in the entry's `divergence` strings, not as a per-value state.

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

One native request field is refused rather than honored: `n_cmpl` (and its `n` alias) above 1, because mlxcel serves one completion per request and has no path that returns a JSON array of result objects. At one slot that refusal is byte-equivalent to upstream's own, whose schema limit is `1 <= value <= n_parallel` and which refuses rather than clamps: the pinned binary with `-np 1` answers `Field 'n_cmpl': Value must be between 1 <= value <= 1, but got 2`. The value `1` is accepted, so a client that sends the whole schema at its defaults is not turned away.

The other four left that list in #1477 and are honored now, each against a measurement of the pinned binary rather than against the schema's description.

- `return_tokens` fills the top-level `tokens` array with the raw generated ids and leaves it `[]` otherwise. The ids are the sampled ones, not a re-tokenization of `content`: with `stop: ["Paris"]` the binary answers `content: " "` and `tokens: [12095]`, so the token whose text the stop sequence excluded is still counted and still reported. Streaming is separate and unconditional: every per-token frame carries that token's id in `tokens` whether or not the field was sent, and the final streaming frame carries `[]`.
- `return_progress` opens a streaming response with `prompt_progress: {total, cache, processed, time_ms}` frames that carry no content, one when the slot starts and one per prefill chunk. `processed` counts cache-supplied tokens, so a repeat of a cached prompt opens at `processed == cache` rather than at zero. The field is inert without `stream`, which is upstream's own `stream && return_progress` gate.
- `n_indent` stops the completion when a generated line's indentation falls below the given number of whitespace characters, reported as `stop_type: "limit"`. Upstream's cursor advances one line per decoded token, so the stop lands a token or two after the offending line's text appeared, and the response text is cut at the first character after that line's leading whitespace; the streamed frames already carried the extra text, in mlxcel exactly as in b10621. The value domain is `0..=2147483647`, refused rather than clamped outside it.
- `t_max_predict_ms` bounds the prediction phase for one request, measured from the first token and checked only on a piece whose emitted text contains a newline, also reported as `stop_type: "limit"`. It is per request and does not touch the server-wide `--decode-timeout` / `MLXCEL_DECODE_TIMEOUT` watchdog. The value domain is `-1..=9223372036854775807`, with `-1` and `0` both disabling it.

`generation_settings` grew from 39 to 47 of b10621's 49 `task_params` keys in the same change: `samplers`, `timings_per_token`, `logit_bias`, `lora`, `chat_format`, `generation_prompt`, `reasoning_format` and `reasoning_in_content` are reported now, each from the value mlxcel actually resolved. The two still absent are `backend_sampling`, whose entry records that mlxcel's sampler IS the backend graph with no CPU chain to switch to, and `speculative.types`, which names an upstream draft-model type mlxcel has none of. Both are omitted rather than reported with an invented value, which is the same policy `GET /props` follows and the same policy that moved those eight keys into the reported set as mlxcel gained an analogue for each.

Two streaming-frame shapes changed with them. Every partial frame now carries `id_slot: -1`, upstream's sentinel: its `send_partial_response` never stamps a slot id and only the final frame names the slot. And `tokens_predicted` on a partial frame is the slot's generated-token count rather than the number of frames sent, which differ whenever the stop matcher or the incremental detokenizer held a piece back.

`verbose` left that list in #1477, on evidence rather than on effort: b10621 writes its `__verbose` debug block only from the OAI-compat response builders (`server-task.cpp`), and the native completion object IS `to_json_non_oaicompat()`, so `verbose: true` changes nothing upstream either. Measured against the pinned binary, the top-level key set with the field set is identical to the key set without it. mlxcel now accepts it and ignores it, which is what upstream does; refusing it was the divergence.

`tokens_cached` also changed meaning in #1477 and is a **breaking change for a client that read it as a cache-hit figure**. b10621 reports the slot's cache occupancy after the request, which is `tokens_evaluated + tokens_predicted - 1` (saturating), not what the prefix cache supplied for it. Six measurements against the pinned binary agree, including the `n_predict: 0` prompt-only case, which upstream still answers with `tokens_predicted: 1`, and a fully cache-hit request, where the figure is unchanged by the hit. The cache-supplied count is `timings.cache_n`, which is unchanged and is what `timings.prompt_n` is derived from; a client that wants the hit size should read `cache_n`.

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

Two `/infill` differences remain, deliberate and permanent; its manifest entry records them as `by_design` (kind `policy`) rather than claiming support:

- Text in `input_prefix`, `input_suffix`, `input_extra[].text`, `input_extra[].filename` or `prompt` that contains one of the model's own FIM marker spellings is **refused** with a diagnostic naming the field and the marker. b10621 tokenizes those fields with special-token parsing off, so the marker stays literal text there. mlxcel's generation entry point takes a prompt string that is re-tokenized with parsing on, and a string cannot express "these characters are not that token", so carrying the text through would let a file the client did not write restructure the FIM prompt and return a fluent wrong completion. Failing loudly is the smaller divergence.
- The FIM context is not truncated to the prefill batch. b10621 keeps only the last `3*(n_batch/4)` prefix tokens and the first `(n_batch/4) - (2 + prompt)` suffix tokens; mlxcel serves the whole prefix and suffix, consistent with its policy everywhere else of clamping an over-long request rather than dropping prompt text.

### Embedding and reranking mode, pooling and normalization (#1452)

b10621's `--embedding` / `--embeddings`, `--rerank` / `--reranking`, `--pooling` and `--embd-normalize` are accepted on both server binaries, with `LLAMA_ARG_EMBEDDINGS`, `LLAMA_ARG_RERANKING` and `LLAMA_ARG_POOLING` bound. The two mode flags do the same two things upstream's do, and one more that upstream never needs.

They **restrict**: generation routes answer `501` naming the flag, so a client sees the same "this server does not generate" it would see from `llama-server --embeddings`. They **select**: a mlxcel server runs a dedicated embedding or reranking worker rather than reusing one set of weights, so `--embeddings` requires an embedding checkpoint to resolve from `-m` or `--embedding-model`, and `--reranking` requires a reranker from `-m` or `--reranker-model`. A command line that asks for a mode nothing can serve fails at startup naming the flag, instead of booting a server whose only route answers 501 forever. That startup refusal is the one behavior upstream does not have, because its `--embeddings` embeds with whatever `-m` happens to be and mlxcel has no path that produces embeddings from a causal chat checkpoint.

`--pooling` maps `mean`, `cls` and `last` onto mlxcel's own kernels, and the flag outranks both the checkpoint's `1_Pooling/config.json` and the `MLXCEL_EMBEDDING_POOLING` variable. The other two values are not pooling kernels at all: `rank` is how b10621 puts a model on its reranking path, so it is accepted as a synonym for `--reranking`, and `none` asks for one vector per token, which mlxcel cannot produce because every embedding family pools inside its own forward pass before the engine sees the output. `none` is refused at startup with that reason. mlxcel's own `max` pooling has no b10621 spelling and stays reachable through `MLXCEL_EMBEDDING_POOLING` and the checkpoint config.

`--embd-normalize` implements the whole b10621 domain with upstream's arithmetic: `-1` none, `0` the max-absolute rescale into the signed int16 range, `1` taxicab, `2` euclidean, and any value above 2 that p-norm. A zero vector normalizes to zeros rather than NaN, which is upstream's `norm = sum > 0 ? 1/sum : 0` rule and matters here because the embedding route refuses a non-finite response with a 500. `2` delegates to the L2 kernel every mlxcel embedding response already used, so the default path produces exactly the numbers it produced before. The value is also readable per request as `embd_normalize` on `/embedding`, `/embeddings` and `/v1/embeddings`, as upstream reads it, and `mlxcel embed` takes the same `--pooling` and `--embd-normalize` so the offline and server surfaces resolve identically.

One default differs: with the flag unset mlxcel follows the checkpoint's own `normalize` flag from `config.json`, which is euclidean for every checkpoint that does not set it and none for one that sets it to `false`. b10621 always defaults to `2`, because GGUF carries no such metadata.

The four #1452 manifest entries record these divergences as `by_design`: the startup refusals and the `--pooling` value mapping are `architectural` (no embedding path exists for a causal chat checkpoint, no unpooled hidden state exists for `none`, no pooling kernel exists for `rank` to name), and the `--embd-normalize` default is `policy` (defaulting to a constant `2` is producible but would override the checkpoint author's declared `normalize`). Note the granularity limit: `--pooling` is one entry, so the manifest cannot say per value that `mean`, `cls` and `last` are served while only `none` is refused; the entry's `divergence` strings carry that split.

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

### Context shift is off by default, and retention is operator-settable (#1472)

`--context-shift` / `--no-context-shift` (`LLAMA_ARG_CONTEXT_SHIFT`), `--keep`, and the native `n_keep` / `n_discard` request fields now carry b10621's context-retention semantics on both server binaries. **The default changed**: before #1472, whenever `--ctx-size` or `--max-kv-size` bounded the KV window, mlxcel silently front-trimmed the oldest tokens (keeping a fixed 4-token attention sink) and never told the client. Now, matching b10621's disabled-by-default context shift:

- A prompt that does not fit the per-slot context is refused at admission with upstream's wording (`request (N tokens) exceeds the available context size (M tokens), try increasing it`), whether or not shifting is enabled; the shift only governs growth during decode.
- With shifting disabled (the default), a generation that reaches the bound stops gracefully with `truncated: true` and `stop_type: "limit"` (OpenAI `finish_reason: "length"`), which is b10621's stop-at-the-bound shape.
- With `--context-shift`, the scheduler makes room by discarding tokens after a retained prefix: `--keep` (per request `n_keep`; `-1` = the whole initial prompt, one token added for a tokenizer-prepended BOS, clamped to `bound - 4`) survives every shift, and per-request `n_discard` sets the discard depth (`0`, the default, discards half of the non-retained window). When a prefill chunk overshoots the bound the discard is raised to the overshoot so the cap holds.

**To restore something close to the old rolling-window behavior, pass `--context-shift --keep 4`.** The old behavior discarded exactly the excess per step where the new default discard is half the window, so the retained text differs, but generation is again unbounded with the attention sink pinned. Two caveats carried over from the old trim: the shift is a recorded no-op on Turbo-quantized KV layers (warned at startup), and VLM requests are exempt from the whole machinery, as upstream exempts multimodal.

The same issue closed the batching and slot-count entries: `--parallel` now accepts b10621's whole domain with `-1` (auto) as the default, resolving to 4 slots exactly as upstream's own auto does (its `kv_unified` half is tracked on `--kv-unified`, #1473); `--batch-size` is classified `aliased` onto `--prefill-chunk-size` (same logical prefill batch, different default: 512 against upstream's 2048); `--ubatch-size` is `not_applicable` (no physical micro-batch exists on unified memory; the value is accepted, noticed at startup, and reported in `/props`); and `--swa-full` is `not_applicable` and refused when set, because sliding-window families build their own ring caches from the checkpoint's `sliding_window` and the state operations the flag purchases upstream are gated on scheduler-owned caches, not ring size. The `-np` and `-ub` single-dash spellings now rewrite through the argv pre-pass.

### The prompt cache covers every text route (#1473)

Before #1473 `--cache-prompt` and the per-request `cache_prompt` field switched a cache that only the chat-shaped routes reached: `/v1/chat/completions`, `/v1/messages` and `/v1/responses` looked prefixes up and donated them back, while `/v1/completions` and native `/completion` built no prompt-cache request context at all and therefore re-prefilled a shared prefix on every request whatever the flag said. Both raw-prompt routes now build one through the same seam, keyed on the model id and a fixed raw-prompt template signature so a raw prompt can never adopt a chat-rendered prefix (the two raw routes do share one bucket with each other, which is correct: same tokenization, same absence of a template). `cache_prompt` is declared on the native `/completion` schema, where a b10621 client sends it, and `false` withholds the context, which skips both the lookup and the donate-back rather than only the lookup. `/v1/completions` has no such field, so the server-wide switch governs there, and it now reports `usage.prompt_tokens_details.cached_tokens` on both the non-streaming response and the streaming usage chunk, exactly as the chat routes do.

Measured on `qwen3-0.6b-4bit` with a 410-token prompt: `/completion` reports `timings.cache_n` 0 / `prompt_n` 410 cold and 400 / 10 on the repeat, `cache_prompt: false` returns to 0 / 410 with byte-identical text, and a fourth request hits again, so the opt-out does not disturb the store. `tests/prompt_cache_compat_e2e.rs` covers both new routes with the same cold / hit / opt-out / hit-again contract the chat route already had.

`--cache-ram` is `aliased` onto `--prompt-cache-capacity-bytes` (MiB with upstream's `-1` and `0` sentinels; a requested value is honored exactly, and only the unset default differs, 2048 MiB against upstream's 8192). `--cont-batching` is `supported`: `-cb` and `-nocb` rewrite through the argv pre-pass, and `--no-cont-batching` now stops prefill/decode interleaving outright rather than only pinning the decode width, because the scheduler's mixed-step gate is anded with `max_batch_size > 1`. That matches upstream's own gate, which adds pending prompts to the batch only when `params_base.cont_batching || batch.size() == 0`.

### Control vectors are rejected, not ignored (#1449)

b10621's `--control-vector` and `--control-vector-scaled` load activation-steering vectors and add them to the model's residual stream. mlxcel has no control-vector application path, and accepting a vector without applying it would silently change what a deployment believes it is serving, so both flags parse (hidden) and any configured vector fails startup before the model load with a diagnostic naming logit-level steering (`--lang-bias`, per-request `logit_bias`) as the nearest mlxcel feature; a zero scale is still a configured vector set and is rejected as a whole rather than partially applied. `--control-vector-layer-range START END` alone is accepted as inert, because without vectors it configures nothing, exactly as upstream. Since only the no-vector configuration can ever serve, prompt/KV cache reuse across control-vector configurations is impossible by construction.

### Resumable streams and chat-completion control (#1444)

b10621's WebUI can close a tab mid-generation and reopen it without losing the answer: a streaming request that carries an `X-Conversation-Id` header keeps a server-side copy of its SSE bytes, and three routes manage that copy. mlxcel implements the same lifecycle with upstream's constants (4 MiB buffer with front-drop, 300 second post-completion TTL, 60 second GC) on every streaming surface: `/chat/completions` and `/v1/chat/completions`, `/v1/completions`, the native `/completion` and `/completions`, `/v1/responses` and `/v1/messages`.

- `GET /v1/stream?conv_id=<id>&from=N` replays the buffered bytes from byte offset `N`, then follows live output until the generation finishes. The replayed bytes are the exact on-wire SSE frames, so what a client received before disconnecting is byte-for-byte a prefix of the replay and resuming from that offset yields exactly the remainder. An offset that fell below the dropped buffer prefix answers upstream's 400 "Stream offset lost, please restart".
- `POST /v1/streams/lookup` with `{"conversation_ids": [...]}` reports `{conversation_id, is_done, total_bytes, started_at, completed_at}` for the ids the caller already knows, exact or `<id>::<model>` variants; the server never lists sessions it was not asked about.
- `DELETE /v1/stream?conv_id=<id>` is the explicit Stop: idempotent 204, cancels the generation through the same scheduler token a disconnect uses, and evicts the buffer.

While a session is attached, a client disconnect deliberately does NOT cancel the generation; it runs to its natural stop with its output committed to the buffer, which is what makes the resume lossless. Without the header, nothing changes: a disconnect still aborts the sequence, and no session exists to replay.

Realtime control rides the same identifiers. A chat or completions request that sets `reasoning_control: true` arms a runtime force-end flag on the sequence's thinking tracker, and `POST /v1/chat/completions/control` with `{"id": "<chatcmpl-...>", "action": "reasoning_end"}` sets it. The next sampled reasoning token is replaced by the close of the thinking block, exactly as an exhausted `--reasoning-budget` closes it; events already streamed are unaffected. The response contract is upstream's: 400 for a missing id or an unknown action, 200 `{"success": false, "message": ...}` for an unknown id or an unarmed completion, 200 `{"success": true}` when the flag is set.

One deliberate divergence keeps these entries out of `supported`, recorded in the manifest as `by_design` (kind `policy`, revisitable only by deciding that all configured keys form a single trust domain as upstream's do): with more than one `--api-key` configured, sessions and control entries are scoped to the key that created them. Another key's `GET` answers the same 404 as an unknown id, its lookup omits the session, its `DELETE` is a 204 no-op, and its control request answers "no active completion", so none of the endpoints is an existence oracle across keys. b10621 treats every configured key as equivalent; #1444 requires that one key cannot inspect or control another key's stream, and the manifest records the difference instead of hiding it. With zero or one key the behavior is identical to upstream. mlxcel additionally caps retained completed sessions at 256 (oldest evicted first), so replay buffers stay bounded under a churn of distinct conversation ids.

The OpenAI Responses retrieve/cancel surface (`GET /v1/responses/:id`, `POST /v1/responses/:id/cancel`) is unchanged and remains a separate mechanism: no translation between it and the stream lifecycle is claimed.
### Health, props, slots, metrics, and slot persistence (#1440)

`/health` and `/v1/health` now behave exactly like b10621's: the ready answer is `200 {"status": "ok"}` and nothing else, the not-ready answer is the upstream `503 {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}` envelope, and load no longer changes the answer. **A deployment that watched `/health` for the old `503 {"status": "no slot available"}` saturation signal must move to `GET /slots?fail_on_no_slot=1`**, which answers b10621's `503 no slot available` envelope when no slot is idle; a probe that parsed the old rich health body (batch gauges, observability counters) finds that data on `GET /slots` and `GET /metrics` now.

`GET /props` is mounted unconditionally, as upstream mounts it, and reports the b10621 key set: `default_generation_settings` is now the upstream `{params, n_ctx}` shape rather than a flat map (**a client that read `default_generation_settings.temperature` must read `default_generation_settings.params.temperature`**), and the payload carries the model identity (`model_alias`, `model_ftype`, `model_path`), `modalities`, `media_marker`, the three endpoint toggles, `ui` / `ui_settings`, `chat_template` and `chat_template_caps`, `bos_token` / `eos_token`, `build_info`, `is_sleeping` and `cors_proxy_enabled`. mlxcel's own resolved-configuration keys stay as top-level extensions (`kv_cache_mode`, `kv_bits`, `speculative`, `capabilities`, and the context/batch geometry under `geometry`). `--props` now gates `POST /props` alone, which acknowledges `{"success": true}` without changing anything, exactly upstream's handler.

`GET /slots` reports b10621's slot objects from a real per-request slot registry: a generation request occupies the lowest free of `--parallel` slots from its first prefill or decoded token until it finishes, so `id`, `is_processing`, `id_task`, the prompt-token split (`n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`), the resolved `params`, and the `next_token` progress block are live values, and the native `/completion` responses now carry the real `id_slot` instead of `-1`. `prompt` and `generated` stay redacted unless `LLAMA_SERVER_SLOTS_DEBUG` is set, which is b10621's own debug gate. A disabled endpoint (`--no-slots`) answers upstream's 501 diagnostic instead of the former 404, as do `/metrics` without `--metrics` and the slot actions without `--slot-save-path`. A slot is claimed on the request's first progress signal rather than when the route accepts it: binding at route entry let a request that was still waiting in the scheduler queue hold a slot away from one the worker was serving, and with `--parallel 2` and five concurrent streams that left one request emitting every frame with `id_slot: -1`, which b10621 never does because its task waits for a slot before it starts. What remains recorded as a deliberate difference is the `params` key subset, which follows `/props`: a key mlxcel does not act on is omitted rather than invented.

`--slot-save-path DIR` enables `POST /slots/:id_slot?action=save|restore|erase` with upstream's response schemas, filename validation (`fs_validate_filename` ported rule for rule), atomic writes, and canonical-path confinement that refuses traversal and symlink escapes; restores are bound to the saving model and tokenizer identity. What is persisted is the slot's token stream, not its KV cache: a restore rehydrates tokens and the next request re-prefills (or adopts from the server's own prompt cache), which is the entry's recorded divergence.

`GET /metrics` opens with the b10621 `llamacpp:` counter and gauge families name-for-name, carries the `Process-Start-Time-Unix` header, and averages the throughput gauges over the window between two scrapes, so a Prometheus scrape config written for llama-server works unchanged; the `mlxcel_` families follow in the same body. It answers during idle sleep as well, along with `/props`, `/models` and `/health`.

### Vertex AI (GCP) custom-container compatibility (#1456)

b10621 serves Google Cloud Vertex AI custom containers purely from the `AIP_*` environment variables, and mlxcel now does the same on both server binaries. With `AIP_MODE=PREDICTION`: `AIP_HTTP_PORT` (default 8080) overrides `--port` with a logged warning, `AIP_HEALTH_ROUTE` (leading slash ensured) becomes a GET alias of the health handler, and `AIP_PREDICT_ROUTE` (default `/predict`) mounts the prediction fan-out; a predict path colliding with a registered route fails startup before the model load. With `AIP_MODE` unset nothing is registered and the variables are inert.

`POST /predict` accepts `{"instances": [...]}` with at most 128 entries. Each instance names its target in `@requestFormat`, either the camelCase alias of a registered route (`chatCompletions`, `embeddings`, `applyTemplate`, ...) or a registered path verbatim; the field is stripped, a `stream` field is forced off with a warning, and the remainder is dispatched through the composed router in-process with the predict request's own headers, so API-key authentication, request validation, and queue admission apply exactly as they do to a direct call. Results come back as `{"predictions": [...]}` in request order, with per-instance failures as error objects in their slots. The alias table is derived from the same route inventory the router itself mounts, so a newly added route is aliased automatically.

Two internals differ without changing any response, and are recorded in the manifest notes rather than as divergences: instance execution is bounded to eight concurrent dispatches (upstream launches every instance at once; order is preserved either way), and alias collisions such as `completions` (`/completions` native vs `/v1/completions` OpenAI) resolve deterministically to the `/v1` route by registration order, where upstream iterates an unordered map and its winner is unspecified. Neither the predict route nor the health alias is a public endpoint, exactly as in b10621: with API keys configured, Vertex AI must present the key.

The `AIP_*` variables are documented in [`environment-variables.md`](environment-variables.md).

### Web UI, tools, MCP, CORS proxy, and agent mode are refused, not imitated (#1435)

Everything in this group exists for b10621's embedded browser UI: `--ui` / `--webui` serve the SvelteKit bundle at `/`, `--path` replaces it with a directory, `--tools` and `--tools-runtime` expose server-executed tools to that UI through `/tools` (upstream's own developer documentation marks the endpoint UI-internal), `--mcp-servers-config` / `--mcp-servers-json` feed the same endpoint from stdio MCP child processes, `--ui-mcp-proxy` opens the generic `/cors-proxy` URL proxy so the browser can reach remote MCP servers, and `--agent` turns the proxy and every tool on at once. mlxcel ships no web UI and executes nothing server-side on a model's behalf, so the whole group is classified `not_applicable`: nothing is aliased, nothing is accepted and ignored.

Concretely, on both server binaries: `--no-ui`, `--no-webui`, `--no-agent`, `--no-ui-mcp-proxy` and `--no-webui-mcp-proxy` are accepted as inert (they ask for the state the server is permanently in, which is also upstream's default for the last three), and every enabling form (`--ui`, `--webui`, `--ui-config`, `--ui-config-file`, `--path`, `--tools`, `--tools-runtime`, `--mcp-servers-config`, `--mcp-servers-json`, `--ui-mcp-proxy`, `--webui-mcp-proxy`, `--agent`, and the single-dash `-ag`) fails startup with a one-line diagnostic naming the supported alternative, before the model load. The value-taking options bind their `LLAMA_ARG_*` variables through clap; the three bool pairs (`LLAMA_ARG_UI`, `LLAMA_ARG_UI_MCP_PROXY`, `LLAMA_ARG_AGENT`) resolve at runtime with b10621's `parse_bool_value` rules and the `LLAMA_ARG_NO_*` alias, so a falsy value is the inert form and a truthy one reaches the same refusal the flag does.

`GET`/`POST /tools` and `GET`/`POST /cors-proxy` are mounted as b10621's own disabled-feature stub: 403 with `{"error":{"message":"this feature is disabled","type":"feature_disabled"}}`, byte-shaped like upstream's `res_403`. A client of a llama-server deployment that had these features off sees the identical answer here; a deployment that had them on is the recorded divergence. Because the CORS proxy is refused rather than implemented, its SSRF surface (loopback, link-local, metadata-service and DNS-rebinding reachability) does not exist here by construction. Client-declared `tools` in chat-completion requests are unrelated and keep working: mlxcel parses model output for tool calls, it just never executes anything server-side. A server-side MCP tool loop expressed through the OpenAI Responses built-in tool contract is tracked separately in #1457 as a product feature, outside b10621 compatibility.
### `--models-dir` is router mode; the store root moved to `--model-store-root` (#1438)

`--sleep-idle-seconds` is served since #1440, and it is the entry that took the manifest's last `deferred` slot. Upstream's sleep destroys the context and the model after the idle window, snapshots `/props`, `/models` and `/metrics`, and blocks every other route until a request wakes it and `load_model` finishes. `-1` (the default) and any negative value disable it; `0` arms it, which is upstream's own `< 0` gate.

mlxcel sleeps the serving **worker** rather than swapping the model provider. The scheduler's idle arm already blocks on the request channel, so a configured window turns that into a bounded wait whose expiry returns from the loop; the worker then drops the scheduler, which frees the weights along with the `LoadedModel` it owns, clears the prompt-prefix cache (a separate handle whose detached KV blocks would otherwise survive the sleep and defeat half its purpose), parks on the same channel, and on the next request reloads and hands that request to the rebuilt scheduler. Waking is therefore that request's own prefill, exactly as upstream's blocking wake is, and nothing in flight can be lost because reaching the idle arm already means no decode batch and no queued prefill. `/health` keeps answering 200 through the sleep, as upstream's does. Measured on a real checkpoint with `--sleep-idle-seconds 5`: `is_sleeping` flips true after the window and false again 0.39 s after a completion request.

One difference is recorded as `by_design`: **more routes stay reachable during sleep than upstream's four, and they answer without waking the server.** b10621 caches `/props`, `/models` and `/metrics` precisely because its own handlers read the context it just destroyed; mlxcel's read `AppState` (config, tokenizer, chat template, metrics handles), none of which the sleep frees, so they answer live with no snapshot. The same is true of `/tokenize`, `/detokenize`, `/apply-template` and the other prompt-inspection routes, which upstream has to wake the server to answer and mlxcel does not. There is nothing to make them wake for: waking exists to restore the model, and none of them reads it.

Slot save and restore persist the slot's token stream plus the model and tokenizer identity, not its KV image, and an action addressed to a slot that is processing answers `503` where upstream defers it until the slot frees. Both are recorded `by_design` with the reasoning on the manifest entries: a KV image would bind the file to the cache mode, quantization scheme and paged block size that wrote it, where a token stream restores under any of them and rebuilds the same KV by prefilling.

`--models-dir` / `LLAMA_ARG_MODELS_DIR` now carries b10621's router semantics on `mlxcel-server` and `mlxcel serve`: started without a model argument it serves the router surface, an in-process model pool where b10621 spawns child processes. The pool reconciles three sources with b10621's precedence (cache < models-dir < preset): the model cache (the mlxcel store that `--model-store-root` / `MLXCEL_MODELS_DIR` / `MLXCEL_CACHE_DIR` resolve), whose `<owner>/<name>` snapshots list as removable `source: "cache"` entries; the `--models-dir` directory, where each direct subdirectory holding a `config.json` is one model named by the directory (a symlink resolving outside the directory is skipped); and `--models-preset` INI sections. Request `model` names resolve only against the discovered registry (plus preset aliases), and the management routes sit behind the same API keys as everything else. `GET /models` (and `/v1/models`) reports the b10621 router inventory with `status`, `aliases`, `tags`, `architecture`, `source` and `can_remove`, `?reload=1` rescans, `POST /models/load` / `POST /models/unload` answer upstream's exact refusals, `--models-max` (default 4) bounds the loaded set with LRU eviction, and `--models-autoload` plus the per-request `?autoload=` switch control load-on-demand. Requests route by the JSON body's `model` on POST and `?model=` on GET, so `GET /slots?model=X` or `GET /props?model=X` reach the named model's own endpoint, exactly upstream's proxy contract.

`POST /models` downloads a new model into the cache: the name is validated (non-empty, well-formed repo id, then a synchronous metadata probe against the hub, upstream's validate-then-download order; a bare owner-less name expands to `mlx-community/<name>` first, the same expansion `-m` applies), the route answers `{"success": true}`, and a background task fetches the MLX snapshot while the entry lists as `status: "downloading"` with per-url progress. `DELETE /models?model=X` removes a cache-sourced model from disk (cancelling an in-flight download or unloading a running instance first) and answers upstream's exact refusals for everything else; deletion is containment-checked against the store root, so an HTTP-supplied name can never delete outside the cache. `GET /models/sse` streams the full `{"model", "event"[, "data"]}` vocabulary: `models_reload`, `model_status`, `status_change`, `download_progress`, `download_finished`, `download_failed`, and `model_remove`.

`--models-preset` loads b10621's INI presets: `[name]` sections define models (`model =` names a checkpoint directory, `hf-repo =` a cache snapshot) or overlay configuration onto discovered ones, `[*]` is the global preset cascaded under every model, keys accept the long-option spelling without dashes or the `LLAMA_ARG_*` environment spelling, and each section overlays a per-model clone of the router's startup config that is re-resolved through the same pipeline as the CLI (explicitly-given router CLI flags win, upstream's own overlay order). The preset-only `load-on-startup` and `dedup-cache-models` options are honored and `stop-timeout` parses inert. A key outside the translated set (`model`, `hf-repo`, `alias`, `tags`, `ctx-size`, `parallel`, `temp`, `top-k`, `top-p`, `min-p`, `seed`, `n-predict`) fails startup with a diagnostic naming the key and section rather than being silently ignored; giving `--models-preset` together with a model argument fails startup the same way, since presets steer nothing outside router mode.

**The old mlxcel meaning moved.** `--models-dir` used to name the local model-store root that `-m <owner/name>` resolution and auto-download use; that is now `--model-store-root` on both server binaries (`MLXCEL_MODELS_DIR` still works, and the `mlxcel download` / `list` / `rm` / `generate` subcommands keep their `--models-dir` spelling, which was never a llama-server surface). A server command line that combines `--models-dir` with a model argument fails startup with a diagnostic naming the replacement rather than silently picking either meaning. `--tags` sets the informational tags the `/v1/models` model object reports.

Single-model `GET /models` / `GET /v1/models` also moved to b10621's shape: the OpenAI `data` entry carries `aliases` (the full `--alias` list), `tags`, `owned_by: "llamacpp"` (it used to say `"user"`), and a `meta` block of checkpoint facts derived from `config.json` and the safetensors headers (`n_params` unpacks quantized `U32` payloads by their declared bit width), next to b10621's Ollama-compatible `models` block whose `details.format` honestly reads `"safetensors"`.

### Multi-adapter LoRA, served unfused (#1439)

`--lora` takes comma-separated adapter paths (scale 1.0 each, the mlxcel-native `--adapter` spelling stays an alias) and `--lora-scaled` takes comma-separated `FNAME:SCALE` pairs. Adapters serve **unfused**: each one contributes a low-rank term `scale * (x @ A) @ B` that the linear layers add per forward, with the user scale multiplied into the adapter's own `alpha / r`, exactly the product upstream applies. Because the base weights are never modified, the scales are live: they sit behind shared handles the serving layer writes and the layers read. A non-numeric, NaN, or infinite scale, a missing adapter directory, an adapter whose tensor shapes do not fit the base projection, and combining multi-adapter or scaled loading with tensor/pipeline parallelism all fail startup. `GET /lora-adapters` reports every adapter in b10621's entry shape (`id`, `path`, `scale`, `task_name`, `prompt_prefix`) with the live server-default scale, so a swap is visible there.

`POST /lora-adapters` applies a new scale vector at runtime, resolved by upstream's rule (listed ids set scales, unlisted drop to 0.0, unknown ids are ignored). The per-request native `lora` field resolves the same way and applies to that request only. Both carry upstream's timing: a request snapshots its scales at admission, the scheduler writes the shared handles once per batch from the executing group's snapshot, and b10621's own `can_batch_with` rule keeps a batch to one snapshot, so a swap never changes a generation already in flight and two concurrent requests naming different adapter sets cannot leak into each other. `--lora-init-without-apply` is the same mechanism at its starting point: the adapters load and validate at scale 0.0, and a later POST raises them with no restart. A term at scale 0.0 adds no operation at all, so a server holding unapplied adapters answers byte-identically to one started without them.

**`--lora-fuse` (mlxcel-native) opts back into the old behavior**: the adapters are baked into the base weights at load, which costs nothing at decode but makes the scales fixed for the process, so `POST /lora-adapters` and a non-inert per-request `lora` are refused with a diagnostic naming the flag. It is not a b10621 spelling and cannot be reached from a b10621 command line. The tensor-parallel and pipeline loaders imply it, because they shard the base weights in ways the runtime terms do not follow yet. `--no-batch` refuses adapters outright rather than silently serving base weights, since the legacy sequential worker predates both paths.

### Mirostat, dynamic temperature, adaptive-p, min_keep, logit bias, probability reports, and the seed fold (#1485)

The remaining b10621 sampling surfaces landed as Rust sampler stages; no C++ or bridge change was needed. Mirostat (`--mirostat` 1/2, tau/eta, and the native fields) replaces the whole chain while active, exactly as b10621's `common_sampler_init` skips its sampler list then: penalties, DRY, and every truncation filter are bypassed, and the per-sequence surprise target `mu` carries across steps. Dynamic temperature (`--dynatemp-range` / `--dynatemp-exp`), the `min_keep` floor, and adaptive-p run on a Rust extended chain that reproduces the fixed b10621 filter order with the parameters the fused path has no slots for; every config that uses none of them stays byte-identical on the fused path. Adaptive-p activates only when the sampler list names the `adaptive_p` stage, upstream's own rule, so `--samplers` / `--sampler-seq` / the `samplers` field now accept the fixed default order optionally extended with `adaptive_p` / `a`; every other order is still refused. `-l` / `--logit-bias` and the native `logit_bias` field land on the shared token-bias map with upstream's shapes (pairs or object, string keys tokenized, `false` as a ban), and `n_probs` (alias `logprobs`) / `post_sampling_probs` produce b10621's `completion_probabilities` report on the native route, pre-sampling from the raw logits or post-chain in linear probabilities.

Three behavior changes an existing deployment can notice. First, the seed domain folds into b10621's uint32 space on the flag and on every route's `seed` field: `-1` (and `4294967295`) stays the random sentinel, but `--seed -2`, previously random, is now the deterministic seed `4294967294`, and the pre-#1485 422 for below-minus-one field values is gone. Second, `--dry-sequence-breaker` carries b10621's semantics: when the flag is absent and DRY is enabled, upstream's default breaker set (`\n`, `:`, `"`, `*`) now applies where mlxcel previously ran DRY with no breakers; `none` is the explicit empty set; a breaker no longer needs to encode to exactly one token (startup no longer fails on multi-token breakers), because breaker token data is derived by scanning the vocabulary for tokens whose decoded text carries the breaker, upstream's `get_overlapping_token_sequences`. The native `dry_sequence_breakers` field takes a non-empty array of STRINGS now (upstream's value domain and its exact error wording); the OpenAI-shaped routes keep their mlxcel-native token-id array. Third, `/props` and `generation_settings` report the new resolved settings, and `dry_sequence_breakers` is reported as the breaker strings rather than resolved token ids.

`--temp`, `--top-k` and `--top-p` are recorded `by_design`: with the flag absent, the server default still resolves from the checkpoint's `generation_config.json` (mlx-lm parity, the deliberate mlxcel behavior); with the flag given, behavior matches b10621 exactly.

### GBNF grammars and lazy triggers (#1485)

`--grammar`, `--grammar-file`, `--json-schema` and `--json-schema-file` are served, and so are the native `json_schema` field with its `grammar` alias, `grammar_lazy`, `grammar_triggers` and `preserved_tokens` on `POST /completion` and `POST /infill`. A grammar constrains generation for real: every step's logits are masked to the tokens that keep the output parsable, and the request is refused up front if the grammar is not.

GBNF is parsed with b10621's own grammar, including the parts that are easy to get wrong from a syntax summary: `_` is not a legal identifier character while `-` and a leading digit are, `#` runs to end of line and counts as whitespace, an empty alternative is legal and load-bearing, `x{0}` erases the item, an oversized maximum such as `x{0,5000}` is silently widened to unbounded while an oversized minimum is an error, the repetition budget is multiplicative across nesting, repetition operators chain, `.` matches any code point including a newline, and left recursion is rejected across every rule rather than only the reachable ones. Token terminals work too: `<[1000]>`, `<think>` and their `!` negations become token-level elements rather than being mis-parsed as text.

Three things differ from b10621 on purpose. The GBNF source is capped at 256 KiB, and a larger grammar is refused with a diagnostic naming the limit; upstream has no source-size limit, so a large grammar built from many simple non-repeating rules is accepted there and refused here. Upstream's own guard is its multiplicative repetition budget, which bounds how far a grammar can *expand* and therefore already bounds mlxcel's 4 MiB lowered-grammar cap, but it says nothing about source bytes. Two more. A `json_schema` value is compiled through llguidance's JSON-Schema front end instead of being converted to GBNF first, so `generation_settings.grammar` is empty for a schema request where upstream reports the converted grammar text; the output conforms to the schema either way. And lazy-grammar trigger patterns are compiled with `fancy-regex` rather than `std::regex`, with a pattern it cannot compile refused with a 400 instead of throwing at sampler init; when a trigger match starts inside a token, the buffered bytes from the match offset are re-tokenized before replay, because the matcher advances by whole tokens.

Precedence follows b10621's code rather than its own field description: the schema branch is taken only when `json_schema` is present **and** `grammar` is absent, so `{"grammar": "", "json_schema": {...}}` runs with no request grammar at all and leaves any `--grammar` default in place. A non-string `grammar` is ignored the same way upstream's `json_value` falls back to its empty default. Startup failure follows upstream's split as well: a malformed `--json-schema` aborts before the server listens, because upstream converts it inside the argument handler, while a malformed `--grammar` logs a startup error and then refuses each constrained request, because upstream does not parse GBNF until a request builds its sampler.

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

1. `scripts/ci/check_llama_compat_manifest.py` (the `llama-compat manifest` CI job, `make verify-llama-compat`): offline structural validation of counts, states, issue links, shard ownership, the entry key allowlist, the `mlxcel` claim-key allowlist, the `divergence` rule, test ids, and canonical serialization. Every entry's `issue` must belong to its own shard's `pin.json` owner set, an `mlxcel` block may only use its nine recognized keys, a `supported` entry may not record a divergence, and a `by_design` entry must carry a non-empty `divergence`, `notes`, a resolving `test` pointer and a well-formed `rationale` (with `rationale` null on every other state). The CI job adds `--check-issues-open`, so a `deferred` entry pointing at a closed issue fails; `by_design` entries never feed that check, which is what lets an issue close once its only remaining divergences are recorded as permanent. `scripts/ci/check_llama_compat_manifest_test.sh` is that validator's own negative coverage: it mutates a throwaway copy of the manifest (passed via `--manifest-dir`) and asserts each rule actually rejects it, so a rule cannot degrade into one that only ever passes.
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

mlxcel's pre-#1447 behavior was `deepseek` unconditionally. The same placement applies on the disaggregated router front, which serves the same route.

The streamed form carries the delimiters too since #1470, so the concatenation of `delta.content` equals the non-streaming `message.content` byte for byte under `none` and `deepseek-legacy`. It could not before: the streaming filter consumes a thinking delimiter as it matches it, and a generation prompt that primes the block open means the model never emits an open marker at all. The filter now reports the **canonical** marker pair for every delimiter it consumes (Gemma 4's opener is matched as `<|channel>thought` but reported as `<|channel>`, which is what the non-streaming form writes), and the route re-emits those markers around the reasoning fragments; the primed case resolves its open marker from the family the primed close marker names, exactly as the non-streaming rebuild resolves it from the close marker in the raw text.

The non-streaming form changed with it, in a way a client can see: it keeps the whitespace the model emitted between its close marker and its answer, which the parser's trim used to drop. `none` exists to leave the generation unparsed, and the streamed form was already passing those bytes through, so the two shapes disagreed by exactly them. Real-checkpoint validation measured a 608-byte stream against a 606-byte `message.content` on a generation ending `</think>\n\n2 plus 2 equals 4.`; the answer itself is still trimmed under `deepseek`, where the thoughts are reported separately.

One difference stays, and it is `by_design`: `auto` resolves to `deepseek` rather than being detected from the template, because mlxcel's reasoning split uses one marker table for every family it supports and there is no second placement for a detector to choose. For those families upstream's `auto` resolves to `deepseek` as well. A third difference is gone: #1470 settled the `reasoning_in_content` question against the full b10621 tree. Upstream sets that flag in `tools/server/server-schema.cpp` and reports it from `params_to_json`, and no parser reads it, so it is a dead field at this commit and a streamed `deepseek-legacy` response carries the thoughts in both fields exactly as the help text documents, which is what mlxcel already does.

`--reasoning-budget-message` (`LLAMA_ARG_THINK_BUDGET_MESSAGE`) is injected since #1470. b10621 composes the forced run as `message_tokens + end_tag_tokens` and forces that whole sequence token by token from its FORCING state, which it also enters for a runtime `reasoning_end`, so the message appears on the exhausted-budget path and on the force-end path alike. mlxcel matches: the message is tokenized once against the model's vocabulary and forced immediately before the close tag, and a model that closes its own block within budget is untouched.

The native `--reasoning-alias-field <none|reasoning>` policy is orthogonal to that placement table. It defaults to duplicating any emitted `reasoning_content` into an identical `reasoning` field for OpenRouter-style clients; `none` keeps the b10621-compatible field alone.

The thoughts-preserving content form is rebuilt from the parser's own content plus the extracted reasoning, not from the raw text. Rebuilding matters twice over: the raw text still carries the tool-call syntax the parser removed, which would report the same call as `content` and as `tool_calls`, and the raw-text cleaning pass strips the Gemma 4 `<|channel>` delimiters that `none` exists to keep.

`--skip-chat-parsing` supersedes all of it, as it does upstream: everything the model emitted goes to `content` verbatim, reasoning and tool-call syntax included, and no `reasoning_content` or `tool_calls` is produced. Tool-call parsing is gated off with it on every path, so a call is never reported twice.

### The flags that are template kwargs

`--reasoning`, `--reasoning-effort` and `--reasoning-preserve` are not response shaping upstream: b10621 implements all three by writing `enable_thinking`, `reasoning_effort` and `preserve_reasoning` into `params.default_template_kwargs`, the same map `--chat-template-kwargs` fills. mlxcel writes the same keys into the same place, so the merge rule, the per-request override, and the template's freedom to ignore a key are shared with upstream rather than reimplemented.

clap gives no command-line order, so when `--chat-template-kwargs` names one of those keys too, the dedicated flag wins here and the collision is logged; b10621 applies whichever handler ran last. Upstream itself deprecates setting `enable_thinking` through the kwargs in favour of `--reasoning`, so the two agree whenever the flag came last. Same shape for `--chat-template` and `--chat-template-file`, which write one field upstream: the inline template wins here.

### `--chat-template` takes template text, not a name

b10621 accepts either Jinja template text or one of 54 built-in identifiers (`chatml`, `llama3`, `deepseek3`, ...). mlxcel has no built-in template library: an MLX checkpoint carries its own template in `tokenizer_config.json`, which is what mlxcel renders by default. A bare built-in name would become the template itself and every prompt would render to the literal string, so the name set is recognised and refused before the model resolves.

### What is left

No entry in this shard is `deferred` any more. `--reasoning-budget-message` became `supported` with #1470's injection, and `--reasoning-format` became `by_design` on the `auto` resolution alone once the streamed delimiters landed. Every native field in this shard other than `echo` is `not_applicable`: mlxcel's `POST /completion` is a raw-prompt endpoint with no chat template and no chat parsing for them to configure.

### Assistant prefill and `echo` (#1470)

`--prefill-assistant` is served, and it is on by default as it is upstream: when the last message of a chat request is an assistant message it is a prefix the model continues, not a turn it answers. `--no-prefill-assistant` restores the pre-#1470 behavior. Two or more trailing assistant messages are refused with b10621's own wording.

mlxcel reaches the continued prompt through the chat template rather than through per-family C++. b10621 renders `messages[:-1]`, then appends a hand-written per-family generation prompt and the continuation text; mlxcel renders `messages[:-1]` with `add_generation_prompt = true`, which *is* the family's own assistant-turn opener, and appends the continuation text with no closing tag. The result is the same prompt for the families upstream has handlers for, and a correct one for every other family mlxcel loads.

The rendering is a prompt-cache key dimension. The continued prompt is a strict prefix of the answered one, so without the dimension a boundary snapshot recorded under one rendering could be adopted by the other, which continues from a different point in the same string; the boundary render itself is skipped entirely under a prefill for the same reason.

`echo` selects whether the prefill leads the response. It is inert on `POST /completion`, which is upstream's behavior too: despite a help text about echoing input tokens, the field's only consumer primes the chat parser so a *continuation's* prefill is not re-emitted, and a raw-prompt request is never a continuation. On the chat routes `echo: true` puts the prefilled assistant text at the head of `message.content`, and of the streamed deltas.

One case is refused rather than served: a trailing assistant message carrying only `reasoning` and no `content`, where the chat template did not prime an open thinking block. The open marker is the template's, and reconstructing it would mean hard-coding a per-family marker table in the request path.

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

The existing byte and decode limits are unchanged and still apply: `--max-image-payload-size`, `--max-images`, the decoder's width, height and allocation caps, the audio per-clip and per-request ceilings, and the 10-second fetch timeout. For base64 `data:` images in JSON requests, the HTTP body extractor is sized from the configured image budget rather than Axum's 2 MiB default: `--max-image-payload-size * --max-images` is expanded by base64's 4/3 envelope and a small JSON/data-URL overhead. A request whose JSON body exceeds that derived ceiling is rejected before parsing with an OpenAI-shaped `413` error object. The derived extractor ceiling is clamped at 2 GiB so extreme image-limit settings cannot turn buffered JSON parsing into an unbounded allocation path; per-image decoding still enforces the exact configured image payload, count, dimension and allocation limits after the body has parsed.

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

**The shared route now dispatches through the loaded chat model, exactly as upstream does.** Posting the clip to `gemma3n-e2b-4bit` returns its transcript. The Whisper worker stays the implementation when no chat model can take audio, which is the Whisper-server shape b10621 cannot express at all, so it adds no divergence in any configuration upstream can reach; its own limitations are listed below.

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

### Containers

The accepted set is upstream's: **wav, mp3 and flac**, the three its mtmd audio front-end decodes. The container is identified from the clip's own magic bytes, not from the `format` string the client sends, so a mislabelled clip transcribes here exactly as it does upstream and a container outside the set is refused on content. RIFF/WAVE keeps mlxcel's in-tree reader, so every WAV clip decodes byte-for-byte as it did before; mp3 and flac go through [`symphonia`](https://github.com/pdeljanov/Symphonia) (pure Rust, MPL-2.0), enabled for those two codecs only.

### Limits

Every bound is applied before a decoder sees the clip: at most 32 multipart parts, 25 MiB per part (matching the route's body limit), and the geometry read from the container header rather than from a decode: at most 192 kHz, 8 channels and 600 seconds. A `data` chunk that declares more audio than the file carries is clamped to what is present, so an amplifying header costs a header parse. A compressed container states its length in a header orders of magnitude smaller than the samples it expands to, so the per-clip frame budget is re-checked against what the decoder actually produces: a file whose header understates its length stops at the cap instead of growing without bound. A FLAC upload additionally has its metadata-block chain walked before the decoder reads a field from it, because a 12 MiB file carrying a genuine FLAC header and a random body made the probe allocate about 3.9 GiB of resident memory, measured against a same-size control on the same warm server that never reached the probe and grew 8 MiB. That allocation is sized from fields inside the file, so the only place to stop it is ahead of the decoder; with the check in place the same twenty requests grow 2 MiB. A malformed or truncated file is a 400 naming the structural problem, answered in hundredths of a second.

### Streaming

`stream=true` on the chat-model path emits one `transcript.text.delta` per decoded token, then the `transcript.text.done` frame and `data: [DONE]`, which is upstream's granularity as well as its frame shapes. The streamed response drives its own generation rather than re-framing a finished one, so the deltas arrive as the model produces them; their concatenation is the same string the non-streaming response returns in `text`.

### The Whisper-server shape is an mlxcel extension

`mlxcel-server -m models/mlx/whisper-tiny` loads a dedicated STT worker and no chat model, which is a server shape b10621 cannot express at all: it has no speech-to-text model, and its transcription route exists only as a translation layer over a chat model. On that shape three things differ from the chat-model path, and they are limitations of the mlxcel extension rather than b10621 divergences, because there is no upstream behavior to diverge from:

- The `usage` counts report zeros: the STT worker returns a finished string and no token accounting.
- `prompt` steers nothing, for the same reason.
- A streamed response is a single delta carrying the whole transcript, because there is no token stream to split.

Everything b10621 can express goes through the chat-model path, where all three are honored.

## Regeneration

```bash
python3 scripts/compat/extract_b10621_manifest.py \
    --llama-server /path/to/llama-b10621/llama-server \
    --source-dir /path/to/llama.cpp-c1d0e7a \
    --archive /path/to/llama-b10621-bin-macos-arm64.tar.gz
```

The extractor consumes the official binary (`--help` is the authority for options) and the pinned sources (`server.cpp`, `server-http.cpp`, `server-schema.cpp` for routes and native fields). `--archive` verifies the download against the pinned SHA-256 and then requires the `--llama-server` binary to be byte-identical to a member of that verified archive, both before the binary is executed. That second half matters because the extractor runs the binary with `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` pointed at its own directory: hashing the tarball while executing an unrelated path would be assurance about a file that is never used. The archive is compared through `tarfile` in read mode and is never extracted. Omitting `--archive` runs the binary unverified and the extractor warns on stderr. Regeneration is deterministic: facts are rewritten wholesale, policy fields (`state`, `issue`, `test`, `notes`, `divergence`, `rationale`, `mlxcel`) are preserved by entry id and re-emitted in that canonical key order (a policy key an older schema never wrote is backfilled from the skeleton), `pin.json`'s `shards[name].owners` map is preserved by shard name exactly like `mlxcel_baseline` (a brand-new shard starts with an empty owner set, which needs a human before CI accepts entries in it), and running the extractor twice leaves the worktree clean. Entries that are new upstream land in `_unclassified.json`, which the validator rejects until a human classifies them, so bumping the pin to a newer nightly produces a merge-blocking, reviewable diff instead of silent drift.

## RoPE and YaRN runtime overrides

`--rope-scaling`, `--rope-scale`, `--rope-freq-base` and `--rope-freq-scale` are accepted on both server binaries with b10621's spellings, value domains and `LLAMA_ARG_ROPE_*` bindings, and they change the rotation for real. They rewrite the checkpoint's `rope_scaling` block and `rope_theta` before the model is constructed, which is before any KV cache exists, matching where `llama_model_load` applies them upstream.

| Flag | Environment | Effect |
|---|---|---|
| `--rope-scaling {none,linear,yarn}` | `LLAMA_ARG_ROPE_SCALING_TYPE` | `none` drops the checkpoint's block and rotates with the plain `base^(2i/d)` table. `linear` divides positions by `--rope-scale`, or by the checkpoint's own factor when that flag is absent. `yarn` (#1472) builds the YaRN table with the factor from `--rope-scale` / `--rope-freq-scale` and the `--yarn-*` knobs. |
| `--rope-scale N` | `LLAMA_ARG_ROPE_SCALE` | Expands context by a factor of N (`rope_freq_scale = 1/N`). |
| `--rope-freq-scale N` | `LLAMA_ARG_ROPE_FREQ_SCALE` | The reciprocal spelling of the same setting. Passing both with values that are not reciprocals is a startup error rather than a silent precedence choice. |
| `--rope-freq-base N` | `LLAMA_ARG_ROPE_FREQ_BASE` | Replaces `rope_theta`. On Gemma 3 it reaches the global-attention layers only; the sliding layers keep `rope_local_base_freq`, which is llama.cpp's separate `rope_freq_base_train_swa`. |

One composition is refused instead of being approximated: a bare `--rope-scale` or `--rope-freq-scale` on a checkpoint that declares a banded scheme (`llama3`) is a startup error, because llama.cpp multiplies its own `rope_freq_scale` into that rotation, mlxcel's banded table has no such multiplier, and dropping either half silently changes the result. Name the scheme (`--rope-scaling linear` or `--rope-scaling none`) to say which rotation you want. On a checkpoint that declares YaRN the same bare scale composes: it replaces the rotation's factor, which is where b10621 writes `rope_freq_scale` for a YaRN model.

### YaRN (#1472)

Since #1472 the shared RoPE path builds a real YaRN frequency table (the `Yarn` arm of `RopeScalingKind` in `src/models/rope_utils.rs`), porting the in-tree DeepSeek/upstream `YarnRoPE` math generalized with ggml's extrapolation mix, plus the attention-magnitude multiplier applied to Q and K before the rotation. A checkpoint on the shared path whose own `rope_scaling` block declares `yarn` now rotates with it (it previously warned and ran unscaled), and `--rope-scaling yarn` forces it, with the factor from `--rope-scale` / `--rope-freq-scale` (default: the checkpoint's own, else 1.0).

The five `--yarn-*` flags tune whatever YaRN rotation is in force, forced or checkpoint-declared, and are inert against any other rotation, exactly as they are in b10621, whose `llama_context` reads them only under `LLAMA_ROPE_SCALING_TYPE_YARN`. The sentinels (`-1.0`, and `0` for `--yarn-orig-ctx`) mean "use the values the model was trained with": the original context falls back to the block's `original_max_position_embeddings`, then `max_position_embeddings`, then upstream `YarnRoPE`'s 4096; the betas fall back to the block's, then 32 / 1; the extrapolation mix resolves to 1.0 for a YaRN rotation. One upstream quirk is mirrored deliberately: `--yarn-attn-factor` participates only when the resolved extrapolation mix is 0, because b10621's `llama-context.cpp` recomputes the attention factor over the flag whenever the mix is non-zero.

Families that implement YaRN outside the shared path (DeepSeek V2 / V3.2 / V4, gpt-oss, Mellum, TeleChat3) keep building it from their own config; a runtime override installed against one of them refuses to serve (the applications-counter check below) rather than being ignored, and a family whose own RoPE resolver cannot represent the scheme (Gemma 3's linear-only global rope, InternLM's dynamic-NTK path) refuses with a load error naming it.

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

Upstream reuses a cached chunk that is not a prefix of the incoming prompt by deleting the span between the divergence and the resumption point and shifting the rotary positions of everything after it back down. mlxcel's prompt cache reuses a strict token prefix: it adopts a cached KV set whose tokens are a prefix of the request and prefills the remainder. The gap is not a missing switch, it is a missing operation. `KVCache::trim_front_keep_sink` drops the oldest tokens by advancing `live_start` and deliberately leaves `offset` alone, and `gather_positions` / `gather_within_tail` compact surviving slots without touching the rotation already baked into each cached key. Nothing rewrites a cached key's RoPE rotation, so there is no way to express the shift, and a server that accepted the number would behave exactly as it does at `--cache-reuse 0` while the operator believed otherwise. #1473 settled this as a permanent classification rather than a pending task: building the operation needs a rotation primitive threaded with each layer's RoPE descriptor, a dequantize-rotate-requantize pass for Int8, sidecar rebuilds that the Turbo modes already decline for a plain head trim, and a second implementation against the paged block table, for a reuse class the strict-prefix cache covers in the common case. The refusal diagnostic says so instead of pointing at an issue.

### Slot state and context checkpoints are refused with a diagnostic (#1473)

`--slot-prompt-similarity` (`-sps`), `--kv-unified` (`-kvu` / `-no-kvu`), `--cache-idle-slots`, `--ctx-checkpoints` (`-ctxcp`, also spelled `--swa-checkpoints`) and `--checkpoint-min-step` (`-cms`) are declared on both server binaries and accept their inert value; a value that asks for the behavior fails startup with a diagnostic naming what is missing. Each tunes per-slot retained prompts, a shared KV buffer, or a ring of context checkpoints, and mlxcel has none of the three: reuse goes through a process-wide radix trie over token prefixes that every request consults regardless of which sequence last held them, KV is allocated per sequence through the scheduler's cache pool, and at most one history-boundary snapshot is captured per sequence. `--slot-prompt-similarity` makes the point sharpest: its upstream default is `0.10`, not `0`, so accepting it inert would mean honoring a script that passes the upstream default while no slot-selection policy exists to honor. Before #1473 they were not declared at all, and clap's unknown-argument error told an operator nothing about why.

`--kv-unified` is also the other half of `--parallel`'s remaining difference, which is recorded `by_design`: upstream's `-1` auto resolves to 4 slots AND one shared KV buffer, so a default-config single request there can use the whole context. mlxcel resolves the same 4 slots and keeps per-slot context shares. Pass `--parallel 1` for the whole-context case.

### Validating a change to the prompt cache

`tests/prompt_cache_compat_e2e.rs` is the differential gate. It runs a real server and compares per-position tokens and logprobs between a cold evaluation, a prefix-cache hit, a per-request `cache_prompt: false`, and three concurrent requests of different prompt lengths against their own solo runs. What a prompt cache has to be is not "good" but "the same as not having one", which is a statement about two runs that no single-run perplexity can express.

Two properties of that comparison decide whether it means anything. The same-width arms (cache hit, per-request disable) are gated on exact agreement, because they run at the batch width the reference did. The concurrent arm is gated on the **first** divergence and the reference's own top-two gap there, not on the total disagreement: a decode at batch width three does not run the same kernels as one at width one (issue #203), near-ties can land the other way, and once a single token flips every position after it is conditioned on different text, so counting the cascade measures the flip's echo rather than the change. On `qwen3-0.6b-4bit` the concurrent arms part at reference gaps of 0.25 and 0.125 against a decided-position threshold of 2.0, which is that jitter class and not a cache defect; the same-width arms agree on every one of 32 positions.

## Logging, introspection, and built-in presets

`logging-and-presets.json` holds four unrelated surfaces that share one owner: where and how the server logs, the two introspection options that print and exit, `--log-prompts-dir`, and b10621's twelve built-in model presets.

### Log destination

Precedence, highest first: `--log-disable`, then `--log-file PATH` on the command line, then `LLAMA_ARG_LOG_FILE`, then the mlxcel-native `LLAMA_LOG_FILE`, then standard output. `--log-disable` installs no subscriber at all, matching b10621's `common_log_pause`.

An unusable destination is a **startup failure**, never a silent fallback to the terminal. A deployment that asked for a log file and got stdout believes it has an audit trail it does not have, so `--log-file` refuses a path that is a symbolic link (a pre-planted symlink is how an unprivileged local account turns a server log into an append primitive against a file it cannot write itself), a path that is a directory, a path whose parent directory does not exist, and a path that cannot be opened for append. Each refusal names the option and the path.

That check runs where the rest of the b10621 compatibility surface is classified, before the model reference resolves, because `-m mlx-community/...` can mean a multi-gigabyte download and the subscriber itself is not installed until `start_server`. The destination is created there too, which is what b10621 does: its `--log-file` handler opens the file from inside the argument parser.

The file is created `0600`, and an existing file is tightened to `0600` on open, so re-running the server fixes permissions rather than inheriting them. Before #1448 the default umask left it `0644`: a log carrying model paths, request metadata and slot timings was readable by every account on the host.

Every line written to any sink passes through a redaction pass. `--api-key`, the contents of every `--api-key-file`, `--hf-token`, `HF_TOKEN`, `LLAMA_API_KEY`, `MLXCEL_API_KEY` and the file named by `LLAMA_ARG_API_KEY_FILE` are registered before the subscriber is installed, and any occurrence of one of those values in a log line is replaced with `[redacted]`. Redaction is line-buffered rather than per-write because the `tracing` formatter emits one event in several writes, so a value straddling two of them would slip past a per-write scan. `tests/llama_logging_presets.rs` proves the property against real processes with canary values rather than by inspection.

mlxcel logs to standard output where b10621 logs to standard error. That is a pre-existing mlxcel default that #1448 deliberately left alone; the two options this section adds that write machine-readable output to stdout, `--cache-list` and `--completion-bash`, both run and exit before any subscriber exists, so nothing interleaves.

### Log format and verbosity

| Flag | Environment | Effect |
|---|---|---|
| `--log-colors on\|off\|auto` | `LLAMA_ARG_LOG_COLORS` | ANSI escapes. `auto`, the default, colors only when the sink is a terminal, so a log file never receives escapes. Values are parsed with b10621's own case-sensitive vocabulary, not clap's wider boolish set, and anything outside it stops startup. |
| `--log-prefix` / `--no-log-prefix` | `LLAMA_ARG_LOG_PREFIX` | The per-line level tag. On by default, as upstream. |
| `--log-timestamps` / `--no-log-timestamps` | `LLAMA_ARG_LOG_TIMESTAMPS` | The per-line timestamp. On by default, as upstream. |
| `-lv N`, `--verbosity N`, `--log-verbosity N` | `LLAMA_ARG_LOG_VERBOSITY` | Threshold, default `3`. Messages above it are dropped, so a larger number is always at least as verbose. |
| `-v`, `--verbose`, `--log-verbose` | none | Every mlxcel message, unconditionally. |

Both defaults in the middle two rows were read off the pinned macOS arm64 binary rather than assumed: `llama-server` with no logging flags prints `0.00.039.303 I cmn common_param: ...`, and `--no-log-prefix --no-log-timestamps` reduces that to `cmn common_param: ...`.

Verbosity precedence, highest first: `--verbose` on the command line, then `--verbosity N` on the command line, then `RUST_LOG`, then `LLAMA_ARG_LOG_VERBOSITY`, then the compiled-in default. A command-line flag always beats the environment, which is what b10621 does and what mlxcel did not do before #1448: `EnvFilter::try_from_default_env` ran first, so `RUST_LOG=warn mlxcel-server -v` silently ignored `-v` while upstream's `-v` sets the threshold to `INT_MAX` unconditionally. Among environment variables `RUST_LOG` wins, because it is the more expressive per-target form and it is what mlxcel operators already have in their scripts.

The threshold maps onto mlxcel's own levels: `0` and `1` to `error`, `2` to `warn`, `3` to `info`, `4` to `debug`, `5` and above to `trace` for mlxcel's targets over `debug` for dependencies. `--verbose` resolves to the same filter as the top tier, so it is never less verbose than `--verbosity 5`. The top tier does not raise dependencies to `trace` on purpose: a bare `trace` directive turns on hyper and tokio internals and buries the mlxcel messages the operator asked to see.

### `--cache-list` and `--completion-bash`

Both run before any model is resolved, need no `-m`, and exit 0, as upstream's parser-level handlers do.

`--cache-list` (and the llama.cpp short spelling `-cl`) lists mlxcel's model store in b10621's exact output shape, a `number of models in cache: N` header followed by `%4zu. %s` per entry. The store is the directory `--model-store-root` / `MLXCEL_MODELS_DIR` / `MLXCEL_CACHE_DIR` resolve (`--models-dir` before #1438 renamed it on the server binaries), holding `<owner>/<name>` snapshots of MLX SafeTensors checkpoints rather than llama.cpp's GGUF cache, and each entry is printed as the repository id `-m` accepts, so the output pastes straight into the next command. A directory counts only when it actually holds a checkpoint (`config.json` or a `*.safetensors` file), so a half-finished download is not offered as a model that cannot load.

`--completion-bash` prints a source-able bash completion script generated from the live clap surface: one completion function, an `opts` list, a `case "$prev"` block giving file and directory completion to the path-valued options, then a `complete -F` line. `mlxcel-server` registers it against `mlxcel-server`; `mlxcel serve` registers it against `mlxcel` and says so in a header comment. Only **visible** arguments and **visible** aliases reach the script. mlxcel's hidden b10621 compatibility surface (`--n-gpu-layers`, `--mlock`, `--control-vector`, `--log-prompts-dir`, the presets) and the `--dump-flag-surface` machine interface are deliberately omitted; llama.cpp hides no arguments, so upstream has no equivalent choice to make. `tests/llama_logging_presets.rs` runs `bash -n` over both binaries' output and asserts both halves of that rule.

### `--log-prompts-dir` is refused

mlxcel does not write request prompts to disk, and #1448 declined to add it. Accepting the flag as a no-op would leave the named directory empty while the operator believed prompts were being captured; implementing it would put a plaintext copy of user request bodies on the log volume, which is a disclosure surface the project will not create for a debugging aid. The option is accepted by the parser, hidden, and refused before the model reference resolves, with a diagnostic naming `--log-file` plus `--verbosity 4` as the request-level debugging path that records route, slot, token counts and timings and no prompt bodies. b10621 creates the directory from inside its parser; the refusal here leaves no trace on the filesystem.

### The built-in presets are refused, with the MLX equivalent named

Each of b10621's twelve presets rewrites `params.model.hf_repo` to a **GGUF** repository under `ggml-org` and then overwrites the port, context, batch, parallelism and (for the two gpt-oss presets) sampling block. mlxcel serves MLX SafeTensors and has no GGUF reader, so neither half can be honored on its own: mapping only the checkpoint would silently drop the parameter block, and mapping only the parameter block would serve a different quantization than the flag names.

All twelve are hidden, accepted by the parser, and refused before the model reference resolves. Eleven of them print the nearest MLX checkpoint and the exact two command lines that reach it:

| Preset | Upstream GGUF | mlxcel equivalent |
|---|---|---|
| `--embd-gemma-default` | `ggml-org/embeddinggemma-300M-qat-q4_0-GGUF` | `mlx-community/embeddinggemma-300m-4bit` |
| `--fim-qwen-1.5b-default` | `ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF` | `mlx-community/Qwen2.5-Coder-1.5B-8bit` |
| `--fim-qwen-3b-default` | `ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF` | `mlx-community/Qwen2.5-Coder-3B-8bit` |
| `--fim-qwen-7b-default` | `ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF` | `mlx-community/Qwen2.5-Coder-7B-8bit` |
| `--fim-qwen-7b-spec` | same, plus a `0.5B` draft | `mlx-community/Qwen2.5-Coder-7B-8bit` with `--model-draft mlx-community/Qwen2.5-Coder-0.5B-8bit --draft-kind dflash` |
| `--fim-qwen-14b-spec` | `ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF`, plus a `0.5B` draft | `mlx-community/Qwen2.5-Coder-14B-8bit` with the same draft |
| `--fim-qwen-30b-default` | `ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF` | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` |
| `--gpt-oss-20b-default` | `ggml-org/gpt-oss-20b-GGUF` | `mlx-community/gpt-oss-20b-MXFP4-Q8` |
| `--gpt-oss-120b-default` | `ggml-org/gpt-oss-120b-GGUF` | `mlx-community/gpt-oss-120b-MXFP4-Q8` |
| `--vision-gemma-4b-default` | `ggml-org/gemma-3-4b-it-qat-GGUF` | `mlx-community/gemma-3-4b-it-qat-4bit` |
| `--vision-gemma-12b-default` | `ggml-org/gemma-3-12b-it-qat-GGUF` | `mlx-community/gemma-3-12b-it-qat-4bit` |

Those repository names appear in diagnostics only and are never resolved implicitly, so a repository that later moves degrades to a stale suggestion rather than a failed startup or, worse, a silent download of something else.

`--spec-default` is the twelfth and configures no model at all: it enables b10621's n-gram-modulo drafter, which predicts continuations from the context itself. mlxcel's drafters are checkpoint-backed, so the refusal points at `--draft-kind mtp` on an MTP-capable target and `--draft-kind dflash --model-draft <path-or-repo-id>` otherwise. The n-gram tuning knobs the preset would have set are classified under #1433 in the speculative shard.
