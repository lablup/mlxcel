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

## Sharding

The manifest is sharded by area so that the concurrent implementation chains of epic #1431 edit disjoint files. Ownership is machine-readable, not prose: `pin.json`'s `shards` map records, per shard, the set of implementation issue numbers allowed to own entries in it (`shards["authentication"].owners == [1437]`, for example), and `scripts/ci/check_llama_compat_manifest.py` fails an entry whose `issue` is not a member of its own shard's owner set. That is what stops two concurrent chains from editing the same file: the file, not just the reviewer, rejects the second chain's entry.

| Shard | Chain | Owning issues |
|---|---|---|
| `transport-tls-cors.json` | A | #1432 |
| `authentication.json` | A | #1437 |
| `routes.json` | A | #1438, #1440, #1441, #1442, #1452, #1466 |
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
| `sampling-and-grammar.json` | C | #1436, #1377 |
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
