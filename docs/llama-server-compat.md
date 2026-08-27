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

## Sharding

The manifest is sharded by area so that the concurrent implementation chains of epic #1431 edit disjoint files. Ownership is machine-readable, not prose: `pin.json`'s `shards` map records, per shard, the set of implementation issue numbers allowed to own entries in it (`shards["authentication"].owners == [1437]`, for example), and `scripts/ci/check_llama_compat_manifest.py` fails an entry whose `issue` is not a member of its own shard's owner set. That is what stops two concurrent chains from editing the same file: the file, not just the reviewer, rejects the second chain's entry.

| Shard | Chain | Owning issues |
|---|---|---|
| `transport-tls-cors.json` | A | #1432 |
| `authentication.json` | A | #1437 |
| `routes.json` | A | #1441, #1442, #1452 |
| `embeddings-and-rerank.json` | A | #1452 |
| `observability-and-slots.json` | A | #1440 |
| `router-models.json` | A | #1438 |
| `lora-adapters.json` | A | #1439 |
| `model-source.json` | B | #1434 |
| `ggml-runtime.json` | B | #1445 |
| `chat-templates.json` | B | #1447 |
| `multimodal-and-audio.json` | B | #1451, #1446 |
| `ui-tools-mcp-gcp.json` | B | #1435, #1456 |
| `streams-and-realtime.json` | B | #1444 |
| `runtime-and-context.json` | C | #1450, #1453, #1449 |
| `sampling-and-grammar.json` | C | #1436, #1377 |
| `speculative.json` | C | #1433 |
| `logging-and-presets.json` | C | #1448 |

Routes and native request fields live in the shard of the issue that owns them (for example the audio-transcription routes sit in `multimodal-and-audio.json` and the native sampling fields in `sampling-and-grammar.json`), so a chain never has to edit another chain's file.

## Enforcement

Three gates hold the manifest and the binaries together; all three run in CI and in `make verify`:

1. `scripts/ci/check_llama_compat_manifest.py` (the `llama-compat manifest` CI job, `make verify-llama-compat`): offline structural validation of counts, states, issue links, shard ownership, the entry key allowlist, the `mlxcel` claim-key allowlist, the `divergence` rule, test ids, and canonical serialization. Every entry's `issue` must belong to its own shard's `pin.json` owner set, an `mlxcel` block may only use its nine recognized keys, and a `supported` entry may not record a divergence. The CI job adds `--check-issues-open`, so a `deferred` entry pointing at a closed issue fails. `scripts/ci/check_llama_compat_manifest_test.sh` is that validator's own negative coverage: it mutates a throwaway copy of the manifest (passed via `--manifest-dir`) and asserts each rule actually rejects it, so a rule cannot degrade into one that only ever passes.
2. `tests/llama_compat_manifest.rs`: verifies every option claim against the real clap surfaces of both `mlxcel serve` and `mlxcel-server`, hidden compatibility arguments included, via the hidden `--dump-flag-surface` machine interface, and the other direction too: an option entry carrying no claim must be accepted by neither binary, so adding a b10621 flag without flipping its entry fails. It also asserts that the sentinel itself never renders in `--help`. It contains an archive-gated full-inventory conformance test as well: set `MLXCEL_LLAMA_B10621_DIR` to the extracted official archive directory to re-derive the option inventory from the real `llama-server --help` and compare it exactly (CI skips this; it never downloads the archive).
3. `src/server/llama_compat_tests.rs`: verifies route claims against the real router and native-field claims against `NativeCompletionRequest`, in both directions, and restates the `divergence` rule so a `cargo test` run cannot pass a manifest `make verify` would reject. Mounting a b10621 route or accepting a b10621 field without flipping its manifest entry fails, which is what turns silent drift into a reviewable diff. An `aliased` claim is checked both ways as well: the mlxcel identity must resolve and the b10621 identity must not, so an alias cannot be mislabelled as full support.

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
