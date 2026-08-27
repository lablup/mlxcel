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

`pin.json` records the frozen reference, the inventory counts, and the shard map (`shards[name].owners`, the set of implementation issue numbers allowed to own entries in that shard; see [Sharding](#sharding)). Every other `*.json` file is an area shard holding entries of three kinds: `option` (one per help entry, with every accepted spelling, the environment binding, and the default), `route` (method/path), and `native_request_field` (native `/completion` schema fields with aliases). `pin.json` and every shard carry `schema_version: 2`.

Every entry carries exactly one compatibility-policy state, per the policy defined in epic #1431:

| State | Meaning |
|---|---|
| `supported` | Spelling, value domain, default, precedence, and observable behavior match. |
| `aliased` | A different mlxcel spelling, route, or request field provides equivalent behavior, with a tested translation. The claim names the mlxcel identity, which must differ from the b10621 one; when mlxcel answers the b10621 name itself the entry is `supported`, not `aliased`. |
| `not_applicable` | No MLX/CUDA equivalent; mlxcel rejects the option with an actionable diagnostic or accepts only semantically inert forms, with a test and documentation. |
| `deferred` | Not yet true; a linked implementation issue owns it. Accepted-but-ignored flags are `deferred`, never `supported`. |

Every non-`supported` entry carries a linked issue, plus `notes` and a test id where they apply. Divergences that survive the same spelling are marked explicitly in `notes`, for example the semantic collisions on `--timeout`, `--models-dir`, `--cache-type-k/v` and `POST /completions`, the inverted DRY disable sentinel on `--dry-penalty-last-n`, and the penalty-window drift on `--repeat-penalty` / `--frequency-penalty` / `--presence-penalty`. A flag mlxcel parses but does not act on is `deferred` with its acceptance pinned in the `mlxcel` claim block, never `supported`. Entries may also carry an `mlxcel` claim block describing what the current binaries already accept: `accepted_spellings`, `accepted_on_one_binary_only`, `env`, `env_binding` (`clap` or a tested `runtime` bridge, with `env_test`), `defaults`, `hidden`, `route`, `field`. That is a closed key set; the validator rejects any other key in the block, so a typo'd key fails the gate instead of silently recording a claim nothing checks. Claims are verified by CI regardless of state, so partial support cannot silently regress.

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

1. `scripts/ci/check_llama_compat_manifest.py` (the `llama-compat manifest` CI job, `make verify-llama-compat`): offline structural validation of counts, states, issue links, shard ownership, the `mlxcel` claim-key allowlist, test ids, and canonical serialization. Every entry's `issue` must belong to its own shard's `pin.json` owner set, and an `mlxcel` block may only use its nine recognized keys. The CI job adds `--check-issues-open`, so a `deferred` entry pointing at a closed issue fails.
2. `tests/llama_compat_manifest.rs`: verifies every option claim against the real clap surfaces of both `mlxcel serve` and `mlxcel-server`, hidden compatibility arguments included, via the hidden `--dump-flag-surface` machine interface, and the other direction too: an option entry carrying no claim must be accepted by neither binary, so adding a b10621 flag without flipping its entry fails. It also asserts that the sentinel itself never renders in `--help`. It contains an archive-gated full-inventory conformance test as well: set `MLXCEL_LLAMA_B10621_DIR` to the extracted official archive directory to re-derive the option inventory from the real `llama-server --help` and compare it exactly (CI skips this; it never downloads the archive).
3. `src/server/llama_compat_tests.rs`: verifies route claims against the real router and native-field claims against `NativeCompletionRequest`, in both directions. Mounting a b10621 route or accepting a b10621 field without flipping its manifest entry fails, which is what turns silent drift into a reviewable diff. An `aliased` claim is checked both ways as well: the mlxcel identity must resolve and the b10621 identity must not, so an alias cannot be mislabelled as full support.

## Regeneration

```bash
python3 scripts/compat/extract_b10621_manifest.py \
    --llama-server /path/to/llama-b10621/llama-server \
    --source-dir /path/to/llama.cpp-c1d0e7a \
    --archive /path/to/llama-b10621-bin-macos-arm64.tar.gz
```

The extractor consumes the official binary (`--help` is the authority for options) and the pinned sources (`server.cpp`, `server-http.cpp`, `server-schema.cpp` for routes and native fields). `--archive` verifies the download against the pinned SHA-256 and then requires the `--llama-server` binary to be byte-identical to a member of that verified archive, both before the binary is executed. That second half matters because the extractor runs the binary with `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` pointed at its own directory: hashing the tarball while executing an unrelated path would be assurance about a file that is never used. The archive is compared through `tarfile` in read mode and is never extracted. Omitting `--archive` runs the binary unverified and the extractor warns on stderr. Regeneration is deterministic: facts are rewritten wholesale, policy fields (`state`, `issue`, `test`, `notes`, `mlxcel`) are preserved by entry id, `pin.json`'s `shards[name].owners` map is preserved by shard name exactly like `mlxcel_baseline` (a brand-new shard starts with an empty owner set, which needs a human before CI accepts entries in it), and running the extractor twice leaves the worktree clean. Entries that are new upstream land in `_unclassified.json`, which the validator rejects until a human classifies them, so bumping the pin to a newer nightly produces a merge-blocking, reviewable diff instead of silent drift.
