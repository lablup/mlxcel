# PR #3283: --ctx-checkpoints and --checkpoint-min-step with ring-only state checkpoints for Gemma 3, Gemma 4 and AFMoE

**Date**: 2026-11-23
**Status**: Draft PR, not merged (closes #3255, part of #3259); verified with unit tests and an end-to-end replay of `gemma-4-12b-it-4bit` on GB10
**Risk Level**: Medium

## Executive Summary

On Gemma 4 the prompt cache served an exact multi-turn continuation but missed completely on any edit before the end of the prompt. The 1,024-token sliding ring had wrapped, so the history-boundary snapshot could not be truncated, and a client that edits the previous user message saw `cached_tokens=0` on every turn. The llama-server flags for this case, `--ctx-checkpoints` and `--checkpoint-min-step`, were refused at startup above 0.

This PR maps both flags onto checkpoint mechanisms the prompt cache already had behind environment switches (keep-previous-turn, periodic prefill checkpoints, turn-boundary state checkpoints), and gives Gemma 3, Gemma 4 (text, VL and Unified wrappers) and AFMoE ring-only state checkpoints: each sliding layer's ring at a history boundary, restored over the conversation's newest whole snapshot with its full-attention layers cut back. On the real 12B model, an edit one turn further back than the last message now restores the exact boundary (12,853 of 12,853 tokens, against 12,288 on `main` with the switches on). The proposed per-family default (issue item 5) is not shipped, because the measurements show keep-previous-turn alone already serves the common last-message edit, and two ring checkpoints would add about 2 GB per conversation to cover edits further back.

## 1. Problem Statement

### 1.1 Background

The issue's baseline was measured on GB10 with `gemma-4-12b-it-4bit` (FP16 KV, prompts of about 5.8K tokens). An exact continuation hit (5,808 of 5,823 tokens cached, TTFT down from about 4.9 s to 0.4 to 0.56 s). Any divergence before the prompt end was a full miss with `snapshot_diverged`, because a whole snapshot of a wrapped rotating cache can only be reused at its exact length. Qwen 3 under the same conditions reused up to the divergence point.

`main` already had the pieces for multiple restore points, all behind environment variables: `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL` (#2973, whole snapshots at absolute multiples of an interval of at least 256), `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` (#3035, exempts the previous turn's restore source from the session-chain supersede rule), and turn-boundary state checkpoints (#3047), which store only the state that cannot be rebuilt and take the attention rows from a longer donor snapshot of the same conversation. Only the Qwen 3.5, Qwen3-Next and Qwen4-exp families implemented the state checkpoint hooks, so Gemma had none.

### 1.2 Existing Issues

- **Flags refused.** `src/cli/slot_args.rs` rejected `--ctx-checkpoints N` and `--checkpoint-min-step M` above 0 with a message saying mlxcel takes at most one snapshot per sequence through `capture_history_boundary_snapshot`. That function no longer existed, and the statement had been false since #2973 and #3047.
- **No checkpoint hooks for the sliding-window families.** Even with every switch on, Gemma 3, Gemma 4 and AFMoE could only restore from whole snapshots, which are expensive on Gemma 4: 320 KiB per sliding token and 16 KiB per full-attention token, about 431 MB at 5.8K tokens.
- **No per-conversation bound.** State checkpoints had a byte budget and LRU eviction, but nothing limited how many one conversation kept or how close together they were.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Agent and desktop clients that rewrite earlier messages reprefill the whole context on Gemma 4 (about 14.5 s at 14K tokens) | High | High |
| Operators porting llama-server command lines hit a startup refusal on two common flags | Medium | High |
| A ring checkpoint restores wrong state and the output drifts silently | High | Low (guarded by the exactness test and restore validation) |

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/kv_ring_checkpoint.rs` (new) | Shared ring-only capture and restore. `state_checkpoint_from_snapshot` keeps only the rotating tensors and scalars of each sliding layer, returns `None` while every ring is unwrapped (the whole snapshot still truncates then), and stores nothing for KV-shared layers. `restore_sequence_state_from_checkpoint` takes the sliding layers from the checkpoint and the full-attention layers from the donor, cut back after `standard_truncatable_to`. A ring of the wrong length is refused. |
| `gemma3.rs`, `gemma4.rs`, `afmoe.rs`, `vision::{mod, gemma4_vl, gemma4_unified}` | Hooks that call the shared module, forwarded through the VL and Unified wrappers. |
| `src/server/prompt_cache/ctx_checkpoints.rs` (new) | `CtxCheckpointFlags` validation, `resolve_ctx_checkpoint_policy` against the environment switches, and `RING_FAMILY_DEFAULT_CTX_CHECKPOINTS = None`. |
| `state_checkpoints.rs` | `enforce_conversation_checkpoint_rules`: the per-conversation cap and minimum spacing, applied on insert before the byte-budget LRU pass. |
| `kv_arch.rs`, `snapshot_sizing.rs` | `sliding_window_ring_bytes_from_config` and `recommend_ring_checkpoint_capacity_from_config`; `gemma4_unified` added to the ring family match. |
| `startup.rs`, `cli_input.rs`, `scheduler/prompt_cache.rs`, `policy.rs` | Startup resolution and logging; the scheduler reads the resolved interval from the store instead of a process-wide environment read. |
| `slot_args.rs`, `cache_args.rs` | Module doc and help text rewritten; stale `capture_history_boundary_snapshot` text removed. |
| Docs and manifest | `docs/llama-server-compat.md`, `CONTINUOUS_BATCHING.md`, `environment-variables.md`, `CHANGELOG.md`; both b10621 manifest entries move from `not_applicable` to `by_design` with their divergences listed. |
| Benchmark record | `docs/benchmark_results/gemma4-ctx-checkpoints-replay-gb10-2026-11-23.md` with per-turn and cache-stats CSVs. |

36 files, +2,256 / -117 lines.

Flag semantics after this PR:

- `--ctx-checkpoints` unset keeps today's behavior. `N > 0` turns on keep-previous-turn and the family's state checkpoints and caps them at `N` per conversation. `N == 0` sets `state_checkpoint_capacity_bytes = 0` and leaves keep-previous-turn off. That also turns off the state checkpoints the hybrid Qwen families store by default, which is a behavior change for deployments that already pass `0`, so it is logged once at info. Negative values are refused.
- `--checkpoint-min-step M` with `M > 0` sets the periodic prefill checkpoint interval to `M` and keeps no two state checkpoints of one conversation closer than `M` tokens. `0 < M < 256` is refused with a message naming the floor. `0` means no periodic checkpoints and no spacing.
- The flags, and their `LLAMA_ARG_*` variables through clap, override `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` and `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`. `MLXCEL_PROMPT_CACHE_STATE_CHECKPOINT_CAPACITY_BYTES` still sets the byte budget. `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1` still turns everything off, with a warning when a flag is given. Unset flags do not adopt llama-server's defaults (32 and 8192).

## 3. Technical Decisions

### 3.1 Ring sizing deviates from the issue's `window x bytes`

The issue sized a ring checkpoint as the sliding window times the per-token bytes of the sliding layers: 320 MiB on gemma-4-12b. The first build used that formula and the budget held 4 checkpoints instead of the intended 12, so most checkpoints were evicted during the replay.

The cause is how `RotatingKVCache::update_concat` appends a multi-token piece. It keeps the newest `window - 1` prior rows and concatenates the whole new piece, so right after a prefill piece the ring holds `window - 1 + piece` rows, not `window`. A history boundary is captured at the end of a prefill piece, so the captured ring has that size. On gemma-4-12b at the default 2,048-token prefill chunk that is 3,071 rows x 327,680 bytes = 1,006,305,280 bytes (about 960 MiB) per checkpoint. `sliding_window_ring_bytes_from_config` now takes `piece_tokens` and computes `window - 1 + piece` rows per sliding layer (`piece_tokens = 1` gives the decode-shaped ring), and the budget is twelve such entries capped at an eighth of the memory left after the model: 12,075,663,360 bytes on the test host.

| Option | Pros | Cons |
|---|---|---|
| Size at `window` rows (issue) | Matches the theoretical ring | Underestimates real entries by 3x; 4 of 12 fit |
| Trim the captured ring to `window` rows | 320 MiB per checkpoint | Changes the tensors the restore installs; needs its own exactness proof and rerun |
| **Chosen: size at `window - 1 + prefill chunk` rows** | Budget matches what is stored; restore path unchanged | Each checkpoint is about as large as a whole snapshot at 14K tokens |

The budget uses `prefill_chunk_size` at startup. A chunk raised at runtime makes larger entries, so fewer than 12 fit; the byte bound still holds because eviction uses each entry's real `size_bytes`.

### 3.2 `gemma4_unified` in the family match

`gemma-4-12b-it` ships with `model_type` `gemma4_unified` (text config `gemma4_unified_text`). The ring family match listed only `gemma3`, `gemma4` and `afmoe`, so on the real checkpoint `--ctx-checkpoints` resolved a state checkpoint budget of 0 and did nothing. Unit fixtures used `gemma4` and passed. Commit `2efd6a69` adds the name and a sizing test with the Unified config.

### 3.3 The cap and the spacing run over the whole conversation lineage

`--ctx-checkpoints N` is a count per conversation, not per session. A conversation is the set of stored checkpoints in the same `SessionlessBucketKey` (model, LoRA, template, media digest, rope regime) and session as the inserted one, whose tokens are a prefix of it or that it is a prefix of. The first version only counted stored checkpoints that were prefixes of the inserted one. When requests complete out of order, a shorter checkpoint inserted after a longer one saw none of the longer ones, so a conversation could exceed `N` or keep two points closer than `M`. The byte budget still bounded memory, but the per-conversation rule did not hold. The implementation review raised this as MEDIUM, and `6343b7aa` fixed it.

`enforce_conversation_checkpoint_rules` now collects the lineage in both prefix directions, sorts it longest first (ties broken by digest), drops any checkpoint that a kept longer one on its line is within `M` tokens of, then drops the shortest beyond `N`. "Newer" means the longer boundary, not the later insertion, so the result does not depend on insertion order. A test pins this by inserting checkpoints longest first and checking the cap and spacing.

### 3.4 Known limitation: over-eviction across forked branches

Collecting both prefix directions has a cost. When a conversation forks (an edit at turn k+1 creates a sibling branch) and a shorter common ancestor is re-inserted afterwards, for example by a regenerate at turn k, both branches are on the ancestor's line. The cap then keeps the `N` longest across both branches, and one branch can lose all its checkpoints. This over-evicts and never over-retains, so memory and isolation are unaffected. The security review raised it as MEDIUM because it contradicted the doc comment's claim that one chat is not counted against another. The PR documents the behavior on `enforce_conversation_checkpoint_rules`, pins it with a test that the count stays within the cap, and defers a per-branch rule to the generic scheduler-owned checkpoint path in #3254.

### 3.5 Periodic points at absolute multiples of `M`

llama-server spaces checkpoints `M` tokens after the previous one. mlxcel keeps `main`'s absolute-multiple points (`policy.rs`, `checkpoint_points`): with `M = 4096`, the prefill splits at 4,096, 8,192, 12,288 and so on regardless of earlier requests. Spacing relative to history would make the split points depend on what the cache held when the request arrived, so a hit could no longer reproduce a cold run's prefill partition, which the #3040 output-equality comparison depends on. Absolute points also make a checkpoint shareable across conversations with a common prefix: in the replay, sessions B and C reuse a shared 4,096-token prefix on their first turn (1.7 s against about 5.2 s cold). The divergence is recorded in the manifest and in `docs/llama-server-compat.md`.

`--checkpoint-min-step` drives two separate rules. The periodic interval places whole-snapshot points at multiples of `M`. The spacing rule drops the shorter of two boundary state checkpoints on one lineage that are closer than `M`. The `--ctx-checkpoints 3 --checkpoint-min-step 4096` arm shows both: target 5 gets 8,192 tokens from a periodic point, and only one state checkpoint survives because the conversation's turns are about 1,600 tokens apart.

### 3.6 Item 5, the family default, not shipped

The issue proposed that Gemma 3, Gemma 4 and AFMoE get keep-previous-turn plus two ring checkpoints by default, conditional on the measurement showing the hit. `RING_FAMILY_DEFAULT_CTX_CHECKPOINTS` stays `None`:

- An edit of the last user message (target 7) is fully served by keep-previous-turn alone: 14,481 of 14,481 tokens, with or without ring checkpoints, and `state_checkpoint_hits = 0`.
- A ring checkpoint only adds edits two or more turns back.
- At these context lengths a ring checkpoint (960 MiB) costs about as much as a whole snapshot (about 1.0 GB at 14K tokens), so a default of two adds about 2 GB per conversation for that case.

Capturing the ring trimmed to its window (320 MiB) would change this balance, but it changes the restore's exactness path and needs its own parity proof, so it is a follow-up. The implementation review also left open whether keep-previous-turn alone should be defaulted for these families, since that half of item 5 is what serves the common case.

## 4. Measurements

GB10, `gemma-4-12b-it-4bit` (48 layers, 40 sliding with a 1,024 window, FP16 KV), `scripts/bench_prompt_cache_replay.py --edit-target n --verify`, one release server per arm under `gpu-lock`, default 2,048 prefill chunk. Base `main` at `4cd537cf`, branch at `c84ae709`. Session A has ten turns and turn 8 (A8) rewrites the user message of turn `n`; "needed" is the history boundary of turn `n - 1`.

| Arm | A8 cached / needed | A8 TTFT (s) | ckpt inserts / hits | ckpt entries / evictions |
|---|---|---|---|---|
| main, nothing set, target 7 | 0 / 14481 | 14.83 | 0 / 0 | 0 / 0 |
| main, env switches, target 7 | 14481 / 14481 | 1.70 | 0 / 0 | 0 / 0 |
| main, env switches, target 6 | 12288 / 12853 | 3.79 | 0 / 0 | 0 / 0 |
| main, env switches, target 5 | 8192 / 11307 | 7.70 | 0 / 0 | 0 / 0 |
| main, env switches, target 3 | 4096 / 7924 | 11.12 | 0 / 0 | 0 / 0 |
| PR, flags unset, target 7 | 0 / 14481 | 14.54 | 0 / 0 | 0 / 0 |
| PR, `--ctx-checkpoints 3`, target 7 | 14481 / 14481 | 1.74 | 16 / 0 | 10 / 6 |
| PR, `--ctx-checkpoints 3`, target 6 | 12853 / 12853 | 3.36 | 16 / 1 | 11 / 5 |
| PR, `--ctx-checkpoints 3`, target 5 | 0 / 11307 | 14.73 | 16 / 0 | 12 / 4 |
| PR, `--ctx-checkpoints 3 --checkpoint-min-step 4096`, target 5 | 8192 / 11307 | 7.49 | 16 / 0 | 4 / 12 |
| PR, `--ctx-checkpoints 8`, target 3 | 7924 / 7924 | 8.94 | 16 / 1 | 13 / 3 |

The env switches on `main` are `KEEP_PREVIOUS_TURN=1` and `CHECKPOINT_INTERVAL=4096`. With flags unset the PR matches `main`. With `N = 3`, targets 5 and 3 miss because the cap holds only A5 to A7 and `--ctx-checkpoints` does not turn on periodic points. With `N = 8`, 13 checkpoints (11.26 GB) fit under the 12.08 GB budget.

Acceptance criterion 2 read literally asks for checkpoint hits at the edited previous message. At target 7 the turn reaches the previous boundary through the keep-previous-turn whole snapshot, so that arm shows 0 hits; the ring checkpoint is the restore point at target 6 and, with `N = 8`, at target 3. A8 equals its cold rerun in every arm. The `--verify` mismatches are all chained whole-snapshot hits (A2 to A5, B2, B3, C2), occur in the same places on `main`, and are the bf16 differences described in #3040. Capture costs no measurable time: every non-edited turn has the same TTFT with and without `--ctx-checkpoints 3` (within 0.1 s), because the checkpoint shares the boundary snapshot's buffers by refcount.

Ring restore exactness is pinned by `models::gemma4_ring_checkpoint_tests` (ignored by default): a 4-layer fixture with a wrapped window of 8, a checkpoint at boundary 13 and an 18-token donor restores and decodes bit for bit equal to the cold run, and the negative control (the donor's ring under the checkpoint's scalars) changes the logits.

## 5. Review Outcome

The implementation review found one MEDIUM (lineage, fixed in `6343b7aa`) and one LOW (no Gemma 3 or AFMoE sizing coverage, fixed in `392b1021`). Its documentation LOW, that the byte-budget variable alone now enables ring checkpoints for these families with no per-conversation cap, was addressed in `d1661e94`. The security and performance review found no CRITICAL or HIGH items. It checked memory bounds (an entry larger than the budget is refused, eviction uses real sizes, capture shares buffers), isolation (conversation rules filter on the sessionless bucket and session, the donor rule is unchanged, multimodal requests never capture boundary snapshots), restore validation (family, length, offset, window and geometry checked; state installed only after every layer succeeds; a refused restore discards the checkpoint and retries without it), and CLI arithmetic (saturating). Its MEDIUM, the fork over-eviction, is documented and deferred (3.4). Its remaining LOWs, the startup-chunk sizing and the O(n x L) scan under the write lock with n about 12, were accepted.

## 6. Unverified and Open Items

- **The replay predates the lineage fix.** The benchmark ran at `c84ae709`; `6343b7aa` changed which checkpoints the cap and spacing evict, and the replay was not rerun. The entries and evictions columns and the cap arms describe the earlier rule. The replay completes requests in order, so the restore results are expected to hold, but this is not measured.
- **AFMoE** has hooks and sizing tests but no benchmark with a real checkpoint.
- **Gemma 3 and the Gemma 4 VL wrapper** have no end-to-end benchmark. Only the Unified 12B text path was replayed.
- **Parallel CUDA-graph aborts.** Combined runs of the `gemma3` and `gemma4` test filters at full or 4-thread width abort intermittently with `cudaStreamEndCapture ... previous error during capture`, the parallel capture failure the Makefile serializes the CUDA gate for. The same filter set passed once at full width (1,051 passed), every member passes in smaller groups or alone, and a serialized run of the changed areas passed 483 with 0 failures. Remote CI was cancelled for this epic, so no CI run covers the PR.
- **Open decisions and follow-ups.** Whether to default keep-previous-turn for the sliding-window families; the window-trimmed ring capture that would reopen item 5; a per-branch cap with #3254; a ring at a boundary that never wrapped stores no checkpoint even after that boundary's whole snapshot is superseded (as specified); with `M` larger than a turn the spacing rule leaves one state checkpoint per conversation, so `N` is effectively 1 (as specified).

## 7. Learning Points

- **Size a cache entry from what the append path stores, not from the data structure's nominal capacity.** A rotating cache with window `W` holds `W - 1 + piece` rows after a multi-token append. The 3x error showed up as a budget that held 4 entries instead of 12, which only the real-model replay with eviction counters revealed.
- **Family matching by `model_type` needs a test with the shipped config.** Fixtures that use the canonical name pass while the released checkpoint uses a variant (`gemma4_unified`) and silently gets a zero budget.
- **A per-group rule over a prefix tree must look in both directions.** Counting only ancestors makes the result depend on completion order. Counting ancestors and descendants makes it order independent but merges sibling branches through their common ancestor; the two failure modes trade over-retention for over-eviction, and choosing the one that never exceeds the bound is the safe interim.
- **Measure before shipping a default.** The issue made item 5 conditional on the hit, and the replay showed the cheaper mechanism (keep-previous-turn) already served the common case, so the costlier default was withheld with the numbers recorded next to the constant.

## Related Issues

- #3255: this issue. #3259: the parent epic.
- #3254: generic scheduler-owned checkpoints, where the per-branch cap belongs.
- #3256: paged sliding blocks, after which Gemma's sliding layers reuse without checkpoints.
- #2973, #3035, #3047: the periodic, keep-previous-turn and state checkpoint mechanisms the flags map onto.
- #3040: output-equality rules for `--verify`.
