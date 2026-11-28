# PR #3037: Keep the previous turn's restore point

**Date**: 2026-11-12
**Status**: Completed (closes #3035, refs #2885)
**Risk Level**: Low (opt-in switch, default behavior unchanged)

## Executive Summary

An agent that replaces or compacts its last user message got no prompt-cache reuse beyond the system-prompt root on snapshot-only families. The snapshot that served that edit, the one the previous turn restored from, was removed by the session-chain supersede rule when the previous turn stored its own. `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN=1` keeps that one snapshot per session. On Qwen3.8-Flash-Next at the model-aware default budget, the edited turn goes from 4,744 cached tokens and 22.47 s TTFT to 25,804 cached tokens and 2.90 s, the same as an unedited turn.

## 1. Problem Statement

### 1.1 Background

Snapshot-only families (Qwen3.8 Flash Next, Mamba hybrids) cannot truncate recurrent state, so the prompt cache stores whole-prefix snapshots: the history boundary captured during prefill, the completion donated at finish, the opt-in system-prompt root and periodic checkpoints. Since #1146 the store supersedes a session's shorter same-origin snapshot whenever a longer one extends it, which keeps a conversation at one boundary and one completion snapshot.

### 1.2 Existing Issues

- Turn N restores from turn N-1's snapshot; storing turn N's snapshot supersedes it. A request that rewrites turn N's user message needs exactly that superseded state. Turn N's own boundary ends after the edited message and is not a prefix of the edited request.
- The model-aware default budget is sized for 6 snapshots of 8,192 tokens (2.1 GB on this model). At 28K tokens one snapshot is about 1 GB, so even retained checkpoints are evicted before use. Checkpoints served the edit only at 8 GiB, reusing 16,384 tokens (11.6 s).
- Earlier replays used scripted assistant replies, which never matched the completion snapshots. That made completions look like leaked entries and overstated snapshot pressure.

## 2. Technical Decisions

### 2.1 Retain the restore source, not a fixed depth

| Option | Pros | Cons |
|---|---|---|
| Keep the two longest same-origin ancestors (depth 2) | Simple rule in the supersede filter | Doubles both chains (boundary and completion), so 4 entries per conversation, and keeps a boundary that a completion already dominates |
| Strata-style extra message-boundary capture | Independent of lookups | Needs another prefill split and state copy per turn |
| **Chosen: record what each lookup adopted, per session, and exempt it** | Exactly the state the previous turn started from, whichever origin it was (completion with real replies, boundary when replies retokenize differently); one extra entry | Adds a per-session map to the store |

The exemption moves with each restore: when turn N+1 adopts turn N's snapshot, turn N-1's snapshot loses protection and the next insert supersedes it. Steady state is three entries per conversation.

### 2.2 Opt-in with a matching default budget

The switch costs one snapshot per conversation (about 1 GB at 28K tokens on this model), so it stays off by default. When it is on and the budget is the model-aware default, the target rises from 6 to 9 representative entries (two per conversation become three), still bounded by a quarter of free memory. Operator caps are not touched.

## 3. Implementation

- `PromptCacheConfig::keep_previous_turn`, set at startup from `keep_previous_turn_enabled()` before sizing.
- `Inner::restore_sources: HashMap<(sessionless bucket, session key), digest>`, written on a snapshot hit with a session key, read by the supersede filter, cleared by `remove_snapshot` when its entry goes and by `clear()`.
- `bounded_capacity` takes the entry target; `recommend_model_snapshot_capacity` takes it from `model_aware_snapshot_target_entries`.
- `--live-replies` in the replay harness appends each served reply to the next turn's history.

## 4. Results

| Edit | Switch off | Switch on |
|---|---|---|
| Last user message, 28,553-token prompt | 4,744 cached, 22.47 s | 25,804 cached, 2.90 s |
| Three turns back, 28,195-token prompt | 4,744 cached, 22.18 s | 4,744 cached, 22.06 s |

The store ends at 7 entries (3.08 of 3.15 GB) with the switch on and 4 (1.75 of 2.10 GB) without. Unedited outputs are identical across arms. Adopting a completion snapshot differs from the cold run in both arms; that predates this PR, because the completion tail was written by decode steps.

## 5. Change Summary

| Area | Change |
|---|---|
| Store | Per-session restore-source record and supersede exemption |
| Sizing | Entry target parameter, 9 entries with the switch |
| Startup | Switch copied into the config before sizing |
| Tests | 3 store tests, 1 sizing test |
| Docs | Env var row, CONTINUOUS_BATCHING paragraph, benchmark result with raw rows |
| Harness | `--live-replies` |

## 6. Follow-up

- An edit more than one turn back is still served only by checkpoints and a larger budget. The default budget's 8,192-token representative length is the binding limit for long agent contexts.
- Completion-snapshot restores are not byte-identical to cold prefills at temperature 0; whether that needs fixing or documenting is open.
- Small-Mac memory behavior (#2883) is unmeasured.
