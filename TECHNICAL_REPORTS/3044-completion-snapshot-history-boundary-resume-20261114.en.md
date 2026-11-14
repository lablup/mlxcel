# Technical Report: PR #3044 - fix(prompt-cache): resume chat turns from the history boundary

**Date**: 2026-11-14
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

For snapshot-only families (Qwen 3.x hybrids, Gemma 4, the SSM hybrids) the prompt cache stored a completion snapshot at the end of generation, keyed by `prompt + reply`. Its reply rows were written by single-token decode steps, a partition no cold prefill makes, so restoring it never reproduced a cold run of the next turn: on Qwen3.8-Flash-Next with live replies, 13 of 16 `--verify` requests differed from their cold runs (#3040). A chat turn with a history boundary now stores no completion snapshot at all, so the next turn resumes from the boundary; a request without a boundary stores the prompt-end state when its prefill finishes. The replay then matches 16 of 16 at a cost of 33 to 36 re-prefilled tokens per turn (mean +0.054 s TTFT, about 2%). `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1` restores the old behavior.

---

## 1. Problem Statement

### 1.1 Background

Snapshot families cannot share KV blocks across requests because their recurrent or linear-attention state lives in model-owned buffers. The prompt cache instead copies that state at chosen points and restores it on a later request whose tokens extend the stored vector. Since #1143 a chat turn stores two snapshots: the history boundary (the conversation rendered without its generation prompt, captured mid-prefill) and the completion snapshot (donated at finish). #2974 established that a hit reproduces a cold run only when the hit forwards exactly the pieces a cold prefill of the same prompt forwards after the restore point.

### 1.2 Existing Issues

- With `--live-replies` (the model's own replies fed back unedited), every turn adopted the previous turn's completion snapshot, because it was the longest stored prefix.
- That snapshot's tail was decode-written. The model's kernels depend on the row count, so state written one row at a time is not byte-identical to state written by a prefill of the same tokens. `qwen4_exp_decode_tail_tests` pins this on the tiny fixture: decode-written state matches neither a prefill split at the prompt boundary nor a one-piece prefill, while restoring the prompt-boundary state and prefilling the reply is bit-identical to the cold split.
- The gap is rounding-sized, but greedy decoding flips as soon as two candidates are within a few bfloat16 units, so 13 of 16 replayed requests produced different text from their cold runs.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A cached turn answers differently from the same request on an empty cache | Non-reproducible output, failing `--verify` gates | High before the fix (13 of 16) |
| Re-prefilling the reply on every turn costs TTFT | Slower chats | Certain, measured at about 2% |
| `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` (#3037) loses its restore source | Edits fall back to the system-prompt root | Low: the restore source is tracked by digest, whatever its origin |

---

## 2. Technical Review

### 2.1 Security

No new input surface. The new environment variable is a boolean read once through a `OnceLock`, parsed by the existing `parse_switch`.

### 2.2 Performance

- The next turn prefills the generation prompt and the reply again: 33 to 36 tokens per turn on the replay, +0.022 to +0.098 s TTFT on the twelve unedited turns that adopt a previous turn.
- For a request without a boundary, the prompt-end copy runs in `finish_prefill` after the first token was emitted, so it adds nothing to TTFT and replaces the end-of-generation copy one for one.
- A chat turn with a boundary now copies the state once per turn instead of twice, and a conversation holds one fewer snapshot, which frees snapshot budget.

### 2.3 Compatibility & Dependencies

- Raw completions that relied on an exact `prompt + reply` entry now restore the prompt end and prefill the reply.
- Speculative bursts do not finish their prefill through the classic path, so they keep the end-of-generation snapshot.
- Requests that finish inside the prefill (immediate EOS, one-token bounds) still donate at finish, as before; their state holds only the prompt.

### 2.4 Code Quality

The decision lives in a new module, `scheduler/prompt_end_snapshot.rs`, and a `HashSet<SequenceId>` (`prompt_cache_prompt_end_stored`) tells `donate_finished_sequence_cache` to skip the finish-time copy. The set is cleared both at donate and in `release_sequence_caches`, so aborted or errored sequences cannot leak entries.

---

## 3. Technical Decisions

### 3.1 Resume from the history boundary instead of keying by the prompt

**Context**: The first commit stored the completion snapshot at the prompt end, keyed by the prompt alone, which removed the decode-written reply.

**Options Considered**:
1. Prompt-end snapshot: still 3 of 16 on the replay.
2. No completion snapshot for chat turns with a boundary: 16 of 16.
3. Keep the end-of-generation snapshot and accept divergence (old default).

**Decision**: Option 2 by default, option 3 behind `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1`.

**Rationale**: The rows between the history boundary and the prompt end are the generation prompt, forwarded as a five-row last piece that runs the small-row kernels. The next turn's cold run forwards those tokens inside its history segment, so any state that includes them reproduces no cold run. Only the boundary itself ends a piece that the next turn's cold plan also ends.

### 3.2 Decide at prefill finish, not at generation finish

The state is only prefill-written at the moment `finish_prefill` hands the sequence to decode. Deciding there (and recording the decision in a set the finish consults) avoids rewinding state later and keeps the speculative paths, which never reach that call, untouched.

### 3.3 Gate on the request having a boundary, not on capturing it

During review the gate was changed from `history_boundary_split(seq).is_some()` to `ctx.history_prefix_tokens.is_some()`. A regenerated turn adopts the boundary snapshot an earlier request of the same turn stored, so `boundary_capture_applies` declines (the adopted prefix already reaches the boundary) and the original gate fell through to a prompt-end store. That entry, containing the five-row generation-prompt piece, would then be the longest match for the next turn and reintroduce the divergence. `history_prefix_tokens` is already normalized by `resolve_history_boundary` to a strict prefix of the prompt, or `None` when the boundary feature is disabled or the request is multimodal, so it is the right predicate. The scheduler test now regenerates turn 2 and asserts that no entry is added.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
finish_prefill (classic path)
  ├─ finished inside prefill → donate_finished_sequence_cache (unchanged)
  └─ goes on to decode → insert_prompt_end_snapshot
        ├─ DECODED_TAIL=1 or not a snapshot family → nothing (old path at finish)
        ├─ record seq in prompt_cache_prompt_end_stored
        ├─ request has a history boundary → store nothing
        └─ otherwise → insert_model_state_snapshot(prompt, Completion)
donate_finished_sequence_cache
  └─ seq recorded → skip the snapshot-family copy
release_sequence_caches → remove seq from the set
```

### 4.2 Key Code Changes

- `src/server/batch/scheduler/prompt_end_snapshot.rs` (new): `insert_prompt_end_snapshot`.
- `src/server/batch/scheduler/prefill.rs`: calls it before the transition to `Decoding`.
- `src/server/batch/scheduler/prompt_cache.rs`: `donate_finished_sequence_cache` removes the record first, then skips the copy when it was set.
- `src/server/prompt_cache/decoded_tail.rs` (new): `decoded_tail_snapshot_enabled()` with a thread-local test override.
- `src/models/qwen4_exp_decode_tail_tests.rs` (new): the model-level evidence.
- `src/server/batch/scheduler_completion_snapshot_tests.rs`: prompt-end entry only, opt-out keeps the old entry, a chat turn resumes from its boundary, a regenerate stores nothing extra, and the hit decodes what a cold run decodes.

### 4.3 Data Model Changes

`BatchScheduler` gains `prompt_cache_prompt_end_stored: HashSet<SequenceId>`. No store or entry format change; completion entries keep `SnapshotOrigin::Completion`.

---

## 5. Learning Points

### 5.1 Decode and prefill write different bytes

On these kernels, writing a token by decode and writing the same token by prefill are not interchangeable at the bit level. A cache that wants cold-run identity can only reuse state whose every row was written by the same partition a cold run would use.

### 5.2 The generation prompt is part of the partition

Removing the reply was not enough because the generation-prompt scaffold also sits past the point where the next turn's plan splits. The measurement of the intermediate build (still 3 of 16) is what located that.

### 5.3 Capture and existence are different predicates

"This request captured the boundary" and "this request has a boundary" diverge exactly on regenerate, where the boundary already exists. Gates that decide what not to store should ask the second.

---

## 6. Additional Resources

### Key Keywords

- **History boundary**: the token offset where the chat history ends and the generation prompt starts (#1143).
- **Partition effect**: the dependence of model state bytes on how a prompt is split into forwards (#2974).
- **Restore source**: the snapshot a session's latest turn adopted, kept by `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` (#3035, #3037).

### Related PRs/Issues

- #3040: the issue this PR closes.
- #2974, #2975: partition effect and chained restores.
- #3035, #3037: previous-turn restore point.
- #1143, #1754: history boundary and completion snapshot accounting.

---

## 7. Change Summary

### Statistics

About 18 files: 7 source files, 3 test files, 2 docs, and a benchmark write-up with 3 raw JSON files.

### Changes by Category

- Scheduler: prompt-end decision, donate skip, record cleanup.
- Prompt cache: opt-out switch.
- Tests: model-level decode-tail evidence, scheduler-level storage and cold-identity tests.
- Docs: `CONTINUOUS_BATCHING.md`, `environment-variables.md`, replay write-up.

### Related Commits

- `fix(prompt-cache): store completion snapshots at the prompt boundary`
- `fix(prompt-cache): resume chat turns from the history boundary`
- merge of `origin/main`
- `fix(prompt-cache): skip the prompt-end snapshot on a regenerated turn`

---

## 8. Follow-up Actions

### Monitoring Required

- TTFT on long multi-turn chats with long replies, where re-prefilling the reply costs more than the 32-token replay measured.

### Future Improvements

- A chat turn that finishes inside its prefill (immediate EOS or a one-token bound) still donates a prompt-end entry with the generation-prompt piece. It is rare and was left unchanged.
- Speculative bursts still store the decode-written end-of-generation snapshot; they diverge from cold runs the same way and were not measured here.

---

## Appendix

### A. Test Results

- `models::qwen4_exp::decode_tail_tests`, `models::qwen4_exp::snapshot_tests`
- `server::batch::scheduler_completion_snapshot_tests`
- `server::prompt_cache`

### B. Performance Benchmarks

| Arm | Match cold | Unedited-turn TTFT | TTFT total |
|---|---|---|---|
| `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1` | 3 of 16 | 2.62 to 3.08 s | 70.00 s |
| prompt-end snapshot (intermediate) | 3 of 16 | 2.67 to 3.12 s | 70.61 s |
| default: resume from the boundary | 16 of 16 | 2.62 to 3.13 s | 70.54 s |

M3 Ultra, `Qwen3.8-Flash-Next-OptiQ-2bit`, `--turn-tokens 3072 --edit-turn 9 --edit-target 8 --live-replies --verify`, `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`.

### C. References

- `docs/benchmark_results/qwen4-exp-completion-snapshot-replay-m3ultra-2026-11-14.md`
- `docs/CONTINUOUS_BATCHING.md`, section on the one-prefill plan
