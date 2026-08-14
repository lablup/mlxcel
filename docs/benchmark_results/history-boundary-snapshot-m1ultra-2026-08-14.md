# History-boundary prompt-cache snapshot (issue #1143)

Two-arm measurement of the history-boundary snapshot for snapshot-only families.

- **Date:** 2026-08-14
- **Hardware:** Apple M1 Ultra, macOS 26.6.1
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:**
  - *baseline* — `main` at 9c154ff3, binary built from the primary worktree
  - *fix* — 9c154ff3 plus this issue's branch
- **Machine load:** load average ~22 for the whole session (parallel builds from
  other agents on the same box). Every number below is a counter or a token
  count, both insensitive to load. No wall-clock claim is made here, and none
  should be read into these numbers.

## Method

`scripts`-free probe over the production `/v1/chat/completions` path: one system
message, then three user turns, each turn replaying the previous assistant reply
verbatim as history under a fixed `prompt_cache_key`. Greedy sampling
(`temperature = 0`). Counters read from `/v1/cache/stats` before and after every
turn.

Arm attribution: `snapshot_inserts` advances **once** per turn in the baseline
arm (the end-of-generation donate) and **twice** per turn in the fix arm (the
end-of-generation donate plus the history-boundary snapshot). A run that
reported one insert per turn would not be running the new path, so the table is
self-checking.

## Results

### qwen3.5-0.8b-4bit (Attention + GatedDeltaNet hybrid, snapshot-only)

The discriminating fixture: the template primes `<think>` in the generation
prompt and drops the reasoning block when the assistant turn is re-rendered as
history, which is divergence classes (a) and (b) from epic #1148.

| Arm | Turn 2 `cached_tokens` | Turn 3 `cached_tokens` | `snapshot_hits` | `snapshot_inserts` |
|---|---|---|---|---|
| baseline | 0 / 189 | 0 / 214 | 0 | 3 |
| fix | 150 / 189 | 184 / 214 | 2 | 6 |

Prompt lengths are identical across arms (155 / 189 / 214), so the two rows
describe the same conversation. The hit lengths are exactly the previous turn's
history boundary: turn 1 rendered 155 prompt tokens of which 150 are history and
5 are the generation-prompt scaffold; turn 2 rendered 189 of which 184 are
history.

### llama-3.2-1b-instruct (dense KV, longest-prefix trie) — no-regression control

| Arm | Turn 2 `cached_tokens` | Turn 3 `cached_tokens` | KV `hits` | KV `inserts` | `snapshot_inserts` |
|---|---|---|---|---|---|
| baseline | 320 / 377 | 416 / 447 | 2 / 3 | 3 | 0 |
| fix | 320 / 377 | 416 / 447 | 2 / 3 | 3 | 0 |

Identical in every cell. The boundary path is gated on
`LanguageModel::supports_snapshot_reuse()`, so a dense-KV family never reaches
it and never pays for the extra render, tokenization, or forward.

### falcon-h1-tiny-90m-instruct-4bit — not a discriminating fixture in this probe

| Arm | Turn 2 `cached_tokens` | Turn 3 `cached_tokens` | `snapshot_hits` | `snapshot_inserts` |
|---|---|---|---|---|
| baseline | 243 / 281 | 331 / 358 | 2 | 3 |
| fix | 252 / 290 | 350 / 377 | 2 | 6 |

Reported for completeness, and it does **not** support any claim about this
change. Epic #1148 chose falcon-h1 as the retokenization-drift fixture (120
sampled completion tokens versus 118 re-tokenized), but the replies this probe
happened to generate re-tokenize canonically, so the end-of-generation snapshot
already prefixed the next turn and the baseline arm hit too. The two arms also
generated different replies (different completion lengths), so the token counts
are not comparable cell to cell. The only thing this row establishes is that the
boundary snapshot is being produced (`snapshot_inserts` doubles) and that adding
it does not break a family that was already hitting.

### The kill switch reproduces the baseline exactly

Same binary as the fix arm, with `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`, same
three-turn qwen3.5-0.8b-4bit probe: turn 2 `cached_tokens` 0 / 189, turn 3
0 / 214, `snapshot_hits` 0, `snapshot_inserts` 3 (one per turn). Cell for cell
identical to the `main` baseline arm above, which is what "restores the previous
behavior" has to mean to be worth documenting.

### Greedy output is not bit-exact across the split

Third arm pair, same binary, differing only in `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT`
(`snapshot_inserts` after one turn: 1 with the split off, 2 with it on, so the
arms are confirmed distinct). Single greedy turn against qwen3.5-0.8b-4bit,
identical 155-token prompt and 120 completion tokens in both arms:

- The two outputs agree for the first 168 characters and then differ by one
  word (`**Task:**` versus `**Specific Task:**`), staying coherent and on topic
  thereafter.
- Each arm is deterministic in itself: three repeats of the split-on arm
  produced byte-identical text.

This is the documented #203 / #325 / #326 near-tie flip class, not corruption:
two forwards do not reduce in the same order as one. It is also already the
status quo on this path for two independent reasons, `--prefill-chunk-size`
splits prefills the same way and any prompt-cache hit forwards only the suffix.
Recorded here because the boundary split is on by default and a reader
comparing outputs across versions deserves to know why they moved.

## Not measured

- **Wall-clock latency and TTFT.** The box carried load average ~22 throughout;
  an absolute timing claim would not have survived it. What the boundary split
  costs on the foreground request (one extra graph launch plus one model-state
  copy, against a saved re-prefill of the whole history on the next turn) is
  open and belongs to whoever needs the latency number.
- **gemma-4-31b-it-4bit**, the rotating-attention representative named in the
  issue's acceptance criteria. Out of scope for this run by size; it is the
  orchestrator's verification.
- **Peak memory.** The boundary snapshot is captured mid-prefill, so a snapshot
  copy is resident alongside a live prefill graph for a moment. Negligible at
  these model sizes, unquantified at 31B.
- **Concurrency.** All runs were single-client. Two conversations against the
  same snapshot-only model now insert four snapshot entries per turn-pair rather
  than two; whether that pressures the snapshot byte budget is the subject of
  issue #1146 and is not measured here.
