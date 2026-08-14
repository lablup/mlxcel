# Post-completion prompt-cache warm-up (issue #1144)

Two-arm measurement of the background warm-up that extends a conversation's
history-boundary snapshot (issue #1143) to cover the previous assistant reply.

- **Date:** 2026-08-14
- **Hardware:** Apple M1 Ultra, macOS 26.6.1
- **Build:** `cargo build --release --features metal,accelerate`, this branch
- **Arms:** one binary, `MLXCEL_DISABLE_CACHE_WARMUP=1` versus unset
- **Machine load:** load average 4-20 across the session (parallel builds from
  other agents). Every number below is a token count or a counter, both
  insensitive to load. No wall-clock claim is made.

## Method

Three-turn conversation over the production `/v1/chat/completions` path under a
fixed `prompt_cache_key`, greedy sampling, each turn replaying the previous
assistant reply verbatim as history, with a 4 second think-time between turns so
the scheduler has the idle window a warm-up requires. Counters read from
`/v1/cache/stats` before and after every turn.

`qwen3.5-0.8b-4bit` with `chat_template_kwargs.enable_thinking = false` is the
fixture. That combination is divergence class (a) from epic #1148 (the template
injects an empty `<think>\n\n</think>\n\n` into the generation prompt), so the
end-of-generation snapshot cannot prefix the next turn and the boundary snapshot
is the only thing a turn can hit. It is therefore the only small checkpoint on
hand that can show the warm-up's contribution in isolation.

Arm attribution: `snapshot_warmups_run` can only advance when a warm-up actually
restored an ancestor snapshot, prefilled the delta, and stored the result. It is
0 in the disabled arm by construction.

## Why the fixture matters, and where falcon-h1 does not

`falcon-h1-tiny-90m-instruct-4bit` was tried first and is **not** a usable
fixture for this issue. Its replies re-tokenize canonically, so the
end-of-generation snapshot already prefixes the next turn: with the warm-up on,
turn 3 cached 372 of 397 tokens, and the warm-up's own delta was 2 tokens
because it restored the completion snapshot rather than the boundary one. The
warm-up demonstrably ran (`snapshot_warmups_run` 0 to 2) but had almost nothing
left to contribute. Reported here so the falcon numbers elsewhere in this branch
are not mistaken for evidence about this feature.

`qwen3.5-0.8b-4bit` in default thinking mode is also unusable as a fixture,
for a different reason: the model never closes its `<think>` block within any
token budget tried (110, 400, 600), so `content` comes back empty and there is
no reply text for the next turn to echo. That is a property of a 0.8b model, not
of the feature.

## Results

### qwen3.5-0.8b-4bit, `enable_thinking = false`

| Arm | Turn 2 cached / prompt | Turn 2 uncached | Turn 3 cached / prompt | Turn 3 uncached | `warmups_run` |
|---|---|---|---|---|---|
| warm-up off | 150 / 227 | 77 | 220 / 279 | 59 | 0 |
| warm-up on | 194 / 227 | 33 | 251 / 275 | 24 | 2 |

Turn 2 is the cleanest cell: identical prompt length in both arms, and the
cached prefix grows from the history boundary (150) to the history boundary plus
the previous reply (194). Uncached tokens fall 77 to 33, a 57% reduction. On
turn 3 the prompt lengths differ slightly (279 versus 275) because the replies
diverge across arms once the prefill shape changes, so uncached tokens are the
comparable figure there: 59 to 24, a 59% reduction.

`warmups_run` is 0 in the disabled arm and 2 in the enabled arm, which is what
makes this an attribution and not a coincidence.

### The construction that did not work, and why

The first implementation built the warm-up target from
`render(messages + reply, add_generation_prompt = false)` and clipped it against
a single probe render. Measured on the same fixture:

| Arm | Turn 2 cached | Turn 3 cached | `warmups_run` |
|---|---|---|---|
| warm-up off | 150 / 227 | 220 / 279 | 0 |
| warm-up on (single probe) | 153 / 227 | 223 / 275 | 2 |

**+3 tokens.** The warm-up ran and stored a snapshot, and the snapshot was
almost entirely useless. Templates render the *final* assistant message
differently from an earlier one, so the `add_generation_prompt = false` form and
the probe disagree immediately after the assistant header, and the clip threw
the reply away, keeping only `<|im_start|>assistant\n`.

Before the clip existed at all, the same construction was actively harmful:
`cached_tokens` went to **0 / 227 and 0 / 275**, worse than not warming up. A
warm-up supersedes the boundary snapshot it chains from, so storing a vector the
next turn cannot match does not merely waste a background prefill, it destroys a
working hit.

The fix is to render two probes that differ only in a trailing placeholder user
turn. Both place the reply where the next turn will place it, so their common
prefix ends exactly where the next turn's own words begin, and it contains the
reply.

### Warm-ups yield to foreground work

Two concurrent clients driving six back-to-back turns each with no think-time,
against `--parallel 4`:

| Phase | `warmups_run` | `warmups_skipped` | Foreground errors |
|---|---|---|---|
| 12 requests over 2.6s of continuous load | 0 | 10 | 0 |
| after the load stops | 2 | 10 | 0 |

Warm-ups are queued throughout and run none of them while foreground work
exists; the 10 skips are the bounded queue dropping stale jobs, which is the
intended newest-wins behavior. They drain only once the scheduler goes idle.

A counter that is pinned versus advancing is the right instrument here: it says
the same thing on a loaded box that it says on a quiet one, which a latency
percentile would not.

## Not measured

- **Wall-clock latency, TTFT, and the actual time saved.** The whole point of
  the warm-up is to move prefill off the foreground path, and this run does not
  quantify that in seconds. The box was never quiet enough for an absolute
  timing claim. What is measured is how many tokens the next turn no longer has
  to prefill, which is the mechanism; converting that to latency needs a quiet
  machine.
- **gemma-4-31b-it-4bit**, the checkpoint named in the issue's acceptance
  criteria. Out of scope for this run by size; left to the epic-level
  verification.
- **Idle-GPU cost.** A warm-up spends a background forward per turn. On an
  otherwise idle server that is free; on a server that would have slept it is
  not, and the power cost is unmeasured.
- **Behaviour under sustained partial load**, where idle windows are short but
  nonzero. The concurrency check below covers only the two extremes, saturated
  and idle.
