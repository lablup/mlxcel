# System-prompt root snapshot on Qwen3.8-Flash-Next (#2885 Part B)

Replay of an agent-shaped chat workload with the system-prompt root snapshot off (the Part A default) and on (`MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`).

## Environment

| Item | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2 |
| Build | `cargo build --release --features metal,accelerate`, `main` @ `261080ba` plus this PR |
| Target | `models/Qwen3.8-Flash-Next-OptiQ-2bit`, `mlxcel-server` defaults otherwise (prompt cache on, snapshot store 2,096,805,888 bytes) |
| Harness | `scripts/bench_prompt_cache_replay.py --verify`, wrapped in `scripts/with_indexers_paused.sh`, one server per arm |
| Sampling | temperature 0, 32 completion tokens per request |

## Workload

All 16 requests share one system prompt (about 4K tokens of repository docs) and a list of six tools. Assistant replies in the history are scripted, so both arms send byte-identical requests.

- Session A: ten turns, each adding about 1.5K tokens of user text. Turn 8 replaces the user message of turn 3 with a one-line summary (an agent compacting its history about 9K tokens back).
- Sessions B and C: new chats with the same system prompt and tools, three turns each.

No request sets `prompt_cache_key` or `user`, so all of them share the anonymous session bucket.

## Results

| Arm | Cached tokens | Total TTFT | Total TTFT with cache reset | Snapshot hits / lookups |
|---|---|---|---|---|
| Root off | 122,978 of 175,486 | 51.16 s | 161.65 s | 12 / 16 |
| Root on | 137,210 of 175,486 | 38.37 s | 162.96 s | 15 / 16 |

The requests where the arms differ (cached tokens, TTFT):

| Request | Prompt tokens | Root off | Root on |
|---|---|---|---|
| A8 (history edited) | 15,772 | 0, 14.55 s | 4,744, 10.34 s |
| B1 (new chat) | 6,191 | 0, 5.70 s | 4,744, 1.53 s |
| C1 (new chat) | 6,080 | 0, 5.60 s | 4,744, 1.44 s |

Every other request hits its own conversation's history boundary in both arms and takes 1.4 to 1.7 s. A1, the first request, runs the extra split and stores the root: 6.13 s with the root on against 6.06 s off.

The root covers 4,744 tokens (system prompt, tool list and the first user turn's header). Its snapshot is 254,080,436 bytes; the history-boundary snapshot of A1 at 6,414 tokens is 304,772,372 bytes.

## Output identity

- Within each arm, all 16 completions are byte-identical to the same request sent after `POST /v1/cache/reset` (`identical_to_cold: 16`).
- Across arms, all 16 completions are byte-identical too. The extra split at the root did not flip a greedy token on this workload. That is not guaranteed in general, since splitting a prefill can change the reduction order (see [CONTINUOUS_BATCHING.md](../CONTINUOUS_BATCHING.md#one-prefill-plan-and-when-a-prompt-cache-hit-reproduces-a-miss)).

## Memory

The store reached 2,003,438,184 bytes in 6 entries with the root on, against a 2,096,805,888-byte budget. Both arms recorded 15 LRU self-evictions: at 250 to 350 MB per entry the default budget holds about six snapshots. The root stayed resident through them, since LRU skips roots while they hold at most half of the budget. `vm_stat` was not sampled; the host has 512 GB, so this run says nothing about the macOS guardrails of #2883 on a small Mac.

Raw per-request rows: `qwen4-exp-system-root-replay-m3ultra-2026-11-11/replay-off.json` and `replay-on.json`.
