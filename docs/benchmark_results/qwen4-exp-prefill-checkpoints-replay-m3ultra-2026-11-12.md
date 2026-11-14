# Periodic prefill checkpoints on Qwen3.8-Flash-Next (#2885 Part B)

Replay of an agent-shaped chat workload whose history edit lands deep in a long context, with the system-prompt root on in both arms and periodic checkpoints (`MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL=1`, every 16,384 tokens) on in the second.

## Environment

| Item | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2 |
| Build | `cargo build --release --features metal,accelerate`, `main` @ `ce6855bc` plus this PR |
| Target | `models/Qwen3.8-Flash-Next-OptiQ-2bit`, `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`, `--prompt-cache-snapshot-capacity-bytes 8589934592` |
| Harness | `scripts/bench_prompt_cache_replay.py --turn-tokens 3072 --edit-turn 9 --edit-target 6 --verify`, wrapped in `scripts/with_indexers_paused.sh`, one server per arm |
| Sampling | temperature 0, 32 completion tokens per request |

The snapshot budget is raised to 8 GiB in both arms because at these lengths one snapshot of this model is 0.5 to 1.2 GB; the 2 GB default would evict the checkpoints before they are used.

## Workload

Same generator as the [root replay](qwen4-exp-system-root-replay-m3ultra-2026-11-11.md), with about 3K tokens of user text per turn. Session A grows from 8,025 to 30,890 prompt tokens over ten turns; turn 9 replaces the user message of turn 6, about 20K tokens into a 28K-token prompt. Sessions B and C are new chats with the same system prompt and tools.

## Results

| Arm | Cached tokens | Total TTFT | Total TTFT with cache reset | Snapshot hits / lookups |
|---|---|---|---|---|
| Root only | 197,007 of 268,730 | 70.18 s | 250.34 s | 15 / 16 |
| Root and checkpoints | 208,647 of 268,730 | 58.66 s | 249.73 s | 15 / 16 |

Only the edited turn changes:

| Request | Prompt tokens | Root only | Root and checkpoints |
|---|---|---|---|
| A9 (history edited at ~20K) | 28,232 | 4,744 cached, 22.16 s | 16,384 cached, 11.25 s |

Every other request adopts its own conversation's history boundary (or the root, for B1 and C1) in both arms and takes 2.6 to 3.3 s, A1 aside (7.6 s and 7.5 s cold). Splitting the history segment at 16,384 did not slow the turns that cross it (A4: 2.90 s against 3.11 s; cold A5 to A10 within 0.2 s of the root arm).

## Output identity

- A9, which adopts the 16,384 checkpoint, is byte-identical to the same request after `POST /v1/cache/reset` in both arms, and so are A1 to A4, A10, B1 to B3 and C1 to C3.
- All 16 cold completions are byte-identical across arms, although the checkpoint arm's cold prefills also split at 16,384.
- A5 to A8 differ from their cold runs in the checkpoint arm (`identical_to_cold: 12`) and matched in the root-only arm. This is not caused by the checkpoints. Each of these turns adopts the previous turn's history boundary, whose earlier rows were written by earlier turns' pieces, so its partition differs from the cold run's in both arms (see [CONTINUOUS_BATCHING.md](../CONTINUOUS_BATCHING.md#one-prefill-plan-and-when-a-prompt-cache-hit-reproduces-a-miss)). Re-running A1 to A5 against fresh servers gave the same served A5 (`9fb0627a`, cold `8e972158`) in both arms and twice within each process; at the first completion token the served run gives `The` a logprob of -0.5 against -0.125 cold. The root-only replay process that matched was the exception. #2974 traced the gap to the partition itself: it is one to three bfloat16 units in the last place of the logits, the same size as switching a cold prefill between chunk sizes (see [CONTINUOUS_BATCHING.md](../CONTINUOUS_BATCHING.md#one-prefill-plan-and-when-a-prompt-cache-hit-reproduces-a-miss)).

## Memory

Both stores ran near the 8 GiB budget with 14 entries at the end (8.20 GB with checkpoints, 8.54 GB without), averaging 586 MB and 610 MB per entry, with 8 and 7 LRU evictions. Checkpoints are not superseded within a conversation, so they compete with history-boundary snapshots under LRU; at the 2 GB default they would mostly be evicted before an edit reaches back to them. `vm_stat` was not sampled; the host has 512 GB, so this run says nothing about the macOS guardrails of #2883 on a small Mac.

Raw per-request rows: `qwen4-exp-prefill-checkpoints-replay-m3ultra-2026-11-12/replay-root.json` and `replay-ckpt.json`.
