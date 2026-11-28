# Keeping the previous turn's restore point on Qwen3.8-Flash-Next (#3035)

Replay of an agent-shaped chat whose turn 9 rewrites an earlier user message, with the model's own replies fed back (`--live-replies`), at the model-aware default snapshot budget. The second arm sets `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN=1`.

## Environment

| Item | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2 |
| Build | `cargo build --release --features metal,accelerate`, `main` @ `e33150cb` plus this PR |
| Target | `models/Qwen3.8-Flash-Next-OptiQ-2bit`, `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`, no snapshot capacity flag |
| Harness | `scripts/bench_prompt_cache_replay.py --turn-tokens 3072 --edit-turn 9 --edit-target {8,6} --live-replies`, one server per arm |
| Sampling | temperature 0, 32 completion tokens per request |

The model-aware default budget is 2,096,805,888 bytes with the switch off (6 representative entries of 8,192 tokens) and 3,145,208,832 bytes with it on (9 entries).

## Workload

Same generator as the [checkpoint replay](qwen4-exp-prefill-checkpoints-replay-m3ultra-2026-11-12.md): about 3K tokens of user text per turn, session A grows from 8,025 to about 31K prompt tokens over ten turns, and sessions B and C are new chats with the same system prompt and tools. With `--live-replies` each turn appends the reply the server actually generated, so every unedited turn can adopt the previous turn's completion snapshot. Turn 9 replaces the user message of turn 8 (the last user message, about 3K tokens back) or of turn 6 (about 9K back).

## Results

| Arm | Edited request | Cached tokens | TTFT | All requests cached | Final entries |
|---|---|---|---|---|---|
| Edit turn 8, switch off | A9, 28,553 tokens | 4,744 (root) | 22.47 s | 198,214 of 269,998 | 4 |
| Edit turn 8, switch on | A9, 28,553 tokens | 25,804 | 2.90 s | 219,274 of 269,998 | 7 |
| Edit turn 6, switch off | A9, 28,195 tokens | 4,744 (root) | 22.18 s | 197,856 of 269,282 | 4 |
| Edit turn 6, switch on | A9, 28,195 tokens | 4,744 (root) | 22.06 s | 197,856 of 269,282 | 7 |

- Editing the last user message: with the switch on, A9 adopts the completion snapshot of turn 7 (25,804 tokens), the one A8 restored from, and its TTFT matches an unedited turn. With it off that snapshot was superseded when A8 stored its own, and A9 fell back to the system-prompt root.
- Editing three turns back is not served by either arm: the switch keeps one restore point, and the next one back is gone. Periodic checkpoints cover that case only with a larger budget (11.6 s at 8 GiB in the checkpoint replay).
- Every other request takes 2.5 to 3.4 s in all four arms (A1 cold: 7.4 to 7.6 s), so keeping the extra snapshot did not slow the unedited turns.

## Output identity

Both edit-turn-8 arms were run again with `--verify` (every request resent after `POST /v1/cache/reset`).

- Every request except A9 and A10 returned the same completion in both arms, so keeping the extra snapshot changes no output it does not serve. A9 adopts a different snapshot in each arm, and A10 extends A9's reply.
- In both arms only the three requests that adopt the system-prompt root or nothing (A1, B1, C1) match their cold runs; the 13 requests that adopt a previous turn's completion snapshot do not, A9 with the switch on included. This predates the switch: a completion snapshot's tail was written by decode steps rather than by a prefill, so its state differs from a cold prefill of the same tokens. The scripted-reply replays never adopted completion snapshots, which is why they reported 12 matches.

## Memory

With the switch on the store ended at 7 entries, 3.08 GB of the 3.15 GB default, against 4 entries and 1.75 GB with it off. Entries average 440 MB. LRU evicted 5 to 7 entries in every arm, all from the conversation being inserted (`snapshot_self_evictions`), since at 28K tokens one snapshot of this model is about 1 GB. `vm_stat` was not sampled; the host has 512 GB, so this run says nothing about the macOS guardrails of #2883 on a small Mac, where the quarter-of-free-memory ceiling may hold the budget below 9 entries.

Raw per-request rows: `qwen4-exp-keep-previous-turn-replay-m3ultra-2026-11-12/replay-{t8,t6}-{off,on}.json` and the `--verify` runs `replay-t8-{off,on}-verify.json`.
