# Completion snapshots and cold-run identity on Qwen3.8-Flash-Next (#3040)

Live-replies replay of the [keep-previous-turn workload](qwen4-exp-keep-previous-turn-replay-m3ultra-2026-11-12.md) with `--verify`: every request is sent again after `POST /v1/cache/reset`, and its greedy completion is compared with the cached run's.

## Environment

| Item | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2 |
| Build | `cargo build --release --features metal,accelerate`, `main` @ `e4ed09e7` plus #3040 |
| Target | `models/Qwen3.8-Flash-Next-OptiQ-2bit`, `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`, no snapshot capacity flag, `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` unset |
| Harness | `scripts/bench_prompt_cache_replay.py --turn-tokens 3072 --edit-turn 9 --edit-target 8 --live-replies --verify`, one server per arm |
| Sampling | temperature 0, 32 completion tokens per request |

## Results

| Arm | Next turn adopts | Match cold | Unedited-turn TTFT | TTFT total |
|---|---|---|---|---|
| `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1` (end-of-generation snapshot, the old default) | prompt + reply, decode-written | 3 of 16 | 2.62 to 3.08 s | 70.00 s |
| completion snapshot keyed by the prompt (intermediate build, not shipped) | prompt end, 5 tokens past the history boundary | 3 of 16 | 2.67 to 3.12 s | 70.61 s |
| default: no completion snapshot for a turn with a history boundary | the history boundary | 16 of 16 | 2.62 to 3.13 s | 70.54 s |

- The old default reproduces the issue: only A1, B1 and C1, which adopt the system-prompt root or nothing, match their cold runs.
- Dropping the decode-written reply is not enough. The prompt-end state still holds the generation prompt, forwarded as a five-row last piece, and the next turn's cold run forwards those tokens inside its history segment. The mismatch set is unchanged.
- Resuming from the history boundary makes every request match its cold run, A9 (the edit, which adopts the system-prompt root) included.
- Cost: the next turn prefills 33 to 36 more tokens (the generation prompt and the reply). On the twelve unedited turns that adopt a previous turn, TTFT rose by 0.022 to 0.098 s, mean 0.054 s (about 2%). The edited turn and the turns that adopt the root are unchanged.

Raw rows: `qwen4-exp-completion-snapshot-replay-m3ultra-2026-11-14/replay-{decoded-tail,prompt-end,boundary-resume}-verify.json`.
