# Turn-boundary state checkpoints on hybrid models (#3047)

Live-replies replay with an edit one turn back (`--edit-target 8`) and three turns back (`--edit-target 6`) of turn 9. The off arm sets `MLXCEL_PROMPT_CACHE_STATE_CHECKPOINT_CAPACITY_BYTES=0`, which is the behavior of `main`. The on arm uses the default budget.

## Environment

| Item | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2 |
| Build | `cargo build --release --features metal,accelerate`, `main` @ `ab45ac59` plus #3047 |
| Targets | `qwen3.5-4b-4bit` at `MLXCEL_MEMORY_LIMIT=16G`, `qwen3.8-27b-4bit` at `32G`, `Qwen3.8-Flash-Next-OptiQ-2bit` with no limit |
| Server | `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1`, default snapshot capacity, one server per run under `/usr/bin/time -l` |
| Harness | `scripts/bench_prompt_cache_replay.py --turn-tokens 3072 --edit-turn 9 --edit-target {8,6} --live-replies`, with `--verify` on the first run of each cell |
| Sampling | temperature 0, 32 completion tokens per request |
| Order | off, on, off, on per cell, in one GPU-lock hold. Load average at start was 1.2 to 7.7, with one 4b run at 21.3 and one at 14.0. Spread across those reps stays within 0.4 s of total TTFT. |

## Edited turn A9

| Model, edit | Arm | A9 cached tokens | A9 TTFT (reps) | Replay TTFT total (reps) |
|---|---|---|---|---|
| 4b, 1 turn back | off | 4,706 (system root) | 17.05 to 17.38 s (3) | 43.40 to 43.96 s |
| 4b, 1 turn back | on | 25,625 (checkpoint) | 1.70 to 1.71 s (3) | 28.02 to 28.31 s |
| 4b, 3 turns back | off | 4,706 | 16.69 to 16.86 s (3) | 43.02 to 43.32 s |
| 4b, 3 turns back | on | 19,895 (checkpoint) | 5.22 to 5.25 s (3) | 31.55 to 31.78 s |
| 27b, 1 turn back | off | 4,744 | 87.04 to 87.07 s (3) | 219.36 to 219.62 s |
| 27b, 1 turn back | on | 25,885 (checkpoint) | 7.89 to 7.98 s (2) | 140.24 to 140.72 s |
| 27b, 3 turns back | off | 4,744 | 85.58 to 85.64 s (2) | 217.95 to 218.09 s |
| 27b, 3 turns back | on | 20,081 (checkpoint) | 24.96 to 24.98 s (2) | 157.29 to 157.45 s |

Every other request has the same cached-token count in both arms.

## Memory

| Model | Whole snapshots at end | Checkpoints at end (budget) | Peak footprint off / on, 1 turn back | 3 turns back |
|---|---|---|---|---|
| 4b at 16G | 1,059 MB in 3 entries (budget 1,687 MB) | 151 MB in 6 entries (590 MB) | 25.7 / 13.5 GB | 27.2 / 17.3 GB |
| 27b at 32G | 2,201 MB (budget 3,401 MB) | 449 MB (1,701 MB) | 51.7 / 33.5 GB | 52.7 / 36.4 GB |

Peak footprint comes from runs without `--verify`. It includes the model weights and the prefill activations, so the off arm peaks higher: its edited turn prefills about 23K tokens from the system root. The whole-snapshot budget and its use are unchanged. The checkpoints live in their own budget.

## Identity with cold runs

| Model, edit | off match cold | on match cold |
|---|---|---|
| 4b, 1 turn back | 14 of 16 (A6, A8 differ) | 12 of 16 (A6, A8, A9, A10) |
| 4b, 3 turns back | 14 of 16 (A6, A8) | 13 of 16 (A6, A8, A9) |
| 27b, 1 turn back | 14 of 16 (B2, B3) | 14 of 16 (B2, B3) |
| 27b, 3 turns back | 14 of 16 (B2, B3) | 14 of 16 (B2, B3) |
| Flash-Next, 1 turn back | n/a | 16 of 16 |

The off-arm mismatches are the partition effect described in `docs/CONTINUOUS_BATCHING.md`. A chained hit forwards the history in different pieces than a cold run, which moves bf16 logits by a few ULP. On the 4b model, A9 served from a checkpoint follows the same rule. A chained run that held a whole snapshot at 25,625 tokens (`MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN=1`, 128G, `qwen3.5-4b-4bit_t8_kpt128_reference.json`) produces the same 16 completions, byte for byte, as the checkpoint run, A9 and A10 included. The checkpoint reproduces the whole snapshot it replaces, and the tiny-fixture tests pin the same equality bit for bit. On Flash-Next (`qwen4_exp`) the checkpoint arm and the keep-previous-turn arm both match all 16 cold runs. The checkpoint arm holds 1,191 MB of whole snapshots plus 662 MB of checkpoints, against 2,946 MB of whole snapshots in the keep-previous-turn arm.

Raw rows: `hybrid-turn-state-checkpoints-replay-m3ultra-2026-11-15/`.
