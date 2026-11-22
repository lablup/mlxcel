# Context checkpoints on Gemma 4 (issue #3255), GB10, 2026-11-23

`--ctx-checkpoints N` and `--checkpoint-min-step M` now configure the prompt cache's checkpoints instead of refusing to start. On the sliding-window families (Gemma 3, Gemma 4, AFMoE) the flag turns on ring-only turn-boundary state checkpoints: every sliding layer's ring at a history boundary, rebuilt over the conversation's newest whole snapshot with its full-attention layers cut back. This record measures what that buys on the agent-shaped replay, against `main` with the switches that already existed.

Host: NVIDIA GB10 (driver 580.178.04), CUDA build with `MLX_CUDA_ARCHITECTURES=121`. Model `gemma-4-12b-it-4bit` (`model_type` `gemma4_unified`, 48 layers, 40 sliding with a 1,024-token window, FP16 KV). Base is `main` at `4cd537cf`; the change is branch `feature/issue-3255-ctx-checkpoints` at `c84ae709`. Both are release `mlxcel-server` builds, copied aside before measuring. Every arm runs under `gpu-lock`, one server per arm, default prefill chunk (2,048 tokens).

Run setup: `scripts/bench_prompt_cache_replay.py --edit-target <n> --verify`. Session A has ten turns, and turn 8 (A8) rewrites the user message of turn `n`. Sessions B and C are new chats with the same system prompt and tools, three turns each. `--verify` resends every request after `POST /v1/cache/reset` and compares the outputs. Requests carry no session key, so all three sessions share the anonymous session. Per-turn rows are in [`turns.csv`](gemma4-ctx-checkpoints-replay-gb10-2026-11-23/turns.csv), and the end-of-run `/v1/cache/stats` are in [`cache_stats.csv`](gemma4-ctx-checkpoints-replay-gb10-2026-11-23/cache_stats.csv).

The restore point an edit of turn `n` needs is the history boundary of turn `n - 1`. That is exactly the `cached_tokens` turn `n` reported, so it is the "needed" column below.

## Results

| Arm | Configuration | Edit target | A8 cached / needed | A8 TTFT (s) | A8 equals cold | Identical to cold | ckpt inserts / hits | ckpt entries / evictions |
|---|---|---|---|---|---|---|---|---|
| `main-default-t7` | main, nothing set | 7 | 0 / 14481 | 14.83 | yes | 10/16 | 0 / 0 | 0 / 0 |
| `main-default-t6` | main, nothing set | 6 | 0 / 12853 | 14.27 | yes | 12/16 | 0 / 0 | 0 / 0 |
| `main-env-t7` | main, `KEEP_PREVIOUS_TURN=1`, `CHECKPOINT_INTERVAL=4096` | 7 | 14481 / 14481 | 1.70 | yes | 11/16 | 0 / 0 | 0 / 0 |
| `main-env-t6` | same env | 6 | 12288 / 12853 | 3.79 | yes | 13/16 | 0 / 0 | 0 / 0 |
| `main-env-t5` | same env | 5 | 8192 / 11307 | 7.70 | yes | 13/16 | 0 / 0 | 0 / 0 |
| `main-env-t3` | same env | 3 | 4096 / 7924 | 11.12 | yes | 13/16 | 0 / 0 | 0 / 0 |
| `pr-default-t7` | PR, nothing set | 7 | 0 / 14481 | 14.54 | yes | 12/16 | 0 / 0 | 0 / 0 |
| `pr-ctx3-t7` | PR, `--ctx-checkpoints 3` | 7 | 14481 / 14481 | 1.74 | yes | 12/16 | 16 / 0 | 10 / 6 |
| `pr-ctx3-t6` | PR, `--ctx-checkpoints 3` | 6 | **12853 / 12853** | 3.36 | yes | 12/16 | 16 / **1** | 11 / 5 |
| `pr-ctx3-t5` | PR, `--ctx-checkpoints 3` | 5 | 0 / 11307 | 14.73 | yes | 12/16 | 16 / 0 | 12 / 4 |
| `pr-ctx3-t3` | PR, `--ctx-checkpoints 3` | 3 | 0 / 7924 | 14.29 | yes | 12/16 | 16 / 0 | 12 / 4 |
| `pr-ctx3-cms4096-t5` | PR, `--ctx-checkpoints 3 --checkpoint-min-step 4096` | 5 | 8192 / 11307 | 7.49 | yes | 11/16 | 16 / 0 | 4 / 12 |
| `pr-ctx8-t3` | PR, `--ctx-checkpoints 8` | 3 | **7924 / 7924** | 8.94 | yes | 12/16 | 16 / **1** | 13 / 3 |

What the arms show:

- **Unset flags change nothing.** `pr-default-t7` matches `main-default-t7`: the edited turn is a full miss in both (0 cached, about 14.5 s TTFT), and every other turn reuses the same prefix.
- **An edit of the last user message (target 7) is served by keep-previous-turn, not by a ring checkpoint.** `--ctx-checkpoints 3` turns keep-previous-turn on, and the snapshot A7 restored from (14,481 tokens) is kept as a whole snapshot. A whole snapshot that reaches as far wins over a checkpoint, so `state_checkpoint_hits` stays 0 in this arm. The env switch alone does the same on `main`.
- **An edit one turn further back (target 6) is where the ring checkpoint is the only exact restore point.** On `main` nothing short of the periodic 12,288 checkpoint reaches it. With `--ctx-checkpoints 3` the request restores A5's history boundary exactly (12,853 tokens): A5's ring checkpoint is rebuilt over the conversation's newest snapshot, with `state_checkpoint_hits = 1`. The turn's output equals its cold rerun.
- **`N` is a hard per-conversation cap.** With `N = 3`, session A keeps the checkpoints of its three newest boundaries (A5 to A7), so targets 5 and 3 find nothing and miss. They do not fall back to a periodic checkpoint because `--ctx-checkpoints` does not turn those on. `--checkpoint-min-step 4096` adds the periodic points and gets target 5 the same 8,192 tokens `main` gets with the interval switch. It also spaces state checkpoints at least 4,096 tokens apart; this conversation's turns are about 1,600 tokens apart, so one checkpoint per conversation survives.
- **A larger `N` reaches further back.** With `--ctx-checkpoints 8`, target 3 restores A2's history boundary exactly (7,924 tokens, `state_checkpoint_hits = 1`). `main` with the env switches gets 4,096 tokens there, from the periodic checkpoint. The 13 checkpoints of that run (11.26 GB) are just under the 12.08 GB budget.

## Output equality (#3040 rules)

Every edited turn (A8) equals its cold rerun in every arm. The mismatches are all plain whole-snapshot chained hits (A2 to A5, B2, B3, C2), never A8, never a cold first turn, and never a checkpoint restore. The set moves from run to run on the same binary and occurs on `main` in the same places (for example `main-default-t7`: A2, A3, A4, B2, B3, C2). That is the bf16 chained-hit difference #3040 describes, not something this change introduces. The exactness of the ring restore itself is pinned by `models::gemma4_ring_checkpoint_tests` (a 4-layer fixture with a wrapped window of 8: the restored logits equal the cold run's bit for bit, and the donor's ring under the checkpoint's metadata changes them).

## Cost

A ring checkpoint is larger than the issue estimated. A multi-token append keeps the newest `window - 1` rows plus the whole new piece (`RotatingKVCache::update_concat`), so a ring captured at a history boundary holds up to `window - 1 + piece` rows. On this model that is 3,071 rows at the 2,048-token chunk, 1,006,305,280 bytes (960 MiB), against 320 MiB for the window alone. A whole snapshot here is about 1.0 GB at 14K tokens (`main-default-t7`: four snapshots, 4,178,695,424 bytes), so at these lengths a ring checkpoint saves only the full-attention share. The budget is sized from `window - 1 + prefill chunk`, 12 checkpoints capped at an eighth of the memory left after the model: 12,075,663,360 bytes on this host. A first build sized at the window alone held 4 checkpoints and evicted the rest.

Capture costs no measurable time. On the same binary, every turn except A8 has the same TTFT in `pr-default-t7` and `pr-ctx3-t7` (within 0.1 s). The checkpoint shares the boundary snapshot's buffers and is taken while that snapshot is stored. The lower TTFT totals of the `main-env` arms come from the 4,096-token periodic interval: sessions B and C reuse a shared 4,096-token prefix on their first turn (1.7 s against about 5.2 s cold). They do not come from anything `--ctx-checkpoints` adds or removes.

## Default for the sliding-window families (item 5): not shipped

`RING_FAMILY_DEFAULT_CTX_CHECKPOINTS` stays `None`, so Gemma 3, Gemma 4 and AFMoE keep today's defaults unless an operator passes the flag. The measurements do not justify a default of two checkpoints plus keep-previous-turn:

- The most common edit, the last user message, is fully served by keep-previous-turn alone. No ring checkpoint is involved.
- A ring checkpoint only serves an edit two or more turns back. Each one costs about as much as a whole snapshot at these context lengths (960 MiB against about 1 GB), so a default of two adds about 2 GB per conversation for that case.
- The checkpoint is cheaper than a whole snapshot only once the context grows well past the ring. Capturing the ring trimmed to its window (320 MiB) would change that balance, but it touches the restore's exactness path and needs its own parity proof and rerun, so it is left as a follow-up.
