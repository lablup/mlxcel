# Penalty pipelining and server prompt lookup (GB10, 2026-10-11)

End-of-run measurement for #2229 (penalty and DRY requests on the engine's B=1 client, PR #2256) and #2255 (prompt lookup on the server, PR #2262). It sets the two values #2255 left provisional: `DEFAULT_PROMPT_LOOKUP_MAX_BATCH` and `PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT`.

## Host and method

NVIDIA GB10 (sm_121), driver 580.178.04, host `spark-102`, release profile, `--features cuda`, Qwen3-1.7B 4-bit and Llama-3.2-1B-Instruct 4-bit. Run 2026-10-10 17:33 to 21:50 UTC under the host `gpu-lock`, with no other build, test or GPU job. Three builds:

- **base**: `4c44e317` (PR #2194), pre-epic; `mlxcel generate` ran on `CxxGenerator`.
- **prev**: `c2523e5f`, main before #2229; the engine client decoded penalty requests on its synchronous loop.
- **new**: `28078507`, main after #2229 and #2255.

base and prev carry a measurement-only patch that adds `--repetition-penalty` to `mlxcel-bench-engine` (new has the flag since #2256).

Single-stream numbers come from `scripts/engine_bench_rounds.py`: 5 interleaved rounds, rotating order, the #1820 quiet-host gate before every arm, 128 generated tokens after 16 warmup tokens, and a null arm (the first arm run twice per round). Serving numbers start a fresh `mlxcel-server` per round and side, send two warmup requests, then run the drivers in `data/server-prompt-lookup-gb10-2026-10-11/`: `m_client.py` (raw `/completion`, greedy, `ignore_eos`), `m_chat.py` (streamed `/v1/chat/completions`, greedy, decode speed timed on the client because the chat response carries server timings only under speculative dispatch) and `scripts/bench_serving_concurrency.py`. Flag on and off alternate order across rounds.

## #2229: penalty requests on the CLI path

`--path cli --repetition-penalty 1.1`, decode tok/s, median over 5 rounds (range).

| Model | Prompt | new | prev | base | new without penalty | Null (new repeat) |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 256 | 189.99 (185.10..191.24) | 154.41 (149.92..157.54) | 189.71 (188.54..191.19) | 196.79 | 191.61 |
| Qwen3-1.7B | 8192 | 115.45 (111.73..116.00) | 99.22 (91.29..100.33) | 113.42 (113.13..113.99) | 117.24 | 115.64 |
| Llama-3.2-1B | 256 | 284.10 (282.89..288.21) | 216.59 (214.72..222.76) | 276.03 (274.87..276.93) | 293.55 | 284.34 |
| Llama-3.2-1B | 8192 | 218.73 (217.94..219.90) | 178.43 (173.28..183.61) | 216.03 (181.10..216.25) | 225.35 | 217.52 |

- **#2229 recovers the lost overlap.** Against prev, penalty requests decode 23 % (Qwen3, 256), 16 % (Qwen3, 8192), 31 % (Llama, 256) and 23 % (Llama, 8192) faster; every round of new is above every round of prev.
- **Against the pre-epic `CxxGenerator`**, new is equal on Qwen3 at 256 tokens (paired delta -0.41 %, inside the null range) and 1.5 to 2.7 % faster in the other three cells.
- A penalty request now costs 1.6 to 3.3 % against the same request without one, the host-side draw the pipeline cannot hide.
- TTFT is unchanged within the null range in every cell.

## #2255: server prompt lookup

### (a) Plain chat, B=1, flag on vs off

Server defaults (`--parallel 4`, so the lone sequence is on paged storage), Qwen3-1.7B, 256 tokens.

Chat through the chat template (Qwen3 reasons first, so the reply copies little of the prompt), 8 questions, 5 rounds. Decode tok/s summed over the 8 requests per round:

| Round | on | off | on vs off |
|---|---|---|---|
| 1 | 157.1 | 162.9 | -3.6 % |
| 2 | 154.6 | 160.2 | -3.5 % |
| 3 | 164.5 | 169.9 | -3.2 % |
| 4 | 153.5 | 159.5 | -3.7 % |
| 5 | 156.0 | 161.6 | -3.5 % |

**The criterion (no worse than -1.0 %) is not met: -3.49 % (-3.73..-3.22).** Per question the delta runs from -0.6 % to -6.1 %, with 7 to 51 of 26 to 113 proposed tokens accepted (about 25 to 45 percent). At that acceptance a verify round emits fewer tokens than it costs. On the server each proposal also ends the lookahead for a tick, and re-priming waits for three asks with no proposal. The draft governor (`DraftPolicy::Gated`, tuned on the CLI where a round costs only its forward) keeps such rounds running. Without the flag, decode is unchanged; the flag is opt-in. Follow-up: server-side gating that stops verify rounds at this acceptance (#2264).

Raw completion of the same questions (`/completion`, no template) copies far more of the prompt once the answer starts repeating under `ignore_eos`, and there the flag gains 16 to 79 % on three questions. The one question that proposed almost nothing (1 of 4 tokens accepted) measured -3.3 % (-8.1..+4.1), consistent with the chat result.

### (b) Copy-heavy prompt, B=1

"Repeat the following paragraph exactly", raw completion, 256 tokens, 4 requests per round, 5 rounds: on 241.5 to 245.7 tok/s, off 145.9 to 149.9 tok/s, **+53 to +67 %** per request (median of paired deltas), 172 of 249 proposed tokens accepted in 41 verify rounds (4.2 tokens per verify, counting the target's own token).

### (c) Concurrency

`--parallel 8`, 256 tokens, aggregate tok/s, median over 3 rounds (range). **off**: no flag; **on**: `--draft-kind prompt-lookup` with the default `--prompt-lookup-max-batch 2`; **mb8**: the gate opened to 8.

| Workload | Conc. | off | on | mb8 |
|---|---|---|---|---|
| `bench_serving_concurrency.py --prompt-style plain` | 2 | 138.9 (121.2..144.5) | 133.0 (130.4..133.5) | 132.1 (131.9..133.9) |
| | 4 | 246.7 (246.0..265.0) | 267.6 (253.6..275.0) | 254.7 (254.1..256.8) |
| | 8 | 119.0 (115.1..122.9) | 121.4 (118.6..124.5) | 126.2 (124.8..126.5) |
| `bench_serving_concurrency.py --prompt-style copy` | 2 | 216.8 (204.8..219.7) | 202.4 (201.4..209.3) | 200.4 (195.3..208.1) |
| | 4 | 265.3 (264.9..269.9) | 257.6 (255.6..258.7) | 229.8 (229.8..231.5) |
| | 8 | 120.0 (118.6..126.9) | 121.3 (121.1..123.4) | 128.5 (127.9..129.9) |
| `m_client.py chat` (raw completion) | 2 | 179.7 (166.4..189.8) | 210.1 (163.9..230.6) | 203.6 (185.2..219.2) |
| | 4 | 183.7 (165.1..188.0) | 187.1 (172.8..195.1) | 210.8 (188.5..232.2) |
| | 8 | 144.1 (96.6..182.2) | 143.3 (137.8..174.3) | 177.6 (173.6..183.2) |
| `m_client.py copy` (raw completion) | 2 | 164.8 (147.8..176.7) | 243.7 (230.6..263.2) | 245.7 (234.4..247.1) |
| | 4 | 173.3 (161.3..184.2) | 179.7 (169.2..243.1) | 247.4 (239.9..251.4) |
| | 8 | 140.0 (91.0..149.0) | 144.2 (136.5..152.8) | 228.8 (197.3..236.4) |

- `bench_serving_concurrency.py` sends its prompt through the chat template (Qwen3 reasons first), and its synthetic paragraph is one sentence repeated, so neither style is a clean "copy" or "plain" case.
- **Default 2 stays.** With the default gate, concurrency 4 and 8 run as without the flag in every workload (the gate is closed there except as requests drain). Opening the gate to 8 gains 15 to 63 % on raw-completion workloads at concurrency 4 and 8, but loses 13 % on the `bench_serving` copy workload at concurrency 4 (229.8 vs 265.3, ranges disjoint), so a wider default would fail the "no worse than without" criterion.
- At concurrency 2 the gate is open in both flagged arms. The chat-template workloads read 4 to 7 % below off with overlapping ranges; this is the (a) cost at B=2 and goes with #2264.

### (d) Long prompts on paged and dense storage

B=1, raw completion, 256 tokens, 3 requests per round, 3 rounds (1024 to 4096 tokens: one follow-up run with 3 rounds). `long-copy` asks to repeat a paragraph after a non-repetitive document; `long-chat` asks an open question after it. Decode tok/s, median (on / off ratio), accepted of proposed tokens.

| Storage | Prompt | long-copy on / off | long-chat on / off |
|---|---|---|---|
| paged | 1084 / 955 | 314.8 / 117.1 (2.69x), 1971 of 2034 | 153.8 / 121.0 (1.27x), 750 of 1158 |
| paged | 2090 / 1961 | 255.3 / 91.5 (2.79x), 1971 of 2034 | 154.6 / 94.3 (1.64x), 1224 of 1566 |
| paged | 4099 / 3970 | 187.5 / 132.2 (1.42x), 1989 of 2052 | 101.7 / 90.8 (1.12x), 876 of 1428 |
| paged | 8205 / 8076 | 123.7 / 108.0 (1.15x), 1989 of 2052 | 65.7 / 106.4 (0.62x), 1134 of 1923 |
| dense | 8205 / 8076 | 237.7 / 113.2 (2.10x), 1989 of 2052 | 111.3 / 112.7 (0.99x), 947 of 1539 |

- The multi-token verify forward on paged storage goes through the pool gather (ADR 0001), whose cost grows with context. At 8192 tokens it halves the copy gain against dense (1.15x vs 2.10x) and turns a 59 % acceptance prompt into a 38 % loss.
- **`PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT` is set to 4096.** Paged sequences verify up to 4096 tokens of context, where both prompt styles still gain, and decode plainly past it. Dense storage needs no limit at 8192.
- Off-arm paged decode moves non-monotonically with prompt length (91.5 at 2048, 132.2 at 4096 with one round at 77.9); the ratios are taken within the same rounds, so they are the comparison to read.

## Decisions

| Value | Before | After | Evidence |
|---|---|---|---|
| `DEFAULT_PROMPT_LOOKUP_MAX_BATCH` | 2 (provisional) | 2 | (c): a gate of 8 loses 13 % on one workload at concurrency 4 |
| `PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT` | `None` (provisional) | `Some(4096)` | (d): gains at 4096 in both styles, 0.62x at 8192 |

## Reproduce

`data/server-prompt-lookup-gb10-2026-10-11/run.sh` (single-stream and serving (a) to (d)), `run2.sh` (the paged follow-up at 1024 to 4096 tokens) and `run3.sh` (the chat-template (a) run), with `pick.sh` dispatching to the three builds (`WT_PREFIX` points at their worktrees, `MODELS` at the checkpoints) and `analyze.py` producing the tables. The first (a) chat run (`run2.sh`'s a2 block) read server timings the chat response does not carry without speculative dispatch and was discarded; `run3.sh` is its replacement with client-side timing.
