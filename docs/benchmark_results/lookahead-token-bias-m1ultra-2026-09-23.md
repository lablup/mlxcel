# Token-biased requests on the lookahead decode pipeline (issue #1950)

Two-binary measurement of the change that lets a request carrying a token bias
(`ignore_eos`, `logit_bias`) stay on the lookahead decode pipeline instead of
falling back to the synchronous path.

- **Date:** 2026-09-23
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** `main` at `5ae51f6c` versus this branch, two separate binaries, same
  model directory and same request bodies
- **Model:** `models/mlx/c4ai-command-r7b-12-2024-4bit`
- **Machine load:** background indexers suspended with
  `scripts/with_indexers_paused.sh` for every run; load average 2 to 3

## Method

`mlxcel-server --metrics` on one port at a time, warmed with two requests (one
plain, one `ignore_eos`) that are discarded. Each measured request posts to
`/completion` with a fixed 512-token prompt (the deterministic corpus
`bench_decode.rs` synthesizes, recovered as text so both paths tokenize to the
same 512 ids), `n_predict: 128`, `temperature: 0`, `cache_prompt: false`.

Attribution guard: `mlxcel_batch_decode_lookahead_steps_total` is read before
and after every request, so each row states which decode path served it. A row
claiming a speedup without the counter moving would be measuring something else.

`predicted_per_second` is the server's own `(predicted_n - 1) / predicted_ms`.

## Decode throughput

| Binary | Request | Decode tok/s (3 runs) | Lookahead steps for 128 tokens |
|---|---|---|---|
| main | plain | 105.05, 104.96, 106.10 | 125 |
| main | `ignore_eos: true` | 98.60, 98.45, 98.83 | 0 |
| main | `logit_bias` | 98.07, 98.60 | 0 |
| branch | plain | 104.96, 105.57, 105.31 | 125 |
| branch | `ignore_eos: true` | 104.70, 103.34, 105.75 | 125 |
| branch | `logit_bias` | 104.79, 104.61 | 125 |

A biased request gains 6.2% (`ignore_eos`, median 98.60 to 104.70) and 6.5%
(`logit_bias`, 98.34 to 104.70), and lands within noise of an unbiased request
on the same binary. Unbiased requests are unchanged (105.05 to 105.31 median),
which is the expected result of a bias stage that adds no graph nodes when every
row's map is empty.

The gap this closes is larger on a build that carries the decode-only
command-buffer budget (PR #1947). That budget raises the pipelined path only, so
on 2026-09-21 with it applied the same two request shapes measured 111.0 to
113.3 against 97.0 to 97.4, a 13 to 16% gap. Both effects land on the same
requests, and neither reaches the synchronous path.

## Output equivalence

Greedy output must not move. A bias on a token the model never emits proves
nothing, so the check bans id 17939 (`Ġprompt`), which is the first token of the
unbiased continuation, and compares SHA-1 of the returned text at
`n_predict: 48`:

| Request | main (synchronous) | branch (lookahead) |
|---|---|---|
| plain | `bde985dad50c`, 45 steps | `bde985dad50c`, 45 steps |
| ban 17939 | `ef3caa8646c3`, 0 steps | `ef3caa8646c3`, 45 steps |
| ban 17939 + boost 39637 | `ef3caa8646c3`, 0 steps | `ef3caa8646c3`, 45 steps |

The bias changes the text (`" prompt tokens are dominated"` becomes `" tokens
are dominated"`), and the two paths produce the same text for the same bias,
while only the branch keeps the pipeline.

## Not measured

- Batched decode with more than one active sequence, and batches that mix biased
  and unbiased rows. The per-row delta matrix is unit-tested (`sampling.rs`,
  `apply_token_bias_rows_biases_each_row_independently`) but was not benchmarked
  under concurrent load.
- CUDA and ROCm. The gate and the fused dispatch are backend-independent, and
  the bias stage is ordinary MLX ops, but no non-Metal run was taken.
- Stochastic sampling. Every arm is greedy, so the documented batched-versus-B=1
  RNG sequencing difference is out of scope here.
