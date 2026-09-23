# Cohere2 prefill projects only the sampled row (issue #1954)

Two-binary measurement of Cohere2 overriding the `forward_last_logits` entry
points so a prefill slices the hidden state to the sampled row before the tied
256k-vocabulary LM head and the logit scale, instead of computing every row and
keeping one.

- **Date:** 2026-09-23
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** `main` at `63923ccd` versus this branch, two separate binaries
- **Model:** `models/mlx/c4ai-command-r7b-12-2024-4bit`
- **Machine load:** background indexers suspended with
  `scripts/with_indexers_paused.sh`; load average 6.6 to 4.5 (CLI), 3.9 to 3.5
  (server)

## CLI harness

`scripts/bench_decode.sh` at its standard condition (`--ignore-eos`,
same-process warmup of 20 tokens, `--max-tokens 128`) with `--prompt-tokens`
512 and 2048. Each round runs both lengths in ABBA order (main, branch, branch,
main), three rounds, six runs per arm per cell. The harness runs
`./target/release`, so each arm ran from its own directory whose
`target/release` links to that arm's binaries; the `main` arm set
`BENCH_ALLOW_STALE_BINARY=1` because its binary predates the tree on purpose.

| Prompt | main prefill tok/s (ms) | branch prefill tok/s (ms) | Change |
|---|---|---|---|
| 512 | 727.2 (704.0), range 727.07 to 727.56 | 831.7 (615.6), range 831.52 to 832.00 | +14.4%, -88 ms |
| 2048 | 699.9 (2926), range 699.80 to 699.94 | 796.6 (2571), range 796.60 to 796.71 | +13.8%, -355 ms |

Decode is unchanged, as expected for a change that only touches multi-row
calls: 114.1 to 115.4 versus 114.2 to 116.7 tok/s at pp512, 98.2 to 99.3 versus
98.1 to 99.5 at pp2048.

The six runs of each arm fall within 0.1% of each other and the arms are 14%
apart, so the attribution does not rest on a noisy median.

## Server

`mlxcel-server` on one port at a time, `/completion` with the 512-token synthetic
prompt, `n_predict: 16`, `temperature: 0`, `cache_prompt: false`, two discarded
warmup requests, five measured requests per server start, and two starts per
binary in ABBA order. The value is the server's own `timings.prompt_ms`.

| Binary | prompt_ms, all ten measured requests | prefill tok/s |
|---|---|---|
| main | 713.0 every request | 718.1 |
| branch | 624.0 to 625.0 | 819.2 to 820.5 |

## mlx-lm reference

`scripts/bench_mlxlm.py` with mlx-lm 0.31.3 at pp512/tg128, three runs, same
session: prefill 781.01, 774.66, 792.47 tok/s (median 781.0), decode 94.34,
94.34, 94.27. mlxcel prefill moves from 6.9% behind (727.2) to 6.5% ahead
(831.7). mlx-lm computes logits for the last prompt token alone and evaluates
only the cache for the rest, which is what this change does.

## Output equivalence

The first sampled token's logits now come from a one-row matmul instead of a
row of a 512-row matmul. Greedy generation of 100 tokens produced identical
text on main and the branch for three short prompts and the 512-token synthetic
prompt.

## Not measured

CUDA, and prompts longer than 2048 tokens (chunked prefill, where every chunk
already went through `forward_last_logits`).
