# DeepSeek V2 default MLA cache layout: decompressed against absorbed (issue #3102), M5 Max, 2026-11-15

Issue #3102 asks whether DeepSeek V2 should cache the MLA latent (`MLXCEL_MLA_ABSORBED=1`) by default instead of the decompressed per-head K and V, and sets the rule: switch only if absorbed decode holds throughput within a few percent and the teacher-forced trace shows no material disagreement.

**Outcome: the default stays decompressed.** The trace passes at every width (0 disagreements on 3,653 decided positions), and the latent cache saves 6.7 GB of peak at a 28.7K-token chunked prefill. But absorbed decode is 10% slower at pp512/tg128 and 4% slower behind a 2K prompt, and is 57% faster only behind a 16K prompt. That fails the issue's short-context throughput condition, so `MLXCEL_MLA_ABSORBED=1` remains the opt-in for long-context and memory-bound use.

Host: Apple M5 Max, 128 GB, macOS 27, shared with five other agents' builds and GPU jobs (every GPU run held the shared lock, but CPU contention from concurrent compiles was present throughout). Build: release, `00cd2d69` (main) with the `examples/logit_trace.rs` stats fix from this branch, which is why the CSVs read `00cd2d69-dirty`; no `src/` file differs from main, so both arms are main's model code. MLX `9f806b36`. Model `deepseek-v2-lite-chat-4bit-mlx`. Both arms run one binary; the arm is `MLXCEL_MLA_ABSORBED=0` (decompressed) or `=1` (absorbed). Every absorbed run printed `mla: absorbed decode folded 27/27 layers; KV cache 10240 -> 1152 bytes/token/layer (8.9x)`.

## Decode throughput

`scripts/bench_decode.sh` (one model load per run, warmup discarded), 128 generated tokens, `--prompt-tokens` as listed, 75 s idle before every run, arms alternated A-B-B-A. Every sample is in `mla-v2-default-layout-m5max-2026-11-15/decode.csv`.

| context | decompressed tok/s (samples) | absorbed tok/s (samples) | median delta |
|---|---|---|---|
| pp512/tg128 | 114.6, 203.0, 200.6, 198.1, 203.4 (median 200.6) | 181.6, 182.1, 119.6, 180.7, 177.8 (median 180.7) | **-9.9%** |
| 2048 prompt, tg128 | 168.1, 169.9, 160.5 (median 168.1) | 147.4, 162.5, 161.9 (median 161.9) | -3.7% |
| 16384 prompt, tg128 | 73.1, 68.6 (mean 70.9) | 114.2, 108.2 (mean 111.2) | **+57%** |

The two outliers (114.6 and 119.6 at pp512) ran while a sibling compile was active: their prefill also read about 2x slow (`prefill_ms` 1398 and 933 against about 490 for every other row). Excluding them leaves the pp512 gap unchanged (201.8 against 180.7 by median, -10.5%).

Per token, the pp512 gap is about 0.55 ms and the 2K gap about 0.23 ms, so absorbed decode carries a fixed per-step cost that its smaller cache read pays back only as the context grows; interpolating the two points puts the crossover near 3K tokens of context. Bytes alone do not explain the fixed cost: at 512 tokens the decompressed step reads about 141 MB of cache plus the 4-bit `kv_b_proj` (28 MB), the absorbed step about 113 MB of dense folded `W_UK`/`W_UV` plus 16 MB of latent. This was not profiled further here.

## Prefill peak and time (28,705-token prompt)

`/usr/bin/time -l mlxcel generate -n 48 --temp 0 --profile --show-reasoning`, prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000`, 75 s idle before every run, A-B-B-A per chunk mode. Rows: `prefill.csv`.

| cell | decompressed peak | absorbed peak | decompressed prefill | absorbed prefill |
|---|---|---|---|---|
| 2048-token pieces | 20.20 / 20.20 GB | **13.52 / 13.52 GB** | 90.8 / 77.3 s | 75.8 / 74.4 s |
| single pass (`MLXCEL_PREFILL_CHUNK=0`) | 48.55 / 48.43 GB | **41.02 / 41.03 GB** | 94.8 / 59.3 s | 89.7 / 85.9 s |

Peaks are stable to 0.1 GB. The prefill times are not usable for a comparison: the same cell spread from 59 to 95 s under the host's CPU contention, against the 42 s both arms read at 2048 pieces on a quieter host in #3084 and #3091. Those quieter runs found no prefill difference between the layouts, and nothing here contradicts them.

Greedy output is not byte-identical between the arms, and the split does not follow the layout: the 2048-piece absorbed run and the single-pass decompressed run produce one text, and the other two cells produce the other. The texts first differ at a digit inside an answer both arms get wrong (the model misreports item 621's weight as 733 or 333; the prompt says 405), which is a near-tie flip, not a layout effect.

## Teacher-forced logit trace

`examples/logit_trace` over `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` concatenated (49,965 tokens), decompressed as the reference, compared with `scripts/compare_logit_traces.py --decided 2`. `MLXCEL_TRACE_MLA_STATS=1` counters are recorded per arm so each trace proves which path it took. Rows: `logit_trace.csv`.

| shape | arguments | positions | top-1 disagree | decided disagree | ppl decompressed / absorbed | absorbed arm's path counters |
|---|---|---|---|---|---|---|
| cold prefill, width 2048 | `2048 2 8 0` | 4096 | 0 | **0 / 1642** | 17.7765 / 17.7765 | `decompressed=54` (cold prefill materializes) |
| prefill continuation, width 2048 behind 2048 | `2048 2 8 2048` | 4096 | 27 (0.66%) | **0 / 1724** | 15.2771 / 15.2730 | `decompressed=81 absorbed_composed=27` |
| decode width 1 behind 1024 | `64 8 8 1024`, `MLXCEL_TRACE_BATCH_ROWS=1`, scalar reference | 512 | 9 (1.76%) | **0 / 209** | 14.6514 / 14.6509 | `decompressed=216 absorbed_composed=13824` |
| decode width 1 behind 8192 | `64 2 8 8192`, same | 128 | 1 (0.78%) | **0 / 78** | 4.5824 / 4.5815 | `decompressed=54 absorbed_composed=3456` |

The decompressed arm's counters read `decompressed` only in every row. In the decode rows the absorbed arm took the absorbed path on every traced step (512 x 27 and 128 x 27 layer calls) and materialized only the context passes. Every disagreement sits where the reference's top two were within 0.19 logits, and perplexity moves by at most 0.03%. By the docs/benchmarks.md gate this is the rounding class: the trace alone would allow the switch.

## Other families

Only `deepseek_v2` reads `MLXCEL_MLA_ABSORBED`. `deepseek_v3` and `deepseek_v32` absorb unconditionally through their own `embed_q` / `unembed_out` decomposition. `minicpm3` and `dots1` cache decompressed K/V but have no absorbed path to switch to, so the decision has nothing to apply to there.

## Reproduce

```bash
python3 scripts/bench/metal_sdpa_chunk/make_prompt.py 28000 > prompt_28000.txt
# Arms: MLXCEL_MLA_ABSORBED=0 (decompressed) or =1 (absorbed).
MLXCEL_MLA_ABSORBED=1 ./scripts/bench_decode.sh models/deepseek-v2-lite-chat-4bit-mlx \
  --prompt-tokens 512 --max-tokens 128 --output absorbed-512.csv          # also 2048, 16384
MLXCEL_MLA_ABSORBED=1 MLXCEL_PREFILL_CHUNK=2048 /usr/bin/time -l ./target/release/mlxcel generate \
  -m models/deepseek-v2-lite-chat-4bit-mlx -p "$(cat prompt_28000.txt)" -n 48 --temp 0 --profile --show-reasoning
cargo build --release --features metal,accelerate --example logit_trace
MLXCEL_MLA_ABSORBED=1 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  ./target/release/examples/logit_trace models/deepseek-v2-lite-chat-4bit-mlx corpus.txt 64 8 8 1024 > absorbed.tsv
python3 scripts/compare_logit_traces.py decompressed.tsv absorbed.tsv --decided 2
```
