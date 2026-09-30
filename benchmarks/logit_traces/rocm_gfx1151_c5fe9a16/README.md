# ROCm logit traces for the remaining rows of lablup/mlxcel#1809

Teacher-forced logit traces from `examples/logit_trace` on the Radeon 8060S (`gfx1151`), for the five checkpoints the first matrix run (`../rocm_gfx1151_bec64748/`) did not cover: a second dense model, a sliding-window model, two SSM hybrids and a VLM. They extend the correctness matrix in `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`. `METADATA.txt` records the host, versions, commit, checkpoint revisions and binary hash; `RUNS.txt` has the exit status and row count of every run; `SHA256SUMS` covers every trace.

**The Metal reference is `../metal_m5_d1128266/`**, traced on an Apple M5 Max (Apple GPU generation 17, NAX path available) at mlxcel `d1128266`, the merge commit of the PR that added this directory, with the same corpus, arguments and checkpoint revisions. The per-pair comparison is in `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`: fifteen of the sixteen pairs have no disagreement on a decided position at `--decided 2.0`, and `nemotron-3-nano-30b-a3b` `w1` is inconclusive because its reference has no decided position. Two more rows, `w1ctx512` for the two SSM hybrids, are traced at a later commit, `3c9edea0`, on both sides (`../metal_m5_3c9edea0/`, `../rocm_gfx1151_3c9edea0/`; see below). The reference host is not the M1 Ultra of the first run, and the document says what that changes.

## What was traced

| Tag | Checkpoint | Path it covers |
|---|---|---|
| `qwen2.5-7b-instruct` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | dense, second model |
| `gemma-3-4b-it` | `mlx-community/gemma-3-4b-it-4bit` | sliding window (1024 tokens, five local layers per global one) |
| `granite-4.0-h-tiny` | `mlx-community/granite-4.0-h-tiny-4bit` | SSM hybrid: Mamba2 plus attention, 64-expert MoE |
| `nemotron-3-nano-30b-a3b` | `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | SSM hybrid: Nemotron-H, 128-expert MoE |
| `qwen2.5-vl-3b-instruct` | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | VLM loaded on the default path; the trace is text-only, so it covers the language model and not the vision tower |

Widths, as `CHUNK_TOKENS MAX_CHUNKS TOPK PREFILL`: `w1` = `1 128 8 0`, `w8` = `8 80 8 512`, `w256` = `256 2 8 0`, the same as the first run. `w8ctx1536` = `8 40 8 1536` with `MLXCEL_TRACE_START_TOKEN=1536` is added for `gemma-3-4b-it` only: the other widths never give it more than 520 tokens of context, which is inside its 1024-token window, so without this shape the sliding-window mask and the rotating cache are never exercised past the window.

Every trace is `default`: no MoE or kernel override is set. granite's MoE goes through `SwitchGLU::forward` (`gather_qmm`) on every backend, since that family does not call the fused MoE decode kernel. On ROCm Nemotron-H runs `forward_nonfused`, and the Mamba2 layers of both hybrids run the SSD graph path at every width, because neither the fused MoE nor the SSM update kernel has a ROCm port (lablup/mlxcel#1814).

## How the Metal half was produced

On an Apple host, at the commit this directory was merged in, with the same five checkpoint revisions (`METADATA.txt`). This is the loop `../metal_m5_d1128266/` was produced with:

```bash
cargo build --release --features metal,accelerate --example logit_trace
T=./target/release/examples/logit_trace
C=tests/fixtures/wikitext2_excerpt.txt
P=metal_<host>_<commit>
for pair in qwen2.5-7b-instruct:Qwen2.5-7B-Instruct-4bit gemma-3-4b-it:gemma-3-4b-it-4bit \
            granite-4.0-h-tiny:granite-4.0-h-tiny-4bit nemotron-3-nano-30b-a3b:NVIDIA-Nemotron-3-Nano-30B-A3B-4bit \
            qwen2.5-vl-3b-instruct:Qwen2.5-VL-3B-Instruct-4bit; do
  tag=${pair%%:*}; dir=models/mlx/${pair#*:}
  $T $dir $C 1 128 8 0   > ${P}_${tag}_default_w1.tsv
  $T $dir $C 8 80 8 512  > ${P}_${tag}_default_w8.tsv
  $T $dir $C 256 2 8 0   > ${P}_${tag}_default_w256.tsv
done
MLXCEL_TRACE_START_TOKEN=1536 $T models/mlx/gemma-3-4b-it-4bit $C 8 40 8 1536 > ${P}_gemma-3-4b-it_default_w8ctx1536.tsv
MLXCEL_TRACE_START_TOKEN=512 $T models/mlx/granite-4.0-h-tiny-4bit $C 1 128 8 512 > ${P}_granite-4.0-h-tiny_default_w1ctx512.tsv
MLXCEL_TRACE_START_TOKEN=512 $T models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit $C 1 128 8 512 > ${P}_nemotron-3-nano-30b-a3b_default_w1ctx512.tsv
```

Then, for each pair. The Nemotron-H traces on both sides begin with five `[NemotronH] ...` loader lines that the loader prints to stdout; remove them first (`grep -v '^\[NemotronH\] '`), or the script stops with a `ValueError`:

```bash
python3 scripts/compare_logit_traces.py <metal.tsv> benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/rocm_gfx1151_c5fe9a16_<tag>_default_<width>.tsv --decided 2.0
```

Record the host, OS, commit, pin and binary hash in a `METADATA.txt` beside the Metal traces, as `../metal_m1u_bec64748/` does. An M1 Ultra (Apple GPU generation 13) is the reference host of the first run; an M5-class host takes different kernels (NAX), so say which generation was used.

Two pairings were expected not to be like-for-like even when both sides run `default`:

- `granite-4.0-h-tiny` and `nemotron-3-nano-30b-a3b` at `w1`: expected here to compare Metal's fused SSM update kernel with ROCm's SSD graph. The code at `d1128266` shows it does not: the fused step needs `seq_len == 1`, `ssm_kernel_available()` and an SSM state already in the cache, `logit_trace` builds a fresh cache per chunk, and a `w1` chunk has no prefill, so Metal runs the same SSD graph. `w1` is graph against graph, and no trace in this directory reaches the fused SSM kernel.
- `nemotron-3-nano-30b-a3b`: Metal takes `fused_moe_forward` (its C++ graph path unless `MLXCEL_FUSED_MOE_RELU2` is set), ROCm takes `forward_nonfused`. No environment variable selects `forward_nonfused` on Metal.

If the merged commit changes model code relative to `c5fe9a16`, retrace the ROCm side at the same commit before comparing; the four things that must match are the commit, the corpus, the arguments and the checkpoint revision.

`logit_trace` does not prepend BOS to chunk 0 (lablup/mlxcel#1785), which is why the `w1` perplexities are very large: each one-token chunk is scored with no context at all. Both backends run the same code, so it does not bias a comparison.

## The `w1ctx512` rows: fused SSM kernel against the graph with state

`w1ctx512` = `1 128 8 512` with `MLXCEL_TRACE_START_TOKEN=512`, for `granite-4.0-h-tiny` and `nemotron-3-nano-30b-a3b` only. Each single-token forward follows a 512-token prefill, so it has SSM state: Metal runs its fused SSM update kernel (`ssm_update_kernel`; inside `fused_mamba2_forward` for Nemotron-H) and ROCm, with no port, runs the SSD graph with that state. Apple has no runtime switch that forces the graph path, so fused against graph could not be A/B tested on Metal.

- Metal: `../metal_m5_3c9edea0/`, at `3c9edea0`. The first Metal trace of these rows, `../metal_m5_d1128266/metal_m5_d1128266_<tag>_default_w1ctx512.tsv`, has byte-identical data rows.
- ROCm: `../rocm_gfx1151_3c9edea0/`, at `3c9edea0`, a later commit than the sixteen ROCm traces here; its `METADATA.txt` lists the ROCm fixes merged in between.

With the same criterion (`--decided 2.0`, Metal the reference, Nemotron-H loader lines filtered from copies):

| Model | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity Metal / ROCm | Verdict |
|---|---|---|---|---|---|
| granite-4.0-h-tiny | 4 / 128 | 0 / 60 | 0.125 | 7.090 / 7.234 (+2.024%) | pass, rounding class |
| nemotron-3-nano-30b-a3b | 7 / 128 | 0 / 71 | 0.250 | 7.489 / 7.489 (-0.003%) | pass, rounding class |

Both are also zero at `--decided 0.5` (0 / 104, 0 / 106) and `1.0` (0 / 84, 0 / 89). The per-position check of granite's perplexity shift and the list of fixes between the commits are in `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`.
