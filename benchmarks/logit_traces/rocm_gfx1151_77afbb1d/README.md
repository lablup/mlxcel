# ROCm logit traces with the HIP fused MoE decode kernels (lablup/mlxcel#2065)

Teacher-forced traces from `examples/logit_trace` on the Radeon 8060S (`gfx1151`) after `moe_gateup` and `moe_down` gained HIP ports. The fused MoE path runs only on a single-token forward, so the `w1` rows are the ones that reach the kernels on Qwen3-30B-A3B; the `w8` rows check that the multi-token path is unchanged. `METADATA.txt` records the host, versions, commit and binary hash; `RUNS.txt` the exit status and row count of every run; `SHA256SUMS` covers every trace. The comparisons are in `docs/benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md`.

```bash
cargo build --release --features rocm --example logit_trace
T=./target/release/examples/logit_trace
C=tests/fixtures/wikitext2_excerpt.txt
P=rocm_gfx1151_77afbb1d
for w in "w1:1 128 8 0" "w8:8 80 8 512"; do
  tag=${w%%:*}; args=${w#*:}
  $T models/mlx/Qwen3-30B-A3B-4bit $C $args > ${P}_qwen3-30b-a3b_default_$tag.tsv
  MLXCEL_FUSED_MOE=0 $T models/mlx/Qwen3-30B-A3B-4bit $C $args > ${P}_qwen3-30b-a3b_fused0_$tag.tsv
  $T models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit $C $args | grep -v '^\[' > ${P}_nemotron-3-nano-30b-a3b_default_$tag.tsv
done
MLXCEL_TRACE_START_TOKEN=512 $T models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit $C 1 128 8 512 | grep -v '^\[' > ${P}_nemotron-3-nano-30b-a3b_default_w1ctx512.tsv
$T models/mlx/Mixtral-8x7B-Instruct-v0.1-4bit $C 1 128 8 0 > ${P}_mixtral-8x7b-instruct_default_w1.tsv
```

Unlike the older Nemotron-H sets, the files here are stored without the five `[NemotronH]` loader lines. Filter the older references with `grep -v '^\[NemotronH\] '` before `scripts/compare_logit_traces.py`.
