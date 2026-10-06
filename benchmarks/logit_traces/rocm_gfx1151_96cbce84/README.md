# ROCm logit traces with the HIP SSM update kernel (lablup/mlxcel#2067)

Teacher-forced traces from `examples/logit_trace` on the Radeon 8060S (`gfx1151`) after `ssm_update_kernel` gained a HIP port. Before it, ROCm ran the Mamba2 SSD graph for every single-token step, so the `w1ctx512` rows in `../rocm_gfx1151_3c9edea0/` compared Metal's fused kernel with ROCm's graph. These compare kernel with kernel, and the `ssmkernel0` rows give the graph on the same binary for a same-backend A/B. `METADATA.txt` records the host, versions, commit and binary hash; `RUNS.txt` the exit status and row count of every run; `SHA256SUMS` covers every trace. The comparison is in `docs/benchmark_results/rocm-ssm-update-kernel-gfx1151-2026-10-04.md`.

```bash
cargo build --release --features rocm --example logit_trace
T=./target/release/examples/logit_trace
C=tests/fixtures/wikitext2_excerpt.txt
P=rocm_gfx1151_96cbce84
for pair in granite-4.0-h-tiny:granite-4.0-h-tiny-4bit nemotron-3-nano-30b-a3b:NVIDIA-Nemotron-3-Nano-30B-A3B-4bit; do
  tag=${pair%%:*}; dir=${pair##*:}
  MLXCEL_TRACE_START_TOKEN=512 $T models/mlx/$dir $C 1 128 8 512 > ${P}_${tag}_default_w1ctx512.tsv
  MLXCEL_SSM_KERNEL=0 MLXCEL_TRACE_START_TOKEN=512 $T models/mlx/$dir $C 1 128 8 512 > ${P}_${tag}_ssmkernel0_w1ctx512.tsv
  $T models/mlx/$dir $C 1 128 8 0 > ${P}_${tag}_default_w1.tsv
done
```

The Nemotron-H traces carry five `[NemotronH]` loader lines on stdout; filter them with `grep -v '^\[NemotronH\] '` on both sides before `scripts/compare_logit_traces.py`.
