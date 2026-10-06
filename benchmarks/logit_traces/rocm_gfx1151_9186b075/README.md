# ROCm Nemotron-H traces with the HIP fc1 squared-ReLU MoE kernel (lablup/mlxcel#2069)

Teacher-forced `w1ctx512` traces of Nemotron-3-Nano-30B-A3B-4bit on the Radeon 8060S (`gfx1151`), with and without `MLXCEL_FUSED_MOE_RELU2=1`, after `moe_fc1_relu2` gained a HIP port. Every traced step is a single-token forward with a 512-token state, so the `relu2` trace runs the HIP fc1 and down kernels for the routed experts and the `default` trace runs `gather_qmm`. `METADATA.txt` records the host, versions and commit; `RUNS.txt` the exit status and row count; `SHA256SUMS` covers both traces. The comparison is in `docs/benchmark_results/rocm-metal-only-ports-gfx1151-2026-10-05.md`.

```bash
cargo build --release --features rocm --example logit_trace
T=./target/release/examples/logit_trace
C=tests/fixtures/wikitext2_excerpt.txt
P=rocm_gfx1151_9186b075
M=models/mlx/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
MLXCEL_TRACE_START_TOKEN=512 $T $M $C 1 128 8 512 | grep -v '^\[' > ${P}_nemotron-3-nano-30b-a3b_default_w1ctx512.tsv
MLXCEL_FUSED_MOE_RELU2=1 MLXCEL_TRACE_START_TOKEN=512 $T $M $C 1 128 8 512 | grep -v '^\[' > ${P}_nemotron-3-nano-30b-a3b_relu2_w1ctx512.tsv
```
