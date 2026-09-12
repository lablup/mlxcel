# ROCm logit traces for lablup/mlxcel#1809

Teacher-forced logit traces from `examples/logit_trace` on a Radeon 8060S (`gfx1151`), built with `--features rocm` at mlxcel `bec64748` (MLX pin `81ba1c6a`). They are the ROCm side of the cross-backend correctness matrix in lablup/mlxcel#1809, taken with the same checkpoints (same Hugging Face revisions), corpus and arguments as the Metal reference in `../metal_m1u_bec64748/`. `METADATA.txt` records the host, ROCm version, binaries and revisions; `RUNS.txt` has the exit status and row count of every run; `SHA256SUMS` covers every trace.

## What was traced

The same four affine 4-bit checkpoints at the same three shapes (`w1` = `1 128 8 0`, `w8` = `8 80 8 512`, `w256` = `256 2 8 0`) over `tests/fixtures/wikitext2_excerpt.txt`:

- `qwen3-0.6b` and `llama-3.1-8b-instruct` with the default settings (`default`).
- `qwen3-30b-a3b` and `mixtral-8x7b-instruct` with `MLXCEL_FUSED_MOE=0` (`fused0`), because the fused MoE kernel has no ROCm port and aborts until lablup/mlxcel#1803. Compare them with the Metal `fused0` traces.

The Mixtral traces use a `logit_trace` built from `bec64748` plus only the f16 `gather_qmm` dispatch change in lablup/mlxcel#1822 (an edit inside `src/lib/mlx-cpp/patches-rocm/`, with no mlxcel Rust or bridge change). Without it, this f16 checkpoint spends about 205 ms per MoE `gather_qmm` call in a generic kernel, and the traces would take hours instead of minutes; with it, the numerics are those of the kernel a user of the fix runs.

## Comparing

```bash
python3 scripts/compare_logit_traces.py \
    benchmarks/logit_traces/metal_m1u_bec64748/metal_m1u_bec64748_<model>_<variant>_<tag>.tsv \
    benchmarks/logit_traces/rocm_gfx1151_bec64748/rocm_gfx1151_bec64748_<model>_<variant>_<tag>.tsv
```

The results and how to read them are in `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-12.md`.
