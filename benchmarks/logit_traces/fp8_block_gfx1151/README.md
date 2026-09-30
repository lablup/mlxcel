# FP8 block checkpoint logit traces for lablup/mlxcel#1807

Teacher-forced logit traces from `examples/logit_trace` on a Radeon 8060S (`gfx1151`), built with `--features rocm`, for a vendor FP8 block checkpoint (`Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8`) that mlxcel requantizes to mxfp8 at load. The write-up is `docs/benchmark_results/rocm-fp8-block-gfx1151-2026-09-30.md`. `METADATA.txt` records the host, revisions and how each arm was produced; `RUNS.txt` has the exit status and row count of every run; `SHA256SUMS` covers every trace.

## Arms

- `rocm`: the GPU as shipped, at `w1` (`1 128 8 0`), `w8` (`8 80 8 512`), `w256` (`256 2 8 0`) and `w32` (`32 8 8 0`).
- `rocm_cpuquant`: the GPU with the load-time mxfp8 quantization moved to the CPU stream by a temporary switch, at `w1`, `w8` and `w256`. Compare with `rocm` at the same width, `rocm_cpuquant` as the reference argument as in the write-up, to isolate the effect of GPU tie rounding in `quantize`.
- `cpu`: `MLXCEL_DEVICE=cpu`, the reference device in place of Metal, at `w32` only; compare it with `rocm_w32`.

```bash
python3 scripts/compare_logit_traces.py realigned-qwen3.5-0.8b-fp8_cpu_w32.tsv realigned-qwen3.5-0.8b-fp8_rocm_w32.tsv --decided 2.0
```
