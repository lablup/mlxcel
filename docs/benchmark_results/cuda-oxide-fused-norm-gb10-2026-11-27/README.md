# cuda-oxide fused residual-add + RMSNorm on GB10, 2026-11-27 (issue #2834)

Raw rows and commands for [the benchmark evidence report](../../mlxcelverse/cuda-oxide-benchmark-evidence.md), which holds the frozen protocol, the tables, the gate evaluation and the recorded result. Host `spark-101` (NVIDIA GB10, driver 580.178.04), checkpoint `llama-3.2-1b-instruct-bf16`, one release binary built with `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1 MLXCEL_CUDA_ARCHITECTURES=121`. Arms: `graph` (`add` + `fast_rms_norm`), `fused` (existing CUDA kernel), `oxide` (cuda-oxide node), `fused-null` (the fused arm again).

| File | Measurement |
| --- | --- |
| `run.sh` | Every command, one function per measurement |
| `provenance.json` | Host, driver, toolkit, commit, MLX pin, binary, artifact and checkpoint digests |
| `host-state.csv` | Per stage: start, end, load average, whether a CI job was running at its end |
| `m1-end-to-end.csv`, `m1-end-to-end-summary.csv` | M1: 7 interleaved rounds, 6 cells, 4 arms; route counters, peak memory and token ids per pass; paired deltas and null floor per cell |
| `m2-logit-trace-summary.csv`, `m2-logit-trace-comparison.txt` | M2: route line and SHA-256 of each teacher-forced logit trace; `compare_logit_traces.py` output for every arm against `fused` |
| `m2c-adapter-matrix.csv` | M2(c): the 31 `cuda_oxide` adapter tests reduced by `scripts/oxide_fused_norm_m2c_summarize.py` from the test binary's capture (command, commit, binary SHA-256, each test's outcome, every field of the 130-case matrix `RESULT` line, libtest totals); a correctness rerun, no timing. The capture itself stays in the run's work directory, as raw logs are not committed here |
| `m3-standalone-events.csv` | M3: standalone driver-API launches, CUDA events, 5 processes, 23 shapes |
| `m4-nsys-kernels*.csv`, `m4-nsys-cuda-api*.csv` | M4: kernel durations and CUDA API calls from `nsys`, standalone and MLX with graphs off; `-repeat` is the second set |
| `m5-op-level-mlx*.csv` | M5: op level through MLX with graphs on, `iso` and `pipe`; `-repeat` is the second set |
| `m6-cold-start.csv`, `m6-cold-start-steady.csv` | M6: first use per state in a fresh process, and the steady-state blocks it is measured against |
| `m6b-*.csv` | M6 and the standalone program with `CUDA_CACHE_DISABLE=1` |
| `m7-kernel-resources.csv` | M7: registers, stack, shared memory, constant bank and SASS instruction count per kernel |
| `m8-cuda-api-per-arm.csv` | M8: CUDA API calls per arm, one arm per process, graphs off and on |
| `m9-node-build-cost.csv` | M9: host time to build the lazy node, nothing evaluated |

Profiler captures, logit traces and the PTX caches were written to a scratch directory outside the checkout and are not kept.
