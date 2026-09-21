# Does compiling MLX architecture-specific cost decode? (GB10, 2026-09-21)

Issue #1934. Host: NVIDIA GB10 (sm_121, DGX Spark), Linux aarch64 `7.0.0-1019-nvidia`, CUDA 13.0.88, MLX pin `81ba1c6a`, toolchain 1.97.1. Source tree `01fe5bbc` (this PR's branch, rebased onto `76c849b5`). Pairing: `models/mlx/qwen3.5-4b-4bit` with `models/mlx/qwen3.5-4b-dflash`, draft width 4 and classic. Driver: `harness/ab_interleave.py`, which imports `run_arm` from the #1797 served harness (`data/draft-block-width-gb10-2026-09-20/harness/`) rather than reimplementing the host gate, the NVRM trip wire and the DFlash diagnostics parsing.

## Why interleaved

A sequential A/B on this pairing reported the architecture-specific build about 2.6% slower at width 4 with disjoint ranges. Two arms measured one after the other are only as comparable as the host is still between them, and on this box that is the thing most likely to be false. This driver alternates at round granularity instead: one server start, one measured request, teardown, next binary, and the arm that opens rotates every round, so no binary systematically owns the warm or the cool half of the session.

## Arms

All three binaries are built from the same source tree and differ only in how MLX was compiled.

| arm | `MLX_CUDA_ARCHITECTURES` | `libmlx.a` | cubin images in the binary |
|---|---|---|---|
| `121` | `121` | 178,388,958 B | 95 `sm_121` |
| `121a-real;121` | `121a-real;121` | 191,110,534 B | 95 `sm_121`, 95 `sm_121a` |
| `121+fp_quantize-a` | `121`, plus the per-source injection | 178,438,494 B | 95 `sm_121`, 1 `sm_121a` |

The third arm is also the experiment's negative control. Its decode kernels are compiled with byte-identical flags to the first arm's, and their disassembly hashes match (`qmv.cu.o` sm_121 section SHA-256 `6eb4cf943169` in both, 3,640,476 lines; `fp_qmv.cu.o` `53faf7a4e1d5`, 65,040 lines), with no second image for the driver to prefer. Any difference it shows cannot be a decode-kernel-speed difference.

## Result: no decode cost survives interleaving

30 arms, 0 errors, `nvrm_delta` 0 everywhere, no CI job overlapping any arm, load1 before each arm between 0.01 and 0.47.

| width | arm | n | min | max | mean | ratio to `121` | ranges disjoint |
|---|---|---|---|---|---|---|---|
| 4 | `121` | 5 | 65.96 | 67.81 | 66.97 | | |
| 4 | `121a-real;121` | 5 | 66.04 | 68.18 | 67.31 | 1.0052 | no |
| 4 | `121+fp_quantize-a` | 5 | 65.40 | 67.17 | 66.21 | 0.9887 | no |
| classic | `121` | 5 | 55.72 | 59.39 | 57.21 | | |
| classic | `121a-real;121` | 5 | 55.94 | 57.24 | 56.76 | 0.9923 | no |
| classic | `121+fp_quantize-a` | 5 | 55.09 | 57.25 | 56.34 | 0.9849 | no |

Paired within-round differences against `121`, which removes round-level drift:

| width | arm | deltas (tok/s) | mean |
|---|---|---|---|
| 4 | `121a-real;121` | +0.81, +1.67, +1.54, -0.60, -1.67 | +0.54% |
| 4 | `121+fp_quantize-a` | -0.26, -1.10, +1.21, -1.55, -2.08 | -1.11% |
| classic | `121a-real;121` | -1.24, +0.22, -0.09, +1.13, -2.22 | -0.73% |
| classic | `121+fp_quantize-a` | -1.18, -0.63, -0.31, -0.05, -2.14 | -1.48% |

The architecture-specific arm is not slower. At width 4 it is nominally faster, its paired deltas change sign across rounds, and every range overlaps.

What bounds the method is the negative control. `121+fp_quantize-a` runs decode kernels that are the same bytes as `121`, and it measures 1.11% slower at width 4 and 1.48% slower at classic with all five paired deltas negative. So five same-sign deltas at about 1.5% are reachable here with no code difference at all, which is what the earlier 2.6% sequential result has to be read against.

## Reproducing

```bash
MLXCEL_SWEEP_HARNESS=<repo>/docs/benchmark_results/data/draft-block-width-gb10-2026-09-20/harness \
python3 harness/ab_interleave.py \
  --arm "121=<server built at 121>" \
  --arm "121a-real;121=<server built at 121a-real;121>" \
  --arm "121+fp_quantize-a=<server built at 121 on this branch>" \
  --target <models>/qwen3.5-4b-4bit --drafter <models>/qwen3.5-4b-dflash \
  --rounds 5 --widths 4,classic --n 1 --out ab.jsonl --logdir logs
```

`ab.jsonl` is one record per arm, carrying the host-gate readings, the server command line, the DFlash diagnostics and the generated text. `logs/` holds each arm's server log; the `CUDA compute capability ...; compiled for [...]` startup line in each one names the architecture list that binary was compiled for, which is how an arm's label is checked against the binary rather than trusted.
