# Raw data for CUDA bf16 SSM parity on GB10 (2026-10-11)

This directory preserves the numeric diagnosis and real-checkpoint parity evidence for issue #2230.

## Stage deviations

`stage-deviations.log` contains the exact command records and stdout from the original seed-2069 probe. They were recovered from the original execution record and were not recreated for this report. The extraction places the fixed run before the failing baseline run; the timestamps and command tags identify their actual order and status.

The baseline compiled `dt'` had bf16 dtype and diverged from the stable float32 graph at normalized RMS/max `1.9571408311071774e-3 / 5.875451501701449e-3`. Its decay and final state deviations were `3.008783834355594e-3 / 1.0758810176786593e-2` and `2.5324028008192334e-3 / 4.127457397370858e-2`. After explicitly casting `dt_bias` to float32 beside `dt`, the three deviations fell to `2.890811209876877e-8 / 2.0028127562385632e-7`, `6.809246821557169e-10 / 4.717584582485562e-9`, and `2.838613609145525e-8 / 1.7010096528383182e-6` without changing the tolerance.

## Granite checkpoint parity

`granite-parity/` records a deterministic greedy comparison on `models/mlx/granite-4.0-h-tiny-4bit`:

- `run.sh`: exact kernel-versus-graph command and comparison driver.
- `provenance.txt`: source commit, release binary SHA-256, and shell-escaped commands.
- `kernel.stdout`, `kernel.stderr`: fused CUDA SSM arm output.
- `graph.stdout`, `graph.stderr`: graph arm output with `MLXCEL_SSM_CUDA_KERNEL=0`.
- `comparison.json`: 16 prompt IDs, 64 generated IDs, both complete ID arrays, equality result, and generated-token SHA-256.

Both arms used prompt `Explain the Mamba state update.`, `-n 64`, `--temp 0`, `MLXCEL_SDPA_DETERMINISTIC=1`, and token-ID printing. All 64 generated IDs were identical. Their SHA-256 is `9df6f4fc2abbfcb121fa5dca3ad1e140e8db5b67d28118a905b93022f8905829`.

The recorded release binary SHA-256 is `0cc49217021dab12b5add25dc2bb8de7e90c405c0b406607b244a8060d3ed800`. `SHA256SUMS` covers every other file in this directory.

Validate the preserved files from this directory with:

```bash
sha256sum --check SHA256SUMS
```
