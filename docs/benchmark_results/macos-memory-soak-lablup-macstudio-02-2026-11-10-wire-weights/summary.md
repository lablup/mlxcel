# macOS memory soak: models/Qwen3.8-Flash-Next-OptiQ-2bit

> Retention note (2026-11-10): final results are retained here. Superseded experiments are indexed in [benchmark index](../README.md); retired run captures and any historical artifact paths resolve in the [frozen evidence tree](https://github.com/lablup/mlxcel/tree/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/). Measurement values and their original limitations have not been remeasured.

Host: Apple M3 Ultra, 512 GiB, macOS 26.6.2. Writes: 20 40 GiB. Allocation: none. MLXCEL_WIRE_WEIGHTS: 1.

| MLXCEL_WIRED_LIMIT | TTFT before (s) | TTFT after (s) | Compressions | Swapouts |
|---|---|---|---|---|
| default | 0.244 | 0.525 | 0 | 0 |
