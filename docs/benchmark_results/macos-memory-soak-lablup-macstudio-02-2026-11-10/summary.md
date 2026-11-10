# macOS memory soak: models/Qwen3.8-Flash-Next-OptiQ-2bit

Host: Apple M3 Ultra, 512 GiB, macOS 26.6.2. Writes: 20 40 GiB. Allocation: none. MLXCEL_WIRE_WEIGHTS: 0.

| MLXCEL_WIRED_LIMIT | TTFT before (s) | TTFT after (s) | Compressions | Swapouts |
|---|---|---|---|---|
| default | 0.234 | 0.549 | 0 | 0 |
| none | 0.709 | 0.426 | 0 | 0 |
| 60G | 0.298 | 0.503 | 0 | 0 |
