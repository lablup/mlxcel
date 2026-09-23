# In-place decode KV row write (issue #1959)

A single-token FP16 decode step used to write its K/V row with `slice_update`,
which copies the whole cache buffer unless the buffer is donatable. During
pipelined decode it never is: MLX's Metal backend keeps each kernel's input
buffers alive in the command buffer's completion handler, and the next step is
encoded while the previous one is still running, so the previous step's
attention still references the cache buffer. Every layer's K and V cache was
copied every token, a cost that grows with context. The write now goes through
`inplace_slice_write`, a primitive whose output adopts the cache buffer and
writes only the new row, whenever the cache provably owns the rows it writes.

- **Date:** 2026-09-23
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** one binary, `MLXCEL_KV_INPLACE_WRITE=0` (copying `slice_update`)
  against the default (in-place); plus `MLXCEL_DIAG_SKIP_DECODE_KV_WRITE` (no
  write at all, output invalid) as the upper bound
- **Machine load:** background indexers suspended with
  `scripts/with_indexers_paused.sh`; load average 4 to 6

## Premise, before building

Skipping the write entirely (`MLXCEL_DIAG_SKIP_DECODE_KV_WRITE`) on command-r7b
4-bit lifted decode 4.4% at context 16, 8% at 512 and 11% at 2048. A one-row
write should not scale with context, which pointed at a whole-buffer copy. Plain
MLX reproduces it: 64 `slice_update`s on `[1, 8, CTX + 256, 128]` fp16 took
0.52 ms (CTX 512) and 0.56 ms (2048) with no outstanding reference to the
previous buffer, and 1.00 ms and 1.39 ms with the previous step's view alive.

## command-r7b 4-bit, `scripts/bench_decode.sh` (tg128)

Each round runs copying, in-place, skip, in-place, copying; three rounds, six
runs per arm for the first two.

| Context | copying (median) | in-place (median, range) | skip bound |
|---|---|---|---|
| 16 | 121.6 | 124.6, +2.5% | 127.7 |
| 512 | 115.5 | 120.2 (117.1 to 121.4), +4.1% | 121.1 to 124.6 |
| 2048 | 99.1 | 107.3 (104.8 to 109.2), +8.2% | 110.0 to 111.9 |

At 512 the two first-round runs (117.1, 117.9) were taken at load 5.9; the other
four were 119.6 to 121.4. The remaining gap to the skip bound is the two write
kernels and the dependent stage the attention waits on.

At prompt 2048 with 512 generated tokens: decode 109.0 and 109.9 against 98.5
and 100.0 tok/s, and the process peak memory footprint 6.80 and 6.82 GB against
7.41 and 7.53 GB (`/usr/bin/time -l`), since no step allocates a copy of the
cache any more.

## Server

`mlxcel-server`, `scripts/bench_serving_concurrency.py --concurrency 1
--prompt-tokens 512 --max-tokens 128`, four requests per server start, two
starts per arm in ABBA order. The first request of each start is cold; the rest
hit the prompt cache (TTFT about 60 ms), so they decode on a cache restored
from a snapshot.

| Arm | decode tok/s, requests 2 to 4 |
|---|---|
| copying | 115.2 to 116.7 |
| in-place | 119.6 to 120.6 |

## Other families (`mlxcel-bench-decode` with the harness arguments, pp512/tg128, two runs per arm)

| Model | copying | in-place |
|---|---|---|
| Llama 3.1 8B Instruct 4-bit | 107.2, 107.8 | 112.3, 112.5 (+4.6%) |
| Qwen2.5 7B Instruct 4-bit | 111.3, 111.6 | 114.2, 114.8 (+2.7%) |
| Gemma 3 4B IT 4-bit | 96.3, 97.2 | 96.8, 97.9 (+0.6%) |

Gemma 3 moves little because most of its layers use the rotating sliding-window
cache, which this change does not touch.

## Correctness

- Greedy output is byte-identical between the arms for command-r7b (three short
  prompts and the 512-token prompt, 300 tokens each, which crosses the buffer
  growth boundary) and for the three families above (two prompts, 300 tokens).
- Server: eight greedy chat requests that share prefixes, replayed so the
  prompt cache restores snapshots (208 to 224 cached tokens), return identical
  replies in both arms.
- Unit tests: the ownership proof holds after the cache's own writes and fails
  after an external buffer replacement or a trim; a snapshot sharing the buffer
  keeps its rows while the cache keeps decoding in place, and a cache restored
  from it writes into its own copy. Dropping the ownership check makes the
  second test fail (the restored cache's row lands in the original).

## Not measured

CUDA and ROCm. `copy_gpu_inplace` exists on both and the ownership argument is
backend-independent, but the completion-handler retention diagnosed here is the
Metal backend's, so the gain there may differ. A cache with a paged backing
returns from `update_and_fetch` before the dense write, so paged decode is
unchanged; its gain, if any, was not measured.
