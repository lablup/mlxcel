# Paged KV pool writes in place under lookahead decode (issue #1964)

With default server settings, paged-capable families decoded a single request
at about half the dense-cache speed. The default storage for them is the paged
pool, and the default decode path is the lookahead pipeline. Each new K/V row
was written with `slice_update` on the layer's pool slab
(`PagedBlockPool::write_block`), which relies on buffer donation. Under
lookahead the previous step's command buffer still holds the slab, so every
write copied the whole slab (1024 blocks x 32 x 8 x 128 x 2 bytes = 64 MiB for
K and again for V, per layer, per token). The write now goes through
`inplace_slice_write` when the block is held by exactly one sequence and
already owned its pool row before the write.

- **Date:** 2026-09-24
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** one `mlxcel-server` binary, default settings (paged storage, 4
  slots, lookahead on), `MLXCEL_KV_INPLACE_WRITE=0` (copying) against the
  default (in-place), two server starts per arm in ABBA order
- **Harness:** `scripts/bench_serving_concurrency.py --concurrency 1,1,4,4
  --prompt-tokens 512 --max-tokens 128`
- **Machine load:** indexers suspended; load average 3.5 to 5

## Diagnosis, before the fix

Llama 3.1 8B 4-bit, concurrency 1, lookahead step counter
(`mlxcel_batch_decode_lookahead_steps_total`) read around each run:

| Server setting | lookahead steps | decode tok/s |
|---|---|---|
| default | +250 | 46.5, 48.2 |
| `MLXCEL_FORCE_SYNC=1` | 0 | 81.5, 81.6 |
| `--kv-cache-budget none` | +250 | 46.5, 47.9 |
| `--ctx-size 8192` (sets `max_kv_size`, which disables lookahead) | 0 | 81.4, 81.5 |
| `--decode-storage-backend dense` | | 104.7 to 109.5 |

With lookahead forced on through a measurement-only switch, decode fell with
the slab size (the copy size) while sync decode did not:

| Setting | slab | decode tok/s |
|---|---|---|
| lookahead, `--ctx-size 2048` | 256 blocks | 76.1 to 78.7 |
| lookahead, `--ctx-size 32768` | 4096 blocks | 52.7 to 54.6 |
| sync, `--ctx-size 32768` | 4096 blocks | 83.9, 84.0 |

## Results

| Model | Arm | conc 1 decode | conc 4 per-request | conc 4 aggregate | conc 4 TTFT |
|---|---|---|---|---|---|
| Llama 3.1 8B 4-bit | copying | 46.4 to 47.9 | 25.3, 25.4 | 98.0 to 98.2 | 199 to 209 ms |
| Llama 3.1 8B 4-bit | in-place | 100.7 to 105.1 | 35.2, 35.3 | 135.0 to 135.7 | 172 to 193 ms |
| Qwen2.5 7B 4-bit | copying | 67.2 to 69.5 | 30.4, 30.5 | 117.4 to 117.6 | 176 to 186 ms |
| Qwen2.5 7B 4-bit | in-place | 103.4 to 109.0 | 34.3 to 35.7 | 132.1 to 137.4 | 168 to 183 ms |

The first concurrency-1 request of each server start is cold; the rest hit the
prompt cache. Concurrency 1 now matches the dense storage backend (104.7 to
109.5 on Llama), and the paged pool keeps its concurrency-4 TTFT advantage over
dense (1.1 to 1.4 s there).

## Correctness

- Four different greedy chat prompts sent concurrently (two rounds) and seven
  sequential requests with shared-prefix prompt-cache restores return identical
  replies in both arms, on Llama 3.1 8B and Qwen2.5 7B (15 replies each).
- Unit test: an in-place write into one sequence's block leaves another block's
  rows intact, and a write into a block shared with another holder copies, so
  an existing view of the slab keeps its bytes. Dropping the exclusivity check
  makes the test fail.

## Not measured

CUDA and ROCm, concurrency above 4, and the remaining sync gap between the paged
and dense backends (81 to 87 against 105 to 110 before this change) is not
re-measured here beyond the concurrency-1 result above.
