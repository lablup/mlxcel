# DeepSeek V2 chunked prefill: KV cache growth parked in MLX's allocator (issue #3084), M5 Max, 2026-11-15

Issue #3084: DeepSeek V2's default (decompressed) MLA prefill peaked 12.85 GB above `MLXCEL_MLA_ABSORBED=1` at 2048-token pieces. The default path does not re-expand the latent cache per piece, as the issue suspected: it caches decompressed per-head K and V (`16 x (192 + 128)` values per token per layer) and appends to them. The gap has two parts, attributed below: the decompressed cache itself, and the previous cache buffers that every piece's growth leaves in MLX's allocator cache.

Host: Apple M5 Max, 128 GB, macOS 27. Main is `05312c60`; the branch is that commit plus this change; release builds. Model `deepseek-v2-lite-chat-4bit-mlx`, 28,705-token prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000`, greedy, 48 tokens, reasoning included in the compared text. Peak is the `/usr/bin/time -l` footprint, prefill is `--profile`. Each run follows a 75 s idle period. The main arm is the attribution build below with its probe off, which takes no new branch at runtime.

## Attribution (last full 2048-token piece, 26,624 tokens cached)

An attribution build (not committed) evaluated `h` and the layer's cache after every layer and printed MLX active, per-layer peak and allocator-cache memory. Every row is in `mla-v2-prefill-cache-growth-m5max-2026-11-15/per-layer-last-piece.csv`.

| path | weights | live KV, 27 layers | active at layer 26 | allocator cache, layer 0 to 26 | footprint |
|---|---|---|---|---|---|
| decompressed (default) | 8.8 GB | 7.9 GB (0.294 GB/layer) | 17.21 GB | 3.05 to 10.47 GB | 27.94 GB |
| absorbed | 8.8 GB | 0.9 GB (0.033 GB/layer) | 10.29 GB | 3.18 to 4.32 GB | 14.88 GB |

The allocator cache climbs 0.27 GB per decompressed layer and 0.04 GB per absorbed layer: each layer's growth concatenates its cache into a larger buffer, and the freed old buffer cannot serve any later request in the piece (they all ask for the larger size, and MLX only reuses a cached buffer within two pages of the request). The engine's per-piece flush (#672) releases them only after the piece. So of the 12.85 GB gap, about 7.0 GB is the decompressed layout (live KV) and about 6.2 GB is parked dead KV. Attention transients (per-layer peak minus active, about 2.1 GB on the default path and 2.6 GB absorbed) are bounded already.

## Change

`KVCache::reserve_rows` grows the buffer before the layer's graph is built, using the same growth rule `update_fp16` uses, queues the copy with `async_eval`, and empties the allocator cache at the next layer's reservation. DeepSeek V2 calls it on both the decompressed and the latent caches. Batching all layers into one reservation does not work: it holds every old and new buffer at once, and measured 27.97 GB.

## Results

| cell | peak main | peak branch | prefill main | prefill branch | greedy output |
|---|---|---|---|---|---|
| default, 2048 pieces | 27.98 GB | **21.04 GB** | 42.18 s | 42.77 s | identical |
| `MLXCEL_MLA_ABSORBED=1`, 2048 pieces | 15.12 / 15.13 GB | **14.35 / 14.36 GB** | 42.19 / 42.21 s | 42.51 / 42.25 s | identical |
| default, single pass | 54.44 GB | 54.43 GB | 52.72 s | 52.55 s | identical |

The single-pass cell never grows a cache (a first fill allocates once), so it does not change; its remainder includes the full-position logits tracked in #3070. In both arms the 2048-piece default output differs from the single-pass and absorbed outputs; that is the existing chunking behavior, not this change.

Variants from one diagnostic binary (`DIAG3084` mode switch), default path, 2048 pieces, in run order:

| mode | peak | prefill (two runs) |
|---|---|---|
| off (main behavior) | 27.98 GB | 41.97 / 42.15 s |
| eager growth, `async_eval`, flush at next layer (shipped) | 21.04 GB | 43.13 / 42.87 s |
| eager growth, blocking `eval` and flush per layer | 20.87 GB | 43.50 / 43.48 s |
| eager growth, no flush | 27.96 GB | 42.94 / 42.77 s |

The shipped mode costs about 2% prefill (0.8 s over 378 growths) for 6.94 GB less peak. Earlier sweeps on a shared host read 52 to 55 s for some branch runs and stayed near 43 s on main, but the single-binary sequence above, with the off mode as control, does not reproduce that.

Arithmetic is unchanged: the grown buffer is the one the lazy path built, and every cell's greedy output is byte-identical, so no logit trace is needed. Raw rows: `mla-v2-prefill-cache-growth-m5max-2026-11-15/runs.csv`.

## Reproduce

```bash
python3 scripts/bench/metal_sdpa_chunk/make_prompt.py 28000 > prompt_28000.txt
MLXCEL_PREFILL_CHUNK=2048 /usr/bin/time -l ./target/release/mlxcel generate \
  -m models/deepseek-v2-lite-chat-4bit-mlx -p "$(cat prompt_28000.txt)" -n 48 --temp 0 --profile --show-reasoning
# MLXCEL_PREFILL_CHUNK=0 for single pass; add MLXCEL_MLA_ABSORBED=1 for the absorbed rows.
```
