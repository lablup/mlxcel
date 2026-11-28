# Releasing superseded FP16 KV buffers during chunked prefill, every model (issue #3092), M5 Max, 2026-11-15

Issue #3092: a chunked-prefill piece grows every layer's dense FP16 `KVCache`, and each superseded buffer is freed into MLX's allocator cache, where no later request in the piece can reuse it, until the engine flushes after the piece (#672). #3091 released those buffers for DeepSeek V2 only, through explicit `KVCache::reserve_rows` calls. The release now runs inside the shared FP16 growth in `KVCache::update_fp16` for every model: a multi-token growth of an existing buffer empties the allocator cache, builds the grown buffer, and queues its copy with `async_eval`. DeepSeek V2's explicit calls are gone.

Host: Apple M5 Max, 128 GB, macOS 27. Main is `49ea6cc9`; the branch is that commit plus this change; release builds (`--features metal,accelerate`), each copied aside before measuring. Prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000` (Llama 3.1's tokenizer reads it as 19,742 tokens, the others as 27,668 to 28,706), greedy, 48 tokens, reasoning included in the compared text. Peak is `/usr/bin/time -l` "peak memory footprint"; prefill is `--profile`. 75 s cooldown before every run, the shared GPU lock held only during the run. Each 2048-token cell runs twice in reversed order (pair `a`: main then branch; pair `b`: branch then main), and the table reports the mean of the two. Every row is in `kv-growth-release-all-models-m5max-2026-11-15/runs.csv`.

## 2048-token pieces (`MLXCEL_PREFILL_CHUNK=2048`)

| model | prompt tokens | peak, main | peak, branch | change | prefill, main (ms) | prefill, branch (ms) | change | output |
|---|---|---|---|---|---|---|---|---|
| qwen3-4b-4bit | 27,668 | 10.94 GB | 7.26 GB | -3.68 GB (-34%) | 10,558 | 10,446 | -1.1% | identical |
| meta-llama-3.1-8b-instruct-4bit | 19,742 | 10.39 GB | 8.09 GB | -2.30 GB (-22%) | 8,919 | 9,180 | +2.9% | identical |
| gemma-3-4b-it-4bit | 28,706 | 5.25 GB | 4.86 GB | -0.39 GB (-7.5%) | 4,130 | 4,243 | +2.7% | identical |
| deepseek-v2-lite-chat-4bit-mlx | 28,705 | 20.20 GB | 20.20 GB | 0 | 43,706 | 43,775 | +0.2% | identical |

The saving scales with the dense FP16 layers that grow: Qwen3 4B grows all 36 layers, Llama 3.1 8B all 32, Gemma 3 4B only its 5 global layers of 34 (the 29 sliding-window layers use `RotatingKVCache`, which this path does not touch). DeepSeek V2 already released through `reserve_rows` on main, so an unchanged peak is the expected result: it confirms the shared path covers both its decompressed and its latent caches.

For Qwen3 and Llama the spread between the two runs of one cell is larger than the change, and the first run of each pair was the slower one (Llama: 10,029 and 9,895 ms first, 8,466 and 7,809 ms second), so the order-reversed mean is the figure to read. It stays within 3% for every family.

## Single pass (`MLXCEL_PREFILL_CHUNK=0`)

| model | peak, main | peak, branch | prefill, main (ms) | prefill, branch (ms) | output |
|---|---|---|---|---|---|
| qwen3-4b-4bit | 9.94 GB | 9.94 GB | 9,361 | 9,756 | identical |
| meta-llama-3.1-8b-instruct-4bit | 10.50 GB | 10.56 GB | 10,197 | 8,824 | identical |
| gemma-3-4b-it-4bit | 7.03 GB | 7.02 GB | 5,154 | 4,173 | identical |

A single pass is a first fill followed by decode rows, neither of which takes the releasing growth, so these are one run each as a no-change check; the timings are single samples. With the release, 2048-token pieces now peak below a single pass on all three families.

## Allocator-cache guard

`src/lib/mlxcel-core/tests/kv_growth_release.rs` runs a 12-layer piece (8 heads, head dim 128, 4,096 rows held, 2,048 appended) through `KVCache` and through the lazy growth main used, and reads MLX's allocator cache after the piece's graph runs. On this host the releasing piece leaves 151 MB and the lazy one 403 MB (16.8 MB of old buffer per layer). With the release disabled, both read 403 MB and the guard fails.
