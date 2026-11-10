# Qwen3.8-Flash-Next measurements deferred from #2880, #2881, #2882, #2883, #2885 and #2917

These are the timing and memory measurements that the implementing PRs (#2908, #2906, #2912, #2909, #2911, #2918) left for a quiet host. Every phase ran one process at a time on an otherwise idle machine, in the order below, on one release binary.

## Environment

| Item | Value |
|---|---|
| Host | Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2 (25G83) |
| Apple GPU generation | 15 (`applegpu_g15d`) |
| Build | `cargo build --release --features metal,accelerate` |
| Branch @ sha | `main` @ `60a3270c` plus the two script changes in this PR |
| Harness | A: `scripts/bench_speculative.sh flashnext` and `flashnext4` (ABBA, 2 discarded warm-ups, 3 blocks, 6 samples per arm). B to D: offline `mlxcel generate` and a streaming chat-completion client against `mlxcel-server`, 5 measured repetitions per arm after one discarded warm-up. E: `scripts/macos_memory_soak.sh` |
| Wrapper | `scripts/with_indexers_paused.sh` |
| Targets | `models/Qwen3.8-Flash-Next-OptiQ-2bit` (`mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit`), `models/Qwen3.8-Flash-Next-4bit-mtp` (`mlx-community/Qwen3.8-Flash-Next-4bit-mtp` @ `825df623`), drafter `models/Qwen3.8-Flash-Next-MTP` (`sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` @ `be6e2bc3`) |
| Sampling | temperature 0 everywhere |

Background load: `audiomxd` held about 112% CPU (about one of 32 cores) from before the run until 11:39 local time, then about 78% after a restart, and dropped to 0% at 11:44 when the host's Jump Desktop session was reconnected. The cause was traced to that session, not to mlxcel. The quiet gate of `scripts/lib/bench_quiet.sh` was raised to `QUIET_LIMIT=130` so this load did not block the run; phase A's first pairing ran entirely with it, the second pairing's third prompt saw the restart, and phases B to E ran without it. No row of phase A was flagged for contention and every spread stayed at or below 1.5%.

`MLXCEL_QMV_WIDE` was left at its default. On this GPU generation the MTP exactness probe turns `qmv_wide` off for an MTP process, so the classic arm of phase A runs with `qmv_wide` and the MTP arm without; the ratio is what a `--draft-model` user gets.

## A. Native MTP head (#2880, #2917)

`scripts/bench_speculative.sh`, block size 4 (the drafter's own), prompts and token counts as defined in the script.

| Pairing | Prompt | Tokens | Classic tok/s | MTP tok/s | Ratio | Acceptance | Emitted per verify |
|---|---|---|---|---|---|---|---|
| OptiQ 2-bit + bf16 split head | source code | 300 | 42.8 | 55.7 | 1.30x | 0.644 | 2.93 |
| OptiQ 2-bit + bf16 split head | prose | 400 | 43.1 | 47.6 | 1.10x | 0.489 | 2.46 |
| OptiQ 2-bit + bf16 split head | enumeration | 400 | 43.1 | 76.0 | 1.76x | 0.983 | 3.95 |
| 4-bit + its embedded head | source code | 300 | 45.4 | 62.9 | 1.38x | 0.642 | 2.90 |
| 4-bit + its embedded head | prose | 400 | 45.7 | 45.5 | 1.00x | 0.370 | 2.11 |
| 4-bit + its embedded head | enumeration | 400 | 45.8 | 79.8 | 1.74x | 0.895 | 3.66 |

Spreads were 0.3% to 1.5% of the median in every arm. The ratio follows acceptance: at 0.37 on prose the 4-bit embedded head breaks even, while the bf16 split head on the 2-bit target still gains 10%. Output identity for these pairings was established in #2908 and #2918 and is not re-checked here.

## B. SSD expert streaming (#2882)

OptiQ 2-bit target, forced streaming (`MLXCEL_EXPERT_STREAM=1`) at two arena sizes against resident (`MLXCEL_EXPERT_STREAM=0`), no contiguous file. Arms interleaved per repetition.

Decode, offline `mlxcel generate -p "Write a short story about a lighthouse." -n 256 --temp 0`:

| Arm | tok/s (5 runs) | Hit rate | Misses | Read | I/O wait | Router sync |
|---|---|---|---|---|---|---|
| Resident | 42.70, 42.90, 42.96, 42.71, 42.90 | - | - | - | - | - |
| 24 GB arena (`25769803776`) | 22.88, 22.98, 23.06, 23.05, 22.94 | 89.8% | 14,065 | 25.3 GB | 1.67 s | 8.0 s |
| 8 GB arena (`8589934592`) | 18.38, 18.36, 18.30, 18.30, 18.38 | 74.9% | 34,464 | 65.2 GB | 4.17 s | 8.0 s |

The `[Expert arena: ...]` counters were identical in every repetition of an arm, so routing is deterministic. A separate run per arm with stdout captured gave the same 256-token text in all three (timing, load and arena lines removed before hashing), and the load lines reported 75.19 GB resident for the resident arm, 56.81 GB at the 24 GB arena and 41.33 GB at the 8 GB arena. Router synchronisation costs the same 8.0 s per run at both arena sizes and is larger than the I/O wait even at 8 GB: on this SSD the per-layer host sync, not the reads, sets the streamed decode speed.

Cold 4K-token prefill and decode through `mlxcel-server` (3,998 prompt tokens, 128 completion tokens, `POST /v1/cache/reset` before each request):

| Arm | Time to first token (5 runs) | Decode tok/s |
|---|---|---|
| Resident | 3.66 s every run | 37.2 to 37.3 |
| 24 GB arena | 5.05 to 5.08 s | 19.6 to 21.2 |
| 8 GB arena | 5.80 to 5.81 s | 16.7 to 17.1 |

The 128-token completion was byte-identical across all 15 server requests of the three arms (one SHA-256 prefix, `1420bb9c470f4486`).

Not covered: a Mac with less memory than the checkpoint, the contiguous file from `mlxcel prepare-expert-stream`, and streaming combined with the external PLE table.

## C. External PLE table (#2881)

`mlxcel prepare-external-ple -m models/Qwen3.8-Flash-Next-OptiQ-2bit -o models/Qwen3.8-Flash-Next-OptiQ-2bit-external-ple` (interleaved layout) wrote the 320,001,536-row, 29.80 GiB table in 15.7 s and linked 14 shards. Each mode served cold 4K and 32K prompts (3,998 and 30,436 tokens, 256 completion tokens, cache reset before each request).

| Mode | Footprint after load | RSS after the runs | 4K TTFT | 32K TTFT | Decode tok/s (4K / 32K) |
|---|---|---|---|---|---|
| Resident table (original checkpoint) | 78.5 GB | 78.6 GB | 3.66 to 3.73 s | 28.49 to 28.59 s | 37.3 / 31.6 |
| `MLXCEL_PLE_IO=direct` | 47.3 GB | 47.4 GB | 3.66 to 3.73 s | 28.48 to 28.77 s | 38.2 / 32.3 |
| `MLXCEL_PLE_IO=mmap` (default) | 47.3 GB | 52.3 GB | 3.66 to 3.73 s | 28.48 to 28.69 s | 38.1 / 32.2 |
| `MLXCEL_PLE_IO=ram` | 78.5 GB | 78.6 GB | 3.66 to 3.72 s | 28.46 to 28.66 s | 38.1 / 32.2 |

Footprint is the process RSS reported by `ps`. All four modes produced byte-identical completions for each prompt (`ab1ee99d52844563` at 4K, `0fa69716af52d30f` at 32K). The external table saves 31 GB resident with no prefill cost, and decode is about 2.4% faster than with the resident table. `vm_stat` swap-outs stayed at 0 throughout, which on a 512 GB host says nothing about the 96 GB and 64 GB machines the issue targets.

## D. Prompt-cache second turn (#2885)

One `mlxcel-server` with defaults. Each repetition resets the cache, sends turn 1 (system prompt plus a long user message cut from `docs/*.md`, 128 completion tokens), appends the returned answer and a new user question, sends turn 2 with the cache in place, then resets the cache and sends the same turn 2 again.

| Conversation | Turn 2 prompt tokens | Turn 2 TTFT, cached | Turn 2 TTFT, cache reset | Turn 1 TTFT |
|---|---|---|---|---|
| 16K | 16,240 | 0.51 to 0.56 s | 15.05 to 15.12 s | 14.96 to 15.02 s |
| 32K | 30,598 | 0.56 to 0.60 s | 28.64 to 28.90 s | 28.53 to 28.79 s |

Each cached turn 2 counted one snapshot hit. Snapshot size was about 584 MB per entry at 16K and 995 MB at 32K against the default store capacity of 2,147,483,648 bytes, so at 32K the store held two entries and evicted by LRU. The cached turn 2 was byte-identical to the cache-reset turn 2 in all 10 repetitions. This shape (the next turn appends to the previous one) is the one the history-boundary snapshot covers; the earlier divergence seen when a hit lands inside a cold history piece was not exercised here.

## E. macOS memory soak (#2883)

The first attempt failed before any write: `scripts/macos_memory_soak.sh` sent a chat completion without a `model` field and every setting got HTTP 400 (`missing field model`). This PR adds the field. Results of the rerun:

`scripts/macos_memory_soak.sh models/Qwen3.8-Flash-Next-OptiQ-2bit 40G`: one server per `MLXCEL_WIRED_LIMIT` setting, a TTFT probe before and after 20 GiB and 40 GiB file writes, with `vm_stat` and the process footprint sampled throughout. A second pass added `--wire-weights` (`MLXCEL_WIRE_WEIGHTS=1`, default limit only).

| `MLXCEL_WIRED_LIMIT` | `MLXCEL_WIRE_WEIGHTS` | TTFT before (s) | TTFT after (s) | Compressions | Swap-outs |
|---|---|---|---|---|---|
| default | 0 | 0.234 | 0.549 | 0 | 0 |
| none | 0 | 0.709 | 0.426 | 0 | 0 |
| 60G | 0 | 0.298 | 0.503 | 0 | 0 |
| default | 1 | 0.244 | 0.525 | 0 | 0 |

Compressions and swap-outs are deltas over each soak. Peak process footprint was 81.9 GB in every setting (MLX active memory 81.0 GB). Each TTFT cell is a single request, so differences of a few hundred milliseconds between settings are within what one sample can show: an earlier plain pass gave 0.234/0.558, 0.719/0.430 and 0.302/0.505 s. With 512 GB of RAM and about 82 GB in use, 40 GiB of page cache never pressured the wired weights, so this run confirms the script and the settings work end to end but cannot show the eviction the issue is about. That needs a 64 GB or 96 GB Mac.

Raw summaries and samples: `docs/benchmark_results/macos-memory-soak-lablup-macstudio-02-2026-11-10/` and `docs/benchmark_results/macos-memory-soak-lablup-macstudio-02-2026-11-10-wire-weights/`.
