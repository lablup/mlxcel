# Metal command-buffer input budget during decode, M1 Ultra, 2026-09-21

Starting point: command-r7b 4-bit decode on this host read 102-104 tok/s, and a Python MLX replica of the same decode graph (same primitives, same kernels, same MLX era) read 108.5. The 0.37 ms per token between them was independent of context length (16, 500 and 2048 tokens), so it was not attention.

Headline: **mlxcel's decode on M1 through M4 has been splitting every token into about 23 Metal command buffers since `MLX_MAX_OPS_PER_BUFFER` was raised to 1000 (#353), because MLX's second commit trigger, the input budget `MLX_MAX_MB_PER_BUFFER`, was left at its default of 50.** Raising that budget to 1000 for decode moved decode by +3% to +21% on eight of the nine checkpoints measured (Gemma 3 4B was flat) and removed most of the gap to the replica. The same budget during prefill roughly doubles peak memory on long prompts, so it is applied around decode steps only, through a runtime override in the `mlx/backend/metal/device.cpp` overlay.

## Environment

| Field | Value |
|---|---|
| Host | Mac Studio, Apple M1 Ultra, 128 GB unified memory, macOS 27.0 (26A428) |
| Apple GPU generation | 13 (`d`, Ultra): MLX defaults are 50 ops and 50 "MB" per command buffer; mlxcel raises ops to 1000 |
| Base | `main` at `0bbfa95d`, MLX pin `81ba1c6a` |
| Build | `cargo build --release --features metal,accelerate` |
| Harness | `mlxcel-bench-decode` with the `scripts/bench_decode.sh` shape: `--prompt-tokens 500 -n 128 --ignore-eos --warmup-tokens 20` unless noted; cells interleaved, three runs per cell unless noted |
| Background | Photos library analysis running throughout (`mediaanalysisd` 60-140% CPU for two days, `com.apple.photos.ImageConversionService` 3.2-3.5% GPU time), load average about 4, Time Machine not running |

The background load depresses both arms of every comparison below equally. By the final comparison it had subsided, and suspending the indexers outright did not move `main`'s absolute number (see the last section).

## Mechanism

MLX commits a Metal command buffer when either counter passes its cap (`CommandEncoder::needs_commit`, `mlx/backend/metal/device.cpp`):

```cpp
return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
```

`buffer_sizes_` sums `array::data_size()` over the distinct input arrays of the buffer. That is an element count, not bytes. A 4-bit 7B model stores about 1.1G packed `u32` elements, and decode reads all of them every token, so with the op cap at 1000 the input cap of 50 (52M elements) commits a buffer every one to two layers: about 23 per token on command-r7b. A bf16 checkpoint stores one element per weight rather than eight weights per packed word, so it commits more often still.

A Metal System Trace attached to a running decode (`xctrace record --attach`, 2 s) showed the GPU busy 89.5% of the time, with an idle gap between consecutive command buffers of 42.8 us median (p90 52 us). The Python replica, traced the same way, showed the same gap structure (88.1% busy, 41.8 us median), which is why the replica also gains from a larger budget, only less (+2.5%, see below).

## Budget sweep, command-r7b 4-bit

`MLX_MAX_MB_PER_BUFFER` set in the environment, op cap at mlxcel's 1000.

| Budget | Prefill tok/s | Decode tok/s |
|---|---|---|
| 50 (default) | 665-668 | 101.4-103.3 |
| 100 | 661-676 | 107.4-109.2 |
| 200 | 657-678 | 106.4-107.5 |
| 400 | 655-671 | 107.3-109.6 |
| 1000 | 656-675 | 109.4-110.3 |
| 4000 | 638-654 | 109.2-109.9 |
| 100000 | 650-654 | 108.7-109.6 |

Decode saturates by 1000; prefill starts losing at 4000. The Python replica under the MLX wheel's own defaults (50 ops, 50 budget): 107.9-108.4 at 50, 110.7 at 1000.

## Across families, default versus 1000

| Checkpoint | Decode, default | Decode, 1000 | Prefill, default | Prefill, 1000 |
|---|---|---|---|---|
| command-r7b 4-bit | 101.4-103.3 | 109.4-110.3 (+7%) | 665-668 | 656-675 |
| Llama 3.1 8B Instruct 4-bit | 99.0-99.2 | 103.4-105.1 (+5.6%) | 713-716 | 696-700 |
| Qwen2.5 7B Instruct 4-bit | 98.2-100.1 | 106.4-108.1 (+8%) | 754-762 | 722-745 |
| Gemma 3n E4B 4-bit | 59.7-60.6 | 62.1-62.4 (+3%) | 759-769 | 735-762 |
| Gemma 3 4B 4-bit | 95.6-97.3 | 96.3-96.9 (flat) | 917-935 | 913-953 |
| Granite 4.0 H Tiny 4-bit | 103.3-104.0 | 113.4-113.8 (+10%) | 1605-1612 | 1607-1620 |
| Qwen3-30B-A3B 4-bit | 72.9-76.0 | 90.0-90.4 (+20%) | 837-845 | 780-851 |
| Mixtral 8x7B Instruct 4-bit | 51.0-52.1 | 61.9-62.8 (+21%) | 323-326 | 320-327 |
| Llama 3.1 8B Instruct bf16 | 34.2-34.7 | 40.1-40.4 (+17%) | 769-777 | 751-755 |

A second, interleaved pass on the two dense models that showed a prefill loss, at two prompt lengths:

| Checkpoint | Prompt | Budget | Prefill | Decode |
|---|---|---|---|---|
| Llama 3.1 8B 4-bit | 500 | 50 / 400 / 1000 | 717-718 / 691-714 / 715-721 | 96.3-99.6 / 103.6-104.9 / 103.6-105.6 |
| Llama 3.1 8B 4-bit | 2048 | 50 / 400 / 1000 | 751-759 / 737-742 / 742-755 | 91.3-93.0 / 92.5-93.0 / 92.5-93.9 |
| Qwen2.5 7B 4-bit | 500 | 50 / 400 / 1000 | 747-762 / 749-760 / 732-747 | 98.6-100.6 / 106.6-108.0 / 106.6-108.3 |
| Qwen2.5 7B 4-bit | 2048 | 50 / 400 / 1000 | 799-808 / 790-797 / 790-794 | 96.7-96.9 / 99.0-100.5 / 100.3-101.5 |

No decode cell regressed. Prefill at 1000 is flat to -1.4% on the 4-bit models and -2.7% on the bf16 one.

## Why the budget is scoped to decode

The input budget is also what bounds how long prefill activations stay alive: a buffer's intermediates are released when it completes. MLX peak memory at a 2048-token prompt, 32 generated tokens:

| Checkpoint | 50 | 100 | 200 | 400 | 1000 |
|---|---|---|---|---|---|
| Qwen2.5 7B 4-bit | 6.01 GB | 6.90 | 8.93 | 11.09 | 12.77 |
| Qwen3-30B-A3B 4-bit | 19.75 GB | 20.74 | 23.36 | 28.78 | 36.19 |
| command-r7b 4-bit | 7.11 GB | | | | 13.94 |
| Llama 3.1 8B bf16 | 17.93 GB | | | | 24.24 |

With one generated token the delta is the same (Qwen2.5: 6.01 to 12.59 GB at 2048, 5.07 to 6.56 GB at 512, +0.43 GB at a 64-token prompt), so it is prefill, and it grows with prompt length. No constant serves both phases, and `Device` latches the budget from the environment once. The overlay adds a process-wide override that `needs_commit` consults when it is non-zero; `DecodeCommandBufferBudget` raises it after a prompt has been encoded and restores it when the decode loop or decode step ends.

## Result with the decode-only switch

Peak memory at a 2048-token prompt with the switch: Qwen2.5 7B 4-bit 6.07 GB, Qwen3-30B-A3B 4-bit 19.75 GB, the same as the device default (6.01 and 19.75), against 12.77 and 36.19 with the budget applied to both phases.

command-r7b 4-bit, same binary, `MLXCEL_DECODE_MB_PER_BUFFER` unset versus `0`, three interleaved runs: decode 110.1-111.2 versus 104.2-104.4, prefill 671-680 versus 674-676.

Final interleaved comparison against `main` at `0bbfa95d`, eight ABBA pairs per arm, `--prompt-tokens 500 -n 128 --ignore-eos`. The measured branch also carried a cohere2 residual-add fusion that is submitted separately; on its own it is worth +1.1% (measured with the budget pinned on both arms), so about 6% of the gain below is this switch. The same-binary on/off comparison above isolates it.

| Condition | `main` decode, median (range) | Branch decode, median (range) | Prefill, `main` / branch |
|---|---|---|---|
| As found (Photos analysis idle by then, load average 3.3) | 102.69 (100.33-103.02) | 109.88 (108.94-110.37) | 670.5 / 671.5 |
| Indexers suspended (`scripts/with_indexers_paused.sh`, plus `suggestd`) | 102.77 (102.23-103.18) | 109.50 (108.73-110.06) | 670.4 / 670.6 |

The op cap was re-checked with the switch active: 50 gives 106.3, 100 gives 108.1, 200 and 400 give 108.4-109.8, and 1000 (mlxcel's default since #353) gives 109.2-110.1, so it stays.

## Server path

`mlxcel-server` on command-r7b, 128-token `/completion` requests at temperature 0, ten consecutive requests per server start, two starts per arm. Requests the lookahead gate admits (the common case) decode pipelined; requests it rejects (`ignore_eos` and other token bias, penalties, per-token logprobs, grammar) decode synchronously, one step encoded and then waited on.

| Path | Switch off | Switch on |
|---|---|---|
| Lookahead (no `ignore_eos`), `mlxcel_batch_decode_lookahead_steps_total` +125 per request | 106.8-110.1 (median 110.0) | 115.8-119.1 (median 118.5), +7.7% |
| Synchronous (`ignore_eos`), budget raised around the step | 97.8-101.8 | 67.6-102.1, erratic |
| Synchronous, budget left on the device default (shipped) | 100.6-101.1 | 100.5-102.1 |

A synchronous step with one large buffer cannot start on the GPU until the whole step is encoded, so the raised budget removes the overlap between CPU encoding and GPU execution inside the step, and the step time picks up the variance of CPU encoding. The switch is therefore applied only around pipelined work: the server's steady lookahead tick and its prime, and the generate loops (except under `MLXCEL_FORCE_SYNC`).

Two earlier readings in this investigation were artifacts of measuring the server with `ignore_eos`, which takes the synchronous path: "server decode runs 9% below the CLI harness" (it does not; pipelined server decode is faster than the harness, which itself decodes with the `ignore_eos` bias), and "the first request after a start is slower with the switch on" (it was the synchronous-path instability, seen on whichever request it hit). The CLI generate loop has no first-request penalty either: single-shot `mlxcel generate -n 128` with no warmup reads 104.4-104.8 on `main` and 113.9-114.2 on the branch.

## Not measured

- M2, M3 and M4 base/Pro/Max. The switch follows the `MLX_MAX_OPS_PER_BUFFER` gate (M1 through M4), which was also set from Ultra measurements; the mechanism is the same, the magnitude is not known.
- M5 and later: MLX's defaults are kept in both phases, as for the op cap.
- Speculative decoding loops (DFlash round loop, MTP verify): not wired to the switch.
- Decode beyond 2048 tokens of context. A 600-token generation, which crosses the 256-token cache clear twice, holds the gain (104.5-104.7 on `main`, 110.8-111.2 on the branch, three runs each).
