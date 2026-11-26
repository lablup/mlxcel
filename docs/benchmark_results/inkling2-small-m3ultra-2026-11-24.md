# Inkling 2 Small on the real checkpoint, M3 Ultra (#3373)

First runs of `thinkingmachines/Inkling-2-Small` (revision 885d78bf, `model_type: inkling2`, 48 layers, hidden 4608, 384 experts, no `loop_config`) in mlxcel, against mlx-vlm 466069f5 on the same host. The checkpoint is 366.48 GiB on disk (24 shards plus `mtp.safetensors`), and the target weights alone take 363.27 GiB, so only one runtime can hold it at a time: every mlxcel and mlx-vlm run below was a separate process, one after the other, under `/usr/bin/lockf -k /Users/devops/Development/.mlxcel-gpu.lock`.

## Environment

| Item | Value |
|---|---|
| Host | Apple M3 Ultra, 512 GB, macOS 27.2.1, `iogpu.wired_limit_mb=491520`, shared with other worktrees (load averages 1.9 to 3.8 during the measured runs) |
| mlxcel | `cargo build --release --features metal,accelerate --bin mlxcel --bin mlxcel-server --example logit_trace` on `9fc73820` plus this PR (the "before" load row uses `9fc73820` alone) |
| Reference | mlx-vlm 466069f5 (`uv run --python 3.12 --with-editable references/mlx-vlm`), loaded the way its Inkling 2 README loads Small: `mx.set_cache_limit(0)`, `lazy=True`, then the vision tower and each layer evaluated in turn |
| Compare | `scripts/compare_logit_traces.py REF.tsv MLXCEL.tsv --decided 2` |

## Load and memory

`mlxcel generate -m models/inkling-2-small -p "Explain why the sky is blue." -n 128 --temp 0 --show-reasoning --profile`, with `MLXCEL_PROFILE_TTFT=1` and `/usr/bin/time -l`. "Lowest free" is the lowest `memory_pressure` free percentage, sampled every 5 s.

| Build | Load | MLX peak after load | First prefill (22 tokens) | Process peak footprint | Lowest free |
|---|---|---|---|---|---|
| `9fc73820` | 54.7 s | 362.66 GiB | 8202 ms | 458.9 GiB | 10% |
| this PR | 59.3 s | 367.04 GiB | 336 ms | 368.5 GiB | 27% |

On `9fc73820` the eager materialization reads the 1348 checkpoint tensors (389.40 GB), but sanitizing turns every interleaved expert `w13` into contiguous gate and up copies, and those copies were left to the first token. That token built all 48 layers' copies in one graph while the sources and the freed buffers in MLX's cache were still held, which took the process to 458.9 GiB and the host to 10% free memory. The PR evaluates the sanitized tensors one decoder layer at a time with the MLX cache limit at 0 and restores the limit afterwards, as upstream does. The first token then costs a normal prefill, and the peak stays 4.4 GiB above the resident weights. The `[TTFT] load` line now reports 868 tensors materialized by the policy and 480 unreferenced, which are the `w13` sources already consumed layer by layer.

mlx-vlm with its README procedure: load 62 s, MLX peak 394.8 GB (367.7 GiB).

`mlxcel inspect -m models/inkling-2-small` (8192-token context): weights 363.27 GiB from the safetensors headers, KV 1.50 GiB, and "This load" (weights, KV and activations) 364.82 GiB, 0.6% below the measured MLX peak of 367.04 GiB. The fit check adds the generic 1.2x allocator headroom on top (437.78 GiB total, 48.62 GiB to spare), which is a margin, not a peak prediction. Before this PR the measured process peak of 458.9 GiB was above even that total. `memory_estimate.rs` needs no change.

## Parity with mlx-vlm

| Check | Result |
|---|---|
| Greedy, `Explain why the sky is blue.`, 128 tokens, default effort (0.9) | text identical to mlx-vlm, before and after the PR, and in all four kernel arms below |
| Text trace, 4 x 256 positions, corpus: bytes 4000 to 12000 of `docs/supported-models.md` at `fd879f20` (the #3378 corpus), `logit_trace ... 256 4 8` against `mlx_lm_logit_trace.py ... --mlx-vlm` | 30 / 1024 top-1 differences, 0 / 409 at decided positions, largest gap at a disagreement 0.320, every disagreement at reference rank 2 or 3; perplexity 7.0850 reference, 7.0372 mlxcel (-0.68%) |
| Image prompt ids, `tests/fixtures/sapiens2/person.jpg` (1280 x 720), `What is in this image?` | 600 ids, identical |
| Image trace, the first 128 tokens of mlxcel's own greedy reply (126 scored rows), `MLXCEL_TRACE_IMAGE=... logit_trace ... 128 1 8` against `mlx_vlm_image_logit_trace.py ... --bounded-load` | 4 / 126 top-1 differences, 0 / 88 decided, largest gap 0.260; perplexity 1.6326 reference, 1.6167 mlxcel (-0.98%) |

`--bounded-load` on the image script and the cache limit around the text script's sliced evaluation are this PR's; one `mx.eval` over every Small parameter does not fit beside its transient copies.

## Server

`mlxcel-server -m models/inkling-2-small` (native batched decode, `max_batch_size=4`), temperature 0, each request sent non-streamed and then streamed:

| Request | Result |
|---|---|
| Text, `Explain why the sky is blue in two sentences.` | `finish_reason: stop`; streamed content and reasoning equal the non-streamed ones |
| `image_url` (the same JPEG as a data URL), `What is in this image?` | `finish_reason: stop`, describes the two men on the sideline; streamed equals non-streamed |
| `get_weather` tool, `What is the weather in Warsaw right now?` | `finish_reason: tool_calls`, `get_weather` with `{"city":"Warsaw"}` in both modes, empty content |

## Decode and prefill, kernels off and on

A sets `MLXCEL_INKLING2_KERNELS=0`, B leaves the kernels on. Order A, B, A, B with a 20 s pause before each run. "Sky" is the 128-token command above (22 prompt tokens). "Long" is the 8000-byte corpus as the prompt (2274 tokens), 32 generated tokens.

| Run | Sky prefill tok/s | Sky decode tok/s | Long prefill tok/s | Long decode tok/s |
|---|---|---|---|---|
| A1 | 64.26 | 18.91 | 359.47 | 16.29 |
| B1 | 65.03 | 19.39 | 362.50 | 16.95 |
| A2 | 64.14 | 18.91 | 362.47 | 16.43 |
| B2 | 62.44 | 19.50 | 362.42 | 17.01 |
| Mean A / B | 64.20 / 63.74 | 18.91 / 19.45 (+2.8%) | 360.97 / 362.46 | 16.36 / 16.98 (+3.8%) |

mlx-vlm decodes the sky prompt at 20.50 and 20.53 tok/s (two runs in one process), 5% ahead of mlxcel's B arm. Its 22-token prefill reads 73.5 tok/s once warm. The four sky runs print the same text (md5 `a27d7b44`).

Kernel gate evidence, from `RUST_LOG=warn,mlxcel::models::inkling2=debug`:

```
Inkling 2 kernels: short convolution on, attention head preparation off, grouped expert matvec off
Inkling 2 short convolution kernel dispatched (first use)
```

The head preparation and grouped expert kernels need `loop_config`, which Small does not have, the same gate as upstream. Both are switched off when the model is built, so neither can run, including in the drafter's verify forwards below, which is where the grouped kernel would otherwise run. The short convolution kernel takes decode steps (`L <= 8`) and the prefill runs the ops path, which is why the kernels move decode and not prefill.

## Batched decode (#3392 path)

Server aggregate throughput, distinct 3- to 7-word prompts, 128 tokens each (`ignore_eos`), temperature 0, in the order C1, C4, C1, C4, C1, C4:

| Concurrency | Runs (tok/s aggregate) |
|---|---|
| C1 | 10.84 / 17.20 / 10.75 |
| C4 | 36.88 / 36.45 / 36.98 |

Four concurrent requests reach 36.8 tok/s in aggregate, 1.9x the CLI's single-stream decode. The C1 wall times vary between 7.4 and 11.9 s for the same 128 tokens and are recorded as measured; the C4 runs agree within 1.5%.

## DSpark drafter (#3384)

Small's `mtp.safetensors` (3.42 GB) holds a 5-layer drafter (hidden 4608, local layers 0, 1, 3, 4) that reads 8 target visits (5, 11, 17, 23, 29, 35, 41, 48), against Mini's 3 layers and 4 visits. It binds through the same code with no change (`Inkling 2 DSpark drafter loaded and bound (block_size = 8)`).

Exactness probe: `mlxcel generate --draft-model models/inkling-2-small` declines, as on Mini. Verify block position 0 differs from the single-token chain in 503855 of 800232 logit bytes, also with `qmv_wide` disabled, and the command exits with the named error. The numbers below were taken with `MLXCEL_MTP_ALLOW_INEXACT=1`.

256 tokens at temperature 0, `--show-reasoning`, A = classic decode, B = drafter, order A, B, A, B per prompt with a 20 s pause before each run. Prompt 1: `Explain why the sky is blue.` Prompt 2: `Write a Python function that returns the n-th Fibonacci number, with a short docstring.`

| Prompt | Classic tok/s (A1 / A2) | Drafter tok/s (B1 / B2) | Speedup | Rounds | Acceptance rate | Mean accepted per round | Emitted per verify |
|---|---|---|---|---|---|---|---|
| 1 | 19.31 / 19.31 | 19.11 / 19.43 | 1.00x | 106 | 0.203 | 1.42 | 2.41 |
| 2 | 19.28 / 19.26 | 32.74 / 33.35 | 1.71x | 60 | 0.464 | 3.25 | 4.25 |

An 8-row verify forward costs about 107 ms on Small (`verify_sync_ms / rounds`), twice a 52 ms decode step, and drafting adds about 13 ms per round, so a round must emit more than about 2.3 tokens to pay off: at 2.41 prompt 1 breaks even, at 4.25 prompt 2 gains 71%. Both arms are deterministic (the two runs of each arm print the same text). Because the verify block is not byte-identical to the decode chain, drafter-on text leaves classic text after about 629 characters on prompt 1 and 286 on prompt 2, at near ties ("at sunrise/sunset longer path" against "at sunset path longer"). mlx-vlm's drafter was not run on Small; on Mini #3384 found its proposals equal to mlxcel's in every round where the contexts agree.

## Raw logs

The per-run logs were kept outside the repository; every value above is copied from them.
