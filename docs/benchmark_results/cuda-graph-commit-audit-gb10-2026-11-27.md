# CUDA graph commits per decode step on GB10: which launches pay for a buffer they read part of (issue #3387), 2026-11-27

MLX's CUDA encoder commits the graph it is capturing when the summed `data_size()` of the graph's inputs passes a per-device budget, 25 << 20 elements on DGX Spark (`get_graph_limits`, `needs_commit` in the `device.cpp` overlay). #3406 took the fused paged-attention kernels' inputs out of that count, because they are handed a whole pool slab and read the pages their page table names. Issue #3387 asks which other paths pay the same way. This page records the audit, the one change it led to, and the decode rate before and after.

Host: NVIDIA GB10 (DGX Spark, sm_121, 121 GiB unified memory, driver 580.178.04), Ubuntu 24.04.5, kernel 7.0.0-1019-nvidia, CUDA 13.0 (`nvcc` 13.0.88), Nsight Systems 2025.3.2. `mlxcel` 0.9.0-beta.2: base is `main` at `944dbfc2`, MLX pin `fc344c98`, Rust 1.97.1. Every binary is `cargo build --locked --release --features cuda --bin mlxcel-server`, copied aside with its tree hash and sha256 before measuring. No build, test or other model ran beside a measurement.

Checkpoints: `mlx-community/Qwen3-0.6B-4bit` (revision `73e3e38d`), `mlx-community/Qwen3-30B-A3B-4bit` (`d388dead`), `mlx-community/gemma-4-26b-a4b-it-4bit` (`0d77464e`), and local copies of `Kimi-VL-A3B-Thinking-4bit` and `LFM2-8B-A1B-4bit` whose download revision was not recorded (`config.json` sha256 `de2a7d0d5123b3e1` and `3aab6586df470e01`).

## Method

One server per cell, `mlxcel-server -m <model> --parallel 4 --no-prompt-cache -c 65536`, under `nsys launch --trace=cuda --cuda-graph-trace=node`. Two warm-up requests, then `nsys start`, the cell, `nsys stop`, so the collection holds the cell only and is flushed while the server is alive. A profile ended by stopping the server lost the last second of events; those traces were discarded. A cell is `B` concurrent streaming `/v1/completions` requests of a `P`-token prompt (the prompt builder of [`bench_decode.py`](issue-3385-qwen3-8b-gb10-2026-11-24/bench_decode.py)), greedy, `ignore_eos`.

The count is `cudaGraphLaunch` calls per decode step in a window in which every row streams: the last 400 steps of a 528-token generation at `B = 1`, and the stretch between the latest first token and the earliest last token at `B = 4` (197 to 766 steps; for the three larger models the collection itself was started once every row was streaming and stopped 300 steps later). The per-layer kernels come out at whole numbers per step in every cell (28.0 paged-attention launches and 1.0 argmax on Qwen3-0.6B, 48.0 of each MoE kernel on Qwen3-30B-A3B, 78.0 `gather_qmv_kernel` on Kimi-VL-A3B), which is the check that the window holds whole steps. "Raised" is the same binary with `MLX_MAX_MB_PER_BUFFER=1000`, the no-byte-budget reference the issue names; the op budget stays at 20 in both arms.

[`scripts/nsys_cuda_graph_launches.py`](../../scripts/nsys_cuda_graph_launches.py) reads an exported trace without the client's timestamps: it takes the argmax kernel as the step marker, groups the kernels by the `cudaGraphLaunch` that ran them, and reports the graphs per step and the kernels held by graphs of at most 20 nodes, which are the ones a byte-budget cut or the end of an eval left short. Its median per-step count agrees with the windowed one in every cell it was run on.

Rows: [`graph-launches.csv`](cuda-graph-commit-audit-gb10-2026-11-27/graph-launches.csv). `binary` is `main`, `main-without-the-3406-name-match` (the known-case arm) or `change` (the tree of the commit that added this page).

## The known case first

The method was first held to what #3385 recorded. The `noexempt` binary is `main` with the `mlxcel_paged_attention` name match of `custom_kernel.cpp` turned off, which is the accounting before #3406. Qwen3-0.6B-4bit, 8,193-token prompts, fused paged v2 decode, graph launches per step by batch size (one trace of four rows that join and leave; every step of a batch size gave the same count):

| binary | budget | B = 1 | B = 2 | B = 3 | B = 4 |
|---|---|---:|---:|---:|---:|
| `noexempt` | default | 29 | 57 | 57 | 85 |
| `main` | default | 26 | 42 | 55 | 68 |
| `main` | raised | 26 | 42 | 55 | 68 |
| `noexempt` | raised | 26 | 42 | 55 | 68 |

With the exemption off, 28 graphs per step end at the `mlxcel_paged_attention_v2_partial` launch, one per layer; on `main` none does.

The #3385 record ([`qwen3-0.6b-profile.txt`](issue-3385-qwen3-8b-gb10-2026-11-24/qwen3-0.6b-profile.txt)) is 47.8 launches per step before the change and 37.6 after, over a window whose batch size varied: its `qmv_kernel` count puts 33% of the steps at `B = 1` and its `rope_single` count puts the mean batch at 1.74. Weighting the rows above by that mix gives 47.8 to 48.7 for `noexempt` and 37.6 for `main`, depending on how the remaining 7% of row-steps split between `B = 3` and `B = 4`. The 54-against-29 figure in the issue text is Gemma 3 on the sliding-window branch of #3334; Gemma 3 is not pool-backed on `main`, so that cell cannot be run here.

## The audit

Graph launches per decode step, default budget against raised, on `main`:

| Path | Model | Context | B | Default | Raised | Cut by |
|---|---|---:|---:|---:|---:|---|
| Fused paged decode (v2 partial, merge) | Qwen3-0.6B-4bit | 8,193 | 1 | 26.0 | 26.0 | nothing: exempt since #3406 |
| Fused paged decode | Qwen3-0.6B-4bit | 8,193 | 4 | 68.0 | 68.0 | nothing |
| Fused paged decode | Qwen3-0.6B-4bit | 3,073 | 4 | 68.0 | 68.0 | nothing |
| Paged sparse decode, synthetic | MiniMax-M3 geometry | 65,536 | 1 | 1.00 | 1.00 | nothing: same kernels, same name match |
| Pool gather, decode under the v2 floor | Qwen3-0.6B-4bit | 3,073 | 1 | 58.0 | 27.0 | builtin `take` on the K and V slab |
| Pool gather, decode under the v2 floor | Qwen3-0.6B-4bit | 257 | 4 | 281.0 | 90.0 | builtin `take` |
| Pool gather, decode, `MLXCEL_PAGED_ATTENTION_NATIVE=0` | Qwen3-0.6B-4bit | 8,193 | 1 | 58.0 | 27.0 | builtin `take` |
| Pool gather, decode, `MLXCEL_PAGED_ATTENTION_NATIVE=0` | Qwen3-0.6B-4bit | 8,193 | 4 | 309.0 | 97.0 | builtin `take` |
| Pool gather, prompt-lookup verify, per emitted token | Qwen3-0.6B-4bit | 3,073 | 1 | 10.6 | 5.5 | builtin `take` |
| MLA absorbed decode (composed) | Kimi-VL-A3B-Thinking-4bit | 9,217 | 1 | 134.0 | 76.1 | builtin `gather_qmm` on the expert stacks |
| MLA absorbed decode (composed) | Kimi-VL-A3B-Thinking-4bit | 9,217 | 4 | 215.0 | 155.1 | builtin `gather_qmm` |
| Fused decode-MoE kernels | Qwen3-30B-A3B-4bit | 8,193 | 1 | 147.0 | 60.1 | `moe_gateup_kernel_cu`, `moe_down_kernel_cu` |
| Fused decode-MoE kernels | LFM2-8B-A1B-4bit | 9,218 | 1 | 48.0 | 29.0 | `moe_gateup_kernel_cu` |
| MoE at B > 1 (`gather_qmm`) | Qwen3-30B-A3B-4bit | 8,193 | 4 | 290.0 | 169.6 | builtin `gather_qmm` |
| MoE at B > 1 (`gather_qmm`) | LFM2-8B-A1B-4bit | 9,218 | 4 | 66.0 | 56.0 | builtin `gather_qmm` |

Whole cells where no decode-only window exists, default against raised, with identical text in both arms: one 8,193-token prefill at `B = 1` launches 1,235 graphs against 814 on Qwen3-0.6B (of its 538 graphs of at most 20 kernels, 221 have a slab gather as their last kernel; the others hold launches that do read more than the budget, the cuDNN SDPA, the SwiGLU graph and RoPE over a 2,048-token chunk), 1,822 against 541 on Kimi-VL-A3B, 2,407 against 730 on Qwen3-30B-A3B and 1,122 against 380 on LFM2-8B-A1B. Four concurrent 3,073-token prompt-lookup requests of 1,024 tokens each launch 46,242 graphs against 24,056; their rows finish before the later ones start streaming, so that cell has no `B = 4` window.

What each row is:

- **Fused paged decode.** Unaffected. The exemption of #3406 is what keeps it so (first table).
- **Paged sparse decode.** `paged_v2::sparse` runs the same two kernels over a `page_size = 1` table, so the name match covers it. The only family wired to it is MiniMax-M3, whose checkpoint does not fit this host (`docs/sparse-paged-decode.md`), so the trace is of `examples/sparse_paged_decode_bench` (MiniMax-M3 geometry, 2,048 selected rows of a 65,536-row allocation, one attention call per eval): 1.00 graph per step on `main` in both arms, 2.00 with the exemption off. The M3 layout (K one head wider than V) declines beyond batch 1, so there is no `B = 4` cell.
- **Pool gather.** Affected, and a builtin. Every step the fused kernel declines gathers each sequence's K and V out of the slab with `take`, and `Gather::eval_gpu` counts the slab whole (67.1M elements here). A step holds one graph that ends at the first gather and one that holds only the second, per layer and per row: 56 short graphs at `B = 1`, 224 at `B = 4`, none under the raised budget. Prompt-lookup verification runs on paged storage only while the context is under 4,608 tokens (`docs/CONTINUOUS_BATCHING.md`); the 8,193-token verify cells ran no verify round and are the plain fused decode. At 3,073 tokens a round emits 8 tokens, so the row is 85 graphs per round against 44.
- **MLA absorbed decode.** `mlxcel_mla_absorbed_decode_partial`, `_partial_quantized` and `_merge` have no CUDA port (`mla_absorbed_decode_ports`, `.cuda = nullptr`); CUDA takes the composed path, which attends over the slice `update_and_fetch` returns, not the preallocated buffer. Nothing in the attention cuts a graph at 9,217 tokens. What does is the model's MoE: `deepseek_v3` is not on the fused MoE kernel, and all 78 `gather_qmv_kernel` launches of a step sit in graphs cut short.
- **Fused decode-MoE kernels.** Affected, and custom kernels. Found by going through every `cuda_kernel` registration rather than from the issue's list. `moe_gateup_kernel_cu` takes the gate and up weight, scale and bias stacks of a layer and `moe_down_kernel_cu` the down stacks, each `[E, rows, ...]`, and both read the `K` experts their `indices` input names. On Qwen3-30B-A3B (128 experts, 8 read) the gate/up launch registers 62.9M elements and the down launch 31.5M, so each layer cuts the graph twice: 96 of the 99 short graphs per step hold one of the two kernels. On LFM2-8B-A1B (32 experts, 4 read) only the gate/up launch is over the budget, 22 per step. Gemma 4 26B-A4B launches the GeGLU form of the pair 30 times per step, but the loader turns CUDA graph capture off for Gemma 4 (#688), so its trace holds no graph at all and nothing is cut there.
- **MoE at B > 1.** `SwitchGLU::forward_fused_kernel` takes one token, so a batched step runs `gather_qmm`, the builtin again.

The other `cuda_kernel` registrations read their inputs whole and stay counted: `mlxcel_fused_add_rms_norm`, `mlxcel_fused_rope_qk_append`, `mlxcel_fused_gelu_tanh_f32`, `mlxcel_gumbel_max_sample`, `mlxcel_rejection_sample`, `bitlinear_matmul_cu`, `ssm_kernel_cu`, `mamba1_selective_scan_cu`. The in-place KV row write registers the row as its input and the cache buffer as its output, and outputs are not counted.

The two builtins need a different mechanism and are reported in a follow-up issue.

## The change

The `custom_kernel.cpp` overlay charges each expert stack of the two MoE kernels for the experts the index names, `data_size / E * K`, instead of `data_size`. An index that names every expert reads the stacks whole and is charged whole, as before. The paged-attention kernels are unchanged.

Graph launches per decode step with the change, default budget:

| Model | Context | B | `main` | change | `main`, raised |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B-4bit | 8,193 | 1 | 147.0 | 62.1 | 60.1 |
| LFM2-8B-A1B-4bit | 9,218 | 1 | 48.0 | 30.0 | 29.0 |
| Qwen3-0.6B-4bit (paged, control) | 8,193 | 1 | 26.0 | 26.0 | 26.0 |
| Qwen3-0.6B-4bit (paged, control) | 8,193 | 4 | 68.0 | 68.0 | 68.0 |

## Decode rate, `main` against the change

[`bench_decode.py`](issue-3385-qwen3-8b-gb10-2026-11-24/bench_decode.py) of the #3385 record: one fresh server per arm (`--parallel 4 --no-prompt-cache -c 65536`), arms `main-a`, `change`, `main-b` rotated every round, `B` concurrent greedy `/v1/completions` streams of a `P`-token prompt with `ignore_eos`, 512 generated tokens per row at `B = 1` and 3,072 at `B = 4`, a cell rejected unless every row returns its full length. The rate is generated tokens per second inside the window in which every row streams. `main-a` and `main-b` are the same binary; "null" is the gap between their medians over the main median, "change" the change median against the median of both main arms. The reply text of every sample is hashed and is identical across the three arms in every cell. No trace, build or test ran beside it.

Three rounds, two samples per cell and round (six per arm), except the second Qwen3-30B-A3B run, which has four samples per cell and round. LFM2 and Gemma 4 ran with a copy of the driver in which the 5% prompt-length check can be switched off by an environment variable; only LFM2 needed it.

| model | P | B | main-a | main-b | change | main range | change range | null | change | per round |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| Qwen3-30B-A3B-4bit | 256 | 1 | 66.44 | 63.95 | 73.06 | 57.83 to 71.19 | 72.57 to 73.79 | 3.8% | +10.2% | +18.9%, +11.9%, +10.0% |
| Qwen3-30B-A3B-4bit | 8192 | 1 | 53.72 | 53.59 | 56.12 | 52.07 to 57.97 | 55.66 to 56.60 | 0.3% | +4.5% | +6.4%, -1.8%, +4.1% |
| Qwen3-30B-A3B-4bit, run 2 | 256 | 1 | 62.09 | 62.08 | 73.07 | 57.07 to 68.78 | 63.67 to 74.98 | 0.0% | +17.7% | +17.7%, +22.0%, +13.2% |
| Qwen3-30B-A3B-4bit, run 2 | 8192 | 1 | 53.33 | 53.31 | 56.53 | 52.33 to 56.56 | 55.39 to 58.55 | 0.0% | +6.0% | +6.0%, +5.8%, +7.9% |
| LFM2-8B-A1B-4bit | 256 | 1 | 123.21 | 122.36 | 123.95 | 121.77 to 123.88 | 122.45 to 124.65 | 0.7% | +0.8% | +1.2%, -0.0%, +0.8% |
| LFM2-8B-A1B-4bit | 8192 | 1 | 109.52 | 109.37 | 109.92 | 108.41 to 111.51 | 109.56 to 110.87 | 0.1% | +0.5% | +0.1%, +0.4%, +1.1% |
| gemma-4-26b-a4b-it-4bit (graph capture off) | 256 | 1 | 43.30 | 43.09 | 43.22 | 42.97 to 43.74 | 41.79 to 43.84 | 0.5% | +0.2% | -1.3%, +0.7%, +0.0% |
| gemma-4-26b-a4b-it-4bit (graph capture off) | 8192 | 1 | 39.56 | 39.35 | 39.39 | 39.11 to 39.61 | 39.25 to 39.49 | 0.5% | -0.1% | -0.2%, +0.0%, -0.2% |
| Qwen3-0.6B-4bit (paged, control) | 8192 | 1 | 108.73 | 107.36 | 107.49 | 105.92 to 115.92 | 106.06 to 113.10 | 1.3% | -0.1% | -0.2%, -0.3%, +0.8% |
| Qwen3-0.6B-4bit (paged, control) | 8192 | 4 | 136.85 | 136.38 | 135.81 | 135.22 to 138.90 | 135.30 to 136.59 | 0.3% | -0.6% | -0.1%, -1.2%, -0.7% |

Rows: [`decode-ab.csv`](cuda-graph-commit-audit-gb10-2026-11-27/decode-ab.csv).

- **Qwen3-30B-A3B** gains at both prompt lengths. On `main` its rate moves between server starts, 57 to 71 tok/s at the short prompt and 52 to 58 at the long one (the [Dff-crossover record](fused-moe-dff-crossover-gb10-2026-11-11.md) also notes a run-to-run spread on fused-path models that it could not explain); with the change the short-prompt samples sit between 71.4 and 75.0 apart from one at 63.7. In the first run one round of the long prompt came out behind `main` (-1.8%), because both main arms of that round started at 55 to 58; the second run, with twice the samples and a null of 0.0%, has all three rounds ahead.
- **LFM2-8B-A1B** is unchanged within the null: +0.8% and +0.5% against nulls of 0.7% and 0.1%. Its tokenizer splits the prompt's words, so the prompts are 290 and 9,218 tokens, and the rows record the real lengths. Losing 18 graph commits of 48 per step is worth less here than 85 of 147 on the larger model.
- **Gemma 4 26B-A4B** runs the GeGLU form of the same two kernels, but the loader turns CUDA graph capture off for Gemma 4 (#688). Its traces hold no `cudaGraphLaunch` in any arm, only 1,041 direct kernel launches per step, so no budget applies, and the rate is the same: +0.2% and -0.1% against nulls of 0.5%.
- **The paged control** launches the same graphs as `main` (first table) and reads -0.1% at `B = 1`, inside a null of 1.3%, and -0.6% at `B = 4`, where the null is 0.3%, the change range lies inside the main range and all three rounds are at or below zero. The change adds one small vector and three substring searches per custom-kernel launch, by its operation count tens of microseconds on a 29 ms step (an estimate, not a measurement), which does not account for 0.6%; the figure is recorded as measured and was not repeated with more samples.

## Not measured

- Metal and ROCm. The ROCm overlay has the same count (`bytes_in_graph_ += arr.data_size()`) under a fixed budget of 200 and no exemption for any kernel; no ROCm host was available.
- MiniMax-M3 on its checkpoint (it does not fit this host), and the sparse path beyond batch 1.
- MTP and full-model draft verification. Prompt lookup was the only drafter traced.
- The GeGLU form of the two kernels with graph capture on. Gemma 4 is its only user and runs with capture off, so it is covered by the commit test on synthetic stacks only.
- The other families on the fused MoE kernel at `B = 1` (the SwitchGLU families besides `qwen3_moe` and `lfm2_moe`, and `qwen3_next`).
- An embedding table over the budget (an 8B model). The three embedding gathers of Qwen3-0.6B are under it.
