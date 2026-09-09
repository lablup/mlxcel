# Model Compatibility & Performance Tests

Per-hardware benchmark results and cross-hardware comparison for mlxcel.

For a public, data-driven Apple Silicon summary that combines M1 Ultra,
M5 Max, and mlx-lm / mlx-vlm baselines, see
[Benchmark Report - 2026-05-19](benchmark-report.md).

## Per-Hardware Results

| Hardware | File | Status | Last Updated |
|----------|------|--------|-------------|
| Mac Studio M1 Ultra 128GB | [model_tests_m1ultra.md](model_tests_m1ultra.md) | Active | 2026-09-06 through 2026-09-08 at `30ab5a39`, with 18 rows re-measured on 2026-09-08 after lablup/mlxcel#1709 and #1711 (mlxcel 0.7.0-beta.1, MLX pin `9a795735`). Full text and VLM re-measurement on the pp512/tg128 shape against same-day mlx-lm 0.31.3 and mlx-vlm 0.6.17 baselines, so this host is now directly comparable to M5 Max. Earlier sweeps used a different shape and were not carried forward. Speculative and batched-serving families not yet re-run |
| MacBook Pro M5 Max 128GB | [model_tests_m5max.md](model_tests_m5max.md) | Active | 2026-09-09 partial pass at `0accedd9`: 21 text checkpoints measured here for the first time, having arrived when the two hosts' stores were synchronised, plus four rows re-measured at `fea72611` to close the commit gap against M1 Ultra's 2026-09-08 pass. Full sweep 2026-09-06 at `a50ff440` (mlxcel 0.7.0-beta.1, MLX pin `9a795735`). First sweep on the fixed pp512/tg128 interval; prefill is not comparable to any earlier sweep on this host. M1 Ultra has since been re-swept on the same shape, so the two hosts are compared under Cross-Hardware Comparison below. Full text, VLM, speculative, batched-serving and embeddings campaign |
| NVIDIA GB10 (DGX Spark) | [model_tests_gb10.md](model_tests_gb10.md) | Active | 2026-07-12 (mlxcel 0.4.0-rc.1, full 159-dir sweep; 7 memory-gated skips) |

## Benchmark CSVs

Current source-of-truth data lives in `benchmarks/`:

| CSV | Hardware | Date | Type |
|-----|----------|------|------|
| `metal_m1ultra_2026-09-06.csv` | M1 Ultra | 2026-09-06 (mlxcel 0.7.0-beta.1 at `30ab5a39`, MLX pin `9a795735`, `--cooldown 30 --big-cooldown 30`; 199 directories, 165 measured, 16 collapsed by checkpoint dedup #1615. The 15 failures are all non-targets or one unsupported architecture, none a runtime defect) | Text |
| `pylm_m1ultra_2026-09-06.csv` | M1 Ultra | 2026-09-06 (mlx-lm 0.31.3, same host and day; 123 measured, giving 106 models both sides ran. Median decode parity 100%, quartiles 98 and 105; MoE median 106% against dense 100%) | Text baseline |
| `metal_m1ultra_vlm_2026-09-07.csv` | M1 Ultra | 2026-09-07 (mlxcel 0.7.0-beta.1 at `30ab5a39`; 87 VLM checkpoints after the non-VLM filter, 70 measured. Every row has 128 generated tokens) | VLM |
| `pylm_m1ultra_vlm_2026-09-07.csv` | M1 Ultra | 2026-09-07 (mlx-vlm 0.6.17; 67 measured. **Single-environment re-run from an empty file**: installing torch, torchvision and timm partway through the first pass changed image preprocessing, moving `granite-vision-3.2-2b-4bit` from 56 prompt tokens to 1540. 3 rows rejected as `FAIL:image_not_applied`) | VLM baseline |
| `metal_m1ultra_2026-09-08.csv` | M1 Ultra | 2026-09-08 (mlxcel 0.7.0-beta.1; 191 rows, 167 measured. Supersedes `metal_m1ultra_2026-09-06.csv` as this host's text record: it carries that sweep plus the checkpoints fetched during the store sync and the rows re-measured after lablup/mlxcel#1709 and #1711. Six rows move by more than 10% against 2026-09-06, of which `gpt_bigcode-santacoder` 3.20x, `pythia-1b` 3.12x and `mistral-small-4-119b-2603-4bit` 2.91x are the fixes landing) | Text |
| `metal_m1ultra_vlm_2026-09-08.csv` | M1 Ultra | 2026-09-08 (mlxcel 0.7.0-beta.1; 86 rows, 76 measured. Same relationship to `metal_m1ultra_vlm_2026-09-07.csv` that the text file above has to the 09-06 sweep) | VLM |
| `metal_m5max_2026-09-06.csv` | M5 Max | 2026-09-06 (mlxcel 0.7.0-beta.1, MLX pin `9a795735`, `--cooldown 30 --big-cooldown 30`, `BENCH_MEM_OVERHEAD_FACTOR=1.209`; first pp512/tg128 sweep after the interval fix `3a746ea8`, 178 dirs, 149 measured, 14 collapsed by checkpoint dedup #1615. `hunyuanocr-mlx-4bit` and `ernie-4.5-vl-28b-a3b-thinking-4bit` arrived after the sweep and were added at `534a6ebd`. Median decode 0.982 vs 2026-09-03; the 4 models that gained >10% all had 1-7 token samples in the old condition. All 12 models down >10% were re-measured in isolation and reproduced within 3%) | Text |
| `metal_m5max_2026-09-09.csv` | M5 Max | 2026-09-09 (mlxcel 0.7.0-beta.1 at `0accedd9`, MLX pin `9a795735`, `--cooldown 20`; 21 checkpoints new to this host, 20 measured. `afm-4.5b` fails as `FAIL:unsupported_arch_arcee_dense`: it is dense `ArceeForCausalLM` and the binary carries only the AFMoE / Trinity MoE entry. Partial pass, not a re-sweep: every other row on this host still comes from 2026-09-06) | Text |
| `metal_m5max_2026-09-09_single_mistral-small-4-119b-2603-4bit.csv` + `_recheck_single_*` + `metal_m5max_vlm_2026-09-09_single_*` | M5 Max | 2026-09-09 (mlxcel 0.7.0-beta.1 at `fea72611`; the row this host had been carrying at 19.04 tok/s since 2026-09-06, re-measured after lablup/mlxcel#1711 landed. Text 98.85 then 98.91 on a repeat, VLM 102.99. The `_recheck_` file also holds `qwen2.5-7b-instruct-4bit`, `meta-llama-3.1-8b-instruct-4bit` and `qwen3-30b-a3b-4bit` as controls, which reproduce 2026-09-06 at 1.00x, 1.00x and 0.98x) | Text/VLM |
| `pylm_m5max_vlm_2026-09-09_mistral4check_single_*.csv` | M5 Max | 2026-09-09 (mlx-vlm 0.4.4, same host and day; the `mistral-small-4-119b-2603-4bit` baseline that puts the row above at 100% of the reference runtime) | VLM baseline |
| `metal_m5max_embeddings_2026-09-09.csv` | M5 Max | 2026-09-09 (mlxcel 0.7.0-beta.1 at `66b8346e`, MLX pin `9a795735`; 102 cells over the 20-checkpoint roster, no skips. First M5 Max embedding CSV with aligned columns, see `benchmarks/README-embeddings-csv-columns.md`. Report: `embeddings-rerank-m5max-2026-09-09.md`) | Embeddings |
| `metal_m5max_vlm_2026-09-06.csv` | M5 Max | 2026-09-06 (mlxcel 0.7.0-beta.1 at `a50ff440-dirty`, which is `a50ff440` plus the fix committed as `34455e42`; same cooldowns and budget; 71 measured rows. **This is the corrected re-run**: the first pass of the day was a byte-for-byte duplicate of the text sweep because `--prompt-tokens` silently discards `--image`, fixed in-campaign) | VLM |
| `metal_m5max_spec_2026-09-06.csv` | M5 Max | 2026-09-06 (mlxcel 0.7.0-beta.1; `speculative_bench --sweep --max-tokens 128`, 16 rows: 4 baselines, 9 measured MTP rows at K=2/4/8 across Gemma 4 31B, Gemma 4 Unified 12B and Qwen 3.8 27B, 3 DFlash deferred. First M5 Max sweep carrying #1621's per-variant MTP dispatch, so the Gemma 4 31B rows exist here for the first time; Gemma 4 Unified reproduces 2026-09-04 within noise) | Speculative |
| `metal_m5max_batch_2026-09-06.csv` | M5 Max | 2026-09-06 (mlxcel 0.7.0-beta.1; `bench_serving_concurrency.py` at `--parallel 4 --max-batch-prefill 4`, `--prompt-tokens 512 --max-tokens 128`; 3 models x B=1/2/4, 0 failed requests; dense aggregate scaling 3.16x / 3.24x at B=4, MoE 1.74x) | Batch |
| `metal_m1ultra_embeddings_2026-09-09.csv` | M1 Ultra | 2026-09-09 (mlxcel 0.7.0-beta.1 at `10f3aae8`; `bench_embeddings.py`, full 20-model roster, 102 cells, 0 skipped, 0 failures. First embedding pass on this host; report in [`embeddings-rerank-m1ultra-2026-09-09.md`](embeddings-rerank-m1ultra-2026-09-09.md)) | Embeddings/Rerank |
| `metal_m5max_embeddings_2026-09-06.csv` | M5 Max | 2026-09-06 (mlxcel 0.7.0-beta.1; `bench_embeddings.py`, **full 20-model roster** for the first time on this host, 102 cells, 0 failures. The two multivector checkpoints are LoRA-only upstream and were merged into their bases locally before the run) | Embeddings/Rerank |
| `metal_m5max_2026-09-03.csv` | M5 Max | 2026-09-03 (mlxcel 0.6.0, MLX pin `9a795735`, `--cooldown 30 --big-cooldown 30`, `BENCH_MEM_OVERHEAD_FACTOR=1.209` for a 90 GB weight budget; version-change full text re-benchmark, 175 dirs, 161 measured; 0 decode regressions vs 0.4.0-rc.1) | Text |
| `metal_m5max_vlm_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0, MLX pin `9a795735`, same cooldowns and budget; version-change full VLM re-benchmark, 77 measured rows; Pixtral/Mistral3 image-token counts drop by design after #792) | VLM |
| `metal_m5max_spec_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0; `speculative_bench --sweep --max-tokens 128`, 12 rows: 3 baselines, 3 measured Gemma 4 Unified 12B MTP rows at K=2/4/8, 3 DFlash deferred, 3 Gemma 4 31B MTP rows that the harness could not drive. That harness restriction was lifted by #1613; an M5 Max re-run against a binary carrying it has not been made) | Speculative |
| `metal_m5max_batch_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0; `bench_serving_concurrency.py` at `--parallel 4 --max-batch-prefill 4`, `--prompt-tokens 512 --max-tokens 128`; 3 models x B=1/2/4, 0 failed requests) | Batch |
| `metal_m5max_embeddings_2026-09-04.csv` | M5 Max | 2026-09-04 (mlxcel 0.6.0; `bench_embeddings.py`, 18 of the 20-model roster present on disk, 90 cells, 0 failures; report in [`embeddings-rerank-m5max-2026-09-04.md`](embeddings-rerank-m5max-2026-09-04.md)) | Embeddings/Rerank |
| `metal_m5max_2026-07-12.csv` | M5 Max | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30 --big-cooldown 30`; version-change full text re-benchmark, 175 dirs, 160 measured; no code regressions) | Text |
| `metal_m5max_vlm_2026-07-12.csv` | M5 Max | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30 --big-cooldown 30`; version-change full VLM re-benchmark, 75 measured rows) | VLM |
| `metal_m5max_2026-06-15.csv` | M5 Max | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec7123; full text re-benchmark, 151 rows, 135 measured) | Text |
| `metal_m5max_vlm_2026-06-15.csv` | M5 Max | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec7123; full VLM re-benchmark, 53 measured rows) | VLM |
| `metal_m5max_2026-05-19.csv` | M5 Max | 2026-05-19 (mlxcel 0.0.28, MLX 0.31.2) | Text |
| `metal_m5max_vlm_2026-05-19.csv` | M5 Max | 2026-05-19 (mlxcel 0.0.28, MLX 0.31.2) | VLM |
| `metal_m5max_vlm_2026-05-20.csv` | M5 Max | 2026-05-20 (mlxcel 0.0.28, MLX 0.31.2; Gemma3n + Molmo v1 + Phi-3.5 vision + Gemma3 4B VLM entries) | VLM |
| `pylm_m5max_2026-05-18.csv` | M5 Max | 2026-05-19 benchmark campaign (mlx-lm 0.31.3 baseline; CSV date crossed midnight) | Text |
| `pylm_m5max_vlm_2026-05-18.csv` | M5 Max | 2026-05-19 benchmark campaign (mlx-vlm 0.4.4 baseline; CSV date crossed midnight) | VLM |
| `metal_m1ultra_spec_2026-09-04.csv` | M1 Ultra | 2026-09-04 (mlxcel 0.7.0-beta.1, MLX pin `9a795735`; `speculative_bench --sweep --batch 1 --max-tokens 128`, 16 rows: 4 baselines, 9 measured MTP rows at K=2/4/8 across Gemma 4 31B, Gemma 4 Unified 12B and Qwen 3.8 27B, 3 DFlash deferred; first sweep with #1613's per-variant MTP dispatch) | Speculative |
| `metal_m1ultra_2026-07-12.csv` | M1 Ultra | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30`; version-change full text re-benchmark, 168 rows; decode median 98% vs mlx-lm, flat vs 0.3.3, nvfp4/minicpm-mxfp4 recovered) | Text |
| `metal_m1ultra_vlm_2026-07-12.csv` | M1 Ultra | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin 57c66cac, `--cooldown 30`; version-change full VLM re-benchmark, 168 rows) | VLM |
| `metal_m1ultra_2026-07-12_cooldown0.csv` + `_vlm_` | M1 Ultra | 2026-07-12 (mlxcel 0.4.0-rc.1; `--cooldown 0` pass of the same build, kept for the thermal-offset comparison: decode reads ~2% lower than the cooldown-30 canonical) | Text/VLM |
| `metal_m1ultra_2026-07-06.csv` | M1 Ultra | 2026-07-06 (mlxcel 0.3.3; full text re-benchmark post VLM-port batch #660-#664 and fixes #666-#668/#671, 169 rows) | Text |
| `metal_m1ultra_vlm_2026-07-06.csv` | M1 Ultra | 2026-07-06 (mlxcel 0.3.3; full VLM re-benchmark, 14 new VLM families measured) | VLM |
| `pylm_m1ultra_*2026-07-06*_single_*.csv` | M1 Ultra | 2026-07-06 (mlx-lm 0.31.3 / mlx-vlm dev; per-model python baselines for 21 newly added models, 13 measured / 8 python-side FAIL) | Baselines |
| `metal_m1ultra_2026-06-15.csv` | M1 Ultra | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec712; full text re-benchmark post #289 fix, 151 rows) | Text |
| `metal_m1ultra_vlm_2026-06-15.csv` | M1 Ultra | 2026-06-15 (mlxcel 0.2.1, MLX pin a6ec712; full VLM re-benchmark, 55 measured rows) | VLM |
| `metal_m1ultra_2026-06-15_pre289_regressed.csv` | M1 Ultra | 2026-06-15 (mlxcel pre-#290; bf16-scale decode regression evidence sweep) | Text |
| `metal_m1ultra_2026-06-12.csv` | M1 Ultra | 2026-06-12 (mlxcel 0.1.4, MLX pin a6ec712; full text re-benchmark, 121 rows) | Text |
| `metal_m1ultra_vlm_2026-06-12.csv` | M1 Ultra | 2026-06-12 (mlxcel 0.1.4, MLX pin a6ec712; full VLM re-benchmark, 49 measured rows) | VLM |
| `metal_m1ultra_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlxcel 0.0.28, MLX commit 84961223; >65GB skipped) | Text |
| `metal_m1ultra_vlm_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlxcel 0.0.28, MLX commit 84961223; >65GB skipped) | VLM |
| `pylm_m1ultra_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlx-lm 0.31.3 baseline, https://github.com/ml-explore/mlx-lm @ `df1d3f3`; >65GB skipped) | Text |
| `pylm_m1ultra_vlm_2026-05-19.csv` | M1 Ultra | 2026-05-19 (mlx-vlm baseline, https://github.com/Blaizzy/mlx-vlm @ `d85ca4d`; >65GB skipped) | VLM |
| `cuda_gb10_2026-07-12.csv` | GB10 | 2026-07-12 (mlxcel 0.4.0-rc.1, MLX pin `57c66cac` / 0.32.1, CUDA 13.0 / SM 12.1; full 159-dir sweep, 142 measured / 0 code failures / 7 memory-gate `SKIP:oom_estimate` at `BENCH_MEM_OVERHEAD_FACTOR=2.0` / 10 N.A. FAILs [drafters, speech/TTS, incomplete checkpoints]; 13 rows amended to the same-day post-reboot #755 singles, median-decode run where n=3, overnight originals in git history) | Text |
| `cuda_gb10_2026-07-12_postreboot_single_*.csv` | GB10 | 2026-07-12 (13 files, 21 rows; post-reboot re-measurement for #755: subjects, controls, and the SSM/hybrid cluster, `--cooldown 30`, including the n=3 repeat sets behind the amended sweep rows) | Text |
| `cuda_gb10_vlm_2026-07-12.csv` | GB10 | 2026-07-12 (mlxcel 0.4.0-rc.1; full VLM sweep over all dirs, 63 measured image rows; text-only models FAIL by design; gemma-4-31b-it-nvfp4 image-input FAIL captured here, since fixed by #749) | VLM |
| `cuda_gb10_2026-07-09.csv` | GB10 | 2026-07-09 (mlxcel 0.4.0-rc.1, MLX pin `57c66cac` / 0.32.1, CUDA 13.0 / SM 12.1; 19-model representative subset re-benchmark, 19 pass / 0 fail, includes gemma-4-31b-it-nvfp4 now functional via the ModelOpt NVFP4 direct-transcode path #692/#693/#697) | Text |
| `cuda_gb10_2026-06-17.csv` | GB10 | 2026-06-17 (mlxcel 0.3.1 [CSV relabeled; Cargo.toml 0.3.0 until release], MLX pin a6ec7123, CUDA 13.0 / SM 12.1, post-#319 CUDA fused decode-MoE; full text re-benchmark, 147 models, 136 pass / 0 fail / 9 not-tested-N.A. / 2 too-large) | Text |
| `cuda_gb10_vlm_2026-06-17.csv` | GB10 | 2026-06-17 (mlxcel 0.3.1; full VLM re-benchmark, 54 measured image rows) | VLM |
| `cuda_gb10_2026-05-28.csv` | GB10 | 2026-05-28 (full text re-benchmark, mlxcel 0.1.0, MLX commit 84961223, warm same-process harness `c9a77f2`, `--cooldown 0`; 109 models, 8 fail/skip) | Text |
| `cuda_gb10_vlm_2026-05-28.csv` | GB10 | 2026-05-28 (full VLM re-benchmark, mlxcel 0.1.0; 38 measured VLM rows, 0 image-path failures) | VLM |
| `cuda_gb10_2026-05-19.csv` | GB10 | 2026-05-19 (mlxcel 0.0.27, MLX 0.31.2) | Text |
| `cuda_gb10_vlm_2026-05-19.csv` | GB10 | 2026-05-19 (mlxcel 0.0.27, MLX 0.31.2) | VLM |

## Cross-Hardware Comparison

Two questions live in these tables and they answer differently. **Does mlxcel beat the Python reference on a given machine**, and **how much faster is one machine than another**. Read them in that order: a cross-hardware ratio taken on mlxcel alone cannot tell a hardware gap from a place where mlxcel fails to exploit the hardware.

M1 Ultra and M5 Max are directly comparable as of the 2026-09-06 sweeps: same mlxcel version (0.7.0-beta.1), same MLX pin (`9a795735`), same pp512/tg128 shape, same cooldowns, each against a same-day mlx-lm 0.31.3 run on its own host. Their mlxcel commits differ by eight (`30ab5a39` against `a50ff440`), of which one touches shared inference code (#1656, pre-Ampere CUDA bf16 handling) and is inert on Metal. Rows added after that date do not share a commit: M1 Ultra re-measured 18 of them on 2026-09-08 and M5 Max added 21 checkpoints on 2026-09-09, so any single row can pair readings taken across a fix. Section 3 says what that cost and how it was caught. GB10 is still on 0.4.0-rc.1 and the old measurement shape, so it is absent from both tables below; see [model_tests_gb10.md](model_tests_gb10.md) until it is re-swept.

### 1. mlxcel against the Python baseline, on each machine

This is the runtime claim, and it holds equally on both machines. Decode, mlxcel over mlx-lm on the same host:

| Host | n | Median | Quartiles | Range |
|---|--:|--:|---|---|
| M1 Ultra | 103 | 100% | 98 / 105 | 26-140% |
| M5 Max | 89 | 100% | 98 / 101 | 23-131% |

Across the 86 models both hosts share, the per-model difference between the two advantages has a median of -1 percentage point. The advantage is a property of the runtime, not of the machine it runs on.

### 2. Where that advantage did not carry over, and what it took to see it

Three checkpoints broke the pattern in the 2026-09-06 sweeps, and they are the reason the two comparisons have to be read together. All three are fixed as of `c07826df`; the numbers are kept here because the way they were found is the reusable part.

| Checkpoint | Architecture | M1 Ultra | M5 Max before | M5 Max after |
|---|---|--:|--:|--:|
| `falcon-h1-tiny-90m-instruct-4bit` | FalconH1 | 108% | 31% | 112% |
| `granite-4.0-h-350m-4bit` | GraniteMoeHybrid | 100% | 23% | 84% |
| `granite-4.0-h-tiny-4bit` | GraniteMoeHybrid | 93% | 33% | 90% |

The cause was a per-mixer `eval` that both families ran at the end of every mixer forward on M5 Max, a NaN workaround from #266. It is a per-layer GPU sync on every decode token. The gate was present and correct; what had never been measured was its M5 arm. `nemotron_h` carries the same construct and was healthy throughout because single-token decode returns early into `forward_fused` and never reaches it.

Neither comparison alone would have surfaced this. The same-machine median is 100% on both hosts, so the runtime table hides it. Read only as hardware, the three say "M1 Ultra is faster on these models", which is the wrong conclusion and the one a cross-hardware table invites. They appear only when the same-machine ratio is computed on both machines and the two are compared.

`plamo-2-1b` carries the same construct and was briefly suspected on the strength of its 0.81x M5-over-M1 ratio. That was the wrong quantity: 0.81x is hardware against hardware, not mlxcel against mlx-lm, and the two cannot corroborate each other. It had no baseline on either host at the time, which turned out to be a missing `numba` in the baseline environment rather than anything about the model. With the baseline measured it is ahead on both machines, at 100% of mlx-lm on M1 Ultra (107.47 against 107.04) and 106% on M5 Max (86.83 against 82.10), and it gains nothing from removing the boundary, so it keeps its gate.

`qwen3-omni-30b-a3b-instruct-4bit` is resolved and there was no defect, but it took four wrong turns to get there and each one is a distinct trap.

It first read as 61% of mlx-vlm on M5 Max and 183% on M1 Ultra. Re-measured on both hosts at the current commit it is 157% and 179%, and mlxcel's cross-host ratio of 2.07x sits with the other 30B-A3B MoE checkpoints at 2.11x to 2.20x. Everything reconciles.

The four turns. **First**, the 61% came from mixing harnesses: an mlxcel VLM row compared against an mlx-lm text baseline. **Second**, corrected to VLM against VLM, it survived at 61% but the mlxcel row was from `a50ff440` while the baseline was measured that day, so a pre-fix runtime was being compared against a current reference. **Third**, the cross-host split correctly identified the baseline as the anomalous quantity, 3.85x against mlxcel's 1.30x and the largest of 61 VLM pairs, which pointed at the baseline when the stale mlxcel row was the actual cause. **Fourth**, the movement was attributed to `5287eb9a`, which the timeline supports and the diff does not: that commit touches `falcon_ocr_rope.rs`, `glm4v.rs` and `paddleocr_vl.rs`, and `qwen3_omni_moe.rs` is unchanged across the whole window. Which commit moved it is not established here.

Two things are worth keeping. A ratio survived all four turns while no absolute value did: the same-host figure stayed near 180% on M1 Ultra across a full re-measurement of both sides (183% then 179%), while both absolute readings moved by more than 1.6x. And measuring mlxcel and the baseline back to back in one session is what settles a case like this, because comparing a stored row against a fresh one cannot separate a code change from machine state. On M5 Max the baseline reads 99.72, 101.45 and 101.23 across three occasions, so it is stable here; M1 Ultra's moved from 25.89 to 42.68 and that remains unexplained.

### 2b. The VLM tables were a commit behind, and the whole thing was re-swept

Chasing `qwen3-omni` surfaced a larger problem than the checkpoint itself: the VLM tables mixed seven mlxcel commits spanning 2026-09-04 to 2026-09-08, and three dtype fixes had landed inside that span. Both hosts were re-swept in full at a single commit rather than patching the rows believed to be affected, because a partial pass leaves the table mixed again and every misreading in this document came from a mixed table.

M5 Max at `f85898eb`: 92 rows, 78 measured, 71 comparable against the previous readings. Median 1.00x, quartiles 1.00 and 1.01. Nine rows moved more than 10%.

| Checkpoint | before | 2026-09-09 | ratio |
|---|--:|--:|--:|
| `mistral-small-4-119b-2603-4bit` | 19.34 | 102.57 | 5.30x |
| `moondream2` | 41.36 | 173.36 | 4.19x |
| `qwen3-vl-30b-a3b-instruct-4bit` | 60.73 | 158.38 | 2.61x |
| `qwen3-omni-30b-a3b-instruct-4bit` | 61.21 | 157.12 | 2.57x |
| `qwen3-vl-32b-instruct-4bit` | 19.34 | 28.28 | 1.46x |
| `llama-3.2-11b-vision-instruct-4bit` | 72.57 | 94.92 | 1.31x |
| `qwen3-vl-8b-instruct-4bit` | 84.66 | 109.63 | 1.29x |
| `paligemma2-3b-ft-docci-448-6bit` | 139.53 | 169.00 | 1.21x |
| `qwen3-vl-2b-instruct-4bit` | 291.01 | 337.83 | 1.16x |

**Selecting by family would have missed a third of the movers.** `moondream2`, `llama-3.2-11b-vision` and `paligemma2-3b` fall outside every family the three commits touch, and `qwen3_omni_moe.rs` is unchanged across the entire window while that checkpoint is one of the two largest movers. Selection was by commit lineage against `f4ecc926`, which is mechanical and needs no judgement about which family a fix reaches. Anyone narrowing a future re-measurement by family should read this row first.

**Re-sweeping only the runtime inflates its own margins.** Dividing the new mlxcel numbers by the 2026-09-07 baseline puts `qwen3-omni` at 298% and at the top of the table, because only one side had moved and the 25.89 tok/s it divides by does not reproduce. The baseline was re-swept in the same pass. With both sides measured on 2026-09-09 the margin is 155% and the top of the table is `jina-vlm-mlx` at 194%.

Same-day margins on M5 Max: 48 pairs, median 105%, quartiles 101 and 113, range 95 to 194%, nothing below 90%. The pair count is 48 rather than the 67 both sides measured, and the 19 dropped rows are not failures: the two runtimes turn the same image into different prompt lengths, so there is no like-for-like comparison to make. They fall into three kinds, and the counts are identical on both hosts, which makes this a property of the two runtimes rather than of either machine.

| Kind | Rows | Shape |
|---|--:|---|
| Image tiling | 3 | `idefics3-8b-llama3-4bit` 189 against 3041, `smolvlm-instruct-bf16` 102 against 1562, `idefics2-8b-4bit` 81 against 340 |
| Chat template | 15 | Exactly +15 tokens on every row, whether the base is 65 or 69, across `qwen3-vl`, `qwen3.5`, `qwen3.6` and `qwen3.8` |
| mlxcel uses more | 1 | `deepseek-vl2-small-4bit` 494 against 436 |

The +15 group is the informative one: a constant offset across three families and sizes from 0.8B to 35B is a fixed template overhead, not a proportional difference, and it may be reconcilable. The tiling group is not, at a factor of 4 to 16.

The last kind extends past the gate. `internvl3-1b-4bit` differs the same way at 293 against 270, and 8.5% clears the 10% threshold, so it stays in the comparison: **its margin is computed across a 23-token difference.** The gate cuts by magnitude and not by cause, so one instance of a single phenomenon is excluded while another is kept. Six further rows differ by 1 to 3 tokens in the same direction, and those are deterministic rather than noise, reproducing exactly on both hosts; they are small enough to disregard at 0.2% to 2.5%, which is a different statement from being unstable. Against a floor where 31 of the compared rows differ by zero tokens, none of this is measurement scatter.

Two rows carry a caveat rather than a number. `minicpm-v-4.6-bf16` and `-mxfp4` are compared across a prompt-length change, 32 tokens against 80, because `94323c20` upscales an image below the scale resolution and the test fixture is 224x224. That is an intended behavior change, not measurement drift, and both read 1.00x and 1.01x, so decode is close to indifferent to prompt length at this scale. The prompt-length gate applied to the runtime-against-baseline comparison was not applied to this before-and-after one; it has been checked since and these two are the only rows it catches.

### 3. Machine against machine

Across the 119 models both hosts measured at an identical prompt length, taking each host's most recent reading per checkpoint rather than a single sweep:

| | n | Median | Quartiles | Range |
|---|--:|--:|---|---|
| Decode, M5 Max / M1 Ultra | 119 | 1.54x | 1.25 / 1.84 | 0.81 - 2.24x |
| Prefill, M5 Max / M1 Ultra | 119 | 4.65x | 4.13 / 5.98 | 1.96 - 7.56x |

Prefill separates the two machines far more than decode does, which is the expected shape: prefill is matmul-bound and decode is bandwidth-bound.

This supersedes a 1.46x decode median taken over the 2026-09-06 pair alone. The join method matters more than the two-point difference: M1 Ultra re-measured 18 rows on 2026-09-08 and M5 Max added 21 checkpoints on 2026-09-09, so a table built from one sweep per host now pairs readings taken at different commits.

**A stale row on one host reads as a hardware gap.** `mistral-small-4-119b-2603-4bit` is the case that showed it. M1 Ultra re-measured it on 2026-09-08 at 55.03 tok/s against 18.89 on 2026-09-06, while M5 Max still carried 19.04 from its own 2026-09-06 sweep, so the pair computed to 0.35x and read as the largest hardware deficit in the set. Re-measured on M5 Max at the current commit it is 98.85 tok/s, and 98.91 on a repeat, which puts the ratio at 1.80x. Neither machine changed. lablup/mlxcel#1711 landed between the two readings and only one host had been re-run.

Three controls measured in the same session decide that this was one stale row and not a stale snapshot: `qwen2.5-7b-instruct-4bit`, `meta-llama-3.1-8b-instruct-4bit` and `qwen3-30b-a3b-4bit` reproduce their 2026-09-06 values at 1.00x, 1.00x and 0.98x. Without them the same evidence would equally support re-sweeping this host entirely, which costs hours.

Eight rows sit below 1.0x and none is a runtime gap. Six are wide-dtype checkpoints, which neither machine is optimised for and both runtimes handle the same way: `plamo-2-1b` at 0.81x (float32), `qwen3.8-27b-hf-bf16` at 0.90x, `phi-3.5-mini-bf16` at 0.95x, `phi-3.5-mini-instruct-hf` at 0.96x, `llama-3.2-1b-instruct` at 0.96x and `qwen3.5-9b-bf16` at 0.99x. The other two are quantized and flat rather than adverse, `llama-3_3-nemotron-super-49b-4bit` and `iquest-coder-v1-7b-instruct-8bit` at 0.98x. `plamo-2-1b` is the clearest read: mlx-lm is slower on M5 Max too, at 0.77x, so both runtimes lose on this model going from M1 Ultra to M5 Max and the ratio is a property of the hardware pair.

That `plamo-2-1b` row is also the shape of the reading error worth guarding against. A sub-1.0x entry here is a claim about two machines, and turning it into a claim about mlxcel needs the baseline on both, which is section 1's job.

### Decode and prefill, selected models (tok/s)

| Model | M1 Ultra decode | M5 Max decode | M5 / M1 | M1 Ultra prefill | M5 Max prefill | M5 / M1 |
|---|--:|--:|--:|--:|--:|--:|
| SmolLM-135M | 370.7 | 812.2 | 2.19x | 12605.1 | 95618.3 | 7.59x |
| ERNIE-4.5-0.3B | 464.2 | 949.3 | 2.05x | 7224.0 | 51892.8 | 7.18x |
| Qwen2.5-0.5B | 331.6 | 624.5 | 1.88x | 6816.2 | 41900.1 | 6.15x |
| Qwen3-0.6B | 249.9 | 519.0 | 2.08x | 4754.1 | 33266.0 | 7.00x |
| Llama-3.2-1B | 402.9 | 524.0 | 1.30x | 4140.0 | 19674.8 | 4.75x |
| StableLM-1.6B | 245.4 | 394.9 | 1.61x | 3287.0 | 16343.7 | 4.97x |
| SmolLM3-3B | 128.4 | 226.5 | 1.76x | 1210.9 | 7475.0 | 6.17x |
| Qwen2.5-7B | 105.3 | 124.0 | 1.18x | 779.2 | 3646.4 | 4.68x |
| Llama-3.1-8B | 105.0 | 114.6 | 1.09x | 750.9 | 3421.1 | 4.56x |
| Qwen3-MoE-30B | 82.3 | 170.5 | 2.07x | 858.4 | 3727.8 | 4.34x |
| Nemotron-H-30B | 96.1 | 178.4 | 1.86x | 349.2 | 762.4 | 2.18x |
| Mixtral-8x7B | 54.5 | 65.8 | 1.21x | 333.4 | 1307.1 | 3.92x |
| Solar-Open-100B | 35.7 | 63.5 | 1.78x | 255.6 | 1113.4 | 4.36x |
| GPT-OSS-120B | 61.2 | 113.6 | 1.85x | 452.6 | 1585.0 | 3.50x |


## Overall Status (M1 Ultra and M5 Max at mlxcel 0.7.0-beta.1; GB10 still at 0.4.0-rc.1)

| Metric | Count |
|--------|-------|
| Supported model architectures | 89+ ModelType variants |
| Text models tested (M1 Ultra, 2026-06-15) | 136 pass, 2 partial, 4 fail, 9 skip/non-standalone (151 dirs; adds apertus, seed-oss, dots.llm1, granite family, lfm2, plamo-2, falcon-h1, BitNet; diffusiongemma loads via #291) |
| Text models tested (M5 Max, 2026-09-03) | 144 pass, 6 partial, 4 fail/skip across 154 table rows (0.6.0 cooldown-30 full sweep, 175 dirs / 161 measured, 90 GB weight budget; 0 decode regressions vs 0.4.0-rc.1). Table rows are below the dir count because 12 duplicate directories, 4 non-text checkpoints and 6 drafter/dflash variants are benchmarked but not listed separately |
| Text models tested (GB10, 2026-07-12) | 141 pass measured + 5 pass carried (memory-gate skips), 0 code failures, 13 not-tested/N.A. (glm-5 pair incomplete/absent; paligemma2 image-only; docling/granite-speech/whisper/kokoro non-text-gen; 4 MTP/DFlash drafters; glm-4.5v + mistral-small-4-119b memory-gated, never measured) (159 dirs; 0.4.0-rc.1 full sweep) |
| VLM models tested (GB10, 2026-07-12) | 63 measured image rows + 1 carried (llama-4-scout, memory-gate skip); gemma-4-31b-it-nvfp4 image-input FAIL since fixed by #749 (0.4.0-rc.1) |
| VLM models tested (M5 Max, 2026-09-04) | 77 measured VLM rows (0.6.0 cooldown-30 full VLM re-sweep). Pixtral/Mistral3 image-token counts fall sharply by design after #792 (aspect-ratio processing, no upscaling), so their prefill and decode are not comparable with the 0.4.0-rc.1 rows |
| VLM models tested (M1 Ultra, 2026-06-15) | 55 measured VLM rows (53 pass + 2 partial) |
| Speculative MTP on M5 Max (2026-09-04) | Gemma 4 Unified 12B + MTP assistant: 1.39x / **1.57x** / 1.55x at K=2/4/8, acceptance 55.6% / 35.0% / 34.6%. Gemma 4 31B MTP produced no number on this run because the harness drove only a Gemma 4 Unified target; #1613 lifted that and the pairing is measured on M1 Ultra below, but this host has not been re-run. The DFlash rows stay deferred on their own blocker |
| Speculative MTP on M1 Ultra (2026-09-04) | First sweep past the harness's Gemma-4-Unified-only target gate (#1613). Every MTP pairing reads below 1.00x on this Apple GPU generation: Gemma 4 31B + assistant 0.93x / 0.75x / 0.75x, Gemma 4 Unified 12B + assistant 0.94x / 0.74x / 0.76x, Qwen 3.8 27B + MTP head 0.91x / 0.69x / 0.37x at K=2/4/8. Acceptance is not the cause; see the matrix for the round-cost reading |
| Batched serving on M5 Max (2026-09-04) | Aggregate scaling at B=4 vs B=1: qwen2.5-0.5b-bf16 3.17x, llama-3.1-8b-4bit 3.25x, qwen3-30b-a3b-4bit (MoE) 1.55x. 0 failed requests across all 9 cells |
| Embedding / rerank on M5 Max (2026-09-04) | 18 checkpoints (11 text embedders, 2 VL embedders, 5 rerankers), 90 cells, 0 failures. The 2 `local/*-merged` multi-vector entries are merge artifacts and are not fetchable, so they stay unmeasured |
| Beating mlx-lm on M1 Ultra (text, >=100%) | 24/74 (32%, 6-15 vs pinned 5-19 baseline) |
| At 90%+ parity on M1 Ultra (text) | 59/74 (80%, 6-15 vs pinned 5-19 baseline) |
| Average vs mlx-lm on M1 Ultra (text) | 96% decode speed (median 98%, 6-15 vs pinned 5-19 baseline) |
| Beating mlx-lm on M5 Max (text, >=100%) | 27/67 (40%) — **prior 0.0.28 campaign; the mlx-lm baseline has not been re-run since, and is now two mlxcel releases stale** |
| At 90%+ parity on M5 Max (text) | 62/67 (93%) — prior 0.0.28 campaign; pending a fresh baseline |
| Average vs mlx-lm on M5 Max (text) | 98% decode speed (median 99%) — prior 0.0.28 campaign; pending a fresh baseline |
| Average vs mlx-vlm on M5 Max (VLM) | 100% decode speed (median 100%; 17 pairs) — prior 0.0.28 campaign; pending a fresh baseline |

## Generating Benchmarks

```bash
# Full text benchmark (auto-names CSV by hardware+date)
./scripts/bench_decode.sh all

# Full VLM benchmark
./scripts/bench_decode.sh all --vlm

# Single model
./scripts/bench_decode.sh models/<model-name>
```

After benchmarking, update the corresponding `model_tests_<hardware>.md` file from the CSV.

## Prompt cache benchmarks

Feature: cross-request prompt-prefix KV cache. Bench driver:
[`tests/prompt_cache_prefill_bench.rs`](../../tests/prompt_cache_prefill_bench.rs) (run with
`cargo test --test prompt_cache_prefill_bench --release -- --ignored --nocapture`).

### What the bench measures

For each conversation depth in `{1, 2, 4, 8, 16}` the bench issues a warmup
turn against the `/v1/chat/completions` streaming endpoint, then a
measurement turn with an identical prefix. It records:

| Column | Definition |
| --- | --- |
| `cache` | `on` = server started with `--prompt-cache-enabled=true`; `off` = disabled. |
| `prompt_tokens` | `usage.prompt_tokens` from the final streaming chunk. |
| `cached_tokens` | `usage.prompt_tokens_details.cached_tokens` when present; otherwise `-`. |
| `ttft_ms` | Time to first content delta (proxy for prefill latency on a non-speculative decoder). |
| `prefill_ms` | Same quantity as `ttft_ms`; kept as a separate column for compatibility with existing CSV readers. |
| `decode_tps` | `completion_tokens / (total - ttft)`. |
| `total_ms` | End-to-end wall-clock time for the measurement turn. |

### Expected qualitative behavior

On a functioning cache at depths >= 2 the measurement turn reports
`cached_tokens > 0` and `ttft_ms` sits below the matching `cache=off` row
for the same depth. The exact per-depth ratio depends on model and host;
target order-of-magnitude (single-digit billion parameter model, dense
backend) is:

* Depth 1: ratio ≈ 1.0 (no preceding conversation to reuse).
* Depth 2–4: ratio 0.3 – 0.8 (partial prefix reuse).
* Depth 8–16: ratio 0.1 – 0.4 (near-constant cache adopt, linear cold
  prefill on the off row).

Record measured numbers for a specific host under a new sub-heading
(e.g. `### M5 Max, qwen3-0.6b-4bit`) when updating this file.

### Validation scope

The harness itself is end-to-end exercised via the integration test
`tests/prompt_cache_e2e.rs`, which asserts the wire contract
(`cached_tokens == 0` on turn 1, `> 0` and monotonic on turns 2..5) and
the prefill-latency ratio bound (≤ 1.3× turn 1) whenever the server is
able to serve the model. Host-specific prompt-cache throughput numbers
should be appended here after running on M1 Ultra, M5 Max, GB10, or Hopper.

## TurboQuant KV cache benchmarks

Feature: TurboQuant KV cache compression (turbo3 / turbo4 modes). Bench
driver: `tests/turbo_kv_e2e.rs` (run with
`cargo test --test turbo_kv_e2e --release -- --ignored --nocapture`).

For the full config guide, tuning knobs, and architectural description see
[`docs/turbo-kv-cache.md`](../turbo-kv-cache.md).

### Source CSV

`benchmarks/turbo_kv/2026-04-26_Mac.localdomain.csv`

### Measured PPL evaluation throughput — 2026-04-26, Mac.localdomain

The quality gate runs wikitext-2 PPL evaluation and records eval throughput
(tok/s) and wall-clock time over a 4K-token evaluation window. Numbers below
are from the first validated run.

| Model | KV mode | PPL eval tok/s | Wall clock ms | Gate result |
|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit | fp16 | 733.76 | 111,617 | baseline |
| Meta-Llama-3.1-8B-Instruct-4bit | turbo4asym | 490.32 | 167,034 | **pass** |

Notes:
- Llama-3.1-8B-Instruct-4bit passes the turbo4asym PPL gate cleanly.
- The active Qwen2.5 quality-gate fixture is `Qwen2.5-1.5B-4bit` (base variant). Numbers for that row are pending a fresh gate run.
- Gemma-3-4b-it-4bit is ready for a quality-gate run but is not represented in this table yet.
- Decode/prefill tok/s measurements (as distinct from PPL eval throughput) are a follow-up item.

## Speculative drafters

This section records the current parity and perf envelope for the speculative
drafter pairings in the local benchmark setup (Gemma 4 MTP, Qwen 3.5 DFlash).

### Methodology

Driven by `src/bin/speculative_bench.rs` and `tests/speculative_parity.rs`:

- Prompt: 17-token-ish instruction (see `DEFAULT_PROMPT` in the bench source).
- Max new tokens: 96 (matches the upstream `mlx-vlm` README perf-table conditions).
- Sampling: greedy (`temperature = 0.0`).
- Decode-only timing (excludes prefill). Numbers come from `GenerationStats::decode_tok_per_sec`, which divides the generated token count by the decode wall-clock — matches the upstream `_dflash_rounds` / `_mtp_rounds` reporting convention.
- Warm-up: one 4-token generation before the timed run so MLX's lazy Metal kernel compilation doesn't inflate the first measurement.

Invocations:

```bash
# Single pairing:
./target/release/speculative_bench \
    --target models/qwen3.5-4b-4bit \
    --kind none \
    --batch 1 \
    --max-tokens 96 \
    2>&1 | tee /tmp/bench-qwen35-baseline.log

# Full sweep across reachable pairings:
./target/release/speculative_bench --sweep --batch 1 --max-tokens 96 \
    2>&1 | tee /tmp/bench-sweep.log
```

### Hardware + MLX pin

- **Hardware**: Apple M1 Ultra, 128 GB unified memory.
- **MLX upstream commit pin**: `84961223c02925bef6bef95d3a0a046779bde935`
  (the `GIT_TAG` in `src/lib/mlx-cpp/CMakeLists.txt` at the time of measurement,
  which is the single place the pin is written down).
- Re-measure after each MLX pin bump so the perf table reflects the active runtime.

### Reachable pairings

These are the pairings whose target + drafter checkpoints are present on
the M1 Ultra reference host. The no-drafter baseline rows are real numbers
captured on the host; the speculative numerator (tok/s) rows were a perf-bench
follow-up when this table was written, and **correctness parity is verified**
end-to-end by the `#[ignore]`-gated tests in `tests/speculative_parity.rs`.

The follow-up has since been measured: the M1 Ultra matrix below carries real
tok/s for every MTP pairing, and the catalog has grown a Qwen 3.8 27B baseline
and MTP pairing (#1613). The table immediately below is kept as the older
reading it was; read the matrix for current numbers.

| Pairing                       | Kind   | B | block_size | tok/s | speedup vs no-drafter | status                                                                |
|-------------------------------|--------|---|------------|-------|------------------------|-----------------------------------------------------------------------|
| Qwen 3.5 4B (no drafter)      | none   | 1 | —          | 95.4  | 1.00×                  | ok                                                                    |
| Qwen 3.5 4B + DFlash          | dflash | 1 | 16         | —     | —                      | parity verified; tok/s row is a perf-bench follow-up                  |
| Gemma 4 31B (no drafter)      | none   | 1 | —          | 20.4  | 1.00×                  | ok                                                                    |
| Gemma 4 31B + MTP assistant   | mtp    | 1 | 4          | —     | —                      | parity verified; tok/s row is a perf-bench follow-up                  |

### M1 Ultra pairing matrix (2026-09-04, mlxcel 0.7.0-beta.1)

Measured on the Mac Studio M1 Ultra 128 GB (Metal) with
`speculative_bench --sweep --batch 1 --max-tokens 128`. Greedy, decode-only
tok/s, the `DEFAULT_PROMPT` (14 tokens under the Gemma tokenizer, 13 under
Qwen's). Source CSV: `benchmarks/metal_m1ultra_spec_2026-09-04.csv`.

This is the first sweep on any host to carry Gemma 4 31B and Qwen 3.8 27B MTP
rows. #1613 replaced the harness's `LoadedModel::Gemma4Unified` match with
per-variant adapter selection and added the Qwen 3.8 baseline and MTP pairing
to `REACHABLE_PAIRINGS`, so the Gemma 4 31B pairing that reads as a blank in
the M5 Max matrix below, and the Qwen 3.8 pairing that is absent from it
entirely, are both measured here.

| Pairing (M1 Ultra Metal)                | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Qwen 3.5 4B (no drafter)                | none | — | 106.1 | 1.00×                 | —          | —                 | ok |
| Qwen 3.5 4B + DFlash                    | dflash | 2/4/8 | — | —                 | —          | —                 | DEFERRED (DFlash loader + public Qwen3NextCache API) |
| Gemma 4 31B (no drafter)                | none | — | 19.9  | 1.00×                 | —          | —                 | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 2 | 18.4  | 0.93×                 | 74.0%      | 0.74              | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 4 | 14.9  | 0.75×                 | 52.9%      | 1.59              | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 8 | 14.9  | 0.75×                 | 52.9%      | 1.59              | ok (effective K=4) |
| Gemma 4 Unified 12B (no drafter)        | none | — | 38.4  | 1.00×                 | —          | —                 | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 36.0  | 0.94×                 | 54.5%      | 0.55              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 28.5  | 0.74×                 | 39.6%      | 1.19              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 29.3  | 0.76×                 | 39.6%      | 1.19              | ok (effective K=4) |
| Qwen 3.8 27B (no drafter)               | none | — | 24.9  | 1.00×                 | —          | —                 | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 2 | 22.5  | 0.91×                 | 76.1%      | 0.76              | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 4 | 17.2  | 0.69×                 | 50.4%      | 1.51              | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 8 | 9.3   | 0.37×                 | 21.6%      | 1.51              | ok (proposals past the fourth are all rejected) |

**Every MTP pairing reads below 1.00x on this host, and acceptance is not the
reason.** Gemma 4 Unified 12B accepts *more* here than on M5 Max at the same K
(39.6% and mean accepted length 1.19 at K=4, against 35.0% and 1.05), and still
lands at 0.74x where M5 Max reads 1.57x. The difference is the verify round:
`speculative-decoding-m1ultra-2026-08-19.md` measures a block-4 round at 2.70
classic decode steps on M1 Ultra against 1.27 on M5 Max, so break-even needs
about 2.7 emitted tokens per verify here and roughly 1.3 there. A mean accepted
length near 1.2 emits about 2.2 per verify, which clears the M5 Max bar and not
this one. This is the same reading behind the static `MLXCEL_ENABLE_MTP_B1`
gate declining B=1 MTP on Apple GPU generation 13, so the rows confirm the
shipped default rather than contradicting it.

**K=8 splits by drafter family.** The Gemma 4 assistants clamp to their
configured block size of 4: acceptance and mean accepted length are identical
at K=4 and K=8, so the wider request is a no-op. The `qwen3_5_mtp` head honors
it instead, proposing 7 per round against 3 at K=4 while accepting the same
1.51, which drops acceptance to 21.6% and throughput to 9.3 tok/s (0.37x).
K=4 or narrower is the operating point for that pairing.

Two baselines appear in both this matrix and the older reachable-pairings table
above: Qwen 3.5 4B at 106.1 tok/s against 95.4, and Gemma 4 31B at 19.9 against
20.4. Those older rows were captured at a different mlxcel version and MLX pin,
so read them as context rather than as a controlled run-over-run comparison.

### GB10 CUDA pairing matrix (2026-07-10, issue #638)

Measured on the NVIDIA GB10 (Grace-Blackwell) CUDA host with
`speculative_bench --sweep --k-values 2,4,8`. Greedy, decode-only tok/s, the
14-token `DEFAULT_PROMPT`, `--max-tokens 128`. Full analysis and the policy
tuning derivation are in
[`speculative-pairing-gb10-2026-07-10.md`](speculative-pairing-gb10-2026-07-10.md).

| Pairing (GB10 CUDA)                     | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Gemma 4 Unified 12B (no drafter)        | none | — | 14.5  | 1.00×                 | —          | —                 | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 19.0  | 1.31×                 | 56.6%      | 0.57              | ok (multirow qmv, #725) |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 21.2  | 1.46×                 | 35.8%      | 1.07              | ok (multirow qmv, #725) |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 20.4  | 1.41×                 | 34.1%      | 1.02              | ok (effective K=4) |

Pre-#725 record (per-row qmv verify; reproducible with `MLXCEL_QMV_MULTIROW=0`,
which measures 7.7 tok/s at K=4 on the same binary):

| Pairing (GB10 CUDA, pre-#725)           | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 11.1  | 0.77×                 | 55.6%      | 0.56              | regression |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 7.6   | 0.52×                 | 35.0%      | 1.05              | regression |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 7.5   | 0.52×                 | 35.0%      | 1.05              | regression (effective K=4) |

The pre-#725 regression came from the verify `[1, K]` forward hitting the CUDA
quantized dispatch's `M*B < 8` per-row qmv fallback, which costs roughly K
classic forwards instead of amortizing to one the way it does on Apple Silicon
(the same pairing measured ~1.87× on M5 Max at the time; the 2026-09-04 0.6.0 re-measurement reads 1.57× at K=4, at unchanged acceptance, so the two hosts have converged somewhat since). The multirow qmv path (#725,
`MLXCEL_QMV_MULTIROW`) removes that fallback's weight re-reads, and B=1 MTP on
GB10 now clears the 1.4× target from issue #638 at K=4/K=8 at unchanged
acceptance (see `qmv-multirow-gb10-2026-07-11.md`). K=8 collapses onto K=4
because the drafter's configured block size is 4 and the acceptance never
clears the adaptive block-expansion gate. Serving note (#736, resolved): the
adaptive policy no longer relies on the `sqrt(K)` shape heuristic (issue
#638), which was calibrated against the pre-#725 verify and could wrongly
decline this favourable pairing; it now settles verdicts from a measured
comparison against classic-step probe rounds (hint format v3), so this
pairing profiles to an enable verdict in serving without a manual override
(`sqrt(K)` remains only as a fallback for windows with no probe signal). The
DFlash and 31B rows remain deferred (no checkpoint / wrong target family, see
the dated note).

### M5 Max pairing matrix (2026-09-06, mlxcel 0.7.0-beta.1)

Measured on the MacBook Pro M5 Max (Metal) with
`speculative_bench --sweep --max-tokens 128`. Greedy, decode-only tok/s, the
14-token `DEFAULT_PROMPT`. Source CSV: `benchmarks/metal_m5max_spec_2026-09-06.csv`.

| Pairing (M5 Max Metal)                  | Kind | K | tok/s | speedup vs no-drafter | acceptance | mean accepted len | status |
|-----------------------------------------|------|---|------:|----------------------:|-----------:|------------------:|--------|
| Qwen 3.5 4B (no drafter)                | none | — | 171.5 | 1.00×                 | —          | —                 | ok |
| Qwen 3.5 4B + DFlash                    | dflash | 2/4/8 | — | —                 | —          | —                 | DEFERRED (DFlash loader + public Qwen3NextCache API) |
| Gemma 4 31B (no drafter)                | none | — | 28.1  | 1.00×                 | —          | —                 | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 2 | 45.9  | 1.63×                 | 80.0%      | 0.80              | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 4 | 53.5  | **1.90×**             | 58.3%      | 1.75              | ok |
| Gemma 4 31B + MTP assistant             | mtp  | 8 | 50.0  | 1.78×                 | 58.3%      | 1.75              | ok (effective K=4) |
| Gemma 4 Unified 12B (no drafter)        | none | — | 44.7  | 1.00×                 | —          | —                 | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 2 | 62.2  | 1.39×                 | 55.6%      | 0.56              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 4 | 70.1  | **1.57×**             | 35.0%      | 1.05              | ok |
| Gemma 4 Unified 12B + MTP assistant     | mtp  | 8 | 69.4  | 1.55×                 | 34.6%      | 1.07              | ok (effective K=4) |
| Qwen 3.8 27B (no drafter)               | none | — | 32.9  | 1.00×                 | —          | —                 | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 2 | 48.0  | **1.46×**             | 73.5%      | 0.74              | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 4 | 44.3  | 1.35×                 | 50.4%      | 1.51              | ok |
| Qwen 3.8 27B + MTP head                 | mtp  | 8 | 25.5  | 0.78×                 | 22.4%      | 1.57              | ok (net loss) |

**The Gemma 4 31B rows exist for the first time on this host.** The 2026-09-04
sweep recorded `MTP bench currently supports a Gemma 4 Unified target` for all
three cells, because `run_mtp` hard-matched `LoadedModel::Gemma4Unified` and the
31B checkpoint loads as another variant. #1621 replaced that with per-variant
adapter dispatch. The pairing turns out to be the strongest on the host: **1.90×
at K=4**, on 58.3% acceptance and a mean accepted length of 1.75.

**Gemma 4 Unified reproduces 2026-09-04 within noise** (1.39× / 1.57× / 1.55×
on both dates, at acceptance identical to three decimal places). It is the
control that makes the other two rows readable: the harness, the prompt and the
drafter path did not move between the sweeps, so the new numbers are new
coverage rather than a shifted baseline. Acceptance also still matches GB10
exactly at K=2 and K=4, which is expected for a greedy run over a fixed prompt
and confirms the drafter is doing the same work on both backends.

**Qwen 3.8 27B at K=8 is a net loss (0.78×)** and is the clearest illustration
of why acceptance belongs next to tok/s in this table. Acceptance collapses from
73.5% at K=2 to 22.4% at K=8 while mean accepted length barely moves (0.74 to
1.57), so the verify block grows without the drafter earning it back and the
pairing lands below its own no-drafter baseline. K=2 is this pairing's operating
point, unlike the Gemma pairings where K=4 wins.

K=8 buys nothing anywhere in this matrix: on both Gemma pairings the mean
accepted length is unchanged from K=4, so every proposal past the fourth is
discarded.

### Deferred pairings

These pairings cannot be measured today because the drafter checkpoint is
not on the reference host AND/OR an upstream dependency is unresolved.

| Pairing                          | Drafter checkpoint                              | Status / blocker                                                                  |
|----------------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|
| Gemma 4 E2B + MTP assistant      | `mlx-community/gemma-4-E2B-it-assistant-bf16`   | drafter checkpoint not on disk; centroid LM head support required                 |
| Gemma 4 E4B + MTP assistant      | `mlx-community/gemma-4-E4B-it-assistant-bf16`   | drafter checkpoint not on disk; centroid LM head support required                 |
| Gemma 4 26B-A4B + MTP assistant  | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` | drafter checkpoint not on disk                                                  |

### Real-model byte-equality parity test

`tests/speculative_parity.rs` carries two `#[ignore]`-gated real-model
tests — `greedy_parity_dflash_qwen35_4b` and `greedy_parity_mtp_gemma4_31b`
— that verify speculative-decoding **correctness** end-to-end.
Each test runs two phases:

1. **Structural phase** (in-process): load the target, assert the model
   variant, resolve the drafter kind, load the drafter, and — for DFlash —
   `bind()` the drafter to the target.
2. **Byte-equality phase** (subprocess): spawn `mlxcel-server` twice
   against the same target — once with `--model-draft --draft-kind
   {dflash,mtp} --draft-block-size {16,4}` and once without any
   `--draft-*` flag — submit the same fixed prompt to
   `/v1/chat/completions` at `temperature = 0`, and assert the two
   responses are byte-identical (same `message.content` *and* same
   `usage.completion_tokens`). The two servers run sequentially so a
   32–48 GB host only holds one target's weights at a time.

#### CI hardware lane / fixed cadence

These tests are `#[ignore]`-gated so `cargo test` on a dev machine (or a
CI host without the model checkpoints) skips them. They are run on the
**hardware lane** — an Apple Silicon runner with the model checkpoints
mounted under `models/` — on a fixed cadence:

```bash
# Run both speculative real-model parity tests serially (required:
# they share GPU memory and each spawns mlxcel-server subprocesses).
cargo test --test speculative_parity --release -- --ignored --test-threads=1 --nocapture
```

A test whose checkpoints are absent self-skips with a log line, so the
invocation is safe to wire into any Apple Silicon CI lane regardless of
which checkpoints that lane has provisioned.

Once the perf-bench numerators are captured, the speculative tok/s rows in
the table above flip on, and the table grows additional rows for the
`(block_size ∈ {2, 3, 4, 5, 6, 8}, B ∈ {1, 4, 8})` MTP sweep and
`(block_size ∈ {4, 8, 16, 24, 32}, B ∈ {1, 4, 8})` DFlash sweep.

### Expected speedup envelope (per upstream `mlx-vlm` README)

For comparison with the eventual measured numbers — these are the
upstream M3 Max / 96 GB results, NOT mlxcel measurements:

| Pairing               | B | block_size | upstream speedup                                                          |
|-----------------------|---|------------|---------------------------------------------------------------------------|
| Gemma 4 26B-A4B + MTP | 4 | 3          | 3.94×                                                                     |
| Gemma 4 31B + MTP     | 4 | 3          | 2.29×                                                                     |
| Gemma 4 E4B + MTP     | 4 | 4          | 1.56×                                                                     |
| Gemma 4 E4B + MTP     | 16| any        | slower than baseline (overhead > speedup at high B on small target)       |

DFlash speedup envelope is not documented as concretely upstream. mlxcel's
measured numbers will become the reference table once the speculative
perf-bench numerators are captured on the hardware lane.
