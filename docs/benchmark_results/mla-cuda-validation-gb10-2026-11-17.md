# CUDA validation of the 2026-11 MLA and attention changes, and the DeepSeek V2 default layout on CUDA (GB10, 2026-11-16 to 2026-11-17)

Issue #3133. The 2026-11 prefill, chunked-decode and MLA changes (#3114, #3117, #3119, #3126, #3129) were measured only on Apple Silicon. This record covers the CUDA side on a GB10. Its main result is that DeepSeek V2's composed absorbed decode, the path CUDA takes because the fused absorbed-decode kernel is Metal-only, meets #3102's default bar. The default layout therefore switches to the latent cache on CUDA as well. ROCm was not measured and keeps the decompressed layout.

## Environment

| Field | Value |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10 (sm_121) |
| mlxcel | upstream `main` at `5e9caf8a` (MLX pin `fc344c98`), release build with `--features cuda` |
| Discipline | quiet box, every GPU run under `gpu-lock`, ABBA order with 60 s idle between runs |
| Driver, toolkit, binary hashes | not recorded in the run reports |

The pin advance in #3134 had never been built for CUDA before this run. It builds and runs.

## DeepSeek V2-Lite: composed absorbed path correctness

deepseek-v2-lite-chat-4bit-mlx, teacher-forced trace with `examples/logit_trace`, compared with `scripts/compare_logit_traces.py --decided 2` against `MLXCEL_MLA_ABSORBED=0`.

| Context | Decided positions that disagree | ppl (`=0` / `=1`) | Metal, same check |
|---|---|---|---|
| 1024 | 0 of 208 | 14.6582 / 14.6543 | 0 of 209 |
| 4096 | 0 of 106 | 12.9960 / 12.9485 | 0 of 106 |

With `=1` the MLA counters show `absorbed_composed` only (`absorbed_fused=0`), and the KV cache drops from 10240 to 1152 bytes per token per layer. The default arm of this build (before the switch) was byte-identical to `=0`. Kimi-VL, which always absorbs, also takes the composed path on CUDA, with ppl 18.01 (Metal 18.04).

## DeepSeek V2-Lite: decode throughput

`mlxcel-bench-decode --ignore-eos -n 128`, two runs per arm in the first round:

| Prompt | `=0` (tok/s) | `=1` (tok/s) |
|---|---|---|
| 512 | 101.8 / 101.1 | 90.6 / 98.0 |
| 2048 | 83.8 / 79.1 | 98.7 / 96.9 |
| 16384 | 30.1 / 30.4 | 72.0 / 73.7 |

Absorbed wins from 2048 tokens up (2.4x behind 16K). At 512, the two `=1` runs were 8% apart, so the gap could not be told from noise. A second round at 512 separated three explanations: noise, a fixed per-step cost in the composed path, and first-seen-shape cost inside the decode window (the default 20-token warmup reaches a cache length of about 532, the measured pass 640).

Median decode tok/s (min to max):

| Cell | `=0` | `=1` | `=1` vs `=0` |
|---|---|---|---|
| `-n 128`, default warmup (20), 5 runs | 96.05 (92.51 to 97.21) | 98.52 (93.23 to 100.20) | +2.6% |
| `-n 128 --warmup-tokens 160`, 5 runs | 98.63 (97.34 to 99.61) | 98.92 (92.69 to 101.49) | +0.3% |
| `-n 512`, default warmup, 3 runs | 93.47 (64.01 to 96.03) | 94.43 (91.43 to 94.90) | +1.0% |

The gap is noise. Each arm varies 4 to 8% from run to run, the medians are within 3% in every cell, and `=1` is never behind. A longer warmup does not change the ratio, so first-seen shapes are not the cause. A 512-token generation keeps the arms level rather than at a fixed per-step gap, so the composed path's per-step cost is not the cause either. Prefill is 1942 to 1999 ms in every run, with no arm difference; the decode window starts after the first token is evaluated, so prefill is not counted in it. The 64.01 run (`=0`, `-n 512`) is a single outlier.

Every `=1` run took `absorbed_composed` (3942, 7722 and 14310 calls per cell, counted over warmup and measured passes). The 54 `decompressed` calls in each `=1` run are the two cold prefills of 27 layers each. For this round, the measurement build printed `mla::stats::snapshot().summary()` after the harness's `Decode path:` line. The counters are always recorded, so the measured code was the same as `main`; the print was not committed. Per-run rows: [decode_pp512.csv](mla-cuda-validation-gb10-2026-11-17/decode_pp512.csv).

**Decision.** With 0 decided disagreements and decode level at 512 and faster from 2048, the composed path meets #3102's bar on CUDA. `mla::COMPOSED_DEFAULT_MEASURED` is true on CUDA builds, so the unset `MLXCEL_MLA_ABSORBED` selects the latent cache there. No length-based switch is needed.

## Prefill peak memory, 28K-token prompt

MLX phase peak from `mlxcel-bench-decode`.

| Model | Path | Before | After | Prefill tok/s |
|---|---|---|---|---|
| GLM-4.7-flash (#3114) | 2048-token chunks | 24.79 GB | 22.73 GB | 421 to 535 |
| GLM-4.7-flash (#3114) | single pass | 74.56 GB | 30.18 GB | 284 to 727 |
| gpt-oss-20b (#3119) | 2048-token chunks | 14.52 GB | 14.52 GB | 1698 to 1699 |
| gpt-oss-20b (#3119) | single pass | 26.63 GB | 26.63 GB | 886 to 890 |

The old GLM single pass did not fail on CUDA; it peaked at 74.56 GB. After #3114 its greedy output matches the chunked runs, where the old single pass differed in wording (still coherent). gpt-oss is unchanged on CUDA by design: `maskless_route_is_fused` is Metal-only, so the sink prefill keeps the band, and the output is identical across trees.

## Llama 4 Scout past 8192 tokens

A 9216-token single pass against two 4608-token chunks gives 0 of 3642 decided disagreements (Metal 0 of 3651), ppl 11.8001 against 11.7992.

Generation from an 8720-token prompt was garbage on CUDA, also on the builds before #3114 and #3119. The CUDA SDPA overlay's `is_kv_cache_slice` lacked the `kv.offset() != 0` check, so the chunked cache's decode window, a slice that starts partway into the buffer, was rebased to the buffer start, and cuDNN read the wrong keys once the window reached 256. #3138 added the check with a regression test (relative rms 1.12 without it). With the fix, Scout answers correctly and a teacher-forced decode after the 8720-token prefill gives 0 of 149 decided disagreements (NLL 1.853 against 1.854).

## Not covered

- The CUDA test gate (`make verify-test-cuda`) did not finish. The host killed it under memory pressure in `maskless_padded_prefill_parity` (olmo3 case) after 72 test binaries: 12688 passed, 0 failed, 279 ignored. The rest needs another run.
- ROCm was not measured. Its default stays decompressed; `MLXCEL_MLA_ABSORBED=1` opts in.
