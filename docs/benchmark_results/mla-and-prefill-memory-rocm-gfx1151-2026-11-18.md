# MLA absorbed decode and prefill memory on ROCm (gfx1151, 2026-11-18)

Issue #3133: the ROCm half of the 2026-11 memory and MLA changes (#3114, #3117, #3119, #3126, #3129), which had only been measured on Apple Silicon and, on 2026-11-18, on CUDA (GB10, [#3149](https://github.com/lablup/mlxcel/pull/3149)). #3133 closed without a ROCm row because no host was available; this record fills that row on a gfx1151 machine.

Per-run rows: [`decode.csv`](mla-and-prefill-memory-rocm-gfx1151-2026-11-18/decode.csv), [`trace.csv`](mla-and-prefill-memory-rocm-gfx1151-2026-11-18/trace.csv), [`prefill-peak.csv`](mla-and-prefill-memory-rocm-gfx1151-2026-11-18/prefill-peak.csv).

## Setup

- **Host:** AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, 20 WGP, wave32), HIP 7.15.26333, ROCm clang 23.0.0git, 30 GiB unified memory, 32 cores, Linux 6.18.12. Every GPU run held the host's ROCm guard lock (`scripts/rocm_gpu_guard.sh --hold`), no build ran during a throughput run, and A/B arms ran in ABBA order with 60 s idle before each run.
- **Versions:** MLX pin `fc344c98`, MLX ROCm overlay commit `75915908` (`src/lib/mlxcelverse/overlays/rocm/UPSTREAM`), backend `rocm-7.15`.
- **Binary:** release build with `--features rocm` from `c27b86f4`. #3149 (`7f8b6c95`) landed after the measurement and changes only the default-resolution rule; both arms here are forced with `MLXCEL_MLA_ABSORBED`, so its two code paths are the ones measured. The section 5 baselines are `b21825cf` (parent of #3114) and `3efb3fb2` (parent of #3119), each in its own worktree.
- **Models:** `deepseek-v2-lite-chat-4bit-mlx`, `kimi-vl-a3b-thinking-4bit`, `glm-4.7-flash-4bit`, `gpt-oss-20b-mxfp4-q4`.
- **Correctness:** teacher-forced traces from `examples/logit_trace`, judged by decided-position disagreement (reference top-1 margin of at least 2 logits), never by byte identity or perplexity alone.

## 3. Absorbed MLA on the composed path (DeepSeek V2-Lite, Kimi-VL)

ROCm has no fused absorbed-decode kernel, so `MLXCEL_MLA_ABSORBED=1` takes the composed path: `MLXCEL_TRACE_MLA_STATS=1` prints `absorbed_composed` for every decode step and `absorbed_fused=0`. The KV cache drops from 10240 to 1152 bytes per token per layer (8.9x).

| shape | decided disagree, `=1` vs `=0` | top-1 disagree | ppl `=0` / `=1` | Metal | CUDA |
|---|---|---|---|---|---|
| decode width 1 behind 1024 | 0 of 208 | 6 of 512 | 14.6489 / 14.6297 | 0 of 209 | 0 of 208 |
| decode width 1 behind 4096 | 0 of 105 | 3 of 256 | 12.9619 / 12.9590 | 0 of 106 | 0 of 106 |

Every disagreement sits where the reference's own top two were within 0.25 logits at 1024 (rank 2 five times, rank 3 once) and 0.234 at 4096 (rank 2 twice, rank 3 once), which is the rounding class rather than a behaviour change. On the measured commit the unset default still resolved to decompressed on ROCm, and it traced byte-identical to `=0` at both lengths, with only `decompressed` steps. Kimi-VL (`deepseek_v3` decoder, always absorbed) behind 1024 tokens counts `absorbed_composed=13824`, `absorbed_fused=0`, and finishes at ppl 17.9719 (Metal composed reference 18.04).

## 4. Should ROCm switch the V2 default?

`mlxcel-bench-decode`, deepseek-v2-lite-chat-4bit-mlx, `--ignore-eos`, tg128, 16 warmup tokens, median decode tok/s.

| prompt | `=0` | `=1` (composed) | |
|---|---|---|---|
| 512 (first round, 2 runs per arm) | 40.30 (42.97, 37.62) | 42.57 (42.89, 42.25) | +5.6% |
| 512 (repeat, 5 runs per arm) | 42.72 (41.62 to 43.57) | 41.28 (36.94 to 41.78) | -3.4% |
| 2048 | 35.28 (35.87, 34.68) | 41.41 (40.88, 41.93) | +17% |
| 16384 | 22.04 (22.05, 22.02) | 36.49 (36.19, 36.78) | +65% |

The 512-token cell is the least certain at two runs: the decompressed arm spread 14% across its two runs and read +5.6%, so it was repeated five times per arm. That repeat puts the composed path 3.4% behind at the median, with a 4.6% spread in the decompressed arm and 11.7% in the composed arm (one run at 36.94); dropping that run moves the composed median to 41.45, still about 3% behind. Prefill is level at 512 and 2048 (151.4 tok/s in every run except one 127.3 outlier) and 4% lower on the composed arm behind 16K (139.5 and 140.3 against 145.9 and 146.5), where the composed arm's phase peak is 3.7 GB smaller. Phase peak behind 16K is 16.06 GB decompressed and 12.39 GB composed; behind 2048 it is 10.89 against 10.19 GB. From 2048 both arms are tight and the composed path is far ahead.

**Verdict:** both of #3102's conditions hold on ROCm: the composed path is 3.4% behind at 512, inside the "within a few percent" band the Metal default was switched under (-1.7% with the fused kernel), and the teacher-forced trace shows 0 decided disagreements at both lengths. The 17% gain at 2048, the 65% gain at 16384 and the 3.7 GB smaller peak there make that a trade worth taking, so the DeepSeek V2 default now absorbs on ROCm too (`mla::default_absorbs_on`).

## 5. Prefill peak memory (#3114, #3119)

`make_prompt.py 28000` prompt (22,754 tokens for GLM and 19,773 for gpt-oss through `generate`); MLX phase peak from `mlxcel-bench-decode --prompt-tokens 28000`. `/usr/bin/time` is not installed on this host, so no max-RSS column.

| model | path | before | after | prefill tok/s (generate) |
|---|---|---|---|---|
| glm-4.7-flash-4bit (`b21825cf` vs branch) | 2048 chunks | 25.26 GB | 20.53 GB | 24.75 to 74.12 |
| glm-4.7-flash-4bit | single pass | failed, 62.72 GB allocation | 21.13 GB | 24.75 to 74.89 |
| gpt-oss-20b-mxfp4-q4 (`3efb3fb2` vs branch) | 2048 chunks | 26.21 GB | 26.21 to 26.43 GB | 324.4 to 325.3 |
| gpt-oss-20b-mxfp4-q4 | single pass | failed, 100.35 GB allocation | failed identically | none |

GLM after #3114 fits a 28K single pass in 21.13 GB on this 30 GiB host, where the old tree asks for a 62.72 GB allocation and fails; with 2048-token chunks the new tree peaks 4.7 GB lower and prefills 3.0x faster (306.9 and 312.3 s against 919.2 s). The before arm's chunked bench cell never produced a throughput figure: the warmup prefill peaked at 25.22 to 25.26 GB and the measured second pass did not finish inside a 40-minute bound, which is why its throughput comes from the single `generate` pass.

gpt-oss is unchanged on ROCm by design: #3119's sink prefill takes MLX's causal mode only where `maskless_route_is_fused` says MLX fuses it, and a 19,773-token single pass asks for a 100.35 GB allocation in both trees. The chunked peak (26.21 GB on both arms, 26.43 GB on one branch run) is also far above CUDA's 14.52 GB, so ROCm keeps a larger per-chunk buffer here; that is a pre-existing property of the backend, not a change in either tree. Greedy text is identical across trees on both models (the ledger echo at temperature 0 digests to `ef011275` on GLM and `b54a2d8f` on gpt-oss in every run). Metal reference: GLM 2048 chunks 28.82 to 20.51 GB, single pass failed to 24.11 GB; gpt-oss single pass 22.54 to 17.86 GB.

## 6. Llama 4 Scout past the 8192 chunk (#3117, #3126)

Not run on this host. Llama 4 Scout is a 61 GB checkpoint and the host has 30 GiB of unified memory, so neither the 9216-token single pass nor the 8720-token decode rows fit. The change is covered on CUDA (GB10, #3149) and on Metal; nothing in #3117 or #3126 is backend-conditional.

## 7. KV sizing follows the runtime layout

`kv_arch` reads `mla::absorbed_enabled()` since #3137, so the sizing follows the ROCm default change with no further code; the prompt-cache snapshot sizing and the KV cache advisor read the same estimate. `mlxcel inspect -m models/deepseek-v2-lite-chat-4bit-mlx --max-tokens 32768` on a build of this change (the throughput and trace arms above use the `c27b86f4` build):

| setting | KV cache | layout | total estimate |
|---|---|---|---|
| unset (now absorbed on ROCm) | 972.0 MiB | MLA compressed latent, 576 elements per token per layer | 12.12 GiB |
| `MLXCEL_MLA_ABSORBED=0` (the previous ROCm default) | 8.44 GiB | decompressed per-head K/V | 21.10 GiB |

The totals differ from the CUDA record (11.12 and 20.10 GiB) because ROCm adds the backend in-flight reserve as a separate term.

## Gate

The full gate did not run on this host. `make verify-rocm-held` was stopped by the harness's memory guard on this 30 GiB machine, so this record carries no local full-gate result. The narrow runs used here, all under `scripts/rocm_gpu_guard.sh --hold`, were `cargo test -p mlxcel-core --features rocm --lib mla::` (61 passed, 0 failed) and `cargo test -p mlxcel --features rocm --lib execution::kv_arch::tests` (35 passed, 0 failed). The default resolution was checked with the branch binary (section 7). Hosted CI carries the rest of the suite.

## Commands

```
M=models/deepseek-v2-lite-chat-4bit-mlx; K=models/kimi-vl-a3b-thinking-4bit
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; MLXCEL_MLA_ABSORBED=0 for decompressed
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > abs-1k.tsv     # 64 4 8 4096; same for $K and for =0
python3 scripts/compare_logit_traces.py dec-1k.tsv abs-1k.tsv --decided 2
MLXCEL_PREFILL_CHUNK=2048 mlxcel-bench-decode -m models/glm-4.7-flash-4bit -p Hello -n 16 --prompt-tokens 28000
```

The corpus for the traces is `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` concatenated.
