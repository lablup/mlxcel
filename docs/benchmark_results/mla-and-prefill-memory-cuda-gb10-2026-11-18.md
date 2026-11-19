# MLA absorbed decode, prefill memory and Llama 4 long context on CUDA (GB10, 2026-11-18)

Issue #3133: the CUDA half of the 2026-11 memory and MLA changes (#3114, #3117, #3119, #3126, #3129), which had only been measured on Apple Silicon. No ROCm host was available.

Per-run rows: [`decode.csv`](mla-and-prefill-memory-cuda-gb10-2026-11-18/decode.csv), [`prefill-peak.csv`](mla-and-prefill-memory-cuda-gb10-2026-11-18/prefill-peak.csv), [`trace.csv`](mla-and-prefill-memory-cuda-gb10-2026-11-18/trace.csv).

## Setup

- **Host:** NVIDIA GB10 (sm_121), driver 580.178.04, CUDA 13.0.88 (`cuda-13.0`), 121 GB unified memory. Every GPU run held the host's `gpu-lock`, no build ran during a throughput run, and A/B arms ran in ABBA order with 60 s idle before each run.
- **Binaries:** release builds with `--features cuda`. Sections 3 to 6 measured main `5e9caf8a`. The section 5 baselines are `b21825cf` (parent of #3114) and `3efb3fb2` (parent of #3119), each in its own worktree. The Scout rows marked "with #3138" use `5e9caf8a` plus that fix. The gate ran on main `fab141b1`.
- **Correctness:** teacher-forced traces from `examples/logit_trace`, judged by decided-position disagreement (reference top-1 margin of at least 2 logits), never by byte identity or perplexity alone.

## 3. Absorbed MLA on the composed path (DeepSeek V2-Lite, Kimi-VL)

CUDA has no fused absorbed-decode kernel, so `MLXCEL_MLA_ABSORBED=1` takes the composed path: the stats show `absorbed_composed` on every decode step and `absorbed_fused=0`. The KV cache drops from 10240 to 1152 bytes per token per layer (8.9x).

| shape | decided disagree, `=1` vs `=0` | top-1 disagree | ppl `=0` / `=1` | Metal |
|---|---|---|---|---|
| decode width 1 behind 1024 | 0 of 208 | 6 of 512 | 14.6543 / 14.6582 | 0 of 209 |
| decode width 1 behind 4096 | 0 of 106 | 4 of 256 | 12.9485 / 12.9960 | 0 of 106 |

On `5e9caf8a` the unset default traced byte-identical to `=0` at both lengths, with only `decompressed` steps. Kimi-VL (`deepseek_v3` decoder, always absorbed) behind 1024 tokens counts `absorbed_composed=13824`, `absorbed_fused=0`, and finishes at ppl 18.01 (Metal composed reference 18.04).

## 4. Should CUDA switch the V2 default?

`mlxcel-bench-decode`, deepseek-v2-lite-chat-4bit-mlx, `--ignore-eos`, median decode tok/s.

| prompt | cell | `=0` | `=1` (composed) | |
|---|---|---|---|---|
| 512 | tg128, 16 warmup, 2 runs (first round) | 101.42 | 94.31 | -7.0% |
| 512 | tg128, 20 warmup, 5 runs | 96.05 (92.51 to 97.21) | 98.52 (93.23 to 100.20) | +2.6% |
| 512 | tg128, 160 warmup, 5 runs | 98.63 (97.34 to 99.61) | 98.92 (92.69 to 101.49) | +0.3% |
| 512 | tg512, 20 warmup, 3 runs | 93.47 (64.01 to 96.03) | 94.43 (91.43 to 94.90) | +1.0% |
| 2048 | tg128, 16 warmup, 2 runs | 81.48 | 97.78 | +20% |
| 16384 | tg128, 16 warmup, 2 runs | 30.23 | 72.85 | 2.4x |

The first round's -7% at 512 came from two runs per arm, one of which (90.64) was low. Five runs per arm show 4 to 8% spread within an arm and medians that never put the composed path behind. A longer warmup leaves the ratio unchanged, which rules out first-seen shapes, and tg512 keeps the arms level, which rules out a fixed per-step cost. The 64.01 run in the tg512 cell is a single outlier. Prefill is 1942 to 1999 ms in every run, with no arm difference. Each `=1` run of the second round prints `absorbed_composed` for every decode step (`decompressed=54` are the two cold prefills, 27 layers each).

**Verdict:** both of #3102's conditions hold on CUDA (within a few percent at 512, 0 decided disagreements), so the DeepSeek V2 default now absorbs on CUDA too (`mla::default_absorbs`). On the switched binary, the unset default traces byte-identical to the `=1` trace above (0 of 208 decided against `=0`, `absorbed_composed=13824`, `absorbed_fused=0`), and `deepseek_v2_greedy_parity` still matches mlx-lm. ROCm, the CPU backend and Metal with `MLXCEL_MLA_FUSED_DECODE=0` keep the decompressed default. Metal reference: -1.7% at 512, +1.8% at 2048, +48% at 16K (fused kernel).

## 5. Prefill peak memory (#3114, #3119)

`make_prompt.py 28000` prompt (22,754 tokens for GLM and 19,773 for gpt-oss through `generate`); MLX phase peak from `mlxcel-bench-decode --prompt-tokens 28000`, which is the primary figure. `/usr/bin/time -v` max RSS stays under 6 GB in every run because it does not count CUDA allocations even on unified memory, so it is in the CSV only.

| model | path | before | after | bench prefill tok/s |
|---|---|---|---|---|
| glm-4.7-flash-4bit (`b21825cf` vs main) | 2048 chunks | 24.79 GB | 22.73 GB | 421 to 535 |
| glm-4.7-flash-4bit | single pass | 74.56 GB | 30.18 GB | 284 to 726 |
| gpt-oss-20b-mxfp4-q4 (`3efb3fb2` vs main) | 2048 chunks | 14.52 GB | 14.52 GB | 1698 to 1699 |
| gpt-oss-20b-mxfp4-q4 | single pass | 26.63 GB | 26.63 GB | 886 to 890 |

Both runs of each arm agree to the hundredth of a GB. The old GLM single pass did not fail on CUDA as it did on Metal; it completed at 74.56 GB. Its greedy text differed in wording from every other GLM run (still coherent), and after #3114 the single pass writes the same text as both chunked arms. gpt-oss is unchanged on CUDA by design: the sink prefill takes MLX's causal mode only where `maskless_route_is_fused` says MLX fuses it, which is Metal only, so CUDA keeps the band; the greedy text is identical across trees. The first `generate` run of each tree includes CUDA kernel compilation, so `generate` prefill times in the CSV vary more than the bench figures. Metal reference: GLM 2048 chunks 28.82 to 20.51 GB, single pass failed to 24.11 GB; gpt-oss single pass 22.54 to 17.86 GB.

## 6. Llama 4 Scout past the 8192 chunk (#3117, #3126, #3135)

| check | result |
|---|---|
| one 9216-wide forward vs two 4608-wide forwards | 0 of 3642 decided (Metal 0 of 3651), top-1 22 of 9216, ppl 11.8001 vs 11.7992, 82 s vs 42 s wall |
| decode after an 8001-token prefill, positions 8001 to 8191 | 0 of 40 decided, NLL 3.565 vs 3.566 |
| decode after an 8001-token prefill, positions 8192 to 8448 | 0 of 60 decided, NLL 3.346 vs 3.234; the last three steps, as the window reaches 256 keys, are the defect's onset (NLL 13.1 vs 4.3, 23.2 vs 10.2, 12.4 vs 5.5) |
| decode after an 8720-token prefill, before #3138 | NLL 21.49 vs 2.85 on the first step, non-finite on the third |
| decode after an 8720-token prefill, with #3138, positions 8720 to 9019 | 0 of 149 decided, top-1 1 of 300, NLL 1.853 vs 1.854 |
| `generate --temp 0 -n 64`, 8720-token prompt, `MLXCEL_PREFILL_CHUNK=0` and default | garbage before #3138; with it, both write the same correct answer |
| MLX phase peak, one 9216-token prefill (`mlxcel-bench-decode`, `MLXCEL_PREFILL_CHUNK=0`, on the branch binary) | 68.03 GB with 60.62 GB of weights, 685 tok/s (Metal: 67.38 GB for the forward through `lm_head`) |

The decode references are single-pass forwards over the same text, where the decode attention path is not involved. Decode past 8192 broke on CUDA once a chunk-local decode window reached 256 keys: the CUDA SDPA overlay took the window, a slice starting mid-buffer, for a KV cache slice and rebased it to the buffer start before cuDNN read it. #3138 fixed that. With it, Scout on the 8720-token prompt prefills at 633 tok/s in one pass and 451 tok/s with the default chunking, and decodes at 10.9 and 11.5 tok/s.

## 7. KV sizing follows the runtime layout

#3137 made `kv_arch` read `mla::absorbed_enabled()`, so the sizing follows the CUDA default change with no further code; the prompt-cache snapshot sizing and the KV cache advisor read the same estimate. `mlxcel inspect -m models/deepseek-v2-lite-chat-4bit-mlx --max-tokens 32768` on the switched binary:

| setting | KV cache | layout | total estimate |
|---|---|---|---|
| unset (now absorbed on CUDA) | 972.0 MiB | MLA compressed latent, 576 elements per token per layer | 11.12 GiB |
| `MLXCEL_MLA_ABSORBED=0` (the previous CUDA default) | 8.44 GiB | decompressed per-head K/V | 20.10 GiB |

## Gate

`cargo test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1` on `fab141b1`, skipping `glm4v_moe_maskless_padded_prefill_matches_unpadded`: all 184 test binaries ran, 15,910 passed, 0 failed, 498 ignored. Without the skip, `make verify-test-cuda` cannot finish on this 121 GB host: the kernel OOM killer stopped it twice in `tests/maskless_padded_prefill_parity.rs`, where glm4v_moe alone takes MemAvailable from 101 GB to 13 GB and memory from earlier tests in the binary is not returned. One run also failed `distributed::stage_assignment::coordinator_names_peers_that_did_not_commit`, a port reservation race in the test helper that passed 5 of 5 times alone and in the full run. Both are #3147. #3140 (a debug-profile `debug_assert` in `paged_detach`) was found on the way.
