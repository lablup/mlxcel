# Turbo-4 WHT latent for absorbed MLA on DeepSeek V2 (M3 Ultra, 2026-11-19)

Issue #3157, epic #3158. Under absorbed decode every Turbo KV mode (`fp16+turbo4`, `turbo4`, `turbo4-delegated`, `fp16+turbo3`) resolves to the fp16 latent on `deepseek_v2`, because the Turbo codecs were built for per-head K/V rows. Stored apart from `k_pe`, the 512-wide latent `ckv` is a power of two, so the Walsh-Hadamard rotation applies to it directly. This run prototypes a Turbo-4 latent, measures it against the shipped fp16 and int8 latents, and decides whether to adopt it.

**Outcome: not adopted.** The Turbo-4 latent is the smallest layout measured (386 bytes per token per layer), but it fails the #3155 quality gate the same way 4-bit affine did, and it is also the slowest decode. No mode or flag reaches it. Turbo requests on an absorbed V2 latent keep resolving to the fp16 latent, and the warning still names `--kv-cache-mode int8`, which is the one quantized latent that ships.

## Design measured

- **Turbo-4 latent.** The cache stores `R ckv` with `R = D2 H D1` (orthonormal WHT between two sign flips): one 4-bit Lloyd-Max index per coordinate (256 bytes), one fp16 rescale `norm / |codebook[idx]|` per token, and `k_pe` fp16. Because `R` is orthogonal, `q . ckv = (R q) . (R ckv)` and `p . ckv = R^T (p . R ckv)`. A decode step rotates the absorbed query once, scores and sums against the gathered codebook values with the rescale folded into the score columns and the probabilities, and inverse-rotates the `[B, H, 1, 512]` output once. One sign-vector pair is used because K and V are the same tensor here. The seed is the cache's `turbo_seed` (`TURBO_DEFAULT_SEED`), so a detached and re-adopted cache rebuilds the same rotation. The codec runs on device and is byte-identical to the host `quantize_v_turbo4` (`mla::turbo_latent_tests::on_device_codec_matches_the_host_turbo4_codec`).
- **WHT + affine int4 latent (second arm).** The same rotation in front of the #3155 affine group-64 codec at 4 bits, decoded by the shipped `quantized_absorbed_attention` on the rotated query. This tests whether whitening is what 4-bit affine was missing. Same 416 bytes as unrotated int4.

Multi-token steps and the cold prefill dequantize the window, inverse-rotate it, and run the existing dense code, as the int8 latent does.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal.
- **Model:** `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx` (27 layers, `kv_lora_rank` 512, `qk_rope_head_dim` 64, 16 heads).
- **Binary:** release build of `43f1c8d4` plus the prototype and a temporary measurement hook ([measurement-hook.patch](mla-turbo-latent-m3ultra-2026-11-19/measurement-hook.patch)). The hook routes an fp16 latent cache through a prototype when `MLXCEL_MLA_LATENT_PROTOTYPE` is `turbo4` or `wht-int4`. It is not in the shipped tree. To reproduce, apply the patch and register the modules in `mla/mod.rs` as `pub(crate) mod turbo_latent;` and `pub(crate) mod turbo_latent_hook;`. The hook stores the Turbo rescale twice (once in the bias slot), so its runtime buffers cost 388 bytes per token per layer, not the codec's 386.
- **Isolation:** every GPU run held the shared GPU lock. The decode sweep ran under `scripts/with_indexers_paused.sh`, one lock hold per prompt length, with the four arms interleaved round by round inside it. Other units were compiling on the host during the traces and the 512/2048 rounds (load average 16 to 24). The 16384 rounds ran at load 1.0 to 1.7. Traces are deterministic, so load affects only throughput.
- **Moonlight and Kimi-VL** (`deepseek_v3`, `kimi_vl`) cache their latent pair through `KVCache::update_and_fetch` directly, not through `MlaLatentCache`, so neither the hook nor any latent codec reaches them yet (#3160 to #3163). They were not run.

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

Reference is the fp16 latent. The corpus is the #3155 one (`docs/architecture.md`, `docs/benchmarks.md`, `docs/turbo-kv-cache.md` from `3c0d74ec` concatenated, 50338 tokens). Width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`. The fp16 reference perplexities (14.6613 and 12.9589) and a same-binary int8 rerun behind 1024 (12 / 512, gap 0.078, ppl 14.7041) match #3155 exactly, so the rows below are comparable with the recorded int4 affine rows. The int8 row behind 4096 was not rerun here; it is quoted from #3155, as are the int4 affine rows.

| candidate | shape | decided disagree | top-1 disagree | flips at gap > 0.11 | flips at gap > 0.5 | largest gap at a disagreement | ppl reference / candidate | ppl delta |
|---|---|---|---|---|---|---|---|---|
| Turbo-4 latent | behind 1024 | 0 / 209 | 45 / 512 | 21 | 3 | 0.719 | 14.6613 / 14.6286 | -0.223% |
| Turbo-4 latent | behind 4096 | 0 / 105 | 16 / 256 | 10 | 3 | 0.641 | 12.9589 / 12.7893 | -1.308% |
| WHT + affine int4 | behind 1024 | 0 / 209 | 42 / 512 | 22 | 6 | 0.922 | 14.6613 / 14.6600 | -0.009% |
| WHT + affine int4 | behind 4096 | 0 / 105 | 11 / 256 | 6 | 1 | 1.125 | 12.9589 / 13.1785 | +1.695% |
| int8 latent (shipped, #3155) | behind 1024 | 0 / 209 | 12 / 512 | 0 | 0 | 0.078 | 14.6613 / 14.7041 | +0.292% |
| int8 latent (shipped, #3155) | behind 4096 | 0 / 105 | 3 / 256 | 0 | 0 | 0.109 | 12.9589 / 12.8767 | -0.634% |
| int4 affine (#3155, not shipped) | behind 1024 | 0 / 209 | 45 / 512 | | | 0.859 | 14.6613 / 14.6607 | -0.004% |
| int4 affine (#3155, not shipped) | behind 4096 | 0 / 105 | 14 / 256 | | | 0.859 | 12.9589 / 13.2174 | +1.995% |

The int4 affine rows are quoted from `mla-quantized-latent-m3ultra-2026-11-19.md`, which did not record the per-gap counts.

The gate is zero decided disagreements and every top-1 disagreement at a near-tie, meaning the reference's top two within 0.11 logits.

- **Turbo-4 fails.** No decided disagreement, but 61 top-1 disagreements across the two shapes, 31 of them outside the 0.11 bar and 6 above 0.5 (largest 0.719). The reference's choice landed as low as rank 6. Logit delta on the reference's choice: p50 0.19 to 0.20, p99 1.03 to 1.41, max 2.66. The Lloyd-Max codebook on a whitened row is not measurably better than per-group affine int4 on the raw latent (45 / 512 for both behind 1024).
- **WHT + affine int4 fails.** 53 disagreements, 28 outside the bar, largest 1.125, and perplexity +1.7% behind 4096. Whitening does not fix 4-bit affine on this latent. The error is the 4-bit step itself, not outlier channels.

Every traced decode step of both prototypes is `absorbed_composed` (13824 behind 1024, 6912 behind 4096), with `absorbed_fused=0` and `absorbed_split_kv=0`. The fp16 reference rows are `absorbed_fused`.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

| prompt | fp16 latent | int8 latent | Turbo-4 latent | Turbo-4 vs int8 | WHT + affine int4 | vs int8 |
|---|---|---|---|---|---|---|
| 512 (3 rounds) | 170.08 | 143.58 | 96.86 | -32.5% | 126.07 | -12.2% |
| 2048 (3 rounds) | 151.41 | 129.82 | 89.94 | -30.7% | 117.49 | -9.5% |
| 16384 (2 rounds) | 109.59 | 71.20 | 57.47 | -19.3% | 72.60 | +2.0% |

Round-to-round spread is under 2% for every cell except one Turbo-4 round at 2048 (68.36, taken while other units were building; the median is unaffected). The fp16 latent runs the fused kernel at 512 and 2048 and the composed path at 16384. Every other arm runs the composed path at every length. Prefill is the same for all arms (1367 to 1386 tok/s at 512 and 2048, 1119 to 1127 at 16384). The MLX peak at 16384 is 11.45 GB fp16, 11.23 int8, 11.87 Turbo-4 and 11.18 WHT + int4.

Turbo-4 is slower than int8 because the codebook is not affine, so `quantized_matmul` cannot read the packed window: each step unpacks it and gathers `[B, 1, S, 512]` f32 codebook values for both contractions. A fused gather kernel could close part of that gap. It cannot close the quality gap, which is why it is out of scope here. WHT + affine int4 matches int8 at 16384 only because it reads a 4-bit window. It pays for the two rotations at shorter lengths.

## Cache size

| layout | bytes per token per layer | 27 layers at 16384 tokens | passes the gate |
|---|---|---|---|
| fp16 latent | 1152 | 0.51 GB | reference |
| int8 latent (shipped) | 672 | 0.30 GB | yes |
| int4 affine latent (#3155, not shipped) | 416 | 0.18 GB | no |
| WHT + affine int4 latent (prototype) | 416 | 0.18 GB | no |
| Turbo-4 latent (prototype) | 386 | 0.17 GB | no |

The 386 is `rank / 2` packed indices, one fp16 rescale and the fp16 rope stream (`turbo4_latent_bytes_per_token`).

## Decision

The adopt criterion was to pass the #3155 quality bar (zero decided disagreements and every top-1 flip at a gap of at most 0.11) and then either be smaller than the int8 latent at that quality or be faster than it at 16384. Turbo-4 is 43% smaller than int8 but fails the bar at both window lengths, with flips up to 0.719 logits. It is also 19% to 33% slower than int8. A 4-bit codec that passed where int4 affine failed would have been adopted on size alone. Neither 4-bit arm passed.

So nothing is wired:

- `latent_layout_supports_mode` still accepts only fp16 and int8 for the latent. Every Turbo mode on absorbed `deepseek_v2` resolves to the fp16 latent with the existing warning (`cache::turbo::allowlist`), which already says to use `--kv-cache-mode int8` for a quantized latent. That wording was checked and is still correct.
- The prototype stays test-only in `src/lib/mlxcel-core/src/mla/turbo_latent.rs` (`#[cfg(test)]`), with its parity tests in `turbo_latent_tests.rs`. It adds no mode, flag, environment variable or user-visible message.
- Turbo-3 was out of scope unless Turbo-4 was adopted. It stores fewer bits, so it would not pass where Turbo-4 does not.

## Not measured

- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one. The decision rests on the trace, which is a codec property and does not depend on the backend.
- **Prefill transient at 16384** beyond the MLX peak above, and the `inspect`/prompt-cache sizing rows. Those exist only for an adopted layout.

## Commands

```
M=models/deepseek-v2-lite-chat-4bit-mlx
# measurement build: apply measurement-hook.patch, register the two modules, then
cargo build --release --bin mlxcel-bench-decode --example logit_trace
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > fp16-1k.tsv          # 64 4 8 4096 for 4k
MLXCEL_MLA_LATENT_PROTOTYPE=turbo4 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > turbo4-1k.tsv        # wht-int4 for the second arm
python3 scripts/compare_logit_traces.py fp16-1k.tsv turbo4-1k.tsv --decided 2
MLXCEL_MLA_LATENT_PROTOTYPE=turbo4 scripts/with_indexers_paused.sh \
  mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # 2048, 16384; --kv-cache-mode int8 for int8
```

Rows: [decode.csv](mla-turbo-latent-m3ultra-2026-11-19/decode.csv) and [logit_trace.csv](mla-turbo-latent-m3ultra-2026-11-19/logit_trace.csv).
