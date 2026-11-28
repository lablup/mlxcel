# Technical Report: PR #3189 - Experimental `--mla-latent-quant` option for MLA latent precision

**Date**: 2026-11-21
**Status**: Completed
**Risk Level**: Medium (opt-in only; `auto` is the default and resolves exactly as before, but an explicit value stores latents the quality gate did not ship, including a new 4-bit latent and a split DSA key slot)

## Executive Summary

`--mla-latent-quant <auto|fp16|int8|int4>` (env `MLXCEL_MLA_LATENT_QUANT`, the CLI flag wins) picks the precision every MLA latent cache of the loaded model is stored at (#3179, follow-up to epic #3158). `auto` reproduces the default gate. `fp16` keeps the latent fp16 under any KV mode, and `int8` and `int4` store the affine group-64 latent on every MLA family, the gated-off GLM-4.7-Flash and DSA families included. On M3 Ultra a 4-bit latent has 0 decided disagreements on Moonlight, LongCat-Flash-Lite, Kimi-Linear-48B and DeepSeek-V2-Lite but misses the 0.11-logit near-tie bar on all four (flips up to 0.84, 1.13, 0.63 and 0.86 logits), and on GLM-4.7-Flash it has 3 decided disagreements behind both 1024 and 4096 tokens with perplexity moving -19.5% and -16.1%. So `int4` stays experimental everywhere. The int8 arms reproduce the epic's numbers exactly.

## 1. Problem Statement

Epic #3158 shipped the affine int8 MLA latent only on the families that passed its quality gate. GLM-4.7-Flash (1 decided disagreement) and the DSA families (indexer top-k overlap below 0.99) stayed fp16, and the 4-bit latent measured in #3155 was not offered at all. An operator who accepts a measured risk had no way to choose those precisions without a custom build.

## 2. What Changed

- `mla/latent_option.rs` (new): the option type, its parsing, process state (`init_mla_latent_quant`, `mla_latent_quant`, a thread-local override for tests and harnesses), and `effective_latent_bits`, the single point where a family, a requested `KVCacheMode` and the option become a latent width. The startup notices and the table of measured results (`EXPERIMENTAL_LATENT_NOTES`, one width per entry) live here.
- Mode resolution (`cache/turbo/allowlist.rs`): an explicit value on a model that holds an MLA latent decides the storage on its own, `Int8` storage when a width is returned and fp16 otherwise. No new `KVCacheMode` variant exists; the width inside `Int8` storage is `latent_cache_bits` (8, or 4 under `int4`). `option_decides_latent` is the one statement of when the option decides a model, used by the resolver, the notices, the DSA key-slot refusal and the `--kv-bits` table, with one case-insensitive rule.
- 4-bit latent (`mla/cache.rs`, `mla/quant_latent.rs`, `mla/latent_keys.rs`): the same affine codec at 4 bits, always decoded on the composed `quantized_matmul` path (the fused Metal kernel reads 8-bit codes only). On the DSA families the indexer key stays 8-bit beside a 4-bit latent: the key slot is two runs of whole groups in the same buffers, and the boundary is derived from buffer shapes.
- Sizing and reporting: `kv_arch` sizes at the stored width (416 B per token per layer at rank 512, 552 with the DSA indexer key), and the memory estimate's `KV dtype` line names a narrower latent width and the flag that chose it.
- Entry points: `generate`, `run`, `serve`, `inspect`, `mlxcel-server` and `mlxcel-bench-decode` take the flag; `tune`, `predict`, `mlxcel-bench-engine`, `mlxcel-engine-parity` and `speculative_bench` validate the environment value at startup. A bad or non-UTF-8 value is a startup error.
- Server: the prompt-cache adoption check compares the stored latent width of each detached layer against the width this process packs, so an entry saved at another precision is declined (`kv_mode_mismatch`).
- Docs: `docs/turbo-kv-cache.md`, `docs/environment-variables.md`, `docs/benchmark_results/mla-latent-quant-option-m3ultra-2026-11-20.md`.

## 3. Technical Decisions

**Derive the latent width from buffer shapes.** A cache that already holds rows reports their width from its packed and scale buffers, and only a fresh cache reads the option. The width then follows the rows through trim, detach, adoption and prompt-cache entries with no extra field that could drift.

**`--kv-bits` steps aside on a model the option decides.** Every KV cache of an MLA family is a latent cache (the KDA layers of Kimi Linear and Kimi K3 hold recurrent state), so the per-layer table has nothing left to decide there, last layer included.

**No disagreement warning when the requested mode is fp16.** fp16 is the default request, so a warning there would fire on every explicit use. The disagreement line is printed only when a quantized mode implies a different latent precision; the experimental notice still prints for unlisted families and for every `int4`.

**A forced latent on a rank that is not a multiple of 64 keeps fp16 at startup.** The whole-groups check that guarded only the DSA slot now covers every latent-pair family and an absorbing DeepSeek V2 under an explicit value, so such a config falls back with a warning naming the flag instead of reaching the per-token (or decompressed) fallback at the first cache write.

## 4. Measurements

M3 Ultra, Metal, teacher-forced against the fp16 latent, `--decided 2`, 8 chunks behind 1024 and 4 behind 4096.

| model | int4 decided (1024 / 4096) | largest flip gap | ppl delta (1024 / 4096) |
|---|---|---|---|
| Moonlight-16B-A3B | 0 / 0 | 0.844 | +2.32% / +1.21% |
| LongCat-Flash-Lite | 0 / 0 | 1.125 | -2.15% / -3.90% |
| Kimi-Linear-48B-A3B | 0 / 0 | 0.625 | -0.46% / +0.23% |
| DeepSeek-V2-Lite | 0 / 0 | 0.859 | -0.004% / +2.00% |
| GLM-4.7-Flash | 3 / 3 | 4.000 | -19.52% / -16.09% |

The int8 arms through the option reproduce every count, gap and perplexity of the epic's documents, GLM-4.7-Flash's failing 1 / 193 included.

LongCat-Flash-Lite decode, median tok/s fp16 / int8 / int4: 124.7 / 135.5 / 124.3 at 512 tokens, 125.1 / 106.9 / 110.0 at 2048, 99.3 / 50.2 / 54.5 at 16384. Moonlight: 156.0 / 140.5 / 132.0 at 512, 139.8 / 121.3 / 123.6 at 2048. Across both models, int4 is 2% to 9% faster than int8 where both run composed, and 6% to 8% slower behind 512 tokens, where int8 has the fused kernel.

`mlxcel inspect --max-tokens 16384`: Moonlight 486.0 MiB fp16, 283.5 MiB int8, 175.5 MiB int4; GLM-4.7-Flash 846.0, 493.5, 305.5 MiB; DeepSeek-V3.2 1.31 GiB, 770.1 MiB, 526.1 MiB (552 B per token per layer: 416 for the latent, 136 for the int8 indexer key).

## 5. Not Measured

- DeepSeek-V3.2 under the option: no teacher-forced trace and no indexer top-k overlap run for the forced int8 slot or the int4 split slot. The split slot rests on tiny-model tests, a coherent 200-token generate run and a 4096-token sparse decode run; its quality on the real checkpoint is unmeasured.
- GLM-5 (`glm_moe_dsa`), LongCat-Flash 560B (`longcat_flash`) and Kimi K3: not on this host or not run. They share the measured code paths and the tiny-model tests.
- CUDA and ROCm: no trace or throughput row. The int4 path is the composed `quantized_matmul` decode those backends already run for int8.
- Decode throughput under int4 on GLM-4.7-Flash, Kimi Linear and DeepSeek-V2-Lite, and prompt-cache adoption on a real served checkpoint (covered by a unit test only).

## 6. Learning Points

**A measured note must be keyed on the width it was measured at.** The GLM int8 note was keyed on both widths, so an int4 run printed an int8 result as if it described int4. Each entry in the table now carries exactly one width, and a test enforces it.

**A rule written three times drifts.** The "explicit value decides this model" rule had three copies, one comparing `deepseek_v2` case-sensitively while the helper it stood beside ignored case. It is now one function with one case rule.

## 7. Follow-up Actions

- Measure the DeepSeek-V3.2 int4 split slot (trace and top-k overlap) on the real checkpoint before anyone relies on it.
- Extending the fused Metal kernel to 4-bit codes would remove int4's loss behind short windows.
- `packed_latent_bits` maps any packed width between the 4-bit and 8-bit sizes to 4, which holds for the two widths a cache can store and needs revisiting if another is added.
