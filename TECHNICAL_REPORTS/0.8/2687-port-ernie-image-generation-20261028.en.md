# PR #2687: Port ERNIE-Image text-to-image and SDEdit

**Date**: 2026-10-28
**Status**: Complete on CUDA (GB10); Metal and ROCm runs not recorded
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Medium (it changes the output of live Ministral 3 text models)

## Executive Summary

PR #2687 (issue #2345, epic #2348) ports mlx-vlm `models/ernie_image` (git 73d366ec) as the `ernie_image` image-generation family: text-to-image and SDEdit img2img for `baidu/ERNIE-Image-Turbo` (8 steps, guidance 1, no CFG) and `baidu/ERNIE-Image` (50 steps, guidance 4, CFG). It decodes through the shared FLUX.2 VAE, loads the raw Diffusers tree, and is registered in `imagegen::FAMILIES` and the model registry, so `mlxcel generate-image`, `mlxcel-server --image-model` (both image routes) and `mlxcel arch --json` see it. The text conditioner and the optional `pe/` prompt enhancer run the in-tree Ministral 3 decoder through a new `Ministral3Model::forward_hidden_states` entry point instead of a copy.

The PR also fixes a bug in shared code: every Ministral 3 checkpoint declares YaRN rope scaling, and the in-tree model ignored it. The fix changes generation output for Ministral 3, the Mistral 3 VLM text tower and Nemotron-3-Embed. Against upstream, the Turbo case is bit-exact (57.00 dB), the Turbo edit and the base 8-step control pass the epic's 30 dB criterion, and the base 50-step case reaches 24.24 dB. The user accepted that case as an exception to the epic criterion.

## 1. Problem Statement

After #2341 the image runtime, FLUX.2 VAE and latent packing existed, but an ERNIE-Image checkpoint had no family and failed at load. The missing pieces were the 36-layer single-stream DiT (hidden 4096, FFN 12288, 32 heads, AdaLN modulation, 3-axis RoPE with a key mask over text padding), a conditioner that runs all but the last Ministral 3 layer and skips the final norm, a static-shift (4.0) flow-match scheduler with an f32 Euler step, SDEdit with `image_strength` in (0, 1] (default 0.6), and the prompt enhancer whose rewritten text upstream returns as `revised_prompt`.

`src/models/ministral3.rs` had no output that stops before the last layer, so it could not serve as the conditioner. Building the port then exposed a second gap: `Attention` rotated with plain RoPE at `rope_theta` and ignored the checkpoint's `rope_parameters.rope_type: yarn`, while mlx-lm and upstream's text encoder apply `YarnRoPE`.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/ernie_image/` (`config`, `blocks`, `transformer`, `text_encoder`, `prompt_enhancer`, `scheduler`, `pipeline`, `weights`, `mod`) implementing `ImageGenerationModel` for Generate and Edit. Request errors are `InvalidImageRequest` (400). |
| DiT blocks | Local `blocks.rs`, not the Z-Image `dit_blocks`: ERNIE repeats each rotary angle on adjacent entries but rotates with half-split `rotate_half`, and runs modulation and gates as f32 islands cast back to the stream dtype. |
| Conditioner | `text_encoder.rs` loads only the `language_model.*` tower of the `Mistral3Model` checkpoint and calls `forward_hidden_states(num_layers = layers - 1, final_norm = false)`. Prompts truncate at `MAX_TEXT_TOKENS` = 2048. |
| Prompt enhancer | `prompt_enhancer.rs`: JSON user turn `{"prompt", "width", "height"}` through `pe_tokenizer`'s chat template, temperature 0.6, top-p 0.95, at most 2048 new tokens, seeded per image. Loaded lazily (about 7 GB) and kept resident. |
| Scheduler | Local `scheduler.rs`: `linspace(1, 0, n + 1)[:-1]` under static shift 4.0. The shared `FlowMatchEulerDiscrete::static_shift` builds `linspace(1, 1/n, n)`, which differs in the last f32 bit at 50 steps, and steps in the sample dtype. |
| Ministral 3 | `forward_hidden_states(input_ids, caches, num_layers, final_norm)` with layer and cache count checks; `forward_with_caches` is now that body followed by the LM head. YaRN frequencies and mscale in `forward` and `forward_batched`, via the existing `gpt_oss_common::yarn_rope_freqs` (default `truncate` true for Ministral 3). |
| Registration and docs | `imagegen::FAMILIES`, registry row, `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`, `docs/embeddings.md` (Nemotron-3-Embed note); test-only `MLXCEL_ERNIE_IMAGE_*` variables allowlisted in `scripts/ci/check_env_var_docs.py`. |
| Fixtures | `tests/fixtures/imagegen/ernie_image_parity.safetensors` (noise checksum and head, 8x8-pooled latents and image) and the generator `tests/fixtures/generate_ernie_image_fixtures.py`. |

Statistics at the pre-merge head `154f618a`: 41 files, +4241 / -30. Main commits: `ea90089f` (hidden-state entry point), `5072fc06` (YaRN in `forward`), `a8a70102` (port), `a7352a95` (YaRN in batched decode), `d25471d8` (docs), `cf1ed59f` (8-step CFG control), `148cef1d` (Nemotron-3-Embed note), `eda88684` (enhancer input cap), `7f1d8ef6` (`// Used by:` comments), plus merges of `origin/main`.

## 3. Technical Decisions

### 3.1 Parity gate and the accepted base exception (user decision, epic #2348)

The issue first asked for final-latents relative RMS below 1e-2. During epic #2348 the user replaced that with an epic-wide criterion: per-case latent bounds that reject a wrong port, PNG PSNR >= 30 dB against the CUDA oracle, and bit-identical initial noise. Measured rel-RMS is recorded as drift and does not gate the merge.

Oracle: mlx-vlm 0.7.8 (git 73d366ec) on mlx 0.32.4, CUDA backend, NVIDIA GB10, `MLX_ENABLE_TF32=0` on both sides, 1024x1024, seed 42, prompt "A red panda serving tea in a moonlit bamboo forest", enhancer off.

| Case | First forward rel-RMS | Final latents rel-RMS (drift) | PNG PSNR | Test bound (latents / PSNR) |
|---|---|---|---|---|
| Turbo, 8 steps, guidance 1 | 0 | 0 | 57.00 dB | 1e-2 / 30 dB |
| Turbo edit, guidance 3, strength 0.7 | 4.90e-2 | 1.35e-1 | 32.23 dB | 2.5e-1 / 30 dB |
| Base, 50 steps, guidance 4 | 5.02e-3 | 3.03e-1 | 24.24 dB | 4.5e-1 / 20 dB |
| Base control, 8 steps, guidance 4 | 5.02e-3 | 1.64e-1 | 31.43 dB | 2.5e-1 / 30 dB |

Initial noise is bit-identical in every case and conditioning matches exactly. 57 dB is the PNG round trip of identical latents.

The base 50-step case misses the 30 dB criterion. The user accepted it as an exception at 24.24 dB under a 20 dB test bound, which issue #2345's acceptance revision records. The cause is one GEMM. With CFG the transformer runs at batch 2, and the small-M bf16 `text_proj` (`[2, 13, 3072] x [3072, 4096]`) picks a different kernel in mlxcel's pinned MLX than in the mlx 0.32.4 wheel and rounds differently in the last bit. 36 layers over 50 steps amplify that. The evidence:

- The 8-step control runs the same checkpoint, prompt, seed and CFG batch-2 forward and reaches 31.43 dB, so the batched forward is sound and the miss comes from 50-step amplification.
- Upstream moves as far against itself. Running only that projection one row at a time gives its base case 2.89e-1 / 23.70 dB and its edit case 1.39e-1 / 32.14 dB. Its batch-2 row 1 also differs from the same row at batch 1 by 5.3e-2.
- The 50-step images are compositionally identical (same panda, teacup, teapot and moon; fine detail differs).

The 20 dB bound only rejects a grossly broken pipeline and is not a wrong-port detector on its own. Wrong-port detection comes from the first-forward bound (`MAX_FIRST_FORWARD` = 1e-1) and the 30 dB gates on the other three cases. The pre-fix port, with GELU computed in f32 and the bias added after the matmul, drifted to 3.3e-1 / 24.8 dB on batch-1 Turbo (first forward 2.0e-1). It would pass the base bound, but the Turbo gate rejects it.

### 3.2 Matching MLX `nn` kernels at batch 1

Batch-1 bit-exactness needed two kernel choices that follow MLX's `nn` layers rather than the mlxcel helpers: GELU computed in the input dtype (`nn.gelu`; `mlxcel_core::gelu` computes in f32), and biased linears as one `addmm` (`nn.Linear`; `Linear::forward` adds the bias after the matmul). These two fixes moved Turbo from 24.8 dB to an exact match.

### 3.3 Precision under the #2341 rule

The transformer stays bf16 (`weights::TRANSFORMER_DTYPE`) on every backend, as upstream casts it. In the f16 A/B the Turbo case produced NaN latents and 9.54 dB. The text encoder and enhancer load through `crate::models::load_text_weights`, the text-model policy. The VAE uses `Flux2Vae::default_dtype`.

### 3.4 Reusing Ministral 3, and the YaRN behavior change

Upstream re-implements a Ministral 3 decoder for the conditioner. The port reuses `Ministral3Model` through `forward_hidden_states`, which the issue required. The refactor alone is behavior-neutral: the logit trace on `pe/` shows 0/256 disagreements and perplexity 40.1104 on both arms. Upstream's encoder omits the Llama 4 attention scale that the in-tree layer applies. That scale is exactly 1 below `original_max_position_embeddings` (16384), and prompts cap at 2048 tokens, so both compute the same values.

Sharing the decoder put the missing YaRN in the image path. The PR fixed it in the shared attention instead of adding a conditioner-only path, so live text models change too:

- **Ministral 3 text models.** Logit trace against mlx-lm at 4096 to 8704 tokens of context: the fixed arm has 0/1024 argmax disagreements, perplexity 46.5053 vs mlx-lm 46.5054. The reverted arm has 12/1024 disagreements and perplexity +0.173%.
- **Batched decode** (`a7352a95`). B1 vs B4 gives 1/128 disagreements and 0/20 on decided positions, the rounding class. `ministral3_yarn_native_isolation` fails with the batched fix reverted.
- **Mistral 3 VLM text tower.** It uses the same attention through `ministral3` in `text_config`, which matches mlx-vlm's `initialize_rope`.
- **Nemotron-3-Embed.** Both checkpoints declare YaRN (`factor` 16) next to a NeMo key `apply_yarn_scaling: false`. Hugging Face Transformers 5.20 ignores that key. Its embeddings now follow the HF reference and differ from those of earlier mlxcel versions, so an index built with an earlier version should be re-embedded (`docs/embeddings.md`). No Nemotron-3-Embed checkpoint was on the host, so this change is not measured.

`RopeParameters::yarn_freqs` follows mlx-lm `initialize_rope`: the legacy `type` key wins over `rope_type`, with mlx-lm defaults (`beta_fast` 32, `beta_slow` 1, `mscale` 1, `mscale_all_dim` 0, `truncate` true). Non-YaRN checkpoints keep the plain `fast_rope` graph.

### 3.5 Prompt enhancer rules and the 4096-token input cap

`use_prompt_enhancer` follows upstream. An explicit value wins. `true` on a checkpoint without `pe/` is 400 with upstream's message. When the option is omitted, text-to-image is enhanced whenever `pe/` exists and an edit never is, because the enhancer cannot see the source image. The enhanced text is the `revised_prompt` of each image. The field is always serialized and is null when nothing was rewritten. The CLI prints `revised prompt:`. The published checkpoints rewrite into Chinese, as upstream does.

The security review found that the enhancer prefilled the whole rendered prompt. `pe/` declares a 262144-token context, the JSON body limit admits megabytes, and the enhancer is on by default and runs once per image, so one request could force a full-context prefill per image on the 7 GB model. Commit `eda88684` caps the rendered input at `MAX_ENHANCER_INPUT_TOKENS` = 4096, twice the conditioner's 2048. A longer prompt returns 400 and the message tells the caller to pass `use_prompt_enhancer: false`, where the conditioner truncates instead. The cap rejects rather than truncates so the JSON user turn is never cut mid-string. The test `an_overlong_enhancer_input_is_the_callers_error` covers it.

## 4. Validation

- CLI Turbo, auto enhancer: 40.8 s, Chinese revised prompt, coherent image. CLI base at defaults: 99.4 s. CLI edit (strength 0.7): 30.3 s, same composition with snow added. Two source images: exit 1.
- Server generations: auto enhancer 200 in 39.8 s with `revised_prompt`; enhancer off 200 in 13.3 s, `revised_prompt` null, image equal to the parity output (57.00 dB). Server edits: 200 in 19.8 s; two images 400; `image_strength` 1.5 400.
- Memory: about 24 GB resident for encoder, transformer and VAE, plus the 7 GB enhancer once loaded.
- Tests: 36 new ERNIE unit tests (they do not compile on main, where the module and entry point are absent) and 4 ignored real-checkpoint tests (three parity tests including the 8-step control, one encoder test). Final head: `imagegen::` 257 passed, 19 ignored; `models::ministral3` 9; `models::registry` 16. The make verify targets, `check_binary_assets.py`, fmt and clippy `-D warnings` pass. The real-checkpoint numbers were measured before the review commits, which changed no numerics.

## 5. Learning Points

- When a family reuses a shared text model, the port's parity work can surface bugs in that model. Fix them in the shared code and measure each live consumer with the teacher-forced logit trace, as `docs/benchmarks.md` describes, not only the image path.
- A batch-1 bit-exact case and a short-loop control separate a sound forward from loop amplification. The 8-step control is what showed the 50-step miss is rounding growth, not a port error.
- Measure upstream against itself (row-by-row `text_proj`, B1 vs B2) before reading a PSNR miss as a defect. Upstream's 23.70 dB against itself sets the floor for this case.
- An optional generative stage that the server runs by default needs its own input bound, separate from the downstream truncation.

## 6. Follow-up and Limitations

Open MEDIUM findings from the implementation and security reviews:

- Only the presence of transformer weights is checked, not their shape against `config.json`. The matmul bridge returns a bare `UniquePtr`, so a malformed checkpoint aborts on the first request instead of returning 500. Other families follow the same pattern.
- ERNIE has no per-side size cap below the shared 16 MP ceiling (FLUX.2 caps at 2048). At 4096x4096 with CFG there are about 135k tokens per step. An OOM surfaces as 500 through `try_eval`.
- The image worker checks the request deadline only before starting, so an abandoned base-model run with `n=10` and the enhancer keeps running. This is shared worker behavior.
- With only the committed fixture, the edit case, the first-forward check and the 8-step control need `MLXCEL_ERNIE_IMAGE_ORACLE_DIR`, so Turbo is the only CI-reproducible wrong-port detector.
- The Nemotron-3-Embed YaRN change is not measured.

Lower findings left open: `random_seed(seed)` reseeds MLX's global RNG for each enhancer call (as upstream and `diffusion_worker` do); the EOS set omits `additional_special_tokens_ids` (this checkpoint ships none); the whole prompt is tokenized before truncation; `rel_rms` and `psnr` in `imagegen/testing.rs` duplicate copies in the FLUX.2 and Z-Image tests.

#2666 is the epic's open precision follow-up for the shared Qwen3 conditioner: it adds opt-in f32 attention for FLUX.2, Z-Image, Bonsai and Ideogram 4. ERNIE conditions on Ministral 3, whose conditioning already matches upstream exactly, so #2666 will not move ERNIE's numbers. The base-case gap comes from the batch-2 `text_proj` GEMM, which #2666 does not cover.

Metal and ROCm runs are not recorded in the PR.

## References

- Issue #2345, epic #2348, runtime issue #2341 (PR #2497), Z-Image PR #2662, follow-up #2666
- Upstream: Blaizzy/mlx-vlm `mlx_vlm/models/ernie_image` and PRs #1954, #2373
- `docs/image-generation.md`, `docs/embeddings.md`, `docs/benchmarks.md`
