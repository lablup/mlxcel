# PR #3278: Validate Inkling omni on real checkpoints, fix NVFP4 and MTP

**Date**: 2026-11-23
**Status**: Draft PR open (closes #1549). Validated on an Apple M3 Ultra (512 GB, macOS 27.2.1) against the real checkpoints.

## Executive Summary

Inkling (`inkling_mm_model`) shipped with synthetic-fixture parity only: `docs/supported-models.md` said the 153.5 GB affine target, the 170.7 GB native NVFP4 export and the 4.5 GB MTP shard had never been run. This PR runs all of them on an M3 Ultra and records the evidence for each item in #1549.

The run found two defects. The native NVFP4 checkpoint decoded every greedy token to id 0, because its quantization lives only in `hf_quant_config.json` and the loader cast its bf16 dense weights to f16. Inkling MTP was enabled with no exactness check, and at draft widths 2 and 3 it emitted greedy streams that differed from classic decode. Both are fixed with regression tests. Text, image, audio and video are correct on both the 4-bit and NVFP4 checkpoints, and CLI and server outputs are byte-identical. MTP is now exact under the gate, but its acceptance is low (about 1.6 tokens per round, matching mlx-vlm's drafter), so it gives no speedup on Inkling-Small. The criterion requiring a positive throughput gain is not met.

## 1. Checkpoints and setup

| Checkpoint | Local path | Notes |
|---|---|---|
| `inference-optimization/Inkling-0.6B-A0.6B` | `models/inkling-0.6b-a0.6b` | HF config dialect, f32 |
| `mlx-community/Inkling-Small-mlx-4bit` | `models/inkling-small-mlx-4bit` | affine 4-bit, `quantization` in config.json |
| `thinkingmachines/Inkling-Small-NVFP4` | `models/inkling-small-nvfp4` | ModelOpt NVFP4 experts, bf16 dense, `hf_quant_config.json`, carries `mtp.safetensors` |

Every GPU run held `/usr/bin/lockf` on the shared GPU lock for its own duration only, one large checkpoint at a time. Peak memory comes from `/usr/bin/time -l` (peak memory footprint).

References: HF `transformers` 5.24.0 `InklingForConditionalGeneration` on CPU in float32 for the 0.6B model, because mlx-vlm cannot read its HF-dialect config (`dense_intermediate_size` is absent). For the Small checkpoints the reference is the vendored mlx-vlm at `references/mlx-vlm` (d998a620), including `mlx_vlm.split_mtp` and its `inkling_mtp` drafter.

Media: `tests/fixtures/sam3/cats.png` (COCO image 39769); a five-second 16 kHz mono WAV produced with macOS `say`, padded to exactly 80000 samples; a 4 s, 10 fps, 320x240 clip of a 60x60 red square moving left to right.

## 2. Defect 1: NVFP4 decoded only id 0

### 2.1 Symptom

Every prompt on `Inkling-Small-NVFP4` (text, image, audio, video) produced a run of `!` (token id 0). Audio and video metadata were still right (100 audio tokens, the same frame pairs), so the failure was in the language model, not the frontends. mlx-vlm on the same directory produced correct text.

### 2.2 Root cause

On Apple Silicon, `bf16_to_f16_at_load(is_quantized, config)` casts bf16 weights to f16 unless the checkpoint is quantized. The VLM loader decides "quantized" by looking for a `quantization` block in config.json. The ModelOpt NVFP4 export has none: its quantization is declared only in `hf_quant_config.json`. Its bf16 dense weights (attention, norms, shared projections, embeddings) were therefore cast to f16, which overflows in Inkling's residual stream. The 4-bit affine checkpoint escaped because its config carries `quantization`.

Two experiments isolated it. `MLXCEL_INKLING_MOE_KERNEL=0` (no fused expert kernel) was still broken, which ruled out the NVFP4 expert path. `MLXCEL_KEEP_BF16=1` alone fixed the output.

The expert sidecar mapping was checked separately and is correct: for each expert, `gate_scale = scale` (E4M3 block scale) and `out_scale = scale * scale2`, the same as mlx-vlm's loader.

### 2.3 Fix

`bf16_to_f16_at_load` returns false for any Inkling config (`is_inkling_config`: `model_type` `inkling`, `inkling_mm_model`, or a `text_config.model_type` of `inkling_text`; `inkling2` is excluded). This covers the text and VLM loaders in one place. Exempting the family rather than teaching `is_model_quantized` about the sidecar was chosen because no Inkling checkpoint benefits from the cast: affine conversions already stay bf16, and the 0.6B checkpoint is f32.

### 2.4 Result after the fix

| Prompt | NVFP4 answer |
|---|---|
| Text, 128 tokens | Coherent reasoning and a correct palindrome function. Matches mlx-vlm's NVFP4 output for the first 354 characters, where a near-tie reorders two words |
| Image (cats) | Two tabby cats (kitten left, adult right), pink blanket or couch, remote controls |
| Audio (5 s WAV) | Word-exact transcript, 100 audio tokens |
| Video (red square) | "moves to the right"; 8 frames, 4 pairs, same timestamps as 4-bit |

Resident after load 154.7 GB, peak footprint 273 to 274 GB, decode about 30.7 tok/s on text.

## 3. Defect 2: Inkling MTP skipped the exactness gate

### 3.1 Symptom

`mtp_capable_target` returned true for `Inkling` and `InklingVLM` with no check. With the 4-bit target and the native MTP head, a 128-token greedy run at `--draft-block-size 2` diverged from classic decode partway through (one inserted word), and width 3 diverged too. The default width 10 (adaptive 3 to 10) happened to match.

### 3.2 Cause

Verify forwards with M >= 2 rows take the Metal `qmv_wide` kernel, while single-token decode takes `qmv`. Their results are not bit-identical on Apple GPU generation 15 and later. With `MLXCEL_QMV_WIDE=0`, widths 2 and 3 matched classic decode exactly. The other MTP families already guard against this with the measured probe in `src/models/speculative_exactness.rs`; Inkling had been wired around it.

### 3.3 Fix

- `InklingModel::probe_block_chain_exactness(block)` builds a chain arm (single-token forwards) and a block arm (one `T = block` forward) on fresh internal caches over three short draws and compares the raw logit bytes per position with `compare_block_against_chain`.
- `InklingModel::mtp_exactness_allows(block)` feeds that probe to `mtp_exactness_gate`, which memoizes per process, retries with `qmv_wide` disabled, and honours `MLXCEL_MTP_ALLOW_INEXACT=1`.
- `mtp_capable_target` (server) and `run_offline_mtp` (`mlxcel generate --draft-kind mtp`) call it. The offline path returns an error that names the override.
- The Inkling MTP drafter no longer applies the shared drafter bf16 to f16 policy. With the target in bf16, an f16 drafter put a dtype boundary at every shared embedding, norm and LM-head call, and MLX promoted that mix to float32. 128 tokens at width 2 went from 11.6 s to 5.1 s at the same acceptance.

On this host the probe reports that even the `qmv_wide`-disabled block forward differs from the chain at position 0 (about 47% of logit bytes), because the short convolutions and attention also take different kernels at `T = 1`. The gate therefore declines Inkling MTP here unless the override is set. This is the intended conservative result: the gate refuses a speculative path it cannot prove exact.

### 3.4 MTP measurements (4-bit target, NVFP4 directory as the drafter, 128 greedy tokens)

| Run | Equal to classic | Mean accepted length | Throughput |
|---|---|---|---|
| Classic decode | n/a | n/a | 41.6 tok/s |
| Gate on, any width | MTP declined | n/a | n/a |
| Override, width 10 (adaptive 3..10) | yes | 1.608 | 10.8 tok/s |
| Override, width 2 | no | 1.494 | 26.6 tok/s |
| Override, width 3 | no | 1.587 | 24.2 tok/s |
| Override, width 2, `MLXCEL_QMV_WIDE=0` | yes | 1.512 | 26.4 tok/s |
| Override, width 3, `MLXCEL_QMV_WIDE=0` | yes | 1.608 | 24.1 tok/s |

Upstream mlx-vlm with the same target and its split drafter accepts 1.59 tokens per round (6.7% of drafted tokens), so the low acceptance is a property of this head and target pair, not of the port. A patched chain-index variant (using block 1 after the seed) did not improve it (6.9%). At about 1.6 tokens per round, a verify forward plus the drafter costs more than the tokens it saves, so no width beats classic decode.

## 4. Validation results by #1549 item

| Item | Evidence |
|---|---|
| 4-bit text, CLI and `/v1/chat/completions`, chat template, 64 tokens | Byte-identical (prompt 25 tokens, completion 64, finish `length`). Default effort is all reasoning; with `reasoning_effort: "none"` the content is a correct Rayleigh-scattering answer. 128-token output equals mlx-vlm exactly. Peak about 154.5 GB |
| 0.6B parity, RAW, greedy, 64 tokens, `MLXCEL_FUSED_MOE=0` | 9 of 9 prompts match HF token for token up to the reference's first `<|endoftext|>` (token counts 9, 22, 52, 45, 21, 64, 45, 53, 48). The 64-token prompt runs without EOS. Peak 3.6 GB. Output text is not reproduced because the model regurgitates memorized material |
| Grounded still image, CLI and server | Two tabby cats, pink blanket, remote controls, identical on both paths; 246 prompt tokens. Peak 161.5 GB |
| 5 s 16 kHz WAV, CLI and server | `audio_tokens=100` = ceil(80000 / 800); transcript word-exact on both paths, WER 0. Peak 155.0 GB |
| Synthetic left-to-right video | 8 sampled frames, pairs [0,1], [2,3], [4,5], [6,7], anchors 0.0, 1.1, 2.2, 3.3 s, 216 video tokens, 319 prompt tokens; answer "to the right", identical on CLI and server. Peak 157 to 158 GB |
| MTP: native shard, 128-token equality, acceptance, throughput | Loaded from `mtp.safetensors`. Equal under the gate's conditions; mean accepted length 1.49 to 1.61; **no throughput gain (not met)**, see 3.4 |
| NVFP4 quality and expert sidecar scales | Fixed (section 2); sidecar mapping identical to mlx-vlm |

## 5. Tests

| Test | What it pins |
|---|---|
| `models::sanitize::tests::inkling_configs_keep_checkpoint_dtype` | Inkling configs without an MLX quantization block keep bf16; `inkling2` is not matched |
| `drafter::inkling_mtp::tests::drafter_load_keeps_bf16_checkpoint_dtype` (mlxcel-core) | The MTP head loads as bf16 |
| `server::batch::speculative_burst_tests::mtp_capable_target_gates_inkling_on_the_exactness_verdict` | `mtp_capable_target` follows the seeded gate verdict for Inkling |
| `models::inkling::tiny_tests::exactness_probe_runs_on_the_tiny_model` | The probe runs at width 3 and returns NotRun at width 1 |

Narrow selectors for `models::inkling`, `models::speculative_exactness`, the sanitize and speculative-burst tests and `drafter::inkling_mtp` pass, as do `coverage_tests`, clippy on lib and tests with `-D warnings`, `cargo fmt --check` and the repository contract checks.

## 6. Limitations and follow-ups

- MTP gives no speedup on Inkling-Small; the gate keeps it off by default on M3 Ultra. Improving acceptance is a model or drafter question, not a port defect.
- Whether a `T > 1` Inkling forward can be made byte-identical to single-token decode (convolution tail and attention kernels at `T = 1`) was not pursued here.
- Validation used one host. Smaller-memory Macs cannot hold the Small checkpoints.
