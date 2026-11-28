# PR #2846: IndicOCR layout plus Qwen3.5 OCR pipeline

**Date**: 2026-11-03
**Status**: Completed (merged; closes #2414)

## Executive Summary

PR #2846 adds the two-stage IndicOCR page pipeline. `mlxcel generate -m <indic-ocr-root> --image page.png` runs IndicDocLayout to find regions and reading order, then IndicBlockOCR (Qwen3.5-0.8B) to transcribe each region with a class instruction, and prints reading-ordered Markdown. The OCR stage also loads directly as a dense Qwen3.5 VLM, so a single crop can be transcribed with `-m weights/ocr` and a prompt.

Three pieces were needed beyond the command itself. The PP-DocLayoutV3 predictor gained an `MlxVlmDetect` decode mode, because IndicDocLayout was trained against mlx-vlm's `detect()` semantics rather than Hugging Face post-processing. The pure routing, cleanup, nested-equation, crop and reconstruction rules moved into `src/vision/indic_ocr/`. Two latent config bugs surfaced on the real checkpoint: duplicate serde aliases in the layout config, and a hardcoded Qwen3.5 EOS pair that never matched IndicOCR's 262146. 27 files, +2084/-34, of which about 900 lines are tests and docs.

## 1. Change Summary

| Area | Change |
|---|---|
| `src/vision/indic_ocr/` (new) | Label routing for both taxonomies, `clean_layout`, nested-equation resolution, area-clamped Lanczos crops, verbatim prompts, Markdown de-hyphenation and math repair |
| `pp_doclayout_v3` | `PpDocLayoutDecode::{HuggingFace, MlxVlmDetect}`, `decode_mlx_vlm_detections`, a square-input relaxation, and optional `out_indices`/`return_idx` and `feat_strides`/`feature_strides` resolution |
| `src/models/detection.rs` | `indic_ocr` routes OCR-stage configs to `Qwen35VLM`; two-stage roots point at the page pipeline |
| `src/models/qwen3_5.rs` | EOS ids now come from `eos_token_id` with the stock pair as fallback |
| `src/commands/generate_indic_ocr.rs` (new) | Stage resolution, per-block OCR through the OCR stage chat template, `--output-json`, `--table-format`, `--layout-detections` replay |
| `src/commands/generate.rs` | Route two-stage roots before model load, including promptless runs; `run_generation_mode` takes an explicit token budget |
| Docs | `docs/supported-models.md` IndicOCR section |

## 2. Technical Decisions

### 2.1 Route the root before any model load

A two-stage IndicOCR root is not a loadable VLM: it has `stages` but no top-level `text_config`. Loading it as Qwen3.5 would fail at tokenizer time with a confusing "no tokenizer" error. `run_generate` therefore checks `model_type: indic_ocr` plus `stages` right after resolving `-m` and routes to the page driver. The same check runs in the promptless branch, because the acceptance command has no `-p` and would otherwise enter the interactive REPL. `get_model_type` returns a named error pointing at the page pipeline for other callers such as `arch`.

### 2.2 Keep the layout decode mode with the predictor

`PpDocLayoutDecode` is a predictor-level choice, not a CLI flag: `mlxcel detect` keeps Hugging Face post-processing and IndicOCR constructs the predictor with `MlxVlmDetect`. The mode changes preprocessing (1024 square versus the processor's 800), filtering (one argmax per query versus flat top-k), and order decoding (kept sub-block versus all queries). Putting it behind one enum keeps those three differences together and keeps the CLI contract unchanged.

### 2.3 The sub-block order decode is deliberate

Hugging Face post-processing decodes the GlobalPointer order over all queries, then filters. mlx-vlm's `detect()` filters first and decodes the retained submatrix. The two can disagree, and the IndicOCR fine-tune was trained on the second. The new unit test pins this by constructing a filtered query whose votes would invert the surviving two if the full matrix were used.

### 2.4 Pure logic separate from model I/O

`src/vision/indic_ocr/` holds every rule that decides what is transcribed and how text is assembled: label routing, containment dedup, marginalia handling, nested equations, crop area clamping, prompts, de-hyphenation, and math repair. That is why 19 tests run without weights. The command file owns stage paths, the detector, the VLM, generation, and JSON output.

### 2.5 Config fields that appear twice

IndicDocLayout's `config.json` carries both `out_indices` and `return_idx`, and both `feat_strides` and `feature_strides`. A serde `alias` treats the second spelling as the same field and rejects the document as a duplicate. They are now separate optional fields with explicit resolution (`out_indices` and `feat_strides` win), and the graph reads the resolved values.

### 2.6 EOS ids from config

`Qwen35Model::eos_token_ids` returned a hardcoded `[248046, 248044]`. IndicOCR's OCR stage declares 262146, so generation never stopped and repeated the answer and `<|im_end|>` until the token cap. The method now reads `eos_token_id` (number or array) and falls back to the stock pair only when it is absent or unusable. The default context for every other Qwen3.5 checkpoint is unchanged because their configs declare the same pair.

## 3. Verification

- 19 `indic_ocr` unit tests: label routing for both taxonomies, prompt selection, transcription skipping, `clean_layout`, header/footer rules, nested equations under each mode, densify, crop clamping and padding, `area_clamp`, Markdown reconstruction.
- 4 `decode_mlx_vlm_detections` tests: per-query argmax, strict confidence filtering, order over the kept sub-block, malformed inputs.
- 87 detection-routing tests, 33 Qwen3.5 tests, 35 PP-DocLayoutV3 tests, and 3 command tests pass.
- Gated real test against `HashNuke/indic-ocr-mlx weights/layout`: loads the fine-tune, decodes with `MlxVlmDetect`, and asserts dense reading order and page-bounded boxes.
- End-to-end on a rendered Hindi page: six blocks in reading order, the header kept with empty text, paragraphs and the equation transcribed correctly, and JSON written with zero-based order and pixel boxes.
- `--layout-detections` replay of `mlxcel detect --format json` reproduces the same six blocks and Markdown after the same cleanup; `--table-format csv` fails before the model loads.
- Single-crop `-m weights/ocr` transcribes the page correctly (56 tokens, greedy).
- `cargo clippy --lib --tests -- -D warnings` and `cargo fmt --check` pass locally. CI's `cargo-clippy` and `CPU-only link` jobs fail in the mlxcel-core CMake build (`Could NOT find BLAS`), unrelated to this diff.

## 4. Unverified

The committed mlx-vlm layout-parity fixture and the first-three-block text comparison against mlx-vlm's `parse()` were not produced: this host has no Python mlx-vlm installation. The pipeline was validated end to end against the real layout and OCR checkpoints and against a rendered page, but not numerically against mlx-vlm. The stock PP-DocLayoutV3 label table is covered by unit tests, not by a replay of a real stock detection file.
