# Technical Report: PR #3245 - Serve LFM2.5 Encoder and ColBERT as embedders

**Date**: 2026-11-23
**Status**: Completed
**Risk Level**: Low (new checkpoints only; LFM2.5-Embedding takes the same path as before, and every new layout is validated at load)

## Executive Summary

The two remaining LFM2.5 bidirectional checkpoints now load on the existing embeddings surface through `ModelType::Lfm2Embedding` (#2366, follow-up to #1325). `LiquidAI/LFM2.5-Encoder-350M` (`Lfm2BidirectionalForMaskedLM`) is a single-vector CLS embedder; before this PR it fell through to the `lfm2` text generator. LFM2.5-ColBERT, in its HF (`1_Dense/`), mlx-community (`dense.*`, bf16 / 4-bit / 8-bit) and mlx-vlm re-export (`projection.*`) layouts, is a multi-vector embedder that returns `[num_tokens, 128]` unit vectors. Before this PR the HF build was rejected, and the mlx-community builds loaded silently as single-vector CLS embedders with the projection unused. Against mlx-vlm d998a620 the encoder's hidden states match at per-token cosine 1.000000 and both full-precision ColBERT builds at minimum 0.999938. The 4-bit build stays above 0.98 against the bf16 reference, and MaxSim ranks the relevant document first on every build.

## 1. Problem Statement

#1325 served LFM2.5-Embedding and deferred ColBERT. Its guard, `reject_late_interaction`, refused a checkpoint only when a `{N}_Dense` folder or a folded `Dense.linear.*` tensor existed. The mlx-community ColBERT conversions store the projection as a top-level `dense.weight`, so they passed the guard and were served as a pooled 1024-wide vector: a wrong answer with no error. The encoder export was not recognized as an embedding architecture at all, and its keys sit under an `lfm2.` wrapper that the shared sanitize does not strip.

## 2. What Changed

- Detection (`src/models/detection.rs`): `Lfm2BidirectionalForMaskedLM` joins `EMBEDDING_ARCHITECTURES`, so `model_type: lfm2` with that architecture resolves to `Lfm2Embedding`.
- Key handling (`src/models/lfm2_embedding.rs`): `sanitize_family_weights` strips a leading `lfm2.` and then runs `sanitize_decoder_embedding_weights` as before. The MLM head is tied to `embed_tokens`, so nothing else is dropped.
- Head selection (`src/models/lfm2_embedding_head.rs`, new): `find_projection` looks for exactly one linear under `dense.0`, `dense`, `projection` or `model.projection`, and refuses two or more. `validate_projection` requires a `1_Dense/config.json` to be Identity with no residual and its widths to match the weight. It also cross-checks `mlx.proj_dim` and refuses a `modules.json` that lists a Pooling module. `projection_widths` reads the widths from the built `UnifiedLinear` and checks `bias`, `scales` and `biases` against them.
- Forward: with a projection, `embed` returns `project_and_normalize(hidden, projection, mask)` (shared with ColIdefics3 / ColQwen2.5), `multi_vector()` is true and `embedding_dim()` is the projection width. Pooling and `--pooling` do not apply on that path.
- Docs: `docs/embeddings.md` and `docs/supported-models.md` replace the "ColBERT is rejected" sentences, describe the encoder as a fine-tuning backbone, and list the known differences from PyLate.
- Tests: synthetic tests for every layout, the quantized head, padding, and each refusal. Real-checkpoint gates (soft-skipped when the checkpoint is absent) compare against committed mlx-vlm fixtures in `tests/fixtures/lfm2_bidirectional/`.

## 3. Technical Decisions

**One `ModelType`, head chosen by the weights.** Encoder, Embedding and ColBERT share `model_type`, the architecture string (except the encoder) and the backbone. A separate `ModelType` would duplicate detection rules that cannot tell the mlx-community ColBERT build from LFM2.5-Embedding by config. What separates them is the presence of a projection tensor, so the loader decides on that.

**Keep the padding-zeroed conv input on the ColBERT path.** Upstream runs ColBERT with `mask_convolutions=False`. Unpadded rows agree either way; keeping the mask makes a padded row equal to the same text run alone, which a batched server needs.

**Validate projection tensors at load.** MLX checks operand shapes inside ops that cross the FFI without a `Result`. A `bias` of the wrong length, or `scales` rows that disagree with the packed weight, would otherwise abort the server on its first request. The security review found this and moved every check to load time, where it is an error naming the tensor. A zero-width projection is refused for the same reason.

**Refuse a Pooling module next to the Dense.** A sentence-transformers model with Pooling followed by Dense is a pooled single-vector head. Without this check it would load as multi-vector. No such LFM2 checkpoint is published, so this guard prevents a future silent misload.

**PyLate behaviors stay caller-side.** `[Q] ` / `[D] ` prefixes, query expansion to 32 mask tokens and punctuation skiplists are not in the mlx-vlm reference either. They are documented as known score differences rather than implemented.

## 4. Measurements

M3 Ultra, Metal, against mlx-vlm d998a620 forward dumps on fixed token ids.

| Checkpoint | Comparison | Result |
|---|---|---|
| lfm2.5-encoder-350m | last hidden state, per-token cosine | min 1.000000 (served CLS vector 1.000001) |
| lfm2.5-colbert-350m (HF, `1_Dense`) | token vectors, per-token cosine | min 0.999938 |
| lfm2.5-colbert-350m-bf16 (`dense.weight`) | same | min 0.999938 |
| lfm2.5-colbert-350m-4bit | vs bf16 reference (floor 0.95) | min 0.982725, mean 0.993 to 0.998 |

MaxSim, relevant vs unrelated document: 6.4645 vs 5.6824 on HF and bf16 (mlx-vlm 6.4688 vs 5.6871), 6.4184 vs 5.6941 on 4-bit. `mlxcel-server -m` returns `(1024,)` vectors for the encoder and `(5, 128)` / `(9, 128)` matrices for ColBERT on `/v1/embeddings`.

## 5. Not Measured

- The `-8bit` ColBERT conversion was not on the host. It shares the quantized `dense.*` path with the 4-bit build.
- No checkpoint exercises the mlx-vlm `projection.*` layout; only synthetic tests cover it.
- PyLate end-to-end retrieval scores were not compared, because the prefix, expansion and skiplist differences above make them differ by design.

## 6. Learning Points

- A shape mismatch that MLX catches inside an op cannot be recovered across the FFI. Any tensor read from an untrusted checkpoint has to be shape-checked before it reaches an op.
- A late-interaction checkpoint can lose its sentence-transformers folder in conversion, so detection should key on the tensors, not on the folder layout.

## 7. Follow-up Actions

- #1426 exposes `multi_vector()` embedders on `/v1/rerank` with MaxSim scoring and should list LFM2.5-ColBERT there.
- Rows for the encoder and ColBERT checkpoints in `docs/model-catalog.tsv` need the per-host inventory columns filled on the other benchmark hosts.
