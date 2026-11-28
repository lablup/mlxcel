# Technical Report: PR #3103 - Opt the remaining families into maskless padded prefill

**Date**: 2026-11-16
**Status**: Completed
**Risk Level**: Medium (GLM4V MRoPE positions change on padded embedding prefill)

## Executive Summary

The last four families #3097 left out now opt into `supports_maskless_padded_prefill`, so a padded or batched prefill piece skips the dense `[B, 1, L, L + offset]` padding mask (#3101).
- **Cohere2-MoE and EXAONE-MoE:** their forwards already handled a maskless piece.
- **GLM4V and GLM4V-MoE:** the text backbones and the VLM wrappers opt in.
- **Nemotron-NAS:** a new `trim_state` trims its model-owned caches. Before this it declined padded prefill entirely.

The work also fixed a bug that affected the masked path too. A padded GLM4V embedding prefill fell back to `index + rope_deltas` for every row, because the stored image positions cover only the real prompt. That put the real tokens after an image at the wrong positions. `glm4v::padded_tail_positions` now keeps the stored rows for the real tokens and continues from them for the pad rows. With that function disabled, the new GLM4V and GLM4V-MoE parity tests fail.

## 7. Change Summary

| Gate | maskless vs masked (prefill / decode) |
|---|---|
| tiny GLM4V and GLM4V-MoE, with and without image positions, ids and embeddings | 0 / 0 |
| tiny Nemotron-NAS, Cohere2-MoE, EXAONE-MoE | 0 / 0 (EXAONE-MoE past the window: the plain-causal padding mask has no window, so maskless is compared with unpadded instead, 0 / 0) |
| glm-4.1v-9b, llama-3.3-nemotron-super-49b, glm-4.5v checkpoints | 0 / 0 |

Tests: the full `mlxcel` lib suite passes serially (11,955), and so does the roster test.

## 8. Follow-up Actions

No real image prompt was run to measure how the GLM4V position fix changes output on padded batches. GLM-OCR was not touched.
