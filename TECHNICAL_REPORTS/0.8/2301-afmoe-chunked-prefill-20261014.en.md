# PR #2301: Correct the AFMoE no-ID chunked prefill contract

**Date**: 2026-10-14
**Status**: Implemented and independently reviewed; CUDA, focused GPU, and contract validation passed
**Risk**: Low (model capability policy and loaded-model regression)

## Summary

AFMoE previously inherited `supports_chunked_prefill() == true` even though a multi-token forward without a sequence ID replaces its model-owned cache set. A legacy caller that split one prompt across no-ID forwards could therefore discard all earlier sliding- and full-attention KV state.

The model now opts out of token-ID chunked prefill. A loaded synthetic regression exercises the production loader and the actual mixed-cache lifecycle across six layers.

## State behavior

AFMoE owns a heterogeneous cache vector. The reduced Trinity fixture follows the real hybrid design with four sliding-attention layers backed by rotating KV caches and two full-attention layers backed by standard KV caches. Its first two layers use dense MLPs and the remaining layers use the MoE path.

The no-ID path deliberately replaces the whole cache vector before every multi-token forward. This behavior keeps independent prompts isolated. Stable sequence IDs instead select per-sequence model-owned state and retain every layer's cache across calls.

DirectEngine already passes the same sequence ID through its prefill pieces, so this issue is not evidence that the current CLI truncated AFMoE prompts. The capability must also be safe for cache-less and legacy no-ID callers.

## Change

- Override `supports_chunked_prefill()` to return `false`.
- Construct a six-layer AFMoE model through `AfmoeModel::from_weights` with deterministically filled nonzero synthetic tensors.
- Run no-ID lengths two and three, then require all six cache offsets to equal three, proving the second prompt replaced earlier state.
- Run the same lengths under one stable sequence ID, then require all six cache offsets to equal five, proving continuation retains both rotating and standard KV state.

The test inspects every cache rather than one representative layer, so a partial reset or a drift between sliding and full-attention cache handling fails the regression.

## Verification

The source passed independent implementation review. The root and surgery CUDA `test-fast` build graph passed in 14 minutes 05 seconds. Under the GPU lock, the complete `models::afmoe::tests::` module passed 29 of 29 tests in 0.85 seconds, including the loaded-model lifecycle regression and existing snapshot restore coverage. The contract gate passed all earlier checks, 123 Python tooling tests, and 49 WebUI schema, DTO, and fixture checks. Formatting and patch-whitespace checks pass. The fixture loads deterministic nonzero synthetic weights; no external AFMoE checkpoint run was performed or claimed.

## Dependency refresh

The validation above was recorded before the branch merged main commit `4b0c4519` from PR #2298. The merge commit `4cc9b87d` brings in the new MLX pin and CUDA/Metal overlay bases without changing the reviewed model files: `afmoe.rs` remains SHA-256 `7f00690db48937f9e5a2810f9c26307e50dd1d5b14ea1810cd216f75d1d3466c`, and `afmoe_tests.rs` remains `844cbce7258e750cface1aa24b7f64a925b2c8c79524159a411c53de25d71a12`. Fresh runtime validation after the final pin integration is recorded below.

## Memory tradeoff

`PrefillCaps` applies this model-wide capability to every token-ID prefill planner. Every honoring caller, including DirectEngine and cache-less callers, therefore uses one whole-prompt graph for AFMoE. This prevents unsafe no-ID state loss but gives up chunk-bounded transient memory, so long prompts can have higher peak memory use.

## Scope

The change preserves independent-prompt reset behavior, stable-ID server state, cache modes, quantization loading, padded-prefill policy, and snapshot behavior. It changes only the chunking capability and adds lifecycle coverage.

## Final integration validation

Integrated main `a2698835`, including MLX `243bdf09`, mlxcelverse, and preceding model fixes, without source conflicts. The rebuilt CUDA `test-fast` binary passes all 29 `models::afmoe` tests, including loaded mixed sliding/full-attention cache lifecycle and snapshot coverage. No-ID prompts reset all six caches, while stable-ID chunks advance every cache continuously. Formatting and diff checks pass. This is loaded synthetic-weight validation, not an external-checkpoint run.
