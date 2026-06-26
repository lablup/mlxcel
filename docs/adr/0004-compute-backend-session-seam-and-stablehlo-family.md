# ADR 0004: The compute-backend seam is an inference session engine, and non-MLX targets converge on a StableHLO/MLIR compiler family

**Status:** Proposed (2026-06-26). Direction agreed with the maintainer. To be marked Accepted once a second reference backend (OpenXLA) validates the session contract on at least one real model. Reframes the seam introduced in issue #338 / PR #446: that PR's `ComputeBackend` trait contract (load-boundary, returns the concrete MLX `LoadedModel`) is treated as provisional and is superseded by the session-level contract described here before any non-MLX engine lands. The `select_backend` selection skeleton and the default-off `experimental-backend` feature gate from #446 are retained.

## Context

PR #446 (issue #338) introduced a `ComputeBackend` seam to abstract the forward-execution engine. It draws the boundary at the model-load point: `ComputeBackend::load_model(path) -> (LoadedModel, MlxcelTokenizer)`, with `MlxBackend` delegating verbatim to `crate::loading`, a single zero-sized `Backend` variant under default features (so the dispatch folds away), and a `cfg`-gated `experimental-backend` scaffold that returns `not_implemented()`. That shipped as a behavior-preserving refactor with byte-identical temp-0 parity. Its `experimental.rs` already carries a note that the concrete return type would have to evolve for a real non-MLX engine.

The refactor's actual motivation is broader than one vendor: hosting FuriosaAI (TCP / RNGD, the `furiosa-opt` Rust eDSL compiling to a virtual ISA), Tenstorrent (TT-Forge / TT-MLIR), and an OpenXLA-based path (StableHLO compiled by XLA and run through PJRT). With those three targets in view, the load-boundary contract is insufficient, for two reasons that sit below the return-type symptom.

First, the forward contract itself is MLX-coupled. `LanguageModel::forward` (`src/lib/mlxcel-core/src/generate.rs`) takes `caches: &mut [KVCache]` and returns `UniquePtr<MlxArray>`. The input KV representation and the output tensor are both MLX types, so a non-MLX engine cannot implement `LanguageModel` at all. Returning a concrete `LoadedModel` (which `impl LanguageModel`) only chooses which loader runs; the executor it produces is still MLX end to end.

Second, all three targets are graph-compiler backends, not eager-op backends. `furiosa-opt`, TT-Forge, and OpenXLA each ingest a whole-graph (or whole-module) description, compile it, and execute through their own runtime, with static shapes, their own memory placement, and their own tensor and KV representations. This rules out two tempting shapes. An op-level `TensorOps` trait (parametrize the existing models over a tensor type) does not fit a compiler backend's programming model and would also push indirection toward the MLX inner loop, which issue #338 explicitly warned against. And reusing the MLX model code with a swapped tensor type does not work either: a Furiosa or XLA Llama is a different implementation (a compiled graph), not the same eager Rust with a different array type.

The control plane above the executor is already backend-neutral and stays reused unchanged: the downloader, SafeTensors parsing, tokenizer, chat template, sampling policy, the OpenAI and llama-server compatible HTTP surface, and the request lifecycle. The coupling that matters is everything from the generation loop down.

## Options considered

### Option A (rejected): op-level `TensorBackend` / `TensorOps` trait

Parametrize models over an associated tensor and cache type and have each backend provide the ops. Rejected because it does not match a compiler backend's whole-graph model (there is no per-op call site to implement when the backend wants the entire graph up front), it would force the MLX hot path through a generic op interface and risk losing graph fusion and `mx.compile`, and it ripples through every one of the 30-plus model implementations. This is the altitude issue #338 already warned against.

### Option B (rejected as the primary shape): per-vendor bespoke backends

Write a separate MLX-style hand-coded engine for Furiosa, Tenstorrent, and XLA each. Rejected as the primary direction because it multiplies the model-porting cost by the number of backends: the feature-parity investment against mlx-lm and mlx-vlm would have to be repeated per vendor. It remains the fallback for any target whose toolchain cannot ingest a shared graph IR.

### Option C (chosen): inference-session seam, with non-MLX targets on a StableHLO/MLIR compiler family

Draw the seam at the inference-session / engine level with a token-level contract, and treat the non-MLX world as a single StableHLO/MLIR compiler family rather than N independent vendor engines.

A backend produces an inference session from `(model_path, config)`. The session exposes prefill and decode-step at the token level: it takes token ids and a sampling spec, runs sampling on-device inside the backend, returns token ids (and log-probabilities only when requested), and owns its KV cache internally. The MLX hot path lives entirely inside the MLX session, so there is no per-op dispatch and the existing graph fusion, `mx.compile`, paged KV, and prompt-cache detach/adopt are preserved. `CxxGenerator` becomes the MLX session implementation. Returning token ids rather than raw logits on the hot path keeps sampling on-device and avoids a per-token device-to-host copy.

For the non-MLX side, OpenXLA (StableHLO) and Tenstorrent (TT-MLIR, which has a StableHLO frontend) converge on the same IR, and IREE/PJRT turn hardware support into a target-plugin problem rather than a rewrite. So the design favors one compiler-family backend that emits a portable StableHLO/MLIR graph over per-vendor engines. The execution families collapse from four to two: MLX (eager, the Apple-Silicon-optimized reference) and StableHLO-compiler (OpenXLA, Tenstorrent, and Furiosa if its compiler ingests StableHLO). Models for the compiler family are defined once as graph emission rather than re-ported per vendor.

### Option D (rejected for now): a full backend-neutral model IR

Define models in a neutral IR that lowers to MLX and to every compiler target, so a model is written once for all backends including MLX. Rejected for now as the largest upfront commitment (it is effectively a mini compiler frontend) and the riskiest. Option C reaches most of its reuse benefit for the compiler family through StableHLO without forcing the MLX eager path into an IR. Option D stays on the table only if write-once across MLX and the compiler family becomes a hard requirement.

## Decision

Adopt Option C.

- The compute-backend seam is an inference session engine, not a load factory and not an op interface. A backend constructs a session; the session runs prefill and decode-step at the token level with on-device sampling and owns its KV representation. The session advertises its capabilities (batching, paged KV, speculative decode, multimodal) so the control plane can gate features it does not support.
- MLX stays the eager, full-featured reference backend, untouched. `CxxGenerator` becomes the MLX session implementation; the per-token forward and the KV optimizations remain MLX-internal.
- Non-MLX targets are served by a single StableHLO/MLIR compiler-family backend that emits a portable graph, rather than by per-vendor hand-written engines. Per-vendor bespoke backends (Option B) are the fallback only for a target whose toolchain cannot ingest the shared IR.
- The `select_backend` selection skeleton and the default-off `experimental-backend` feature gate from PR #446 are kept. The `ComputeBackend` trait contract from #446 (load boundary, concrete `LoadedModel`) is provisional and is replaced by the session contract above before any non-MLX engine is wired in.

### Open problems this ADR names but does not yet resolve

- **Model definition for the compiler family.** How models emit StableHLO/MLIR (a shared graph-builder, reusing config and weight loading) is the central follow-up design and the main cost driver. The first cut may hand-write graph emission for a small hot set (Llama, Qwen, Gemma) rather than all families.
- **KV cache, paged KV, and scheduler coupling.** The batch scheduler, paged KV block table and pool, and speculative decode are built on the MLX `KVCache` type today. They remain MLX-session features initially; abstracting the block-table and pool concepts over a backend-owned KV representation is a separate, later phase, not a prerequisite for the first non-MLX session.
- **Furiosa graph ingestion.** Whether the Furiosa toolchain ingests StableHLO, or needs a bespoke Option B engine, is a feasibility-gate unknown, consistent with the hardware go/no-go gate issue #338 already deferred kernel work behind.

### Validation plan

Prove the session contract with OpenXLA as the second reference backend on one or two hot models before the contract is locked and this ADR is marked Accepted. A second real implementation is what forces the abstraction to be genuine rather than an MLX-shaped trait.

## Consequences

- The `ComputeBackend` trait from PR #446 is reworked from a load-boundary contract returning `LoadedModel` into a session-engine contract. The selection skeleton (`select_backend`, the `Backend` enum, the `experimental-backend` feature gate) survives the rework; only the contract shape changes.
- The control plane (downloader, SafeTensors, tokenizer, chat template, sampling policy, OpenAI / llama-server API, request lifecycle) is confirmed to sit above the seam and is reused across backends unchanged.
- Paged KV, prompt-cache detach/adopt, speculative decode, and cross-request batching stay MLX-session capabilities at first. Multi-backend parity for those is explicitly a later phase, gated on the KV-abstraction problem above.
- The mlx-lm and mlx-vlm feature-parity model-porting investment is preserved for the MLX backend. The compiler family starts with a smaller model set defined through StableHLO emission, and broad model coverage there grows separately.
- If the StableHLO convergence across OpenXLA and Tenstorrent holds (and Furiosa joins it), adding a hardware target becomes a PJRT or MLIR-target problem rather than a per-vendor model rewrite. That convergence is a hypothesis this design bets on and the feasibility gate must confirm.
- This ADR sets direction only. The session-contract design, the StableHLO emission design, and the KV-abstraction phase each get their own follow-up issues, and this ADR is updated to Accepted (or superseded) once the OpenXLA reference backend validates the contract.

## References

- Issue #338 (the seam motivation and scope), PR #446 (the load-boundary seam this ADR reframes), and `src/backend/{mod,mlx,experimental}.rs` (the shipped selection skeleton and the provisional contract).
- `src/lib/mlxcel-core/src/generate.rs` (`LanguageModel::forward`, the MLX-coupled forward contract that the session seam sits above).
- `src/loaded_model.rs` (the concrete `LoadedModel` executor and its multimodal variant dispatch, the coupling that made an engine-neutral return impractical for #338).
- furiosa-opt documentation (https://developer.furiosa.ai/furiosa-opt/book) and repository (https://github.com/furiosa-ai/furiosa-opt); OpenXLA / StableHLO and PJRT; Tenstorrent TT-MLIR. The cross-vendor StableHLO convergence is the hypothesis the feasibility gate validates.
- [ADR 0001](0001-paged-attention-gather-vs-fused-kernel.md), [ADR 0002](0002-turbo-kv-split-dequant-vs-fused.md), and [ADR 0003](0003-release-panic-unwind-with-core-thread-abort.md), the prior records in this series.
