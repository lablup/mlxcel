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

### Implementation decisions (2026-06-26)

The maintainer locked the following to start work. They refine the open problems below.

- **Two parallel tracks to start.** Track A is the session-contract redesign with the MLX path moved behind it (byte-identical), validatable on the backend we already have. Track B is an OpenXLA reference backend. They run in parallel; the contract from Track A stays provisional until Track B validates it, so some rework of the contract is accepted.
- **Compiler-family model definition: export-first, spike-validated.** The intended path is to import an exported graph (HF transformers via torch-export / torch-mlir, or a JAX reference, lowered to StableHLO) and let mlxcel supply weight mapping, tokenizer, KV orchestration, sampling, and serving, so per-model work shrinks to glue. This is validated by a spike inside the Track B milestone before it is committed. In-tree hand-written StableHLO emission is the fallback if the export route does not hold for our model set. (Update 2026-06-27: export-first is validated for the fp16 path, see the Phase 1 outcome below. The remaining open sub-question is the authoring FRONTEND, JAX-reference versus a Rust-native StableHLO emitter, tracked by #451 and evaluated in parallel; the choice is swappable behind the StableHLO artifact boundary.)
- **First-milestone success bar includes quantized decode.** The milestone is not done at fp16 coherent output; it runs through 4-bit quantized decode on the compiler family (the issue's stated real success bar). fp16 single-sequence coherent output is the intermediate checkpoint. The spike must therefore also characterize how 4-bit lowers on XLA (dequant-in-graph versus a custom op or target kernel).
- **KV / paged / scheduler abstraction is a later phase.** The compiler family starts single-sequence; batching and paged KV stay MLX-session features until a later abstraction phase.
- **The StableHLO/MLIR backend lives in its own default-off crate.** XLA / PJRT dependencies must not touch the default Apple-Silicon or CUDA builds.
- **Reference model is a small one** (Llama-3.2-1B class), 4-bit for the success bar with an fp16 variant as the intermediate checkpoint.

### Open problems this ADR names but does not yet resolve

- **Model definition for the compiler family.** Export-first is now validated for the fp16 path (Phase 1 outcome below); the live sub-question is the authoring frontend (JAX-reference, proven; a Rust-native StableHLO emitter, under parallel evaluation in #451; torch.export, unexercised). How 4-bit lowers (dequant-in-graph fusion versus a custom op) is the Phase 2a question, in progress on the JAX harness and frontend-independent.
- **KV cache, paged KV, and scheduler coupling.** The batch scheduler, paged KV block table and pool, and speculative decode are built on the MLX `KVCache` type today. They remain MLX-session features initially; abstracting the block-table and pool concepts over a backend-owned KV representation is a separate, later phase, not a prerequisite for the first non-MLX session.
- **Furiosa graph ingestion.** Whether the Furiosa toolchain ingests StableHLO, or needs a bespoke Option B engine, is a feasibility-gate unknown, consistent with the hardware go/no-go gate issue #338 already deferred kernel work behind.

### Validation plan

Prove the session contract with OpenXLA as the second reference backend before the contract is locked and this ADR is marked Accepted. A second real implementation is what forces the abstraction to be genuine rather than an MLX-shaped trait. Per the 2026-06-26 decisions the two tracks run in parallel: Track A (session contract plus MLX behind it, byte-identical) proceeds on the MLX backend, while Track B (the OpenXLA reference backend) spikes the export route on one small model, reaches fp16 single-sequence coherent output, then carries through 4-bit quantized decode as the success bar. Track B's findings resolve the model-definition strategy and feed back into Track A's contract.

### Phase 1 outcome (2026-06-27)

Track B Phase 0/1 (issue #449, branch `spike/openxla-export-449`) validated the export-first route for the fp16 path. A Llama-3.2-1B model written once as a JAX reference exports to StableHLO via `jax.export` and greedy-decodes token-exact (48 of 48) against an HF transformers temp-0 reference; the same serialized StableHLO runs unmodified on both PJRT and IREE with matching argmax and no `custom_call`, and isolated logits agree with HF to fp32 reduction noise (about 1e-5). This is the result the validation plan required before committing export-first.

Consequences and refinements:

- **Export-first committed for the fp16 path.** Models are authored once per architecture as a graph with weights as inputs; within an architecture family per-checkpoint work is weight-mapping glue, across transformer variants it is scaffold plus per-arch deltas, and new families (MoE, SSM-hybrid, VLM) need new forward authoring. So "per-model shrinks to glue" holds within a family, not across families.
- **Session contract shape confirmed.** The exported `prefill` / `decode_step` graphs with session-owned fixed-capacity KV and on-device argmax match the `InferenceSession` contract landed for #448 (PR #450). Phase 3 integration binds to it.
- **Authoring frontend still open, evaluated in parallel.** Phase 1 proved the JAX-reference route. A Rust-native StableHLO emitter is under parallel evaluation (#451) as the production authoring path that keeps compiler-family model definitions in-tree in Rust with no Python in the authoring path; the StableHLO artifact is the swappable boundary, so this does not affect the runtime or the #448 integration. torch.export-from-HF remains unexercised.
- **GPU deferred; Phase 2 split.** The Phase 1 host (GB10, aarch64 plus Blackwell) has no CUDA jaxlib wheel on PyPI, so Phase 1 ran on CPU (sufficient for the coherence bar). Phase 2 is split: 2a (int4 dequant-in-graph lowering structure, fusion behavior, and correctness) proceeds now on CPU and is frontend-independent; 2b (real GPU perf and memory) is deferred until a GPU host is chosen (an NGC JAX-Toolbox container on the GB10 box, or an x86 CUDA host).

This ADR stays Proposed; it moves to Accepted once the OpenXLA backend integrates behind the #448 session contract (Phase 3) and validates the full contract.

### Low-precision performance decision (2026-07)

Phase 3 landed the OpenXLA backend on macOS (Metal) and profiled it against MLX. On an M1 Ultra (Llama-3.2-1B, greedy) XLA-on-Metal runs about 117x slower than MLX. Profiling with `iree-benchmark-module` (pure runtime, no host glue) shows the Metal decode step is GPU-kernel-bound, not host-bound: a 13-thread CPU beats the Metal GPU on the same StableHLO graph, so the bottleneck is IREE's `metal-spirv` kernel codegen, not invoke overhead or bandwidth.

That split of the performance levers decides where the project invests:

- Graph-level precision and quantization, authored in the StableHLO graph, transfer to every IREE target, including future NPUs, for which low precision is the entry ticket rather than a 2x optimization. f16 / bf16 is landed (#514, #515): about 1.9x on Metal, token-exact, and it speeds up the CPU path too. This is in scope.
- Per-backend kernel codegen (the remaining ~50x to MLX) is upstream IREE's responsibility, is Metal-specific (it does not transfer to non-SPIR-V NPUs), and MLX already owns Apple-Silicon performance. Out of scope.

int8 / int4 weight quantization (#516) is the NPU lever, but its payoff is memory bandwidth: a compute-bound Metal decode cannot demonstrate it, and measuring it needs an actual NPU. It is deferred to a hardware-gated follow-up; on Metal only its token-exactness would be verifiable. Metal's absolute throughput is therefore a pessimistic proxy for an NPU, which brings its own optimized kernels.

### int8 weight quantization: the fusion gate (2026-07-01)

The int8 lever (#516) was picked up on a GB10 (Grace-Blackwell, CUDA via a source-built IREE runtime), the first int-native target available. The packed path (`MLXCEL_XLA_QUANT=packed`) keeps the MLX 4/8-bit weights resident packed (`ui32` + f16 scales/biases) and dequantizes each weight in the StableHLO graph (bit-unpack, then `q*scale + bias`), rather than dequantizing to f32 at load. The whole packed ABI landed: an emitter dequant primitive, a per-weight-dtype weight upload, and `ui32` / f16 device buffers.

The result refines the split above rather than contradicting it. On the GB10 the packed decode is **token-exact** with the f32/f16 path (the in-graph reconstruction is bit-identical to the host dequant) but about **4.3x slower** (roughly 1.6 vs 6.7 tok/s, Llama-3.2-1B 4-bit, greedy), with GPU utilization rising (84% to 96%). IREE's CUDA codegen does not fuse the unpack+dequant into the matmul: the decode step is about 678 dispatches and the reconstructed f32 weight is materialized to DRAM every step, so the graph pays more bandwidth and compute, not less. The aggressive-fusion / generalize-matmul / early-truncate-fusion flags leave the dispatch count unchanged (678 to 677).

So the memory-bandwidth payoff is **not** realized by authoring the dequant in the portable graph alone; it requires the target to fuse dequant into the matmul (a quantized-matmul op, or an int8 `dot_general` lowering to the hardware int8 path). This is the same lever split the f16 profiling reached: the graph-level change is in scope and transferable, but the fused low-precision kernel is upstream IREE's responsibility. The packed path therefore stays behind the `MLXCEL_XLA_QUANT=packed` gate, off by default, as the correctness-verified foundation a fused quantized-matmul reuses. The bandwidth win is now gated on that fusion (in IREE, or on an NPU whose HAL driver lowers a quantized dot to its systolic int8 kernels), not on hardware availability alone.

### Low precision on the Metal target (2026-07-02)

Issue #575 transferred the realized low-precision levers back to the Metal HAL driver on an M1 Ultra, running Llama-3.2-1B 4-bit through the normal `mlxcel generate` path (`MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=metal`) rather than an isolated benchmark harness. The result confirms the transferability thesis for the lever that is realized, and maps the two levers that do not cross to Metal.

f16-resident weights (#572) transfer, run 2.3x faster than f32, and stay token-exact. On Metal the f16 path is 3.45 vs 1.51 tok/s (median of 3, both deterministic across runs), a 2.28x speedup, and it holds the accuracy gate: the f16 greedy trajectory matches the f32 trajectory for 64 steps on the #515 harness (`xla_oracle_check` against an XLA-f32 oracle from the new `xla_traj_dump` example, `TOKEN-EXACT PASS` at 40 and 64 steps). f16 is the per-device default on `metal` (and `cuda`), so the win is on by default; `MLXCEL_XLA_PRECISION=f32` is the reference-run opt-out. The Metal decode responds to precision (f16 roughly halves the per-token compute), so it is compute-bound, consistent with the `iree-benchmark-module` profiling above. This is the same lever #514 / #515 measured (~1.9x), landing on a second IREE target with no per-backend kernel work, which is the point of ADR 0004.

bf16 does not lower on `metal-spirv`. The Metal GPU target advertises no bf16 compute (`compute = fp32|fp16|int64|int32|int16|int8`), so the emitter's `f32` to `bf16` `vector.bitcast` fails to legalize at compile time (`failed to legalize operation 'vector.bitcast'`). f16 is the portable low-precision entry on Metal; bf16 stays a CUDA / CPU lever. `resolve_precision_checked` rejects `MLXCEL_XLA_PRECISION=bf16` on a `metal` device up front, pointing at f16, so it surfaces as a clear load error rather than the `iree-compile` legalization dump (#612).

The packed int8 path (#568) compiles for `metal-spirv` but faults at runtime: the ui32-unpack plus in-graph dequant graph lowers, but the prefill invoke faults in the metal HAL driver with `Metal command buffer failed (status 5)` at `hal.fence.await` (`metal_device.m`), surfaced as IREE `IREE_STATUS_INTERNAL` (#613). The same graph runs token-exact on the GB10 CUDA runtime, so this is IREE metal HAL / `metal-spirv` runtime behavior on the large dequant-then-matmul command buffer, separate from the fusion gate. Because the packed and fused int8 win is a memory-bandwidth win and the Metal decode is compute-bound (the f16 result above shows the decode tracks compute, not bandwidth), it is un-demonstrable on Metal regardless of that fault, and the fused quantized-matmul is upstream IREE work (#574). `check_packed_supported` now rejects `MLXCEL_XLA_QUANT=packed` on a metal device up front rather than surfacing the opaque command-buffer error; the packed path stays out of reach on Metal for now.

The graph-level precision lever that is realized (f16-resident) lands on Metal at 2.3x and token-exact through the normal serve path, confirming ADR 0004's transferability thesis on a second target. The fused low-precision path #575 set out to transfer does not exist in-tree (upstream IREE, tracked by #574), and Metal adds two limits of its own (no bf16 codegen, a packed runtime fault) on top of the shared fusion gate.

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
