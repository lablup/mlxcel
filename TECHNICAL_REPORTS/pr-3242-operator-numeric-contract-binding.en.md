# PR #3242: Bind operator numeric contracts into production auxiliary artifacts

**Date**: 2026-11-23
**Status**: PR open (closes #3211; part of #932 and epic #566). Unit suites pass; real checkpoints verified on GB10 for six vision runtimes on CUDA and Gemma3n audio on `local-task`

## Executive Summary

The OpenXLA/IREE backend compiles the vision encoders and audio towers of multimodal models into cached VMFB "auxiliary artifacts", each guarded by a manifest that must match before `IreeAuxiliaryModule::load` runs the module. #934 and #937 had defined a v3 manifest schema carrying an operator-level numeric contract, but no production family used it: all 11 production call sites built `new_legacy_unqualified` contracts and wrote v1 manifests with no numeric identity. Only the micro-oracle probes wrote v3. #932's first acceptance criterion (a versioned numeric contract in every affected artifact fingerprint, rejected on mismatch before invocation) was therefore unmet even though all seven family ports had merged.

This PR adds one builder that maps a family's declared operator list, the `Precision` its emitter used, the checkpoint dtype classes its loader observed, and the IREE compiler, runtime, and device identity onto an `OperatorNumericContractSet`. Every production family now writes a v3 manifest through it. A cached artifact whose stored contract differs in any field is recompiled before load. Review found and closed one hole, Gemma3n's BF16 host dequantization rounding, which had produced the same contract as F32 dequantization. A second fix keeps v3 artifacts in a schema-named subdirectory so that older binaries sharing the same `/tmp` cache keep working. The per-dispatch kernel IREE selects is recorded as unobserved: the contract binds identity and does not claim production qualification.

## 1. Problem Statement

### 1.1 What the v1 manifest bound, and what it missed

A v1 auxiliary contract records an entry name, a config identity, and a generation identity (compiler path and SHA-256, `iree-compile --version`, flags, MLIR digest). That catches graph and compiler changes, but several inputs that decide the numbers a module produces were not named:

- **The linked IREE runtime.** The runtime archive is part of the consuming binary, not of the compiler invocation, so a runtime rebuild left the manifest unchanged.
- **Host-side preprocessing of weights.** Qwen2-VL, Qwen3-VL, and Gemma3n dequantize affine Q4 weights on the host before upload. The MLIR sees only dense F32 inputs, so the dequantization order and rounding never reached the MLIR digest.
- **Checkpoint dtype classes and per-operator precision.** These appeared, if at all, as free-form strings inside the config identity, not as validated fields.

#932 asked for these as one versioned, shared descriptor rather than family-specific strings. The descriptor existed (`operator_numeric_contract.rs`, `aux_manifest.rs` v3), but its constructor `new_with_operator_numeric_contract` carried `#[allow(dead_code)]`.

### 1.2 Risk of leaving it

| Risk | Impact | Likelihood |
|---|---|---|
| A runtime or host-dequant change reuses a cached artifact whose recorded numerics no longer describe the run | Medium: silent drift that a later oracle cannot attribute | Medium on dev boxes and self-hosted runners that keep `/tmp` caches |
| #3212's operator probes report a contract identity that production artifacts never carry, so probe and production evidence cannot be joined | High for #932 | Certain without this PR |

## 2. Change Summary

| Area | Change |
|---|---|
| `operator_numeric_contract/production.rs` (new) | `production_operator_contract(family, operators, precision, weights, backend)`. Derives each entry's checkpoint dtype, input dtype, accumulator (always F32), result dtype, rounding boundaries, weight execution, and dequantization contract from four inputs. Rejects an input policy that does not fit the operator class. `CheckpointSourceObserver` collects per-tensor dtype classes during weight loading and fails closed on mixed affine layouts. |
| `operator_numeric_contract/families.rs` (new) | One declaration function per runtime: LLaVA/SigLIP/Gemma3, Qwen2-VL/Qwen2.5-VL, Qwen3-VL (main plus DeepStack), Molmo, Molmo2, Youtu-VL, Gemma3n audio encode and merge, Phi4MM audio. Shared helpers build the common encoder layer, attention block, and merger. |
| `iree_identity.rs` (new) | `compiler_version` and `runtime_archive_identity` move here from `numeric_probe.rs` so the probes, production, and #3212 share one implementation. `compiler_version_output` is memoized per compiler path, size, and mtime. `production_artifact_contract` assembles the v3 contract. |
| `build.rs`, `Cargo.toml` | Bakes the SHA-256 of the linked `libiree_runtime_unified.a` into `MLXCEL_XLA_IREE_RUNTIME_SHA256` for the dist, CUDA, ROCm, and macOS layouts. Adds `sha2` as a build dependency. |
| Seven runtime files plus `phi4_audio.rs` | All 11 `new_legacy_unqualified` sites replaced. Loaders feed the observer. Fixed-precision emitters (LLaVA, Qwen3-VL, Molmo, Molmo2, Youtu-VL) and Gemma3n audio take their precision from a named constant that both the emitter and the contract read; Qwen2-VL and Phi4MM pass the resolved value they emitted with. |
| `aux_manifest.rs` | `schema_scoped_vmfb_path` and `qualify_auxiliary_artifact` place non-v1 artifacts in a subdirectory named after the manifest schema. |
| `operator_numeric_contract.rs` | New `AffineQ8Dequantize` class and `SeparateMultiplyThenAddBf16Rounded` evaluation order. Blanket `dead_code` allows removed; variants no production entry builds keep item-level allows with a reason. |
| Tests | `production_tests.rs` and `family_tests.rs`: one stale-cache rejection plus recompile per field class, v1 to v3 replacement, observer behavior, per-family pins at default CUDA and CPU precision, removal sensitivity, emitter/contract precision mismatch, host dequant rounding identity. |
| Docs | `docs/environment-variables.md` describes the v3 manifest, the schema subdirectory, and what the contract does not claim. |

26 files, +2340/-224. Text-only token/decode emitters, their goldens, and every tolerance are unchanged.

## 3. Technical Decisions

### 3.1 Declare operator classes per family, derive everything else

Each family lists the operator classes its graph emits, one entry per distinct operand origin and input policy, for example `weighted("encoder.mlp-down-projection", DenseMatmul)` or `attention(scores, softmax, context)`. A declaration carries only four facts: what the operator reads (`MatrixWeight`, `FloatParameter`, `Activation`), how it materializes inputs (`FollowsPrecision`, `PinnedF32`, `F32Carrier`), its association policy, and an optional explicit output rounding. The builder derives the rest. A `FollowsPrecision` contraction gets the emitter's narrow type as its input dtype and an `OperatorInput` rounding boundary. A `MatrixWeight` operator in an affine checkpoint gets `AffineU4`/`AffineU8`, the dequantization contract, and `HostAffineDequantized` or `PackedAffineInGraph`. Accumulating classes get `AccumulatorResult`.

The alternative was to spell every dtype per entry in each family. That form would let a family declare a BF16 input on a graph emitted at F32. Deriving from the same `Precision` value the emitter received makes that state unrepresentable, and `precision_mismatch_between_emitter_and_contract_changes_identity` checks that a different precision changes the identity. Pooling, RoPE, masking, and gathers are left out of the lists because they round once and carry no accumulator; the MLIR digest in the generation identity still binds them.

### 3.2 One constant drives both the emitter and the contract

For the five fixed-precision vision graphs, the emitter used to call `Builder::new()`, whose default is F32, and the runtime had no value to bind. The PR adds constants such as `LLAVA_VISION_PRECISION: Precision = Precision::F32` and has the emitter call `.with_precision(LLAVA_VISION_PRECISION)`. The emitted MLIR is unchanged because the value equals the default, but a later edit to the constant now changes both the graph and the contract together. Qwen2-VL and Phi4MM already resolve precision per device or per environment variable, so their runtimes pass that resolved value directly.

### 3.3 The Gemma3n BF16 rounding hole (found in review)

Gemma3n audio dequantizes its affine weights with `dequantize_affine_bf16_sequential`: multiply, round to BF16, add the bias, round to BF16 again. That is MLX's BF16 affine behavior. Qwen2-VL and Qwen3-VL use `dequantize_affine`, which stays in F32. Both paths had mapped to `AffineEvaluationOrder::SeparateMultiplyThenAdd`, so the two contracts were identical. Because the dequantization runs on the host, the MLIR digest did not see it either. Switching Gemma3n between the two dequantizers would therefore have left every cached artifact's identity unchanged.

Commit `5c0bf520` adds `SeparateMultiplyThenAddBf16Rounded` and `AffineExecution::HostBf16Sequential`, and introduces one constant, `GEMMA3N_AUDIO_AFFINE`. `decode_weight` matches on it to pick the dequantizer, and the contract resolves the observer with the same value, so the code that rounds and the code that records the rounding cannot disagree. `host_dequantization_rounding_is_part_of_the_identity` asserts that the two host modes produce different identities.

### 3.4 Schema-scoped cache directories

This is the first change to write v3 manifests in production. `reject_manifest_downgrade` refuses to replace a newer-schema manifest with an older one. That is the intended policy, but every auxiliary cache sits at a fixed `std::env::temp_dir()` path. An older binary on the same machine (an unrebased branch on a shared self-hosted runner, or a release build next to a dev build) would find a v3 manifest and fail its vision or audio load with `refusing to replace auxiliary artifact manifest ... from schema v3 with v1`.

| Option | Pros | Cons |
|---|---|---|
| Keep the shared path, document "clear the cache" | No code | Hard failures on shared runners until someone clears `/tmp` |
| Relax the downgrade refusal | Old binaries recompile | Two binaries overwrite each other's entries on every alternation, and the refusal stops guarding anything |
| **Chosen: a schema-named subdirectory** | Each schema owns its entries; the refusal still guards within one schema | One extra compile after upgrading; the old v1 entry stays on disk |

`qualify_auxiliary_artifact` is the single call every production site now uses. It computes `<cache>/mlxcel-xla-aux-artifact-v3-operator-numeric/<file>` for any non-v1 contract, creates the directory, runs `ensure_qualified_auxiliary_artifact` there, and returns the path that must be passed to `IreeAuxiliaryModule::load`. Returning the path, instead of having each caller recompute it, keeps the qualify and load paths from diverging. A v1 contract keeps its original path, so `aux_smoke` and the downgrade tests are unaffected.

### 3.5 Bake the runtime digest at build time

The CPU probes used to hash `lib/libiree_runtime_unified.a` under the IREE dist root at run time. The CUDA, ROCm, and macOS builds link a source-built runtime from `build/runtime/src/iree/runtime/` and have no dist archive to read, and a deployed binary may not see its build tree at all. `build.rs` now hashes the archive it links and bakes the digest as a compile-time env var. `runtime_archive_identity` prefers that value and falls back to the dist layout only when the build could not read the archive. Review noted that the `rerun-if-changed` line is emitted even when the archive is missing, so the build script reruns until it appears. That was kept so the digest self-heals; every linking layout already requires the archive.

### 3.6 Identity, not qualification

`BackendNumericIdentity` takes an algorithm field, and IREE exposes no per-dispatch kernel identity for a compiled VMFB. Every production entry records `unobserved-iree-dispatch-selection` and the emitter revision `mlxcel-xla-stablehlo-emitter-v1`. That keeps #932's three reference claims apart: this PR establishes that an artifact is the one its contract describes. Whether that contract matches production MLX is #3212's job.

### 3.7 One compiler spawn per artifact

Building a contract and building the generation identity both need `iree-compile --version`. Without sharing, each artifact spawned the compiler twice, which review measured at about 45 ms of extra load time per artifact. `compiler_version_output` memoizes the trimmed stdout keyed by path, size, and modification time, so a replaced compiler is queried again. The generation identity strings are byte-identical to before, so no v1 entry was invalidated by this refactor alone.

## 4. Verification

Unit and build checks (final state of the branch):

- `cargo test -p mlxcel-xla --lib --profile test-fast`: 435 passed (433 at the first commit: baseline 422 plus 11 new).
- `--features iree,micro-oracle` over the runtime, manifest, contract, and probe modules: 105 passed.
- `cargo clippy -p mlxcel-xla --lib --tests -D warnings` with no features, `iree`, and `iree,micro-oracle,diagnostics`; `cargo fmt --check`; `cargo check --features cuda,xla-iree --all-targets`; `cargo check --no-default-features --features xla-diagnostics --all-targets`.

The stale-cache tests mutate one field class at a time (operator dtype, accumulator, association, precision, compiler, runtime, device) and assert that the cached VMFB is rejected and recompiled. The family pins fix each family's operator list at default CUDA and CPU precision and fail when an operator, cast, or association policy is removed.

Real checkpoints on GB10, `MLXCEL_BACKEND=xla`, under `gpu-lock`, `-n 24 --temp 0`. A main build ran first, then the branch build loaded twice. Pass 1 replaced the v1 manifest with v3 and pass 2 reused it (VMFB and manifest mtimes unchanged).

| Checkpoint | Runtime | Device | Greedy output across 3 runs |
|---|---|---|---|
| llava-interleave-qwen-0.5b-bf16 | `vision_runtime` | cuda | identical |
| qwen2-vl-2b-instruct-4bit | `qwen2_vl_runtime` (host-dequantized Q4) | cuda | identical |
| qwen3-vl-2b-instruct-4bit | `qwen3_vl_runtime` | cuda | identical |
| molmo-7b-d-0924-4bit | `molmo_vision_runtime` (in-graph Q4, `MLXCEL_XLA_CONTEXT_CAPACITY=512`) | cuda | identical |
| molmo2-4b | `molmo2_vision_runtime` | cuda | identical |
| youtu-vl-4b-instruct | `youtu_vl_runtime` | cuda | identical (see 5) |
| gemma-3n-e4b-it-4bit | `gemma3n_audio_runtime` | local-task | identical |

Coexistence: on qwen3-vl-2b-instruct-4bit, main and branch binaries were alternated against the same cache root. Each reused its own entry (v1 at the old path, v3 in the schema subdirectory), neither hit the downgrade refusal, and greedy output stayed identical.

## 5. Unverified and Open Items

- **Phi4MM.** No checkpoint on this host; covered by unit pins only.
- **Gemma3n audio on CUDA.** Gemma3n's text decode graph fails `iree-compile` on CUDA on main (`failed to distribute`, both e2b and e4b), so its audio runtime was verified on `local-task` only.
- **Planned runs blocked by pre-existing failures on main.** qwen2.5-vl-3b runs out of memory in text prefill at a 16410-token prompt; llava-1.5-7b-4bit has its config rejected for IREE vision; gemma-3-4b hits the request-sized context capacity. None of these paths changed in this PR, but the Qwen2.5-VL and Gemma3 projector contracts were not exercised on a real load.
- **Youtu-VL output quality.** It emits repeated `D` both before and after the change. Byte identity shows the PR did not change it; it does not show the output is correct.
- **Declarations versus graphs.** The family lists are hand-written. The pins catch a removed entry or a changed policy, but nothing checks a list against the MLIR its emitter writes. A new operator class added to an emitter without a matching declaration would still be bound only through the MLIR digest.
- **Single float class per artifact.** The observer records the most frequent floating dtype (ties toward the wider type). Per-tensor dtypes of mixed checkpoints are bound only through the existing `checkpoint_schema_sha256` in the config identity.
- **Pins omit derived fields.** The family pins do not spell `round=` and `dequant=`; those fields are derived from pinned ones and covered by the field-class rejection tests (review LOW, kept).
- **Old v1 entries are never removed.** After an upgrade each auxiliary artifact exists twice in `/tmp` until the cache is cleared.
- **Follow-ups.** #3212 runs the operator micro-oracles against MLX on CUDA, pins `MLX_ENABLE_TF32=0`, and refreshes the post-#923 Qwen VL Q4 evidence; it reuses `iree_identity.rs` so probe reports carry the same contract identity as production. #3248, found in this PR's security review, makes all XLA caches private to the user: every cache, this PR's schema subdirectory included, sits in world-writable `/tmp` without ownership checks, and every manifest field is computable from public inputs, so a manifest proves which artifact it is, not who wrote it. The remaining #932 criteria (shared emitter helpers, CUDA oracles, family gates) stay open.

## 6. Learning Points

- When an input reaches a compiled module through a host-side transform, the MLIR digest cannot see it. Bind that transform into the identity explicitly, and select the implementation and the identity from the same value.
- Deriving contract fields from the value the emitter consumed is stronger than declaring them next to it: the mismatched state cannot be written.
- A downgrade refusal that is correct for one binary becomes an outage when two binary generations share a cache path. Scope the location by schema before turning on a new schema in production.
- A field that cannot be observed should be recorded as unobserved, not omitted, so that a later qualification step has a slot to fill and a reader cannot mistake identity for qualification.

## Related Commits

| Hash | Type | Message |
|---|---|---|
| `666ef90b` | feat | bind operator numeric contracts into production artifacts |
| `5c0bf520` | fix | bind Gemma3n BF16 host dequant rounding into its contract |
| `b322a139` | fix | scope v3 auxiliary caches by schema and spawn version once |
| `3bce42a8` | docs | tidy the v3 auxiliary cache paragraph in environment-variables |
