# Technical Report: PR #1608 - fix: include RT-DETRv2 in arch JSON registry

**Date**: 2026-09-03
**Author**: mlxcel maintainers
**Reviewer**: implementation, security, and finalization review cycle
**Status**: Completed (focused Rust checks and hosted CI passed)
**Languages**: Rust, JSON, Markdown
**Risk Level**: Medium (the registry is a downstream recipes contract, and an inaccurate runtime claim can generate unusable commands)

---

## Executive Summary

PR #1608 corrects a coverage gap left by PR #1606. The initial `mlxcel arch --json` registry was derived only from `ALL_MODEL_TYPES`, which represents text, VLM, embedding, reranking, speech, and audio model loaders. RT-DETRv2 is supported through the separate `mlxcel detect` command, so that derivation silently omitted a real runtime surface.

The correction adds a deterministic extension point for standalone architecture families and registers `rt_detr_v2` with exactly the capabilities the command implements: `detect` runtime, `image` input, `boxes` output, and `detection` category. It deliberately does not advertise generation, serving, distributed execution, or speculative decoding. A security-review follow-up also adds invariants that prevent standalone entries from colliding with loadable family IDs or model-type keys.

---

## 1. Problem Statement

### 1.1 Background

The recipes catalog consumes the versioned architecture registry to decide which modes and command forms it may offer. PR #1606 made that data machine-readable, but its one-entry-per-`ALL_MODEL_TYPES` construction assumed every supported runtime was represented by the main model loader enum.

RT-DETRv2 violates that assumption by design. Object detection has a dedicated predictor and CLI handler because it accepts images and emits bounding boxes rather than tokens. Treating it as a normal generation model would be misleading, but leaving it out makes the registry incomplete.

### 1.2 User-visible impact

- Recipe builders could not discover any `detect` runtime.
- No registry entry exposed the `detection` category or `boxes` output.
- A valid RT-DETRv2 recipe could not be validated against the exported capability catalog.
- Adding RT-DETRv2 to `ModelType` merely to satisfy the registry would risk false `generate` or `serve` support.

---

## 2. Change Summary

### 2.1 Standalone architecture families

`src/models/registry.rs` now defines a small standalone-family descriptor for command runtimes that do not pass through the text/VLM model loader. `build_architecture_registry()` preserves the existing `ALL_MODEL_TYPES` order, then appends standalone entries in a fixed declared order.

The first standalone entry is `rt_detr_v2`:

| Field | Value |
|---|---|
| `id` | `rt_detr_v2` |
| `model_types` | `rt_detr_v2` |
| `runtimes` | `detect` |
| `modalities_in` | `image` |
| `output` | `boxes` |
| `category` | `detection` |
| TP / PP | disabled |
| drafters | none |

The entry does not claim `generate` or `serve`, matching the actual `mlxcel detect` boundary.

### 2.2 Collision invariants

The security review identified a maintenance risk in the extension point: a later standalone entry could reuse an existing family ID or detection key and silently shadow a loadable family in downstream maps. The follow-up test constructs the exported identity sets and rejects:

- duplicate standalone IDs,
- standalone IDs that collide with `ALL_MODEL_TYPES` registry IDs,
- duplicate standalone model-type keys,
- standalone keys that collide with loadable family keys.

### 2.3 Snapshot and documentation

The release CLI regenerated `recipes/registry/0.6.0.json`, adding the detector entry without changing the schema. `README.md` and `docs/supported-models.md` now state that the registry includes both loader-backed families and standalone command runtimes.

---

## 3. Technical Decisions

### 3.1 Keep detector runtimes outside `ModelType`

`ModelType` drives text/VLM loading and capability dispatch. Adding RT-DETRv2 there solely for registry enumeration would couple an image-to-box predictor to token-generation assumptions and increase the chance of false runtime advertising.

The standalone descriptor keeps runtime ownership truthful while still providing one unified JSON document to recipes consumers.

### 3.2 Preserve deterministic ordering

The original loader-backed entries retain their `ALL_MODEL_TYPES` order. Standalone families are appended from a static slice rather than a map, so repeated builds remain byte-stable and committed snapshots stay reviewable.

### 3.3 Extend counts rather than redefine loader coverage

Registry family count is now the number of loader-backed families plus standalone runtime families. Tests express that formula explicitly. This preserves the useful exhaustiveness check for `ALL_MODEL_TYPES` while acknowledging supported commands outside that enum.

### 3.4 Fail tests on ambiguous public identities

Downstream recipe indexes commonly key by family ID and model-type key. Rejecting collisions at test time prevents last-writer-wins behavior and turns an otherwise subtle catalog corruption into an immediate development failure.

---

## 4. Validation

| Check | Result |
|---|---|
| `cargo fmt --check` | passed |
| `cargo test --lib registry --no-default-features` | passed |
| `cargo test --bin mlxcel arch --no-default-features` | passed |
| `cargo check --lib --tests --no-default-features` | passed |
| `cargo clippy --lib --tests --no-default-features -- -D warnings` | passed |
| `make recipes-registry` | passed |
| Release JSON versus committed snapshot | byte-identical |
| RT-DETRv2 capability assertions | passed |
| Standalone ID/key collision assertions | passed |
| Hosted required checks | passed |

The release registry and committed snapshot shared SHA-256 `3360ff0554365c702e9eb501d85c0ec5ae8d4dd7aadddcdc4059be81efcdfddf` at finalization.

---

## 5. Change Statistics

| Metric | Value |
|---|---|
| Files changed | 5 |
| Lines added | 204 |
| Lines removed | 35 |
| Commits | 2 |

Related commits:

- `0ab5509` fix: include RT-DETRv2 in arch JSON registry
- `9cd2806` test: guard standalone registry extension collisions

Files of interest:

- `src/models/registry.rs`: standalone descriptors, registry assembly, capability and collision tests
- `src/main_tests.rs`: CLI-level family-count and detector assertions
- `recipes/registry/0.6.0.json`: regenerated public snapshot
- `README.md`, `docs/supported-models.md`: clarified registry scope

---

## 6. Learning Points

**Runtime catalogs must enumerate command boundaries, not only loader enums.** A single enum is a useful source of truth only when it actually owns every supported execution path.

**Truthful omission is better than capability inflation.** The detector belongs in the registry, but it must not inherit generation or serving capabilities it does not implement.

**Extension points need identity invariants immediately.** Static data can still collide, and downstream JSON consumers often turn collisions into silent overwrites.

---

## 7. Follow-up Actions

- Add future standalone runtimes through the same static descriptor and collision tests.
- Keep recipe builders tolerant of additive families while treating existing IDs and field meanings as stable.
- Revalidate backend support when RT-DETRv2 receives backend-specific qualification data.

