# Technical Report: PR #1606 - feat: add arch JSON recipes registry

**Date**: 2026-09-03
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (the merged PR records focused Rust, CLI, clippy, and registry rebuild validation; the later fix commits also closed a registry-contract mismatch and an artifact-corruption risk)
**Languages**: Rust, Markdown, JSON
**Risk Level**: Medium (`mlxcel arch --json` becomes a downstream contract and commits a large generated snapshot, so schema drift and rebuild safety matter more than the CLI surface alone)

---

## Executive Summary

The recipes site needed a stable machine-readable inventory of model families, but `mlxcel arch` only exposed a human-readable catalog. Downstream automation therefore had no supported way to discover runtime, modality, backend, distributed, drafter, and KV-cache support without scraping prose or duplicating internal enums.

PR #1606 adds `mlxcel arch --json` as that contract, derives one registry entry per `ALL_MODEL_TYPES` variant, commits the first `recipes/registry/0.6.0.json` snapshot plus `CURRENT`, and documents the split between the human catalog and the recipes registry. Two follow-up fix commits then aligned advertised KV modes with the runtime MLA resolver and made `make recipes-registry` update artifacts atomically.

---

## 1. Problem Statement

### 1.1 Background

The repository already had a large internal model taxonomy, but it lived inside Rust enums and capability tables. That was good enough for CLI rendering, yet insufficient for a recipes workflow modeled after static registries such as `recipes.vllm.ai`, where downstream builders expect versioned JSON they can ingest directly.

### 1.2 Existing issues

- `mlxcel arch` was designed for humans, not for recipe builders or static-site generation.
- The canonical architecture facts were spread across `ALL_MODEL_TYPES`, family helpers, distributed flags, and KV-cache capability logic, so external tooling had no single supported source.
- A generated snapshot target that writes directly into tracked files could leave partially rewritten artifacts behind if JSON generation fails mid-command.

### 1.3 Risk of leaving it unfixed

Without a stable exported registry, every downstream consumer would either scrape terminal-oriented output or re-encode the model matrix independently. Both options drift quickly. Worse, if rebuilds can truncate committed registry artifacts on failure, automation may consume a corrupted snapshot while still seeing a nominally updated `CURRENT` pointer.

---

## 2. Technical Review

### 2.1 Registry contract

`src/models/registry.rs` introduces a dedicated serialization layer for the recipes registry:

- `ArchitectureRegistry` carries `mlxcel_version` and a `families` array.
- `ArchitectureFamily` captures stable `id`, display name, category, detection keys, runtimes, modalities, output kind, backend support, tensor/pipeline-parallel flags, drafter support, and KV modes.
- Small enums (`Runtime`, `Modality`, `OutputKind`, `BackendStatus`, `Drafter`, `KvMode`) are all serialized in `snake_case`, which keeps the JSON predictable for downstream static generation.

The important design choice is that registry data is derived from `ALL_MODEL_TYPES` rather than maintained separately. That preserves a single ownership point for family coverage while still exporting a consumable snapshot.

### 2.2 CLI surface

`src/main.rs` adds `arch --json` alongside the existing human-readable `arch` output. The text renderer stays unchanged; JSON is an additive output mode. That separation matters because operators still want the readable catalog, while recipes builders need a deterministic schema.

### 2.3 Snapshot generation

The new `recipes-registry` Make target:

- builds the release CLI,
- extracts the runtime version from `mlxcel --version`,
- writes a versioned JSON snapshot under `recipes/registry/`,
- refreshes `recipes/registry/CURRENT`.

The later fix commit changes this target from direct redirection to a temp-file-and-rename flow. That is the correct hardening step: the snapshot becomes a tracked product artifact, so partial writes are a correctness bug, not a convenience issue.

### 2.4 Reviewer and security corrections reflected in the merge

The merged PR includes two substantive corrections beyond the initial feature commit.

- `fix: align arch registry KV modes with MLA resolver`
  DeepSeek V3 and V3.2 originally advertised quantized KV modes because the generic registry capability table allowed them, but their MLA runtime path downgrades those requests back to `fp16`. The fix removes the misleading modes and adds regression coverage that rejects any registry claim that resolves to `fp16` at runtime.
- `fix: preserve recipes registry on rebuild failure`
  The original Make target redirected JSON straight into the tracked snapshot and could continue into `CURRENT` refresh after a failed write. The fix introduces `set -e`, temp files, a cleanup trap, and rename-on-success semantics so the repository never publishes a half-written snapshot.

These are not cosmetic adjustments. They close the two highest-value gaps in a machine-readable export: contract truthfulness and artifact integrity.

### 2.5 Compatibility and dependencies

- No new crate dependency is added.
- Existing human `mlxcel arch` behavior remains intact.
- The committed snapshot is additive repository data rather than a runtime-breaking change.
- The registry is family-level capability metadata, not a guarantee that every checkpoint/backend pair is qualified; the documentation update makes that distinction explicit.

---

## 3. Technical Decisions

### 3.1 Export from model capabilities instead of maintaining a hand-written catalog

The chosen approach maps every `ModelType` through capability helpers and serializes the result. That keeps the exported registry close to the execution model and avoids a second manual registry that would inevitably drift.

The rejected alternative was to curate a separate recipes JSON by hand or from docs prose. That would have created two sources of truth and delayed every family update until both code and content were edited.

### 3.2 Treat registry IDs as stable product identifiers

`registry_id()` and the `model_type_keys()` mapping make the JSON carry stable per-family identifiers and detection keys, rather than raw enum names only. That matches the intended role-model pattern: downstream pages and recipe builders should bind to exported IDs, not internal Rust spellings.

The trade-off is maintenance overhead whenever new families or alias spellings are added. That overhead is acceptable because the alternative is silent downstream breakage.

### 3.3 Validate exported KV modes against runtime resolution

The follow-up KV-mode fix effectively establishes a rule: the registry may only advertise modes the runtime can actually honor for that family. This is the right boundary. A registry that merely reflects optimistic capability tables is less useful than one grounded in effective runtime behavior.

### 3.4 Make generated artifacts atomic

Generated JSON under version control is part of the product surface. Using temp files and atomic rename acknowledges that fact. The cost is a slightly more complex Make recipe, but it eliminates the much larger risk of broken snapshots after failed rebuilds.

---

## 4. Implementation Details

### 4.1 New source module

`src/models/mod.rs` now exports `registry`, and `src/models/registry.rs` centralizes the registry schema plus capability derivation.

Key implementation seams:

- family-level capabilities are defined in one place,
- `registry_id()` maps public JSON identifiers,
- modality/runtime/backend/drafter/KV helpers normalize the full model taxonomy,
- tests assert uniqueness and parity with existing dispatch contracts.

### 4.2 Tests

`src/main_tests.rs` and `src/models/registry.rs` tests cover:

- CLI JSON emission for `arch --json`,
- unique registry IDs and one-family-per-model-type coverage,
- tensor-parallel and pipeline-parallel parity with execution-time allowlists,
- key acceptance families such as `qwen3`, `whisper`, and rerank-only classifiers,
- rejection of KV modes that runtime resolution downgrades to `fp16`,
- MLA latent-cache families remaining `fp16`-only in the exported contract.

That final class of tests is especially valuable because it prevents a future "looks supported in JSON, silently downgraded in runtime" regression.

### 4.3 Documentation and committed artifacts

- `README.md` now distinguishes `mlxcel arch` from `mlxcel arch --json`.
- `docs/supported-models.md` explains what the registry includes and what it does not promise.
- `recipes/registry/0.6.0.json` and `recipes/registry/CURRENT` become the first committed recipes-facing artifacts.

---

## 5. Validation

The PR body records these checks:

| Check | Result |
|---|---|
| `cargo fmt --check` | passed |
| `cargo test --lib registry --no-default-features` | passed |
| `cargo test --bin mlxcel arch --no-default-features` | passed |
| `cargo check --lib --tests --no-default-features` | passed |
| `cargo clippy --lib --tests --no-default-features -- -D warnings` | passed |
| `make -n recipes-registry` | passed |
| `make recipes-registry` | passed |
| `diff -u /tmp/mlxcel-1606-arch.json recipes/registry/0.6.0.json` | passed |
| JSON parse/assert script | passed |
| Simulated non-zero `arch --json` failure path | tracked registry artifacts stayed unchanged |

This validation is aligned with the real risk surface. It checks both the exported schema and the rebuild mechanics, instead of stopping at unit serialization.

---

## 6. Learning Points

**A generated catalog becomes a public API as soon as other tooling consumes it.** Once the recipes workflow depends on the file, field names, IDs, and capability semantics need the same discipline as CLI flags or HTTP responses.

**Capability export should follow effective runtime behavior, not abstract optimism.** The DeepSeek V3/V3.2 correction is a concrete example: if runtime downgrades a mode, the registry must say so.

**Large generated artifacts need failure semantics, not just success semantics.** The atomic Makefile rewrite prevents stale or partial state from masquerading as a successful rebuild.

---

## 7. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 9 |
| Lines added | 5977 |
| Lines removed | 12 |
| Commits | 3 |

### Related commits

- `3019b63` feat: add arch JSON recipes registry
- `5fbf2b3` fix: align arch registry KV modes with MLA resolver
- `d7acb2f` fix: preserve recipes registry on rebuild failure

### Files of interest

| File | Change |
|---|---|
| `src/models/registry.rs` | Adds the registry schema, capability mapping, and contract tests |
| `src/main.rs` | Adds `arch --json` |
| `src/main_tests.rs` | Covers CLI JSON behavior |
| `Makefile` | Adds and hardens `recipes-registry` rebuild logic |
| `recipes/registry/0.6.0.json` | First committed registry snapshot |
| `recipes/registry/CURRENT` | Tracks the active snapshot version |
| `README.md` | Documents the machine-readable mode |
| `docs/supported-models.md` | Clarifies registry scope and limits |

---

## 8. Follow-up Actions

- Treat new family additions as registry-surface changes and extend `registry_id()` / capability tests in the same patch, not later.
- Consider documenting the registry schema in a dedicated recipes-facing page if external tooling begins depending on exact fields.
- Add a lightweight schema-compatibility check if future versions need additive evolution across multiple committed snapshot files.
