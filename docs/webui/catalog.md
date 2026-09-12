# Model catalog integration

[한국어](catalog.ko.md)

## Scope and ownership

Issue #1840 adds metadata-only catalog projections over the existing `RouterPool`, plus list, detail, and refresh adapters. It does not create a second model registry or start another provider. `create_router_app_with_authenticated_ui` exposes the adapters behind mandatory API-key authentication for integration tests and the future secure startup path. The ordinary `create_router_app` still leaves them unmounted. Production `--webui` startup and browser security integration belong to #1838 and #1837, respectively; these API paths are not a claim that the production flag is already available.

Discovery remains owned by the router: managed cache, explicit `--models-dir`, and presets retain their existing collision precedence (cache < models directory < preset), aliases, and hidden-entry policy. Select the repository store explicitly with `--models-dir models/mlx`; catalog code does not invent a working-directory-dependent default or a filesystem picker. Listing projects discovered entries without downloading a checkpoint, opening a tokenizer, constructing a provider, or loading weights.

## Consuming the projection

The adapter paths below are relative to the server's future validated API prefix. See [the API schema](api.yaml) and [generated TypeScript declarations](generated/ui-api.d.ts) for complete DTO definitions.

| Request | Behavior |
|---|---|
| `GET /ui-api/v1/catalog` | Filtered list, sorted by opaque catalog ID |
| `GET /ui-api/v1/catalog/{id}` | One visible entry, or a structured not-found error |
| `POST /ui-api/v1/catalog/refresh` | Accept a lifecycle-coordinator operation that explicitly rescans configured sources |

The list accepts `limit` (default 50, maximum 200), the returned `cursor`, `q` (maximum 128 bytes), `source`, `task`, `lifecycle`, `support`, and `completeness`. Cursors are at most 512 bytes. Projection is bounded to 1,000 inventory entries. Pagination is deterministic for an unchanged inventory, not a transactional snapshot spanning concurrent refreshes: retain `server_instance_id` and `snapshot_sequence`, and follow the resnapshot rules in [architecture.md](architecture.md) when state changes.

Use `identity.id` for catalog operations and selection; use `identity.inference_id` for inference requests. A display name is not either identity. Content fingerprints describe bounded filesystem metadata observed when metadata is first projected or explicitly refreshed; they are not cryptographic verification of weight contents and are not recomputed on every poll. Revision and lifecycle fields come from the pool rather than a browser-maintained state machine.

Do not collapse these separate facts into a single “works” badge:

- `complete` describes the local checkpoint file layout; it is not a successful load or tensor-integrity check.
- `metadata.support.architecturally_supported` comes from shared model detection and the architecture registry, not a vendor-heading match.
- `runnable_on_backend` reflects registry support for the compiled backend, not a measured inference result.
- `tested_checkpoint` currently remains false with an explicit reason: the catalog has no per-checkpoint evidence database.
- `lifecycle.state` reflects the current provider lifecycle; advertised pre-load capabilities do not imply `ready`.

`metadata.model_type` is the raw bounded `config.json` string, preserving case and returning `null` with a reason when the value is missing, non-string, oversized, or unreadable. `metadata.declared_architectures` is the raw bounded `architectures` array under the same no-truncation rule. `metadata.architecture` is the resolved mlxcel registry identifier from the shared loader detection authority, not an unvalidated copy of either raw field. The catalog supplies restricted probes to that authority; if distinguishing a variant requires weight headers and no bounded sidecar evidence exists, the result remains unknown with a reason. This includes relevant Gemma 4, Inkling, and Kimi K3 distinctions rather than guessing a text variant.

Capabilities carry a phase and an unavailable reason. Image input is confirmed from the existing provider after readiness; metadata alone does not enable image submission. Non-chat output tasks remain separate from chat. Unknown parameter counts and memory estimates remain `null` with reasons, never zero. Disk bytes describe files, not process RSS or allocator memory.

## Filesystem and refresh boundaries

Config and classification sidecars use a 256 KiB read limit; SafeTensors index JSON uses 512 KiB. Catalog probes do not read SafeTensors headers or payloads. Disk accounting uses a maximum of 4,096 visited/pending entries and depth 8, skips symlinks, and reports `null` with a reason when it cannot finish within its limits. Nested `1_Pooling/config.json` evidence is accepted only when its parent components are real directories, not symlinked parents.

Metadata projection runs on blocking workers. Each router or single-model WebUI context owns its own bounded cache, so one pool's refresh or 1,000-entry eviction cannot invalidate another pool's warm projection. The cache avoids repeating bounded metadata acquisition—recursive disk accounting, config parsing, and content-fingerprint work—on ordinary polling cache hits; lifecycle, revisions, provider-confirmed capabilities, and removal eligibility are still refreshed from current pool state. The first cache fill and explicit refresh perform bounded filesystem inspection. Explicit refresh clears only the owning context's cached metadata and performs the existing router rescan; legacy router reloads also advance the catalog epoch so cached metadata is not permanently stale.

Only one refresh owns execution per server instance while its operation is active. Concurrent requests replay that operation without starting another rescan; a later request after completion may start a new one. `changed_entries` compares entry signatures and includes additions, removals, and detected modifications even when the inventory count is unchanged. Client-facing refresh failures are redacted; diagnostic details stay in server logs.

Removal eligibility is advisory, not permission to unlink a path: only managed cache entries can be eligible, and busy entries are unavailable. The actual removal workflow and operation-boundary checks belong to #1841. Single-model mode uses the cache-aware `single_model_entry_from_state_with_cache(&CatalogProjectionCache, &AppState)` handoff to describe the existing provider and its real inference ID, without registering another provider; it reports read-only removal reasons and projects fresh provider/lifecycle state over cached static metadata. Mounting that accessor in production belongs to #1838.

## Regression coverage

Focused tests cover full producer JSON against schema-validated fixtures, raw model-type and declared-architecture bounds, metadata unknowns, cache epoch invalidation, zero-acquisition HTTP cache hits, same-size config/index edits after explicit refresh, fresh lifecycle/provider projection, per-router cache isolation under concurrent 1,000-entry catalogs, shared detection, symlinked evidence, cached single-provider access, and complete traversal of a 1,000-entry catalog before and after HTTP refresh. These tests are not substitutes for inference regression checks after changing shared detection or for later production/browser security acceptance. See [PR #1868](https://github.com/lablup/mlxcel/pull/1868) for the validation record and environment exceptions.

The final combined gate ran at `808994e353fdab5563966e751ca2c71515d08595`: catalog 32 library + 1 CLI, security 26 and discovery 2 passed; workspace all-target clippy, 40 contract fixtures, structural checks, fmt and diff checks passed. The shared detection/loader path remained unchanged. Both independent reviewers cleared the complete router and single-model cache boundaries at this revision.

`single_model_entry_from_state_with_cache` is synchronous. The #1838 startup owner must retain one per-application `Arc<CatalogProjectionCache>` and invoke this helper inside `tokio::task::spawn_blocking`. Never create a cache per poll or call the uncached convenience accessor in an HTTP handler. Cached-helper/provider transitions are tested here; production single-model route offload and responsiveness are #1838 responsibilities. Router HTTP adapters already offload metadata acquisition.
