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

Use `identity.id` for catalog operations and selection; use `identity.inference_id` for inference requests. A display name is not either identity. Content fingerprints describe filesystem metadata changes, not cryptographic verification of weight contents. Revision and lifecycle fields come from the pool rather than a browser-maintained state machine.

Do not collapse these separate facts into a single “works” badge:

- `complete` describes the local checkpoint file layout; it is not a successful load or tensor-integrity check.
- `metadata.support.architecturally_supported` comes from shared model detection and the architecture registry, not a vendor-heading match.
- `runnable_on_backend` reflects registry support for the compiled backend, not a measured inference result.
- `tested_checkpoint` currently remains false with an explicit reason: the catalog has no per-checkpoint evidence database.
- `lifecycle.state` reflects the current provider lifecycle; advertised pre-load capabilities do not imply `ready`.

`metadata.model_type` and `metadata.architecture` are resolved detection/registry identifiers, not an unvalidated copy of `architectures[0]`. The catalog uses the same dispatch authority as loading, but supplies restricted probes. If distinguishing a variant requires weight headers and no bounded sidecar evidence exists, the result remains unknown with a reason. This includes relevant Gemma 4, Inkling, and Kimi K3 distinctions rather than guessing a text variant.

Capabilities carry a phase and an unavailable reason. Image input is confirmed from the existing provider after readiness; metadata alone does not enable image submission. Non-chat output tasks remain separate from chat. Unknown parameter counts and memory estimates remain `null` with reasons, never zero. Disk bytes describe files, not process RSS or allocator memory.

## Filesystem and refresh boundaries

Config and classification sidecars use a 256 KiB read limit; SafeTensors index JSON uses 512 KiB. Catalog probes do not read SafeTensors headers or payloads. Disk accounting uses a maximum of 4,096 visited/pending entries and depth 8, skips symlinks, and reports `null` with a reason when it cannot finish within its limits. Nested `1_Pooling/config.json` evidence is accepted only when its parent components are real directories, not symlinked parents.

Metadata projection runs on blocking workers. A bounded cache avoids repeating recursive disk accounting on ordinary polling; filesystem fingerprints still check relevant metadata, while lifecycle, revisions, and removal eligibility are refreshed from current pool state. This is not a promise of zero filesystem calls on cache hits. Explicit refresh clears cached metadata and performs the existing router rescan.

Only one refresh owns execution per server instance while its operation is active. Concurrent requests replay that operation without starting another rescan; a later request after completion may start a new one. `changed_entries` compares entry signatures and includes additions, removals, and detected modifications even when the inventory count is unchanged. Client-facing refresh failures are redacted; diagnostic details stay in server logs.

Removal eligibility is advisory, not permission to unlink a path: only managed cache entries can be eligible, and busy entries are unavailable. The actual removal workflow and operation-boundary checks belong to #1841. Single-model mode uses `single_model_entry_from_state(&AppState)` to describe the existing provider and its real inference ID, without registering another provider; it reports read-only removal reasons. Mounting that accessor in production belongs to #1838.

## Regression coverage

Focused tests cover full producer JSON against schema-validated fixtures, metadata unknowns, cache invalidation and fresh lifecycle projection, shared detection, symlinked evidence, single-provider access, and complete traversal of a 1,000-entry catalog before and after HTTP refresh. These tests are not substitutes for inference regression checks after changing shared detection or for later production/browser security acceptance. See [PR #1868](https://github.com/lablup/mlxcel/pull/1868) for the validation record and environment exceptions.
