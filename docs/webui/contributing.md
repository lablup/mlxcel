# Contributing to the WebUI

[한국어](contributing.ko.md) | [WebUI documentation index](README.md)

This page is the map for changing the bundled WebUI. The linked documents are the authorities; this page does not restate their contracts.

## Where things live

| Path | What it is | Edited by hand? |
|---|---|---|
| `webui/` | The React and TypeScript source of the WebUI, its tests, and its pinned `package.json` / `pnpm-lock.yaml`. | Yes |
| `src/webui/assets/` | The generated shipping bundle that Cargo embeds into the binaries, with `mlxcel-webui-manifest.json` and `third-party-licenses.txt`. | No, regenerate it |
| `src/server/webui/` | The Rust side: static router, security wrapper, typed UI API adapters. | Yes |
| `docs/webui/api.yaml` | The machine contract for `{api_prefix}/ui-api/v1`. | Yes, contract first |
| `docs/webui/generated/ui-api.d.ts` | TypeScript declarations generated from `api.yaml`. | No, regenerate it |
| `tests/fixtures/webui/` | Executable contract examples and the English/Korean string catalog fixture shared by Rust and frontend tests. | Yes |
| `webpage/` | The marketing website (Next.js). It is not the WebUI, is not embedded in any binary, and shares no code with `webui/`. | Separate project |

## Change the contract first

An API change lands in one PR that updates the producer schema (`docs/webui/api.yaml`), the generated declarations, the fixtures, and at least one consumer. `python3 scripts/ci/check_webui_contract.py` is the drift gate (`--fix` regenerates only the declarations), and `make verify-webui-contract` runs it with its self-tests; point `WEBUI_CONTRACT_PY` at the isolated environment described in [architecture.md](architecture.md#running-the-contract-gate). The ownership and change rules are in [architecture.md](architecture.md); lifecycle and replay rules are in [state-machines.md](state-machines.md).

## Use the shared modules

- `webui/src/api` is the only transport layer. Pages never build backend URLs or their own fetch wrappers.
- `webui/src/state` is the only state authority (`WebUiProvider`, `useWebUi()`, `useWebUiActions()`). Pages do not add their own polling timers, persistence, or model caches.
- `webui/src/design-system` wraps `@lablup/ui-common`; feature pages import its primitives, never the package directly. See [ui-common.md](ui-common.md) and [design-system.md](design-system.md).
- Every user-facing string has a key with English and Korean values in a `strings.ts` next to its feature (`webui/src/features/*/strings.ts`, combined by `webui/src/i18n/catalog.ts`), mirrored in `tests/fixtures/webui/strings.json` ([ux-contract.md](ux-contract.md)).

Page-specific behavior is documented per screen: [catalog.md](catalog.md) (Models), [chat.md](chat.md), [activity.md](activity.md), [settings.md](settings.md), and [runtime.md](runtime.md).

## Build the frontend and regenerate assets

The toolchain is pinned exactly: Node 26.5.1 and pnpm 11.18.0. From the repository root:

```bash
pnpm --dir webui install --frozen-lockfile
pnpm --dir webui run typecheck
pnpm --dir webui run lint
pnpm --dir webui run unit
python3 scripts/webui/build_bundle.py
make verify-webui-bundle
```

`build_bundle.py` checks the toolchain versions, builds, writes the manifest, and validates budgets. Commit the regenerated `src/webui/assets/` together with the source change. `make verify-webui-bundle` rebuilds twice from clean and fails on any byte difference from the committed assets, a stale source digest, or an exceeded budget. `pnpm --dir webui run build` alone is not a shipping build. Details: [bundling.md](bundling.md).

## Rust builds stay offline

Cargo embeds the committed files in `src/webui/assets/`; it never runs Node or downloads frontend packages, so a Rust-only contributor needs no JavaScript toolchain. The UI is behind the `webui` Cargo feature, which is on by default:

| Build | WebUI |
|---|---|
| `cargo build --release --features metal,accelerate` | Included (default features) |
| `cargo build --release --features cuda` | Included (default features) |
| `cargo build --release --no-default-features --features metal,accelerate,surgery` | Omitted; `--webui` then fails startup with the "requires a binary built with the Cargo `webui` feature" message |
| `cargo build --release --no-default-features --features metal,accelerate,webui` | Included, without other default features |

## Gates

`make verify-webui` runs the CPU-safe aggregate gate: contract drift, verifier helper tests, frontend type/lint/unit and browser checks, bundle reproducibility, and the scoped Rust WebUI tests. Installed-artifact, reverse-proxy, startup, real-model, and CUDA evidence are separate targets listed in the [integration matrix](../webui-integration-matrix.md), which also says which of them a release needs. Rust changes still go through the repository-wide `test-fast`, clippy, and fmt gates.

## Licenses

Frontend runtime dependencies that end up in the bundle must have their license text in `webui/public/third-party-licenses.txt` (copied into the bundle) and an entry in the top-level `NOTICE`. A file adapted from another project carries a header comment naming the source, as `webui/src/design-system/themes/glass.css` does. Do not add font files, Apple design-kit assets, SF Symbols, or icons you did not author; the WebUI uses system fonts and project-authored SVG icons.
