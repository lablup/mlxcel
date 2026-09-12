# Developing the bundled WebUI

[한국어](bundling.ko.md)

## Scope

Issue #1836 supplies a minimal React shell, reproducible static assets, and a reusable Rust router. It does **not** enable a production `--webui` CLI flag or model-management API. Startup mounting belongs to #1838; browser authentication belongs to #1837; operational pages follow in the dependent issues of epic #1834. The shell labels planned capabilities rather than claiming they are already operational.

The frontend lives in `webui/`, independently of the existing `webpage/` site. Shipped assets live in `src/webui/assets/`. Cargo embeds those committed files; it neither runs Node nor downloads frontend dependencies. The optional `webui` Cargo feature is enabled by default. `rust-embed` uses `debug-embed`, so development and release artifacts both contain asset bytes rather than reading a source checkout at runtime.

## Canonical build pipeline

Run commands from the repository root, with Python 3, Node **26.5.1**, and pnpm **11.18.0** installed:

```bash
pnpm --dir webui install --frozen-lockfile
pnpm --dir webui run typecheck
pnpm --dir webui run lint
pnpm --dir webui run unit
python3 scripts/webui/build_bundle.py
make verify-webui-bundle
```

The Python entry point checks the exact Node/pnpm versions, runs Vite, writes the manifest, and validates the result. Use it after changing frontend source or build inputs, then commit the corresponding assets and manifest together. `pnpm --dir webui run build` alone is the lower-level Vite step: it does **not** produce a complete manifest-bearing shipping bundle. Do not use that command alone to regenerate committed assets.

`make verify-webui-bundle` (equivalently `pnpm --dir webui run verify-generated`) performs two clean temporary builds, compares their bytes, and compares the result with the checked-in bundle. It rejects missing files, stale source digests, asset-hash drift, and exceeded size budgets. It does not install dependencies. Set `WEBUI_BUNDLE_PY=/path/to/python3` when a specific Python interpreter is needed.

The manifest records pinned package/toolchain versions, a source digest, per-file SHA-256 and byte/gzip sizes, and bundle budgets. It contains no timestamps or absolute build paths. Generated assets exclude source maps; HTML uses relative asset URLs; no runtime CDN, service worker, SSR, or Node service is required. React, React DOM, and scheduler license text ships in `third-party-licenses.txt`, with repository attribution in `NOTICE`.

| Budget | Maximum |
|---|---:|
| Initial JavaScript, gzip | 200 KiB |
| Total JavaScript, gzip | 700 KiB |
| All embedded assets, including manifest | 5 MiB |

Read `src/webui/assets/mlxcel-webui-manifest.json` for current measurements; do not copy a previous bundle's measurements into a new PR. Later chat/visualization owners must preserve these limits with lazy loading when necessary.

## Rust feature and router boundary

```bash
# Apple Silicon shipping build: includes the WebUI feature by default.
cargo build --release --features metal,accelerate
# Omit WebUI while retaining the normally default-on surgery feature.
cargo build --release --no-default-features --features metal,accelerate,surgery
# Explicitly select WebUI without other default features.
cargo build --release --no-default-features --features metal,accelerate,webui
# Linux/CUDA uses its existing backend prerequisite/toolchain.
cargo build --release --features cuda
```

`mlxcel::server::webui::router::<S>()` returns a router whose routes already include `/webui`. Merge it at the root, or nest it once under the **validated server API prefix**, not under a second `/webui`. The parent router retains its root health and inference routes. `/webui` redirects to `/webui/`; the shell's `#models`-style navigation stays client-side. A missing static path returns 404 rather than an HTML fallback that could hide an API-routing error.

Static responses support GET/HEAD and conditional ETags; other methods return 405 with `Allow: GET, HEAD`. HTML and the manifest revalidate; content-hashed assets may be cached immutably. The router rejects encoded/traversal path spellings and sets same-origin CSP, `nosniff`, and no-referrer headers on ordinary, error, redirect, and 304 responses. This static policy does not replace the later Host/Origin/authentication checks on administrative APIs.

## Testing without production startup

```bash
pnpm --dir webui exec playwright install chromium
pnpm --dir webui run browser
cargo test --profile test-fast --features metal,accelerate server::webui::assets::assets_tests
cargo run --example webui_static_harness --features metal,accelerate
```

The browser command starts a Vite preview server and checks the scaffold. It is **not** a production-server, Safari, authentication, or real-model acceptance test. The static harness prints its loopback address and serves `/webui/` alongside stub health/model routes. It loads no checkpoint and does not implement `mlxcel-server --webui`. `MLXCEL_WEBUI_HARNESS_ADDR` can select a local test address.

Relocated-artifact validation must separately build development and release harness artifacts, move them outside the checkout, make source asset directories and `node_modules` inaccessible, and prohibit external network access while retaining loopback HTTP. Fetch the shell, manifest, JavaScript, and CSS from each moved artifact. Account for existing MLX dynamic-library/Metal-resource requirements separately: proving embedded WebUI independence does not prove that all MLX runtime resources are embedded. Record actual execution evidence and unavailable feature/backend gates rather than treating these instructions as completed tests.
