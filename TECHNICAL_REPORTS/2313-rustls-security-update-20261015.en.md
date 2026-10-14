# Technical Report: PR #2313 — Update rustls for four advisories

**Date:** 2026-10-15
**Issue:** #2312
**Scope:** Lockfile-only dependency security patch; no manifest, feature, provider, API, error-behavior, or inference change.

## Summary

The original PR #2313 source commit `e9604cba` updated the sole locked `rustls` package from 0.23.45 to 0.23.46, the upstream patched release for RUSTSEC-2026-0346 through RUSTSEC-2026-0349. Its complete product diff was the rustls version and registry checksum in `Cargo.lock`; 108 other resolved packages, manifests, and the existing provider feature selection remained unchanged.

**Current status:** The same rustls 0.23.46 lock update landed on `main` through PR #2304. After incorporating that merged state, PR #2313 preserves this review and test record as report-only documentation and introduces no additional product change.

No user documentation changes are needed because the patch adds no configuration, command-line option, API, migration, or observable error-contract change. The security correction is supplied by the compatible upstream patch release.

## Advisory assessment

| Advisory | Upstream issue | mlxcel qualification |
|----------|----------------|----------------------|
| RUSTSEC-2026-0346 / GHSA-c8m5-c952-6cm4 | Derived `Debug` output can expose TLS session secrets through a custom client session store | mlxcel has no custom `ClientSessionStore`; updating still removes the vulnerable implementation |
| RUSTSEC-2026-0347 / GHSA-vp88-798v-h67f | Rejected opt-in ECH can disclose buffered application data to the authenticated outer server | mlxcel does not configure ECH |
| RUSTSEC-2026-0348 / GHSA-h2v8-394m-p7vv | Unauthenticated Brotli certificate decompression can allocate up to 1 GiB | The resolved rustls package does not enable Brotli |
| RUSTSEC-2026-0349 / GHSA-w44r-93j7-qv28 | An unauthenticated ServerHello can poison a per-name key-share hint and downgrade a later resumable connection to classical exchange | Outbound clients make the class relevant, but the current rustls feature graph selects the `ring` provider and does not demonstrate product-level post-quantum coverage or loss; no authentication bypass or key compromise is claimed |

All four advisories identify rustls 0.23.46 as patched. No advisory ignore or policy exception was added.

## Validation

The validation below records the original `e9604cba` source and dependency graph; it remains historical evidence rather than a claim that the current report-only diff changes the rustls pin.

- `cargo deny check` passed locally for advisories, bans, licenses, and sources; the hosted `cargo-deny` job also passed on PR #2313.
- `cargo metadata --locked --no-deps --format-version 1` passed and reported the five workspace packages without modifying the lockfile.
- `cargo tree --locked -i rustls -e features` resolved one rustls 0.23.46 package. The feature graph retains the explicit `ring`, `std`, `tls12`, and logging selection and does not enable an alternative rustls crypto provider.
- Parsed diff inspection confirms a two-field lockfile change: version `0.23.45` → `0.23.46` and checksum replacement. `git diff --check` passed.
- `MLXCEL_BUILD_METAL=OFF RUSTC_WRAPPER= cargo test --locked --profile test-fast --no-default-features --lib tls_tests:: -- --test-threads=1` passed 10 tests with none ignored. This covers the existing TLS configuration, certificate, key, and plaintext-fallback guards.
- The equivalent `downloader::tests::` run passed 71 tests and ignored one optional integration test. No live external download was performed.
- An initial default-macOS build of `e9604cba` reached MLX's unrelated `gated_delta_update_nax.metal` MPP compilation failure before running tests. The authoritative focused runs explicitly disabled the unrelated Metal backend and completed with nonzero counts; this failure is not evidence about the rustls pin now on `main`.
- Independent implementation and security review found no blocking issue in the lock-only patch or its qualified advisory wording. Hosted CI had already passed cargo-deny and the repository contract checks observed during finalization; remaining CI status is checked separately before merge.

## Risk and limits

Patch-release resolution risk is low because Cargo selected no transitive update and no source or manifest changed. The four advisory exploits were not independently reproduced; the security behavior comes from the verified upstream release and advisory records. The ring-only rustls provider result is a dependency qualification, not proof that mlxcel previously offered hybrid post-quantum exchange. No inference checkpoint, GPU path, performance benchmark, ECH flow, Brotli certificate, custom session store, or live external downloader TLS session was exercised for this report.

## References

- Issue #2312 and PR #2313.
- rustls 0.23.46 release: https://github.com/rustls/rustls/releases/tag/v/0.23.46
- RustSec records: RUSTSEC-2026-0346, RUSTSEC-2026-0347, RUSTSEC-2026-0348, and RUSTSEC-2026-0349.
