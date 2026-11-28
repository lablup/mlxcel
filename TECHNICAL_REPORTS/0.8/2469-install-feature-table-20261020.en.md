# PR #2469: List every declared Cargo feature in the installation feature table

**Date**: 2026-10-20
**Status**: Implemented; PR open
**Risk**: Low (documentation and one manifest comment; no code or feature semantics change)

## Summary

Closes #1705. The Cargo feature table in `docs/installation.md` documented 9 of the 16 features that `Cargo.toml` declares. The seven missing ones are user-reachable: `webui` is on by default, `voicechat-mic` gates an example, and the five `xla-*` diagnostics features gate the oracle and reference-check examples. The PR adds a row for each, condensed from the feature's existing Cargo.toml comment, corrects the prose that still described the default set as `surgery` alone, and gives the one uncommented `[[example]]` block in the manifest (`xla_aux_smoke`) the explanatory comment every sibling already had.

## Problem

A reader of the installation page could not learn that `webui`, `voicechat-mic`, `xla-reference-diagnostics`, `xla-diagnostics`, `xla-diagnostics-cpu`, `xla-diagnostics-rocm` or `xla-micro-oracle` exist, what they pull in, or which example and test targets need them. The text under the table also said `default = ["surgery"]`, which has been wrong since `webui` joined the default set (#1836): a plain `cargo build` embeds the WebUI bundle, and `--no-default-features` drops it, neither of which the page said. Issue #1705 was filed against a 10-feature manifest with three rows missing; the manifest has grown since, so the gap was seven rows when this landed.

## Technical decisions

- Rows are condensed from the Cargo.toml comments rather than written fresh, so the two descriptions cannot drift in meaning. Each row names the feature's composition (`xla-iree` plus `mlxcel-xla/diagnostics`, `cuda` plus `xla-diagnostics-cpu`, and so on) and the `[[example]]` targets gated by it, taken from the `required-features` entries in the manifest.
- Row order follows the manifest's grouping rather than alphabetical order: `webui` sits next to `surgery` (the two defaults), the `xla-*` diagnostics rows follow `xla-iree` (which they all build on), and `voicechat-mic` closes the table after `test-utils` (both are example and test scaffolding).
- The intro sentence "Shipping builds enable only the platform backend flags" was already untrue with `surgery` on by default; it now says shipping builds enable the defaults plus the backend flags.
- The companion `xla_aux_smoke` comment is derived from the example's `main` and `mlxcel_xla::aux_smoke` (a real-runtime acceptance probe for the generic auxiliary ABI), and only says what the example does and why it builds only under `xla-iree`. The two other example blocks without a leading comment (`voicechat_microphone`, `webui_static_harness`) were outside the issue's scope and are left as they are; their features carry the explanation.

## Changes

- `docs/installation.md`: seven new rows in the feature table, the intro and the default-features paragraph updated, the `--no-default-features` example comment now mentions the WebUI assets.
- `Cargo.toml`: a five-line comment above `[[example]] name = "xla_aux_smoke"`. No manifest semantics change.

## Verification

- `cargo metadata --no-deps --format-version 1` parses the edited manifest.
- A one-off script read the `[features]` block of `Cargo.toml` and compared it with the table: 16 non-default features, 16 rows, nothing missing and nothing undeclared.
- Every relative link added to the table (`webui/bundling.md`, `nemotron-voicechat.md`) resolves to an existing file under `docs/`.
- No Rust source changed, so the fmt, clippy and test gates are unaffected.

## Follow-ups

- The table is maintained by hand. A check in the shape of `scripts/ci/check_kernel_dtype_keys.py` that diffs the `[features]` block against the table would keep this from recurring; it was not added here because the issue is documentation-only and the repository's docs checks do not yet have such a hook.

## Statistics

| Item | Value |
|---|---|
| Files changed | 2 |
| Lines added | 21 |
| Lines deleted | 8 |
| Tests added | 0 |

## Related

- Issue #1705: docs(install): the Cargo feature table omits three declared features.
- Issue #1836: the WebUI epic that added `webui` to the default feature set.
- Issue #1376: the Nemotron VoiceChat microphone example behind `voicechat-mic`.
