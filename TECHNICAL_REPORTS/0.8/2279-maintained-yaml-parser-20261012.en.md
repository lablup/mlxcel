# Technical report: PR #2279 — maintained YAML parser

**Date**: 2026-10-12
**Status**: Verified
**Risk level**: Medium

## Executive summary

The CLI and surgery configuration loaders now use deserialize-only serde-saphyr 1.3.0. This removes serde_yaml and unsafe-libyaml, the dependency chain that blocked CI under RUSTSEC-2024-0449.

## Problem statement

Both configuration paths depended on the unmaintained parser. An advisory suppression would leave the unsupported dependency in the runtime graph. The replacement must preserve document-order language priorities, schema validation, and operation-specific numeric errors.

## Technical decisions

Both loaders retain ordinary YAML merge keys and YAML 1.2 boolean interpretation. Duplicate mapping keys remain errors; a supported language key's duplicate failure maps to DuplicateLanguageCode. The streaming schema rejects unknown fields before they can be misclassified as language keys.

Serde's untagged and internally tagged buffering reads numbers through a typeless representation. serde-saphyr represents allowed non-finite typeless values as canonical strings, so the loaders opt into that representation and narrowly recover .nan, .inf, and -.inf. Surgery's existing validators still reject non-finite factors, coefficients, and ratios. Ordinary quoted surgery numbers remain invalid. Deliberate compatibility change: quoted canonical special values now use the same handling as unquoted values; parser diagnostic formatting also changes.

## Change summary

The lockfile removes both unsupported packages and adds only the replacement parser's dependency tree. Regression tests cover duplicate schema fields, ordered numeric entries, boolean spellings, non-finite operation validation, and all six shipped surgery YAML examples. Existing public signatures and model execution paths are unchanged.

## Validation

All 34 focused language-bias tests and 28 surgery configuration tests pass, including the added factor/alpha/ratio non-finite and quoted canonical regressions. cargo deny check passes advisories, bans, licenses, and sources. Formatting and diff checks pass. Repository contract checks pass, including 123 Python tooling tests and 49 WebUI fixtures. Final tests and CI Clippy, CPU-only link, and cargo-deny were verified on dd36abb9b321fa6eb6c9abaca07f5690ead72a17. The duplicate local CUDA Clippy rebuild was stopped after CI Clippy succeeded.

## References

- Issue #2278
- https://rustsec.org/advisories/RUSTSEC-2024-0449.html
- https://docs.rs/serde-saphyr/1.3.0/serde_saphyr/
