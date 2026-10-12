# Technical report: PR #2279 — maintained YAML parser

**Date**: 2026-10-12
**Status**: Pending final verification
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

All 34 focused language-bias tests and 26 surgery configuration tests pass. cargo deny check passes advisories, bans, licenses, and sources. Formatting and diff checks pass. Repository contract checks pass, including 123 Python tooling tests and 49 WebUI fixtures. The validated source commit is 2fd0ac73b13af0b8e638f2ac401eed505bf7c4cd. Review added explicit alpha/ratio non-finite regressions and quoted canonical special cases after this test run. Execution of those added cases and Clippy verification were pending when this report was prepared; the final PR body records their outcomes.

## References

- Issue #2278
- https://rustsec.org/advisories/RUSTSEC-2024-0449.html
- https://docs.rs/serde-saphyr/1.3.0/serde_saphyr/
