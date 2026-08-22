# Technical Report: PR #1269 - Deterministic language priority from a YAML bias block

## Executive Summary

`LangBiasSet.ordered` is documented as "pairs in priority order (index 0 = highest priority)" and its consumer `to_token_bias` resolves shared tokens under "first-language-wins". The YAML config path built that order by iterating a `HashMap`, so `RandomState` decided the priority. Han script is shared by `ja`, `zh` and `ko`, which means a `--lang-bias-config` file naming two or more CJK languages assigned a different bias to every shared token on every run, silently.

This is the same root-cause class as #1265 (`HashMap` iteration order leaking into ordered state), but in production code rather than a test fixture. The fix collects the `bias:` block through `MapAccess` into an ordered `Vec`, so the author's document order becomes the priority order, matching what `--lang-bias` and the `LLAMA_ARG_LANG_BIAS` fallback have always done.

## 1. Problem Statement

Three entry points produce a `LangBiasSet` and only one of them was wrong.

`parse_lang_bias_entries` walks `s.split(',')` in document order, uses a `seen` map purely as a membership set, and rejects duplicates with `CliError::DuplicateLanguageCode`. `env_fallback_lang_bias` routes `LLAMA_ARG_LANG_BIAS` through that same parser. The YAML path deserialized `bias:` into `Option<HashMap<String, BiasValueStr>>` and pushed the map's iteration order straight into `ordered`.

Two consequences followed, and the second was invisible. Priority became random per run. And because `serde_yaml` resolves a repeated key into a typed `HashMap` last-wins with no diagnostic, the duplicate check in the resolve loop was unreachable, so the YAML path silently accepted input the `--lang-bias` parser has always rejected.

The trigger is not exotic: the schema example in the `LangBiasYamlConfig` doc comment is itself a three-CJK config, so copying the documented example was enough.

## 2. Technical Decisions

### 2.1 A hand-written `Deserialize` rather than an order-preserving map type

Three candidates were considered. `IndexMap` as the field type preserves order but resolves duplicates last-wins, which would leave the dead check dead and keep the YAML path accepting what the CLI rejects. `serde_yaml::Mapping` is `IndexMap`-backed and does reject duplicates, but with serde_yaml's own error rather than `CliError::DuplicateLanguageCode`, so the two entry points would still disagree on the error surfaced.

The chosen shape is a `BiasEntries` newtype over `Vec<(String, BiasValueStr)>` with a hand-written `Deserialize` that collects through `MapAccess::next_entry`. Document order survives, a repeated key arrives as a second entry that the existing resolve loop rejects with the repository's own error, and no new dependency is added. `indexmap` is present in `Cargo.lock` but not in `Cargo.toml`, so both map-type candidates would have promoted a transitive dependency to a direct one.

### 2.2 The accepted YAML schema does not change

The visitor implements `visit_map` and the entry point is `deserialize_map`, so `bias:` remains a plain mapping. `#[serde(deny_unknown_fields)]` is untouched, an absent or empty block still resolves to an empty set, and a sequence-shaped block is still a parse error. Existing config files keep working. A fix that required users to rewrite `bias:` as a list would have been the wrong shape for a bug report about ordering.

### 2.3 The duplicate error message names both surfaces

Making the check reachable from YAML made the existing text, which named only `--lang-bias`, actively wrong for a user who wrote a YAML file. The variant and its `code` field are unchanged; only the message broadened to name both surfaces.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/lang_bias.rs` | `BiasEntries` newtype plus its `Deserialize`; `LangBiasYamlConfig::bias` retyped; duplicate error message broadened; new tests |
| `CHANGELOG.md` | `## [Unreleased]` entries for the two user-visible behavior changes |

## 4. Review Findings

The requirement that carried the most weight was not a review finding but a precondition: the regression tests had to be demonstrated failing against the unfixed code. This bug class is unusually good at producing tests that pass before and after, because a single `resolve()` returns the correct order by luck a fair fraction of the time.

The demonstration reverted only the field type (and the one pre-existing test whose membership assertions would not compile against the ordered type), ran the suite, and restored from a copy rather than using `git stash`, so untracked work was never at risk. Four tests failed, and the output showed three distinct permutations of the same three-key file within a single process run:

```
iteration 0: [(Zh, -10.0), (Ko, 5.0), (Ja, -inf)]
iteration 2: [(Zh, -10.0), (Ja, -inf), (Ko, 5.0)]
           : [(Ja, -inf), (Zh, -10.0), (Ko, 5.0)]
```

That independently reproduces the measurement taken while filing #1267: `RandomState` randomizes per `HashMap` instance, not merely per process (ten maps from the same five keys gave nine distinct orders in one process). It is also why each ordering test runs 32 full `resolve()` calls rather than one.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64).

- `cargo test --profile test-fast --features cuda --lib lang_bias`: 30 passed, exit 0. Against the pre-fix field type: 26 passed, 4 failed, exit 101.
- `cargo fmt --all -- --check`, `cargo clippy --lib --tests --features cuda -- -D warnings`, `cargo check --lib --tests --features cuda`, `cargo check --bins --features cuda`: all exit 0.
- `make verify-test-cuda`: recorded in the PR thread.

## 6. Related Work

- #1267: the issue this closes, filed from the review sweep on PR #1268.
- #1265 and PR #1266: the same root-cause class in four test fixtures, and the origin of the sweep.
- #1277 and #1276: two further instances found by the same sweep, in the distributed registry accessors and in the RT-DETRv2 checkpoint layout sniffer.

Four independent instances of one pattern in a single sweep is the finding that outlives this PR. The pattern is a `HashMap` iteration result becoming an ordered or order-sensitive decision, and nothing in the toolchain flags it: the types are correct, the code compiles, and the tests pass most of the time.
