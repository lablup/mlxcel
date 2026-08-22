// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shared CLI argument types for language bias flags.
//!
//! This module provides:
//! - [`LangBiasCliArgs`]: raw CLI input struct (embedded via `#[command(flatten)]`)
//! - [`LangBiasConfig`]: resolved, validated configuration
//! - [`LangBiasCliArgs::resolve`]: converts raw args to `LangBiasConfig`
//! - Parser for `--lang-bias <entries>` strings (syntax per plan §6.1)
//! - YAML loader for `--lang-bias-config <path>` (plan §6.2)
//!
//! Plan references: `docs_internal/architecture/axis-b-language-steering-plan-20260419.md`
//! sections §6.1–§6.5.

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use clap::Args;
use serde::Deserialize;

use mlxcel_core::lang_analyzer::{
    ExceptionConfig, InclusionPolicy, LangAnalyzerError, LangBiasConfig, LangBiasSet, LanguageCode,
};

/// Error type for CLI argument parsing and YAML loading.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("lang-bias entry {entry:?} is missing '='")]
    MissingEquals { entry: String },
    #[error("lang-bias entry {entry:?} has an empty language code")]
    EmptyLanguageCode { entry: String },
    #[error("lang-bias entry {entry:?} has an empty bias value")]
    EmptyBiasValue { entry: String },
    #[error("unknown language code '{code}'; supported: ja, zh, ko, en, ru, ar, th, hi, he, el")]
    UnknownLanguageCode { code: String, entry: String },
    #[error("unparseable bias value '{value}' in entry {entry:?}: {reason}")]
    UnparseableBias {
        value: String,
        entry: String,
        reason: String,
    },
    /// Raised by both language-bias entry points: the `--lang-bias` string
    /// parser and the YAML `bias:` block, which reject a repeated language code
    /// identically because the repeat makes the priority order ambiguous.
    #[error(
        "duplicate language code '{code}' in language bias entries (ambiguous priority); check --lang-bias and the YAML bias: block"
    )]
    DuplicateLanguageCode { code: String },
    #[error("failed to read lang-bias config file '{path}': {source}")]
    ConfigReadError {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse lang-bias config '{path}': {message}")]
    ConfigParseError { path: String, message: String },
}

/// Parse a bias value string into an `f32`, handling `-inf`, `+inf`, and `inf`.
///
/// Accepts:
/// - `-inf` → `f32::NEG_INFINITY`
/// - `+inf` or `inf` → `f32::INFINITY`
/// - Any valid floating-point literal → parsed via `f32::from_str`
pub fn parse_bias_f32(s: &str) -> Result<f32, String> {
    match s.trim() {
        "-inf" => Ok(f32::NEG_INFINITY),
        "+inf" | "inf" => Ok(f32::INFINITY),
        other => f32::from_str(other).map_err(|e| e.to_string()),
    }
}

/// Parse a `--lang-bias` entries string into a `LangBiasSet`.
///
/// Syntax (plan §6.1):
/// ```text
/// <entry>[,<entry>]*
/// <entry>  = <lang_code>=<bias>
/// <bias>   = -inf | +inf | inf | <float>
/// ```
///
/// Returns `Err` on:
/// - Missing `=`
/// - Empty language code or bias value
/// - Unknown language code
/// - Unparseable float
/// - Duplicate language code
///
/// Note: Leading/trailing whitespace is stripped from each entry, language
/// code, and bias value.
pub fn parse_lang_bias_entries(s: &str) -> Result<LangBiasSet, CliError> {
    let mut ordered = Vec::new();
    let mut seen: HashMap<String, ()> = HashMap::new();

    for raw_entry in s.split(',') {
        let entry = raw_entry.trim().to_owned();
        if entry.is_empty() {
            // Skip empty entries caused by trailing commas.
            continue;
        }

        let eq_pos = entry.find('=').ok_or_else(|| CliError::MissingEquals {
            entry: entry.clone(),
        })?;

        let code_str = entry[..eq_pos].trim();
        let bias_str = entry[eq_pos + 1..].trim();

        if code_str.is_empty() {
            return Err(CliError::EmptyLanguageCode {
                entry: entry.clone(),
            });
        }
        if bias_str.is_empty() {
            return Err(CliError::EmptyBiasValue {
                entry: entry.clone(),
            });
        }

        // Validate and parse the language code via `LanguageCode::from_str`.
        let lang_code = LanguageCode::from_str(code_str).map_err(|e| match e {
            LangAnalyzerError::UnknownLanguageCode(c) => CliError::UnknownLanguageCode {
                code: c,
                entry: entry.clone(),
            },
            _ => CliError::UnknownLanguageCode {
                code: code_str.to_owned(),
                entry: entry.clone(),
            },
        })?;

        // Detect duplicate language codes.
        if seen.contains_key(code_str) {
            return Err(CliError::DuplicateLanguageCode {
                code: code_str.to_owned(),
            });
        }
        seen.insert(code_str.to_owned(), ());

        let bias = parse_bias_f32(bias_str).map_err(|reason| CliError::UnparseableBias {
            value: bias_str.to_owned(),
            entry: entry.clone(),
            reason,
        })?;

        ordered.push((lang_code, bias));
    }

    Ok(LangBiasSet { ordered })
}

/// YAML schema for `--lang-bias-config` files (plan §6.2).
///
/// ```yaml
/// policy: conservative   # or strict (default: conservative)
/// bias:
///   ja: -inf
///   zh: -10.0
///   ko: +5.0
/// exceptions:
///   include_special: false
///   include_numeric: false
///   include_punctuation: false
/// ```
///
/// Unknown top-level keys produce a parse error via `#[serde(deny_unknown_fields)]`.
///
/// The order of the `bias:` entries is significant and is preserved exactly as
/// written: index 0 is the highest priority. `to_token_bias` resolves a token
/// claimed by several languages with first-language-wins, so in the example
/// above the Han tokens shared by `ja`, `zh` and `ko` all receive `ja`'s
/// `-inf`. Writing the same three languages in a different order is a
/// different configuration, not a cosmetic difference.
///
/// A language code repeated inside one `bias:` block is rejected with
/// [`CliError::DuplicateLanguageCode`], the same error the equivalent
/// `--lang-bias` string produces.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LangBiasYamlConfig {
    #[serde(default)]
    pub policy: Option<PolicyStr>,
    #[serde(default)]
    pub bias: Option<BiasEntries>,
    #[serde(default)]
    pub exceptions: Option<ExceptionYaml>,
}

/// The `bias:` block of a YAML config, in document order.
///
/// Deserializing that block into a `HashMap` (what this field used to be) threw
/// away the two properties the resolve loop depends on. `HashMap` iteration
/// order is randomized by `RandomState` per map instance, so the priority order
/// handed to `to_token_bias` was a fresh random permutation on every load, and
/// the resulting bias assigned to a shared Han token changed from run to run for
/// one unchanged config file. `HashMap` also collapses repeated keys during
/// deserialization (serde_yaml resolves them last-wins with no diagnostic),
/// which made the resolve loop's duplicate check unreachable and let the YAML
/// path silently accept input the `--lang-bias` parser rejects. See issue #1267.
///
/// Collecting the entries through `MapAccess` into a `Vec` keeps both: the
/// author's order survives, and a repeated key arrives as a second entry that
/// the resolve loop can reject. The accepted YAML syntax is unchanged, `bias:`
/// is still a plain mapping.
#[derive(Debug, Default)]
pub struct BiasEntries(Vec<(String, BiasValueStr)>);

impl BiasEntries {
    /// The `(language code, bias)` pairs in the order they appear in the document.
    pub fn as_slice(&self) -> &[(String, BiasValueStr)] {
        &self.0
    }

    /// Number of entries, counting a repeated language code once per occurrence.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the `bias:` block is present but empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl IntoIterator for BiasEntries {
    type Item = (String, BiasValueStr);
    type IntoIter = std::vec::IntoIter<(String, BiasValueStr)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'de> Deserialize<'de> for BiasEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BiasEntriesVisitor;

        impl<'de> serde::de::Visitor<'de> for BiasEntriesVisitor {
            type Value = BiasEntries;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a mapping of language code to bias value")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0));
                // Deliberately a Vec push per entry rather than a map insert:
                // repeated keys must reach the caller so it can reject them.
                while let Some(entry) = access.next_entry::<String, BiasValueStr>()? {
                    entries.push(entry);
                }
                Ok(BiasEntries(entries))
            }
        }

        deserializer.deserialize_map(BiasEntriesVisitor)
    }
}

/// Wraps a YAML `policy:` string value with custom deserialization.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyStr {
    Conservative,
    Strict,
}

impl From<PolicyStr> for InclusionPolicy {
    fn from(p: PolicyStr) -> Self {
        match p {
            PolicyStr::Conservative => InclusionPolicy::Conservative,
            PolicyStr::Strict => InclusionPolicy::Strict,
        }
    }
}

/// A bias value in YAML, which may be the special strings `-inf`, `+inf`,
/// `inf`, or a regular float.
#[derive(Debug)]
pub struct BiasValueStr(pub f32);

impl<'de> Deserialize<'de> for BiasValueStr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept both float and string YAML values.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum RawBias {
            Float(f64),
            Str(String),
        }

        let raw = RawBias::deserialize(deserializer)?;
        let value = match raw {
            RawBias::Float(f) => f as f32,
            RawBias::Str(s) => parse_bias_f32(&s).map_err(serde::de::Error::custom)?,
        };
        Ok(BiasValueStr(value))
    }
}

/// Exception configuration as loaded from YAML.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ExceptionYaml {
    #[serde(default)]
    pub include_special: bool,
    #[serde(default)]
    pub include_numeric: bool,
    #[serde(default)]
    pub include_punctuation: bool,
    /// include byte-fragment tokens classified via UTF-8
    /// start-byte analysis. Default: `false`.
    #[serde(default)]
    pub include_byte_fragments: bool,
}

impl From<ExceptionYaml> for ExceptionConfig {
    fn from(e: ExceptionYaml) -> Self {
        ExceptionConfig {
            include_special: e.include_special,
            include_numeric: e.include_numeric,
            include_punctuation: e.include_punctuation,
            include_byte_fragments: e.include_byte_fragments,
        }
    }
}

/// Load and parse a YAML `--lang-bias-config` file.
///
/// Returns `Err(CliError)` if the file cannot be read or the YAML is invalid.
pub fn load_yaml_config(path: &PathBuf) -> Result<LangBiasYamlConfig, CliError> {
    let content = std::fs::read_to_string(path).map_err(|e| CliError::ConfigReadError {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_yaml::from_str(&content).map_err(|e| CliError::ConfigParseError {
        path: path.display().to_string(),
        message: e.to_string(),
    })
}

/// Raw CLI input for language bias flags.
///
/// Embed in command arg structs via `#[command(flatten)]`.
///
/// Used by: `generate` command, `generate-vlm` command.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "Language Bias Options")]
pub struct LangBiasCliArgs {
    /// Language bias entries, e.g. `ja=-inf,zh=-10,ko=+5.0`.
    ///
    /// Syntax: `<lang_code>=<bias>[,<lang_code>=<bias>]*`
    /// where `<bias>` is `-inf`, `+inf`, `inf`, or a float.
    /// Supported language codes: ja, zh, ko, en, ru, ar, th, hi, he, el.
    #[arg(long = "lang-bias", value_name = "ENTRIES")]
    pub lang_bias: Option<String>,

    /// Path to a YAML file containing language bias configuration.
    ///
    /// CLI flags take precedence over YAML config values.
    #[arg(long = "lang-bias-config", value_name = "PATH")]
    pub lang_bias_config: Option<PathBuf>,

    /// Language token inclusion policy: `conservative` (default) or `strict`.
    ///
    /// Conservative: any token containing at least one character of a target script.
    /// Strict: only tokens whose entire script set is contained in the target set.
    #[arg(long = "lang-bias-policy", value_name = "POLICY")]
    pub lang_bias_policy: Option<String>,

    /// Include special tokens (BOS/EOS/PAD/…) in language sets.
    ///
    /// By default, special tokens are excluded from all language sets.
    #[arg(long = "lang-bias-include-special", default_value_t = false)]
    pub include_special: bool,

    /// Include purely numeric tokens in language sets.
    ///
    /// By default, purely numeric tokens are excluded from all language sets.
    #[arg(long = "lang-bias-include-numeric", default_value_t = false)]
    pub include_numeric: bool,

    /// Include purely punctuation tokens in language sets.
    ///
    /// By default, purely punctuation tokens are excluded from all language sets.
    #[arg(long = "lang-bias-include-punctuation", default_value_t = false)]
    pub include_punctuation: bool,

    /// Include byte-fragment tokens in language sets.
    ///
    /// Byte-level BPE tokenizers (Qwen, GPT-2, LLaMA, Mistral) represent
    /// less-common CJK characters as sequences of individual byte tokens.
    /// Each byte decodes to `U+FFFD` on its own and is classified as
    /// `Other` by the standard decode-path classifier, bypassing filters like
    /// `zh=-inf` even though the fragments reassemble into the target
    /// character at generation time.
    ///
    /// Enabling this flag runs a second classification pass that assigns
    /// a likely [`mlxcel_core::lang_analyzer::Script`] to each byte-fragment
    /// token based on its UTF-8 start byte. Start-byte classification is
    /// approximate (for example, the `0xE4`–`0xE9` range covers most CJK
    /// Unified Ideographs but also catches some Latin Extended Additional
    /// blocks), which is why the flag is opt-in. Operators can monitor the
    /// `mlxcel_lang_bias_byte_fragment_suppressions_total` metric to
    /// observe how much suppression comes from byte-fragment entries.
    ///
    /// **Default:** off (behavior is unchanged unless this flag is enabled).
    #[arg(long = "lang-bias-include-byte-fragments", default_value_t = false)]
    pub include_byte_fragments: bool,

    /// Force a rebuild of the `TokenLanguageIndex` cache.
    ///
    /// Normally the cache is rebuilt only when the tokenizer vocab changes.
    /// Use this flag to force a rebuild regardless of cache state.
    #[arg(long = "lang-bias-rebuild-cache", default_value_t = false)]
    pub rebuild_cache: bool,
}

impl LangBiasCliArgs {
    /// Returns `true` if any language bias flag was provided.
    pub fn is_active(&self) -> bool {
        self.lang_bias.is_some()
            || self.lang_bias_config.is_some()
            || self.lang_bias_policy.is_some()
            || self.include_special
            || self.include_numeric
            || self.include_punctuation
            || self.include_byte_fragments
            || self.rebuild_cache
    }

    /// Parse the `--lang-bias-policy` string value into `InclusionPolicy`.
    fn parse_policy(s: &str) -> Result<InclusionPolicy, CliError> {
        match s.trim().to_lowercase().as_str() {
            "conservative" => Ok(InclusionPolicy::Conservative),
            "strict" => Ok(InclusionPolicy::Strict),
            other => Err(CliError::ConfigParseError {
                path: "(--lang-bias-policy)".to_owned(),
                message: format!("unknown policy '{other}'; expected 'conservative' or 'strict'"),
            }),
        }
    }

    /// Resolve raw CLI inputs into a validated [`LangBiasConfig`].
    ///
    /// Precedence rules (plan §6, merge):
    /// 1. Start with defaults.
    /// 2. Apply YAML config file values (if `--lang-bias-config` provided).
    /// 3. Override with explicit CLI flags:
    ///    - `--lang-bias` entries **replace** (not merge with) YAML `bias:` entries.
    ///    - `--lang-bias-policy` overrides YAML `policy:`.
    ///    - `--lang-bias-include-*` flags add to exception config (CLI wins).
    ///    - `--lang-bias-rebuild-cache` is additive.
    ///
    /// Returns `Ok(None)` when no language bias flags are active (fast path).
    /// Returns `Ok(Some(config))` when at least one flag is set.
    pub fn resolve(&self) -> Result<Option<LangBiasConfig>, CliError> {
        if !self.is_active() {
            return Ok(None);
        }

        // Start with defaults.
        let mut policy = InclusionPolicy::Conservative;
        let mut bias_set = LangBiasSet::default();
        let mut exceptions = ExceptionConfig::default();

        // Step 2: Apply YAML config if present.
        if let Some(ref config_path) = self.lang_bias_config {
            let yaml = load_yaml_config(config_path)?;

            if let Some(yaml_policy) = yaml.policy {
                policy = yaml_policy.into();
            }

            if let Some(yaml_bias) = yaml.bias {
                // `BiasEntries` yields the `bias:` block in document order and
                // keeps repeated keys, so `ordered` ends up in the priority
                // order the author wrote and the duplicate check below actually
                // fires. This mirrors `parse_lang_bias_entries` entry for entry.
                let mut ordered = Vec::new();
                let mut seen: HashMap<String, ()> = HashMap::new();
                for (code_str, BiasValueStr(bias)) in yaml_bias {
                    if seen.contains_key(&code_str) {
                        return Err(CliError::DuplicateLanguageCode { code: code_str });
                    }
                    seen.insert(code_str.clone(), ());

                    let lang_code = LanguageCode::from_str(&code_str).map_err(|e| match e {
                        LangAnalyzerError::UnknownLanguageCode(c) => {
                            CliError::UnknownLanguageCode {
                                code: c,
                                entry: format!("{code_str}: (from YAML)"),
                            }
                        }
                        _ => CliError::UnknownLanguageCode {
                            code: code_str.clone(),
                            entry: format!("{code_str}: (from YAML)"),
                        },
                    })?;
                    ordered.push((lang_code, bias));
                }
                bias_set = LangBiasSet { ordered };
            }

            if let Some(yaml_exceptions) = yaml.exceptions {
                exceptions = yaml_exceptions.into();
            }
        }

        // Step 3a: CLI --lang-bias replaces YAML bias entries entirely.
        if let Some(ref entries_str) = self.lang_bias {
            bias_set = parse_lang_bias_entries(entries_str)?;
        }

        // Step 3b: CLI --lang-bias-policy overrides YAML policy.
        if let Some(ref policy_str) = self.lang_bias_policy {
            policy = Self::parse_policy(policy_str)?;
        }

        // Step 3c: CLI exception include flags (additive; CLI wins by OR).
        if self.include_special {
            exceptions.include_special = true;
        }
        if self.include_numeric {
            exceptions.include_numeric = true;
        }
        if self.include_punctuation {
            exceptions.include_punctuation = true;
        }
        if self.include_byte_fragments {
            exceptions.include_byte_fragments = true;
        }

        Ok(Some(LangBiasConfig {
            bias_set,
            policy,
            exceptions,
            rebuild_cache: self.rebuild_cache,
        }))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::lang_analyzer::{InclusionPolicy, LanguageCode};

    // -------------------------------------------------------------------------
    // parse_lang_bias_entries — 5 success cases (plan §10.1)
    // -------------------------------------------------------------------------

    #[test]
    fn parse_single_neg_inf() {
        let set = parse_lang_bias_entries("ja=-inf").unwrap();
        assert_eq!(set.ordered.len(), 1);
        assert_eq!(set.ordered[0].0, LanguageCode::Ja);
        assert_eq!(set.ordered[0].1, f32::NEG_INFINITY);
    }

    #[test]
    fn parse_two_entries() {
        let set = parse_lang_bias_entries("ja=-inf,zh=-10").unwrap();
        assert_eq!(set.ordered.len(), 2);
        assert_eq!(set.ordered[0].0, LanguageCode::Ja);
        assert_eq!(set.ordered[0].1, f32::NEG_INFINITY);
        assert_eq!(set.ordered[1].0, LanguageCode::Zh);
        assert_eq!(set.ordered[1].1, -10.0_f32);
    }

    #[test]
    fn parse_positive_float() {
        let set = parse_lang_bias_entries("ko=+5.0").unwrap();
        assert_eq!(set.ordered.len(), 1);
        assert_eq!(set.ordered[0].0, LanguageCode::Ko);
        assert_eq!(set.ordered[0].1, 5.0_f32);
    }

    #[test]
    fn parse_three_entries_mixed_signs() {
        let set = parse_lang_bias_entries("en=+3,ja=-inf,zh=-5.5").unwrap();
        assert_eq!(set.ordered.len(), 3);
        assert_eq!(set.ordered[0].0, LanguageCode::En);
        assert_eq!(set.ordered[0].1, 3.0_f32);
        assert_eq!(set.ordered[1].0, LanguageCode::Ja);
        assert_eq!(set.ordered[1].1, f32::NEG_INFINITY);
        assert_eq!(set.ordered[2].0, LanguageCode::Zh);
        assert_eq!(set.ordered[2].1, -5.5_f32);
    }

    /// Whitespace convention: leading/trailing whitespace around entries,
    /// codes, and bias values is stripped.
    #[test]
    fn parse_whitespace_tolerance() {
        // Entries separated by ", " with spaces around codes/values are accepted.
        let set = parse_lang_bias_entries(" ja = -inf , zh = -10 ").unwrap();
        assert_eq!(set.ordered.len(), 2);
        assert_eq!(set.ordered[0].0, LanguageCode::Ja);
        assert_eq!(set.ordered[0].1, f32::NEG_INFINITY);
        assert_eq!(set.ordered[1].0, LanguageCode::Zh);
        assert_eq!(set.ordered[1].1, -10.0_f32);
    }

    // -------------------------------------------------------------------------
    // parse_lang_bias_entries — 5 error cases (plan §10.1)
    // -------------------------------------------------------------------------

    #[test]
    fn parse_unknown_language_code() {
        let err = parse_lang_bias_entries("xx=-inf").unwrap_err();
        assert!(
            matches!(err, CliError::UnknownLanguageCode { ref code, .. } if code == "xx"),
            "expected UnknownLanguageCode, got: {err}"
        );
    }

    #[test]
    fn parse_unparseable_float() {
        let err = parse_lang_bias_entries("ja=abc").unwrap_err();
        assert!(
            matches!(err, CliError::UnparseableBias { ref value, .. } if value == "abc"),
            "expected UnparseableBias, got: {err}"
        );
    }

    #[test]
    fn parse_empty_bias_value() {
        let err = parse_lang_bias_entries("ja=").unwrap_err();
        assert!(
            matches!(err, CliError::EmptyBiasValue { .. }),
            "expected EmptyBiasValue, got: {err}"
        );
    }

    #[test]
    fn parse_empty_language_code() {
        let err = parse_lang_bias_entries("=-inf").unwrap_err();
        assert!(
            matches!(err, CliError::EmptyLanguageCode { .. }),
            "expected EmptyLanguageCode, got: {err}"
        );
    }

    #[test]
    fn parse_duplicate_language_code() {
        let err = parse_lang_bias_entries("ja=-inf,ja=+5").unwrap_err();
        assert!(
            matches!(err, CliError::DuplicateLanguageCode { ref code } if code == "ja"),
            "expected DuplicateLanguageCode, got: {err}"
        );
    }

    // -------------------------------------------------------------------------
    // YAML loader tests (plan §10.1)
    // -------------------------------------------------------------------------

    #[test]
    fn yaml_well_formed_parses() {
        let yaml_str = r#"
policy: conservative
bias:
  ja: -inf
  zh: -10.0
  ko: +5.0
exceptions:
  include_special: false
  include_numeric: false
  include_punctuation: true
"#;
        let config: LangBiasYamlConfig = serde_yaml::from_str(yaml_str).unwrap();
        assert!(matches!(config.policy, Some(PolicyStr::Conservative)));
        let bias = config.bias.unwrap();
        // Assert the order, not just membership. Asserting membership alone is
        // what let the randomized `HashMap` ordering of issue #1267 stay hidden.
        let codes: Vec<&str> = bias.as_slice().iter().map(|(c, _)| c.as_str()).collect();
        assert_eq!(codes, ["ja", "zh", "ko"]);
        let values: Vec<f32> = bias.as_slice().iter().map(|(_, v)| v.0).collect();
        assert_eq!(values, [f32::NEG_INFINITY, -10.0_f32, 5.0_f32]);
        let ex = config.exceptions.unwrap();
        assert!(!ex.include_special);
        assert!(!ex.include_numeric);
        assert!(ex.include_punctuation);
    }

    #[test]
    fn yaml_missing_bias_field_defaults_to_empty() {
        // `bias:` is optional (plan: "Missing `bias:` field is an error or defaults to empty").
        // We choose: missing defaults to None (empty set), not an error.
        let yaml_str = r#"
policy: strict
"#;
        let config: LangBiasYamlConfig = serde_yaml::from_str(yaml_str).unwrap();
        assert!(config.bias.is_none());
    }

    #[test]
    fn yaml_unknown_top_level_key_errors() {
        let yaml_str = r#"
policy: conservative
unknown_field: value
"#;
        let result: Result<LangBiasYamlConfig, _> = serde_yaml::from_str(yaml_str);
        assert!(
            result.is_err(),
            "unknown top-level key should produce a parse error"
        );
    }

    // -------------------------------------------------------------------------
    // YAML `bias:` ordering (issue #1267)
    //
    // The `bias:` block used to deserialize into a `HashMap`, whose iteration
    // order `RandomState` randomizes per map instance. Because
    // `TokenLanguageIndex::to_token_bias` resolves a token claimed by several
    // languages with first-language-wins, the bias landing on a shared Han
    // token changed from run to run for one unchanged config file. These tests
    // resolve repeatedly inside one process, which is what makes them
    // load-bearing: a single resolve can pass by luck, and the randomization is
    // per map instance rather than per process, so a fresh resolve inside the
    // same process draws a fresh order.
    // -------------------------------------------------------------------------

    /// Number of repeated resolves the ordering tests perform.
    ///
    /// With three languages there are six possible orders, so a single resolve
    /// against the broken code had a good chance of coming out right and
    /// proving nothing. Repeating the resolve makes an accidental pass
    /// vanishingly unlikely. Measured against the pre-fix code, all three
    /// ordering tests here failed at iteration 0 or 2.
    const ORDER_RESOLVE_ITERATIONS: usize = 32;

    /// Write `contents` to a temp file and return the handle plus its path.
    ///
    /// The handle must stay alive for as long as the path is used: dropping a
    /// `NamedTempFile` deletes the file.
    fn write_temp_yaml(contents: &str) -> (tempfile::NamedTempFile, PathBuf) {
        use std::io::Write;

        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(contents.as_bytes()).unwrap();
        tmpfile.flush().unwrap();
        let path = tmpfile.path().to_path_buf();
        (tmpfile, path)
    }

    /// The three-CJK config from the `LangBiasYamlConfig` schema doc comment.
    ///
    /// Han is shared by all three languages (`scripts_for`: Japanese includes
    /// `Han` under both policies, Chinese is `{Han}`, Korean Conservative
    /// includes `Han`), so this is exactly the case where the priority order
    /// decides the outcome.
    const THREE_CJK_YAML: &str =
        "policy: conservative\nbias:\n  ja: -inf\n  zh: -10.0\n  ko: +5.0\n";

    #[test]
    fn yaml_multi_cjk_bias_keeps_document_order_across_repeated_resolves() {
        let (_tmpfile, path) = write_temp_yaml(THREE_CJK_YAML);

        let expected = [
            (LanguageCode::Ja, f32::NEG_INFINITY),
            (LanguageCode::Zh, -10.0_f32),
            (LanguageCode::Ko, 5.0_f32),
        ];

        for iteration in 0..ORDER_RESOLVE_ITERATIONS {
            let args = LangBiasCliArgs {
                lang_bias_config: Some(path.clone()),
                ..Default::default()
            };
            // A full `resolve()` per iteration, so each one re-reads and
            // re-deserializes the file and gets a fresh map instance.
            let config = args.resolve().unwrap().unwrap();
            let ordered = &config.bias_set.ordered;

            assert_eq!(
                ordered.len(),
                expected.len(),
                "iteration {iteration}: expected {} entries, got {ordered:?}",
                expected.len()
            );
            for (index, (expected_code, expected_bias)) in expected.iter().enumerate() {
                let (code, bias) = ordered[index];
                assert_eq!(
                    code, *expected_code,
                    "iteration {iteration}: entry {index} should be {expected_code:?} but the \
                     resolved order was {ordered:?}; YAML bias: order must be the priority order"
                );
                assert_eq!(
                    bias, *expected_bias,
                    "iteration {iteration}: entry {index} carried the wrong bias value"
                );
            }
        }
    }

    #[test]
    fn yaml_and_cli_paths_agree_on_multi_cjk_order() {
        let (_tmpfile, path) = write_temp_yaml(THREE_CJK_YAML);

        let cli_args = LangBiasCliArgs {
            lang_bias: Some("ja=-inf,zh=-10.0,ko=+5.0".to_owned()),
            ..Default::default()
        };
        let cli_ordered = cli_args.resolve().unwrap().unwrap().bias_set.ordered;

        for iteration in 0..ORDER_RESOLVE_ITERATIONS {
            let yaml_args = LangBiasCliArgs {
                lang_bias_config: Some(path.clone()),
                ..Default::default()
            };
            let yaml_ordered = yaml_args.resolve().unwrap().unwrap().bias_set.ordered;
            assert_eq!(
                yaml_ordered, cli_ordered,
                "iteration {iteration}: the YAML path and the --lang-bias path must resolve \
                 equivalent input to the same LangBiasSet"
            );
        }
    }

    #[test]
    fn yaml_multi_cjk_first_language_wins_on_shared_han_tokens() {
        use mlxcel_core::lang_analyzer::{
            CURRENT_VERSION, Script, TokenLanguageIndex, TokenScriptInfo,
        };

        // A three-token synthetic vocabulary: one pure-Han token claimed by ja,
        // zh and ko alike, one Hiragana token only ja claims, and one Hangul
        // token only ko claims. Building the index by hand keeps the assertion
        // on `to_token_bias` without needing a real tokenizer.
        let token = |token_id: i32, scripts: Vec<Script>| TokenScriptInfo {
            token_id,
            scripts: scripts.into(),
            is_special: false,
            is_numeric: false,
            is_punctuation: false,
            is_whitespace: false,
            is_byte_fragment: false,
        };
        let index = TokenLanguageIndex {
            vocab_hash: "test".to_owned(),
            version: CURRENT_VERSION,
            tokens: vec![
                token(0, vec![Script::Han]),
                token(1, vec![Script::Hiragana]),
                token(2, vec![Script::Hangul]),
            ],
            by_script: HashMap::new(),
        };

        let (_tmpfile, path) = write_temp_yaml(THREE_CJK_YAML);

        for iteration in 0..ORDER_RESOLVE_ITERATIONS {
            let args = LangBiasCliArgs {
                lang_bias_config: Some(path.clone()),
                ..Default::default()
            };
            let config = args.resolve().unwrap().unwrap();
            let bias_map = index.to_token_bias(&config.bias_set, config.policy, &config.exceptions);

            assert_eq!(
                bias_map.get(&0).copied(),
                Some(f32::NEG_INFINITY),
                "iteration {iteration}: the shared Han token must take the first-listed \
                 language's bias (ja = -inf), resolved order was {:?}",
                config.bias_set.ordered
            );
            assert_eq!(
                bias_map.get(&1).copied(),
                Some(f32::NEG_INFINITY),
                "iteration {iteration}: the Hiragana token belongs to ja only"
            );
            assert_eq!(
                bias_map.get(&2).copied(),
                Some(5.0_f32),
                "iteration {iteration}: the Hangul token belongs to ko only"
            );
        }
    }

    #[test]
    fn yaml_duplicate_language_code_is_rejected() {
        // serde_yaml deserializes a repeated key into a typed `HashMap` without
        // any diagnostic (last occurrence wins), which is why the duplicate
        // check in `resolve` used to be unreachable. The ordered representation
        // delivers both occurrences, so the YAML path now rejects what the
        // `--lang-bias` path already rejected.
        let (_tmpfile, path) = write_temp_yaml("bias:\n  ja: -1.0\n  zh: -2.0\n  ja: -3.0\n");

        let args = LangBiasCliArgs {
            lang_bias_config: Some(path),
            ..Default::default()
        };
        let err = args
            .resolve()
            .expect_err("a repeated language code in a YAML bias: block must be rejected");
        assert!(
            matches!(err, CliError::DuplicateLanguageCode { ref code } if code == "ja"),
            "expected DuplicateLanguageCode, got: {err}"
        );
    }

    #[test]
    fn yaml_empty_bias_block_resolves_to_an_empty_set() {
        let (_tmpfile, path) = write_temp_yaml("policy: strict\nbias: {}\n");

        let args = LangBiasCliArgs {
            lang_bias_config: Some(path),
            ..Default::default()
        };
        let config = args.resolve().unwrap().unwrap();
        assert!(
            config.bias_set.ordered.is_empty(),
            "an empty bias: block must resolve to an empty set, not an error"
        );
        assert_eq!(config.policy, InclusionPolicy::Strict);
    }

    #[test]
    fn yaml_bias_block_rejects_a_sequence() {
        // The accepted syntax is unchanged: `bias:` is a mapping, and a list
        // form is still a parse error rather than a silently different config.
        let result: Result<LangBiasYamlConfig, _> =
            serde_yaml::from_str("bias:\n  - ja: -inf\n  - zh: -10.0\n");
        assert!(
            result.is_err(),
            "a sequence-shaped bias: block should still be a parse error"
        );
    }

    // -------------------------------------------------------------------------
    // Precedence: CLI policy overrides YAML policy
    // -------------------------------------------------------------------------

    #[test]
    fn cli_policy_overrides_yaml_policy() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let yaml_str = b"policy: strict\nbias:\n  ja: -inf\n";
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(yaml_str).unwrap();
        let path = tmpfile.path().to_path_buf();

        let args = LangBiasCliArgs {
            lang_bias_config: Some(path),
            // CLI explicitly requests conservative, overriding YAML's strict
            lang_bias_policy: Some("conservative".to_owned()),
            ..Default::default()
        };

        let config = args.resolve().unwrap().unwrap();
        assert_eq!(
            config.policy,
            InclusionPolicy::Conservative,
            "CLI --lang-bias-policy=conservative should override YAML policy=strict"
        );
    }

    // -------------------------------------------------------------------------
    // Exception include flags flip ExceptionConfig fields
    // -------------------------------------------------------------------------

    #[test]
    fn include_flags_flip_exception_config() {
        let args = LangBiasCliArgs {
            lang_bias: Some("ja=-inf".to_owned()),
            include_special: true,
            include_numeric: false,
            include_punctuation: true,
            ..Default::default()
        };

        let config = args.resolve().unwrap().unwrap();
        assert!(
            config.exceptions.include_special,
            "--lang-bias-include-special should set ExceptionConfig.include_special=true"
        );
        assert!(
            !config.exceptions.include_numeric,
            "include_numeric should remain false when flag not set"
        );
        assert!(
            config.exceptions.include_punctuation,
            "--lang-bias-include-punctuation should set ExceptionConfig.include_punctuation=true"
        );
        // Byte-fragment flag defaults to false when unset.
        assert!(
            !config.exceptions.include_byte_fragments,
            "include_byte_fragments must default to false"
        );
    }

    /// `--lang-bias-include-byte-fragments` flips the resolved
    /// `ExceptionConfig.include_byte_fragments` field and otherwise leaves the
    /// exception set identical to Phase 1 defaults.
    #[test]
    fn include_byte_fragments_flag_flips_exception_config() {
        let args = LangBiasCliArgs {
            lang_bias: Some("zh=-inf".to_owned()),
            include_byte_fragments: true,
            ..Default::default()
        };

        let config = args.resolve().unwrap().unwrap();
        assert!(
            config.exceptions.include_byte_fragments,
            "CLI flag must set ExceptionConfig.include_byte_fragments=true"
        );
        // Other exception flags stay at Phase 1 defaults.
        assert!(!config.exceptions.include_special);
        assert!(!config.exceptions.include_numeric);
        assert!(!config.exceptions.include_punctuation);
    }

    /// the CLI flag alone makes `is_active` true so the
    /// resolver runs even without any `--lang-bias` entries. This matters for
    /// operator workflows that only want to rebuild the cache with the
    /// byte-fragment pass enabled.
    #[test]
    fn include_byte_fragments_alone_activates_resolver() {
        let args = LangBiasCliArgs {
            include_byte_fragments: true,
            ..Default::default()
        };
        assert!(args.is_active());
        let config = args.resolve().unwrap().unwrap();
        assert!(config.exceptions.include_byte_fragments);
        assert!(config.bias_set.ordered.is_empty());
    }

    /// YAML `exceptions.include_byte_fragments: true` flows through into the
    /// resolved `ExceptionConfig` when no CLI override is present.
    #[test]
    fn yaml_include_byte_fragments_resolves() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let yaml_str = b"bias:\n  zh: -inf\nexceptions:\n  include_byte_fragments: true\n";
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(yaml_str).unwrap();
        let path = tmpfile.path().to_path_buf();

        let args = LangBiasCliArgs {
            lang_bias_config: Some(path),
            ..Default::default()
        };
        let config = args.resolve().unwrap().unwrap();
        assert!(
            config.exceptions.include_byte_fragments,
            "YAML include_byte_fragments=true must flow into ExceptionConfig"
        );
    }

    /// CLI `--lang-bias-include-byte-fragments` layers additively on top of
    /// YAML exception settings (both end up `true`).
    #[test]
    fn yaml_and_cli_byte_fragments_additive() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // YAML sets include_special=true; CLI sets include_byte_fragments=true.
        let yaml_str = b"bias:\n  zh: -inf\nexceptions:\n  include_special: true\n";
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(yaml_str).unwrap();
        let path = tmpfile.path().to_path_buf();

        let args = LangBiasCliArgs {
            lang_bias_config: Some(path),
            include_byte_fragments: true,
            ..Default::default()
        };
        let config = args.resolve().unwrap().unwrap();
        assert!(
            config.exceptions.include_special,
            "YAML include_special must survive"
        );
        assert!(
            config.exceptions.include_byte_fragments,
            "CLI --lang-bias-include-byte-fragments must flip the flag"
        );
    }

    // -------------------------------------------------------------------------
    // Additional coverage
    // -------------------------------------------------------------------------

    #[test]
    fn resolve_no_active_flags_returns_none() {
        let args = LangBiasCliArgs::default();
        let result = args.resolve().unwrap();
        assert!(result.is_none(), "no active flags should return None");
    }

    #[test]
    fn resolve_lang_bias_only_returns_some() {
        let args = LangBiasCliArgs {
            lang_bias: Some("ko=+5.0".to_owned()),
            ..Default::default()
        };
        let config = args.resolve().unwrap().unwrap();
        assert_eq!(config.bias_set.ordered.len(), 1);
        assert_eq!(config.bias_set.ordered[0].0, LanguageCode::Ko);
        assert_eq!(config.bias_set.ordered[0].1, 5.0_f32);
        assert_eq!(config.policy, InclusionPolicy::Conservative);
    }

    #[test]
    fn cli_lang_bias_replaces_yaml_bias() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // YAML defines zh=-10.0; CLI --lang-bias should replace entirely with ja=-inf.
        let yaml_str = b"bias:\n  zh: -10.0\n";
        let mut tmpfile = NamedTempFile::new().unwrap();
        tmpfile.write_all(yaml_str).unwrap();
        let path = tmpfile.path().to_path_buf();

        let args = LangBiasCliArgs {
            lang_bias_config: Some(path),
            lang_bias: Some("ja=-inf".to_owned()),
            ..Default::default()
        };

        let config = args.resolve().unwrap().unwrap();
        // CLI bias replaces YAML bias: only ja=-inf, no zh entry.
        assert_eq!(config.bias_set.ordered.len(), 1);
        assert_eq!(config.bias_set.ordered[0].0, LanguageCode::Ja);
        assert_eq!(config.bias_set.ordered[0].1, f32::NEG_INFINITY);
    }

    #[test]
    fn parse_bias_f32_special_values() {
        assert_eq!(parse_bias_f32("-inf").unwrap(), f32::NEG_INFINITY);
        assert_eq!(parse_bias_f32("+inf").unwrap(), f32::INFINITY);
        assert_eq!(parse_bias_f32("inf").unwrap(), f32::INFINITY);
        // Use a non-mathematical-constant literal so clippy's
        // `approx_constant` lint does not mistakenly flag this for PI.
        assert_eq!(parse_bias_f32("2.5").unwrap(), 2.5_f32);
        assert_eq!(parse_bias_f32("-5.0").unwrap(), -5.0_f32);
    }
}
