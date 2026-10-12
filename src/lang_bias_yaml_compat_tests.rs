// Copyright 2025-2026 Lablup Inc.
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

use super::*;

fn load(text: &str) -> Result<LangBiasYamlConfig, CliError> {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), text).unwrap();
    load_yaml_config(&file.path().to_path_buf())
}

#[test]
fn duplicate_schema_fields_remain_parse_errors() {
    for text in [
        "policy: strict\npolicy: conservative\n",
        "en: 1\nen: 2\n",
        "exceptions: {en: 1, en: 2}\n",
        "bias: {ja: 1}\nbias: {ko: 2}\n",
        "exceptions: {include_special: true, include_special: false}\n",
    ] {
        assert!(matches!(load(text), Err(CliError::ConfigParseError { .. })));
    }
}

#[test]
fn numeric_spellings_and_order_are_preserved() {
    let config =
        load("bias: {ja: -inf, zh: +inf, ko: .nan, en: 0x10, ru: 1e2, ar: '2.5'}").unwrap();
    let entries = config.bias.unwrap().0;
    assert_eq!(
        entries
            .iter()
            .map(|(code, _)| code.as_str())
            .collect::<Vec<_>>(),
        ["ja", "zh", "ko", "en", "ru", "ar"]
    );
    assert_eq!(entries[0].1.0, f32::NEG_INFINITY);
    assert_eq!(entries[1].1.0, f32::INFINITY);
    assert!(entries[2].1.0.is_nan());
    assert_eq!(entries[3].1.0, 16.0);
    assert_eq!(entries[4].1.0, 100.0);
    assert_eq!(entries[5].1.0, 2.5);
}

#[test]
fn yaml_merge_does_not_bypass_schema_validation() {
    assert!(matches!(
        load("<<: {policy: strict}\n"),
        Err(CliError::ConfigParseError { .. })
    ));
}

#[test]
fn legacy_yaml_boolean_words_are_not_enabled() {
    for word in ["yes", "no", "on", "off"] {
        assert!(load(&format!("exceptions: {{include_special: {word}}}")).is_err());
    }
    for word in ["true", "True", "TRUE"] {
        assert!(
            load(&format!("exceptions: {{include_special: {word}}}"))
                .unwrap()
                .exceptions
                .unwrap()
                .include_special
        );
    }
}
