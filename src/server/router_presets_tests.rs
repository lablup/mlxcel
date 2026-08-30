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

//! Unit tests for the `--models-preset` INI translation (issue #1438):
//! grammar, key canonicalization, the `[*]` global cascade, the strict
//! unknown-key refusal, and the CLI-wins overlay onto per-model startup
//! configs.

use super::{PresetCliOverrides, apply_section_to_startup, parse_preset_text};
use crate::server::ServerStartupConfig;

#[test]
fn sections_keys_comments_and_globals_parse() {
    let presets = parse_preset_text(
        "; file comment\n\
         [*]\n\
         ctx-size = 4096\n\
         \n\
         [chat]\n\
         # inline section\n\
         model = /models/chat\n\
         temp = 0.5\n\
         alias = c1,c2\n\
         version = 1\n\
         [coder]\n\
         model = /models/coder\n\
         top-k = 20\n",
    )
    .expect("parse");
    assert_eq!(presets.global.ctx_size, Some(4096));
    assert_eq!(presets.models.len(), 2);
    let chat = &presets.models["chat"];
    assert_eq!(
        chat.model_path.as_deref(),
        Some(std::path::Path::new("/models/chat"))
    );
    assert_eq!(chat.temperature, Some(0.5));
    assert_eq!(chat.aliases, vec!["c1", "c2"]);
    // The global section cascades underneath each model's own options.
    let merged = presets.for_model("chat");
    assert_eq!(merged.ctx_size, Some(4096));
    assert_eq!(merged.temperature, Some(0.5));
    // A model with no named section still receives the global preset.
    assert_eq!(presets.for_model("unlisted").ctx_size, Some(4096));
}

#[test]
fn env_spellings_and_short_forms_canonicalize() {
    let presets = parse_preset_text(
        "[m]\n\
         LLAMA_ARG_MODEL = /m\n\
         LLAMA_ARG_CTX_SIZE = 2048\n\
         temperature = 0.1\n\
         LLAMA_ARG_ALIAS = a1\n",
    )
    .expect("parse");
    let m = &presets.models["m"];
    assert_eq!(m.model_path.as_deref(), Some(std::path::Path::new("/m")));
    assert_eq!(m.ctx_size, Some(2048));
    assert_eq!(m.temperature, Some(0.1));
    assert_eq!(m.aliases, vec!["a1"]);
}

#[test]
fn unknown_keys_fail_loudly_with_key_and_section() {
    let err = parse_preset_text("[m]\nn-gpu-layers = 99\n").expect_err("must refuse");
    let text = err.to_string();
    assert!(text.contains("n-gpu-layers"), "{text}");
    assert!(text.contains("[m]"), "{text}");
}

#[test]
fn malformed_lines_and_bad_values_are_errors() {
    assert!(
        parse_preset_text("[m\nmodel = /m\n").is_err(),
        "unclosed header"
    );
    assert!(
        parse_preset_text("[m]\njust-a-word\n").is_err(),
        "no equals"
    );
    assert!(
        parse_preset_text("[m]\nctx-size = lots\n").is_err(),
        "bad number"
    );
    assert!(
        parse_preset_text("[m]\nload-on-startup = maybe\n").is_err(),
        "bad boolean"
    );
}

#[test]
fn preset_only_options_parse() {
    let presets = parse_preset_text(
        "[m]\nmodel = /m\nload-on-startup = true\nstop-timeout = 30\ndedup-cache-models = 1\n",
    )
    .expect("parse");
    let m = &presets.models["m"];
    assert!(m.load_on_startup);
    assert_eq!(m.stop_timeout, Some(30));
    assert!(m.dedup_cache_models);
}

#[test]
fn apply_overlays_preset_values_where_the_cli_did_not_speak() {
    let presets = parse_preset_text(
        "[m]\nctx-size = 8192\ntemp = 0.25\ntop-k = 7\nseed = 42\nn-predict = 64\n",
    )
    .expect("parse");
    let section = presets.for_model("m");
    let mut startup = ServerStartupConfig::default();
    apply_section_to_startup(&mut startup, &section, &PresetCliOverrides::default());
    assert_eq!(startup.ctx_size, 8192);
    assert_eq!(startup.temperature, 0.25);
    assert!(startup.temperature_was_set);
    assert_eq!(startup.top_k, 7);
    assert_eq!(startup.seed, Some(42));
    assert_eq!(startup.n_predict, 64);
}

#[test]
fn explicit_cli_flags_win_over_preset_values() {
    let presets = parse_preset_text("[m]\nctx-size = 8192\ntemp = 0.25\n").expect("parse");
    let section = presets.for_model("m");
    let mut startup = ServerStartupConfig {
        ctx_size: 1024,
        temperature: 0.9,
        temperature_was_set: true,
        ..Default::default()
    };
    let cli = PresetCliOverrides {
        ctx_size: true,
        ..Default::default()
    };
    apply_section_to_startup(&mut startup, &section, &cli);
    assert_eq!(startup.ctx_size, 1024, "CLI ctx-size wins");
    assert_eq!(startup.temperature, 0.9, "explicit CLI temp wins");
}

#[test]
fn a_negative_seed_means_random_per_request() {
    let presets = parse_preset_text("[m]\nseed = -1\n").expect("parse");
    let section = presets.for_model("m");
    let mut startup = ServerStartupConfig::default();
    apply_section_to_startup(&mut startup, &section, &PresetCliOverrides::default());
    assert_eq!(startup.seed, None);
}
