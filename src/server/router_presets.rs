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

//! b10621 `--models-preset` INI translation (issue #1438).
//!
//! b10621 loads an INI file whose sections name router models: `[*]` is a
//! global preset merged into every model, and any other `[name]` section
//! either defines a new model (when it carries `model =` or `hf-repo =`) or
//! overlays configuration onto a model discovered from the cache or
//! `--models-dir`. Keys are llama-server option spellings without leading
//! dashes (`ctx-size`, `temp`) or their `LLAMA_ARG_*` environment names, plus
//! the three preset-only options (`load-on-startup`, `stop-timeout`,
//! `dedup-cache-models`). Reference:
//! https://github.com/ggml-org/llama.cpp/blob/main/common/preset.cpp
//!
//! mlxcel translates each section onto its own per-model
//! [`super::ServerStartupConfig`] overlay and rebuilds the model's
//! `ServerConfig` through the same resolution pipeline the CLI uses, so
//! preset keys and CLI flags cannot drift apart. A key outside the translated
//! set fails startup with a diagnostic naming the key and section: loading
//! un-preset models while the operator believes presets apply is the
//! silent-ignore failure epic #1431 exists to remove. Router CLI arguments
//! that were explicitly given override preset values, which is b10621's own
//! overlay order (`preset.merge(base_preset)` with the CLI-args preset
//! winning).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One parsed preset section, already translated into typed directives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PresetSection {
    /// `model =`: checkpoint directory (MLX SafeTensors snapshot).
    pub model_path: Option<PathBuf>,
    /// `hf-repo =`: HuggingFace repo id resolved against the model cache.
    pub hf_repo: Option<String>,
    /// `alias =`: comma-separated alias list.
    pub aliases: Vec<String>,
    /// `tags =`: comma-separated informational tags.
    pub tags: Vec<String>,
    pub ctx_size: Option<usize>,
    pub n_parallel: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    /// `seed =`: b10621 semantics, negative means "random per request".
    pub seed: Option<i64>,
    pub n_predict: Option<i32>,
    /// Preset-only `load-on-startup`: begin loading this model as the router
    /// starts.
    pub load_on_startup: bool,
    /// Preset-only `stop-timeout` (seconds). Accepted for INI compatibility;
    /// the in-process pool unloads by dropping the model's app state, which
    /// needs no force-kill escalation, so the value steers nothing.
    pub stop_timeout: Option<u64>,
    /// Preset-only `dedup-cache-models`: hide a cache model whose snapshot
    /// path this preset also resolves to.
    pub dedup_cache_models: bool,
    /// The raw `key = value` pairs in file order, kept so the model object's
    /// `status.preset` block can reproduce the section as INI text the way
    /// b10621 does.
    pub raw: Vec<(String, String)>,
}

impl PresetSection {
    /// Merge `other` (the `[*]` global section) underneath `self`: values
    /// already present in `self` win, which is upstream's cascade order
    /// (global first, then the section's own options overwrite).
    fn merge_under_global(&mut self, global: &PresetSection) {
        macro_rules! fill {
            ($field:ident) => {
                if self.$field.is_none() {
                    self.$field = global.$field.clone();
                }
            };
        }
        fill!(model_path);
        fill!(hf_repo);
        fill!(ctx_size);
        fill!(n_parallel);
        fill!(temperature);
        fill!(top_k);
        fill!(top_p);
        fill!(min_p);
        fill!(seed);
        fill!(n_predict);
        fill!(stop_timeout);
        if self.aliases.is_empty() {
            self.aliases = global.aliases.clone();
        }
        if self.tags.is_empty() {
            self.tags = global.tags.clone();
        }
        self.load_on_startup |= global.load_on_startup;
        self.dedup_cache_models |= global.dedup_cache_models;
    }

    /// Render the section back as INI text for the b10621 `status.preset`
    /// block (host/port/alias/tags are stripped upstream; mlxcel sections
    /// never carry host/port, and alias/tags are stripped here).
    pub fn to_ini(&self, name: &str) -> String {
        let mut out = format!("[{name}]\n");
        for (key, value) in &self.raw {
            if key == "alias" || key == "tags" {
                continue;
            }
            out.push_str(&format!("{key} = {value}\n"));
        }
        out
    }
}

/// The parsed `--models-preset` file.
#[derive(Debug, Clone, Default)]
pub struct RouterPresets {
    /// `[*]`: merged into every model of every source.
    pub global: PresetSection,
    /// Named sections in file order (BTreeMap: name collisions replace, like
    /// upstream's map semantics).
    pub models: BTreeMap<String, PresetSection>,
}

impl RouterPresets {
    /// The section that applies to `name`, with the global section merged
    /// underneath. For a model with no named section, the global section
    /// alone applies.
    pub fn for_model(&self, name: &str) -> PresetSection {
        let mut section = self.models.get(name).cloned().unwrap_or_default();
        section.merge_under_global(&self.global);
        section
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty() && self.global == PresetSection::default()
    }
}

/// Which router CLI flags were explicitly given, so they win over preset
/// values (b10621 overlays the CLI-args preset on top of every model preset).
/// Detected once at startup from the process argument list.
#[derive(Debug, Clone, Copy, Default)]
pub struct PresetCliOverrides {
    pub ctx_size: bool,
    pub n_parallel: bool,
    pub min_p: bool,
    pub seed: bool,
    pub n_predict: bool,
    pub alias: bool,
    pub tags: bool,
}

impl PresetCliOverrides {
    /// Detect from the live process arguments. Long-form flags only; the
    /// `temperature` / `top-k` / `top-p` flags need no detection here because
    /// [`super::ServerStartupConfig`] already carries their `*_was_set` bits.
    pub fn detect() -> Self {
        use super::long_cli_flag_was_set;
        Self {
            ctx_size: long_cli_flag_was_set("ctx-size"),
            n_parallel: long_cli_flag_was_set("parallel"),
            min_p: long_cli_flag_was_set("min-p"),
            seed: long_cli_flag_was_set("seed"),
            n_predict: long_cli_flag_was_set("n-predict") || long_cli_flag_was_set("predict"),
            alias: long_cli_flag_was_set("alias"),
            tags: long_cli_flag_was_set("tags"),
        }
    }
}

/// Apply `section` onto a per-model startup-config clone, honoring the
/// CLI-wins overlay order. `startup.model_path` is set by the caller to the
/// entry's checkpoint directory before `build_server_config` re-resolves the
/// per-model defaults.
pub fn apply_section_to_startup(
    startup: &mut super::ServerStartupConfig,
    section: &PresetSection,
    cli: &PresetCliOverrides,
) {
    if let Some(ctx) = section.ctx_size
        && !cli.ctx_size
    {
        startup.ctx_size = ctx;
    }
    if let Some(par) = section.n_parallel
        && !cli.n_parallel
    {
        startup.n_parallel = par;
    }
    if let Some(temp) = section.temperature
        && !startup.temperature_was_set
    {
        startup.temperature = temp;
        startup.temperature_was_set = true;
    }
    if let Some(top_k) = section.top_k
        && !startup.top_k_was_set
    {
        startup.top_k = top_k;
        startup.top_k_was_set = true;
    }
    if let Some(top_p) = section.top_p
        && !startup.top_p_was_set
    {
        startup.top_p = top_p;
        startup.top_p_was_set = true;
    }
    if let Some(min_p) = section.min_p
        && !cli.min_p
    {
        startup.min_p = min_p;
    }
    if let Some(seed) = section.seed
        && !cli.seed
    {
        startup.seed = u64::try_from(seed).ok();
    }
    if let Some(n_predict) = section.n_predict
        && !cli.n_predict
    {
        startup.n_predict = n_predict;
    }
    if !section.aliases.is_empty() && !cli.alias {
        startup.model_aliases = section.aliases.clone();
        startup.model_alias = section.aliases.first().cloned();
    }
    if !section.tags.is_empty() && !cli.tags {
        startup.tags = Some(section.tags.join(","));
    }
}

fn parse_truthy(key: &str, section: &str, value: &str) -> anyhow::Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "enabled" => Ok(true),
        "0" | "false" | "off" | "no" | "disabled" => Ok(false),
        other => anyhow::bail!(
            "option '{key}' in preset '[{section}]' expects a boolean, found '{other}'"
        ),
    }
}

fn parse_num<T: std::str::FromStr>(key: &str, section: &str, value: &str) -> anyhow::Result<T> {
    value.trim().parse::<T>().map_err(|_| {
        anyhow::anyhow!("option '{key}' in preset '[{section}]' has an invalid value '{value}'")
    })
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Canonicalize an INI key: strip leading dashes and map `LLAMA_ARG_*`
/// environment spellings onto the long-option spelling, the same dual
/// key space upstream's `key_to_opt` map accepts.
fn canonical_key(raw: &str) -> String {
    let stripped = raw.trim_start_matches('-');
    let lower = stripped.to_ascii_lowercase();
    match stripped {
        "LLAMA_ARG_MODEL" => "model".into(),
        "LLAMA_ARG_HF_REPO" => "hf-repo".into(),
        "LLAMA_ARG_ALIAS" => "alias".into(),
        "LLAMA_ARG_TAGS" => "tags".into(),
        "LLAMA_ARG_CTX_SIZE" => "ctx-size".into(),
        "LLAMA_ARG_N_PARALLEL" => "parallel".into(),
        "LLAMA_ARG_N_PREDICT" => "n-predict".into(),
        _ => match lower.as_str() {
            "m" => "model".into(),
            "hf" | "hfr" => "hf-repo".into(),
            "a" => "alias".into(),
            "c" => "ctx-size".into(),
            "np" => "parallel".into(),
            "n" => "n-predict".into(),
            "temperature" => "temp".into(),
            _ => lower,
        },
    }
}

/// Parse one section's key/value pair into `section`. Unknown keys are a
/// hard error, matching upstream's strict `load_from_ini` (and keeping the
/// epic's no-silent-ignore rule: a key mlxcel cannot honor per model must
/// stop startup, not vanish).
fn apply_key(
    section: &mut PresetSection,
    section_name: &str,
    raw_key: &str,
    value: &str,
) -> anyhow::Result<()> {
    let key = canonical_key(raw_key);
    match key.as_str() {
        "version" => return Ok(()), // reserved, skipped upstream too
        "model" => section.model_path = Some(PathBuf::from(value)),
        "hf-repo" => section.hf_repo = Some(value.trim().to_string()),
        "alias" => section.aliases = split_list(value),
        "tags" => section.tags = split_list(value),
        "ctx-size" => section.ctx_size = Some(parse_num(&key, section_name, value)?),
        "parallel" => section.n_parallel = Some(parse_num(&key, section_name, value)?),
        "temp" => section.temperature = Some(parse_num(&key, section_name, value)?),
        "top-k" => section.top_k = Some(parse_num(&key, section_name, value)?),
        "top-p" => section.top_p = Some(parse_num(&key, section_name, value)?),
        "min-p" => section.min_p = Some(parse_num(&key, section_name, value)?),
        "seed" => section.seed = Some(parse_num(&key, section_name, value)?),
        "n-predict" => section.n_predict = Some(parse_num(&key, section_name, value)?),
        "load-on-startup" => {
            section.load_on_startup = parse_truthy(&key, section_name, value)?;
        }
        "stop-timeout" => section.stop_timeout = Some(parse_num(&key, section_name, value)?),
        "dedup-cache-models" => {
            section.dedup_cache_models = parse_truthy(&key, section_name, value)?;
        }
        other => anyhow::bail!(
            "option '{other}' in preset '[{section_name}]' is not translated by mlxcel presets \
             (#1438). Remove it from the preset, or configure it through the router's own \
             command line, which applies to every model."
        ),
    }
    section.raw.push((key, value.to_string()));
    Ok(())
}

/// Parse a b10621 `--models-preset` INI file.
///
/// Grammar (upstream `common/preset.cpp`): `[name]` section headers, `key =
/// value` pairs, `;` / `#` comment lines, blank lines. Keys before the first
/// header belong to the implicit `default` section, which upstream drops when
/// empty; `[*]` is the global preset merged into every model.
pub fn parse_preset_file(path: &Path) -> anyhow::Result<RouterPresets> {
    let text = std::fs::read_to_string(path).map_err(|err| {
        anyhow::anyhow!(
            "--models-preset {}: cannot read preset file: {err}",
            path.display()
        )
    })?;
    parse_preset_text(&text)
        .map_err(|err| anyhow::anyhow!("--models-preset {}: {err}", path.display()))
}

/// Parse preset INI text. Split from [`parse_preset_file`] for tests.
pub fn parse_preset_text(text: &str) -> anyhow::Result<RouterPresets> {
    let mut presets = RouterPresets::default();
    let mut current_name = String::from("default");
    let mut current = PresetSection::default();

    let finish = |name: &str, section: PresetSection, presets: &mut RouterPresets| {
        if name == "*" {
            presets.global = section;
        } else if name == "default" {
            // Upstream drops an empty implicit default section; a
            // non-empty one is a model named "default".
            if section != PresetSection::default() {
                presets.models.insert(name.to_string(), section);
            }
        } else {
            presets.models.insert(name.to_string(), section);
        }
    };

    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            let Some(name) = rest.strip_suffix(']') else {
                anyhow::bail!("line {}: malformed section header '{raw_line}'", idx + 1);
            };
            let name = name.trim();
            if name.is_empty() {
                anyhow::bail!("line {}: empty section name", idx + 1);
            }
            finish(
                &std::mem::replace(&mut current_name, name.to_string()),
                std::mem::take(&mut current),
                &mut presets,
            );
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            anyhow::bail!(
                "line {}: expected 'key = value', found '{raw_line}'",
                idx + 1
            );
        };
        let key = key.trim();
        if key.is_empty() {
            anyhow::bail!("line {}: empty option key", idx + 1);
        }
        apply_key(&mut current, &current_name, key, value.trim())
            .map_err(|err| anyhow::anyhow!("line {}: {err}", idx + 1))?;
    }
    finish(&current_name, current, &mut presets);
    Ok(presets)
}

#[cfg(test)]
#[path = "router_presets_tests.rs"]
mod router_presets_tests;
