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

use clap::{CommandFactory, Parser};
use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use super::{
    Cli, Commands, FAMILY_ORDER, PipelineParallelOptions, TensorParallelOptions,
    write_supported_models,
};

static CLI_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

struct ScopedEnv(Vec<(&'static str, Option<OsString>)>);

impl ScopedEnv {
    fn set(values: &[(&'static str, &'static str)]) -> Self {
        let saved = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in values {
            // SAFETY: every env-mutating test in this binary holds CLI_ENV_LOCK.
            unsafe { std::env::set_var(key, value) };
        }
        Self(saved)
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..) {
            // SAFETY: the guard is dropped before the outer CLI_ENV_LOCK guard.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

/// The `Default` impls for the parallelism option groups (used by the `mlxcel
/// run` lowering in `commands::run`) MUST match the values clap fills when the
/// corresponding flags are absent on `mlxcel generate`. If a `#[arg(default_*)]`
/// attribute ever changes without updating the matching `Default` impl, the
/// `run`-dispatched one-shot path would silently diverge from a plain
/// `generate`. This test pins the two together.
#[test]
fn run_defaults_match_clap_defaults() {
    let cli = Cli::try_parse_from(["mlxcel", "generate", "-m", "models/foo", "-p", "hi"])
        .expect("minimal generate must parse");
    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    let tp_default = TensorParallelOptions::default();
    assert_eq!(args.tensor_parallel.tp_size, tp_default.tp_size);
    assert_eq!(args.tensor_parallel.tp_moe_mode, tp_default.tp_moe_mode);
    assert_eq!(
        args.tensor_parallel.tp_embedding_mode,
        tp_default.tp_embedding_mode
    );
    assert_eq!(
        args.tensor_parallel.tp_lm_head_mode,
        tp_default.tp_lm_head_mode
    );

    let pp_default = PipelineParallelOptions::default();
    assert_eq!(args.pipeline_parallel.pp_size, pp_default.pp_size);
    assert_eq!(args.pipeline_parallel.pp_layers, pp_default.pp_layers);
    assert_eq!(
        args.pipeline_parallel.pp_micro_batch_size,
        pp_default.pp_micro_batch_size
    );
}

/// Issue #26: the rendered `mlxcel arch` output must mention every model
/// that is registered in `ALL_MODEL_TYPES`. This is the safety net that
/// catches the case where someone adds a `ModelType` variant but the
/// renderer silently drops it.
#[test]
fn supported_models_output_mentions_every_display_name() {
    let mut out = String::new();
    write_supported_models(&mut out).unwrap();

    for &mt in mlxcel::models::ALL_MODEL_TYPES {
        assert!(
            out.contains(mt.display_name()),
            "rendered output is missing display_name {:?} for {:?}",
            mt.display_name(),
            mt
        );
    }
}

#[test]
fn supported_models_output_explains_embedding_and_reranking_interfaces() {
    let mut out = String::new();
    write_supported_models(&mut out).unwrap();

    for expected in [
        "Embedding:",
        "LFM2.5-Embedding",
        "Nemotron-3-Embed",
        "Llama-Nemotron-VL-Embed",
        "Reranker:",
        "Cross-encoder sequence classifier",
        "Qwen3 and Qwen3-VL generative rerankers",
        "mlxcel embed",
        "mlxcel rerank",
        "--reranker-model",
        "docs/embeddings.md",
    ] {
        assert!(
            out.contains(expected),
            "`mlxcel arch` is missing {expected:?}:\n{out}"
        );
    }
}

#[test]
fn supported_models_output_explains_qwen_version_aliases() {
    let mut out = String::new();
    write_supported_models(&mut out).unwrap();

    for expected in [
        "Qwen 3.5 / 3.8 (Attention + GatedDeltaNet hybrid)",
        "Qwen 3.5 / 3.6 MoE (hybrid)",
        "Qwen 3.5 / 3.8 VLM",
        "Qwen 3.5 / 3.6 MoE VLM",
        "model_type: qwen3_5_moe",
        "Qwen 3.8 -> Qwen 3.5 dense/VLM (qwen3_5)",
    ] {
        assert!(
            out.contains(expected),
            "`mlxcel arch` is missing the Qwen alias {expected:?}:\n{out}"
        );
    }
}

/// Issue #26: the header must report the actual `ALL_MODEL_TYPES.len()`
/// instead of the previously-hardcoded `"57+"`. This guards against a
/// future regression where someone re-introduces a fixed count.
#[test]
fn supported_models_header_uses_actual_count() {
    let mut out = String::new();
    write_supported_models(&mut out).unwrap();

    let expected = format!(
        "Supported Model Architectures ({}):",
        mlxcel::models::ALL_MODEL_TYPES.len()
    );
    assert!(
        out.starts_with(&expected),
        "rendered header should start with {expected:?}, got {:?}",
        out.lines().next().unwrap_or("")
    );
}

/// Issue #26: the dead `docs/model_implementations.md` reference was
/// removed. Refuse to let it come back.
#[test]
fn supported_models_output_has_no_dead_doc_link() {
    let mut out = String::new();
    write_supported_models(&mut out).unwrap();

    assert!(
        !out.contains("model_implementations.md"),
        "rendered output must not reference the nonexistent doc \
         `docs/model_implementations.md` (issue #26)"
    );
    // Be slightly broader: the renderer must also not punt readers at
    // any external doc, since the new output is itself exhaustive.
    assert!(
        !out.to_lowercase().contains("for the full list"),
        "rendered output should be self-contained; no `For the full list…` pointer"
    );
}

#[test]
fn supported_models_output_keeps_muse_glimmer_as_an_architecture_name() {
    let mut out = String::new();
    write_supported_models(&mut out).unwrap();

    assert!(out.contains("Muse VLM:"), "missing Muse VLM family: {out}");
    assert!(
        out.contains("Muse Glimmer 30B VLM"),
        "missing Muse Glimmer architecture: {out}"
    );
    for checkpoint_detail in ["BF16/MLX 4-bit", "mixed 2048", "ATEM"] {
        assert!(
            !out.contains(checkpoint_detail),
            "architecture list must not include Muse checkpoint detail {checkpoint_detail:?}: {out}"
        );
    }
}

#[test]
fn top_level_help_keeps_model_specific_guidance_in_the_model_catalog() {
    let mut command = Cli::command();
    let help = command.render_long_help().to_string();

    for expected in ["mlxcel arch", "supported-models.md", "distributed.md"] {
        assert!(
            help.contains(expected),
            "missing {expected:?} in help:\n{help}"
        );
    }
    for model_specific in [
        "Muse Glimmer 30B checkpoints",
        "59.55 GB",
        "Gemma 4 E2B-style",
    ] {
        assert!(
            !help.contains(model_specific),
            "top-level help must leave checkpoint-specific guidance in the model catalog; \
             found {model_specific:?} in:\n{help}"
        );
    }
}

#[test]
fn settings_cli_mlxcel_serve_help_exposes_flag() {
    let mut command = Cli::command();
    let help = command
        .find_subcommand_mut("serve")
        .expect("serve subcommand")
        .render_long_help()
        .to_string();

    assert!(
        help.contains("--settings"),
        "`mlxcel serve --help` is missing --settings:\n{help}"
    );
}

#[test]
fn settings_cli_mlxcel_serve_defaults_off_and_explicit_flag_enables() {
    let default = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo"])
        .expect("serve defaults should parse");
    let Commands::Serve(default) = default.command else {
        panic!("expected serve command");
    };
    assert!(!default.settings, "the settings endpoint must default off");

    let enabled = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--settings"])
        .expect("serve --settings should parse");
    let Commands::Serve(enabled) = enabled.command else {
        panic!("expected serve command");
    };
    assert!(enabled.settings, "--settings must propagate through clap");
}

#[test]
fn embedding_and_reranking_commands_are_discoverable_in_help() {
    let mut command = Cli::command();
    let top_level = command.render_long_help().to_string();
    for expected in [
        "embed",
        "Embed texts (or images) with an embedding checkpoint",
        "rerank",
        "Score query/document relevance with a reranker checkpoint",
    ] {
        assert!(
            top_level.contains(expected),
            "top-level help is missing {expected:?}:\n{top_level}"
        );
    }

    let mut command = Cli::command();
    let embed_help = command
        .find_subcommand_mut("embed")
        .expect("embed subcommand")
        .render_long_help()
        .to_string();
    for option in [
        "--model",
        "--prompt",
        "--image",
        "--instruction",
        "--dimensions",
        "--max-length",
        "--batch-size",
        "--models-dir",
        "--json",
    ] {
        assert!(
            embed_help.contains(option),
            "`mlxcel embed --help` is missing {option}:\n{embed_help}"
        );
    }

    let mut command = Cli::command();
    let rerank_help = command
        .find_subcommand_mut("rerank")
        .expect("rerank subcommand")
        .render_long_help()
        .to_string();
    for option in [
        "--model",
        "--query",
        "--query-image",
        "--document",
        "--image",
        "--instruction",
        "--top-n",
        "--max-length",
        "--batch-size",
        "--models-dir",
        "--json",
    ] {
        assert!(
            rerank_help.contains(option),
            "`mlxcel rerank --help` is missing {option}:\n{rerank_help}"
        );
    }
}

/// `FAMILY_ORDER` controls the rendered section order. If a future
/// `ModelType` is given a brand-new family that the order table has
/// never seen, the renderer still emits it (alphabetically, at the
/// end) but the layout drifts. This test fails fast so a maintainer
/// will update `FAMILY_ORDER` deliberately rather than discovering
/// it via user bug reports.
#[test]
fn family_order_is_exhaustive() {
    let mut missing: Vec<&'static str> = Vec::new();
    for &mt in mlxcel::models::ALL_MODEL_TYPES {
        let family = mt.family();
        if !FAMILY_ORDER.contains(&family) && !missing.contains(&family) {
            missing.push(family);
        }
    }
    assert!(
        missing.is_empty(),
        "FAMILY_ORDER does not list every family used by ModelType::family(); \
         missing: {missing:?}. Add the new family/families to FAMILY_ORDER in \
         src/main.rs in the desired display position."
    );
}

/// `FAMILY_ORDER` should not list a family that nothing currently uses —
/// that suggests stale ordering left over after a family rename or removal.
#[test]
fn family_order_has_no_orphans() {
    let used: std::collections::HashSet<&'static str> = mlxcel::models::ALL_MODEL_TYPES
        .iter()
        .map(|mt| mt.family())
        .collect();
    let orphans: Vec<&'static str> = FAMILY_ORDER
        .iter()
        .copied()
        .filter(|f| !used.contains(f))
        .collect();
    assert!(
        orphans.is_empty(),
        "FAMILY_ORDER mentions families that no ModelType currently uses: \
         {orphans:?}. Remove them or update ModelType::metadata()."
    );
}

#[test]
fn generate_command_parses_tensor_parallel_flags() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "generate",
        "-m",
        "models/foo",
        "-p",
        "hello",
        "--tp-size",
        "2",
        "--tp-moe-mode",
        "within_expert",
        "--tp-embedding-mode",
        "vocab_parallel",
        "--tp-lm-head-mode",
        "replicated",
    ])
    .unwrap();

    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    assert_eq!(args.tensor_parallel.tp_size, 2);
    assert_eq!(args.tensor_parallel.tp_moe_mode, "within_expert");
    assert_eq!(args.tensor_parallel.tp_embedding_mode, "vocab_parallel");
    assert_eq!(args.tensor_parallel.tp_lm_head_mode, "replicated");
}

// (A4): CLI argument-parsing tests for the `--surgery <FILE>`
// flag on the `generate` and `serve` subcommands. These tests only cover
// the clap surface — they do not invoke the surgery pipeline or touch
// any model weights. The end-to-end behavior is exercised by the
// integration test in `tests/surgery_cli.rs`.

#[cfg(feature = "surgery")]
#[test]
fn generate_command_accepts_surgery_flag_with_path() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "generate",
        "-m",
        "models/foo",
        "-p",
        "hello",
        "--surgery",
        "config/surgery.yaml",
    ])
    .expect("clap must accept --surgery <path>");

    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    assert_eq!(
        args.surgery,
        Some(std::path::PathBuf::from("config/surgery.yaml")),
        "--surgery must round-trip through clap as PathBuf"
    );
}

#[cfg(feature = "surgery")]
#[test]
fn generate_command_surgery_flag_defaults_to_none() {
    // Baseline path: omitting the flag yields `None`, which keeps the
    // load path bit-exact with earlier main (acceptance criterion (e)).
    let cli = Cli::try_parse_from(["mlxcel", "generate", "-m", "models/foo", "-p", "hello"])
        .expect("clap must accept generate without --surgery");

    let Commands::Generate(args) = cli.command else {
        panic!("expected generate command");
    };

    assert!(
        args.surgery.is_none(),
        "absent --surgery flag must resolve to None"
    );
}

#[cfg(feature = "surgery")]
#[test]
fn serve_command_accepts_surgery_flag_with_path() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--surgery",
        "config/surgery.yaml",
    ])
    .expect("clap must accept --surgery on serve");

    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        args.surgery,
        Some(std::path::PathBuf::from("config/surgery.yaml")),
        "serve --surgery must round-trip through clap as PathBuf"
    );
}

#[cfg(feature = "surgery")]
#[test]
fn serve_command_surgery_flag_defaults_to_none() {
    let cli = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo"])
        .expect("clap must accept serve without --surgery");

    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };

    assert!(
        args.surgery.is_none(),
        "absent --surgery flag on serve must resolve to None"
    );
}

// ── llama-server model-source flags on `mlxcel serve` (issue #1434) ──
//
// `src/bin/mlx_server.rs`'s test module carries the matching assertions for
// `mlxcel-server`; both binaries must accept the same b10621 spellings, which
// is what `tests/llama_compat_manifest.rs` then holds the manifest to.

#[test]
fn serve_warmup_defaults_to_enabled_and_no_warmup_disables_it() {
    let cli = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo"])
        .expect("clap must accept serve without --warmup");
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };
    assert!(args.warmup, "warmup defaults to enabled");

    let cli = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--no-warmup"])
        .expect("clap must accept --no-warmup on serve");
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };
    assert!(
        !mlxcel::server::resolve_compat_toggle(args.warmup, args._no_warmup),
        "--no-warmup must disable the warmup pass on `mlxcel serve` too"
    );
}

#[test]
fn serve_accepts_every_b10621_model_source_flag() {
    let cli = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "--hf-repo",
        "mlx-community/Qwen3-4B-4bit",
        "--hf-token",
        "hf_example",
        "--offline",
        "--hf-file",
        "model.gguf",
        "--model-url",
        "https://example.com/model.gguf",
        "--docker-repo",
        "ai/gemma3",
    ])
    .expect("clap must accept the b10621 model-source flags on serve");
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve command");
    };
    assert_eq!(args.hf_repo.as_deref(), Some("mlx-community/Qwen3-4B-4bit"));
    assert_eq!(args.hf_token.as_deref(), Some("hf_example"));
    assert!(args.offline);
    assert_eq!(args.hf_file.as_deref(), Some("model.gguf"));
    assert_eq!(
        args.model_url.as_deref(),
        Some("https://example.com/model.gguf")
    );
    assert_eq!(args.docker_repo.as_deref(), Some("ai/gemma3"));
    // `-m` is no longer required when another source is supplied; the
    // model-source translation reports a missing source instead of clap.
    assert_eq!(args.model, None);
}

// ── Drafter flag aliases (issue #464) ───────────────────────────
//
// `mlxcel serve` uses the mlx-lm-style `--draft-model` / `--draft-max`
// spelling as the primary flag names, and `mlxcel-server` uses the
// llama-server-style `--model-draft` / `--draft` spelling. Both binaries
// now accept both spellings via `visible_alias`, so a command line copied
// from one to the other parses unchanged. These tests pin that both
// spellings resolve to the identical `ServeArgs` field value on `mlxcel
// serve`; `src/bin/mlx_server.rs`'s test module carries the matching
// assertions for `mlxcel-server`.

#[test]
fn serve_draft_model_and_model_draft_aliases_resolve_identically() {
    let primary = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--draft-model",
        "models/draft",
    ])
    .expect("--draft-model must parse on `mlxcel serve`");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };

    let aliased = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--model-draft",
        "models/draft",
    ])
    .expect("--model-draft alias must parse on `mlxcel serve`");
    let Commands::Serve(aliased_args) = aliased.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        primary_args.draft_model, aliased_args.draft_model,
        "--draft-model and its --model-draft alias must resolve to the same drafter path"
    );
    assert_eq!(
        primary_args.draft_model,
        Some(std::path::PathBuf::from("models/draft"))
    );
}

#[test]
fn serve_temp_and_temperature_aliases_resolve_identically() {
    let primary = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--temp", "0.37"])
        .expect("--temp must parse");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };
    let alias = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--temperature",
        "0.37",
    ])
    .expect("--temperature must parse");
    let Commands::Serve(alias_args) = alias.command else {
        panic!("expected serve command");
    };
    assert_eq!(primary_args.temp, alias_args.temp);
}

#[test]
fn serve_canonical_llama_envs_and_endpoint_precedence_parse_together() {
    let _lock = CLI_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _env = ScopedEnv::set(&[
        ("LLAMA_ARG_BATCH", "111"),
        ("LLAMA_ARG_BATCH_SIZE", "999"),
        ("LLAMA_ARG_UBATCH", "22"),
        ("LLAMA_ARG_UBATCH_SIZE", "999"),
        ("LLAMA_ARG_SPEC_DRAFT_MODEL", "models/canonical-draft"),
        ("LLAMA_ARG_MODEL_DRAFT", "models/legacy-draft"),
        ("LLAMA_ARG_LOG_FILE", "canonical.log"),
        ("LLAMA_LOG_FILE", "legacy.log"),
        ("LLAMA_ARG_CHAT_TEMPLATE", "{{ messages }}"),
        ("LLAMA_ARG_CHAT_TEMPLATE_FILE", "canonical.jinja"),
        ("LLAMA_ARG_ENDPOINT_METRICS", "true"),
        ("LLAMA_ARG_ENDPOINT_PROPS", "true"),
        ("LLAMA_ARG_ENDPOINT_SLOTS", "false"),
    ]);

    let cli = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo"])
        .expect("canonical llama envs must parse");
    let Commands::Serve(mut args) = cli.command else {
        panic!("expected serve command");
    };
    mlxcel::server::env_fallback_batch_size(&mut args.batch_size);
    mlxcel::server::env_fallback_ubatch_size(&mut args.ubatch_size);
    mlxcel::server::env_fallback_draft_model(&mut args.draft_model);
    mlxcel::server::env_fallback_log_file(&mut args.log_file);
    mlxcel::server::env_fallback_endpoint_slots(&mut args.slots, false, false);
    assert_eq!(args.batch_size, Some(111));
    assert_eq!(args.ubatch_size, Some(22));
    assert_eq!(
        args.draft_model.as_deref(),
        Some(std::path::Path::new("models/canonical-draft"))
    );
    assert_eq!(
        args.log_file.as_deref(),
        Some(std::path::Path::new("canonical.log"))
    );
    assert_eq!(args.chat_template.as_deref(), Some("{{ messages }}"));
    assert_eq!(
        args.chat_template_file.as_deref(),
        Some(std::path::Path::new("canonical.jinja"))
    );
    assert!(args.metrics);
    assert!(args.props);
    assert!(!args.slots);

    let cli_override = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--batch-size",
        "333",
        "--slots",
    ])
    .expect("CLI must override canonical env values");
    let Commands::Serve(mut cli_args) = cli_override.command else {
        panic!("expected serve command");
    };
    mlxcel::server::env_fallback_endpoint_slots(&mut cli_args.slots, true, false);
    assert_eq!(cli_args.batch_size, Some(333));
    assert!(cli_args.slots);

    let no_slots = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--no-slots"])
        .expect("--no-slots must parse with endpoint env");
    let Commands::Serve(mut no_slots_args) = no_slots.command else {
        panic!("expected serve command");
    };
    mlxcel::server::env_fallback_endpoint_slots(&mut no_slots_args.slots, false, true);
    assert!(no_slots_args._no_slots);
    assert!(!(no_slots_args.slots && !no_slots_args._no_slots));
}

#[test]
fn serve_draft_max_and_draft_aliases_resolve_identically() {
    let primary = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--draft-max", "24"])
        .expect("--draft-max must parse on `mlxcel serve`");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };

    let aliased = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--draft", "24"])
        .expect("--draft alias must parse on `mlxcel serve`");
    let Commands::Serve(aliased_args) = aliased.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        primary_args.draft_max, aliased_args.draft_max,
        "--draft-max and its --draft alias must resolve to the same token budget"
    );
    assert_eq!(primary_args.draft_max, 24);
}

#[test]
fn serve_slots_flags_use_last_occurrence() {
    let disabled = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--slots",
        "--no-slots",
    ])
    .expect("--slots --no-slots must parse");
    let Commands::Serve(disabled) = disabled.command else {
        panic!("expected serve command");
    };
    assert!(!(disabled.slots && !disabled._no_slots));

    let enabled = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--no-slots",
        "--slots",
    ])
    .expect("--no-slots --slots must parse");
    let Commands::Serve(enabled) = enabled.command else {
        panic!("expected serve command");
    };
    assert!(enabled.slots && !enabled._no_slots);
}

// ── Server flag spelling parity (issue #1109) ───────────────────
//
// `mlxcel serve` and `mlxcel-server` are two hand-maintained clap
// definitions of the same server, and four flag spellings had drifted
// apart. `--n-parallel` / `--parallel` was the worst case: neither
// spelling worked on both binaries, so a command line copied between
// them failed to parse even though both flags read the same
// `LLAMA_ARG_N_PARALLEL` env var. These tests pin that each spelling
// now resolves to the identical `ServeArgs` field value here;
// `src/bin/mlx_server.rs`'s test module carries the matching assertions
// for `mlxcel-server`, and `tests/cli_help_consistency.rs` asserts the
// two binaries accept the same set of spellings so a fifth divergence
// cannot land silently.

#[test]
fn serve_reasoning_alias_field_defaults_parses_and_rejects_unknown_values() {
    let default = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo"])
        .expect("default serve arguments must parse");
    let Commands::Serve(default) = default.command else {
        panic!("expected serve command");
    };
    assert_eq!(
        default.reasoning_alias_field,
        mlxcel::server::ReasoningAliasField::Reasoning
    );

    for (value, expected) in [
        ("reasoning", mlxcel::server::ReasoningAliasField::Reasoning),
        ("none", mlxcel::server::ReasoningAliasField::None),
    ] {
        let parsed = Cli::try_parse_from([
            "mlxcel",
            "serve",
            "-m",
            "models/foo",
            "--reasoning-alias-field",
            value,
        ])
        .expect("known reasoning alias field must parse");
        let Commands::Serve(args) = parsed.command else {
            panic!("expected serve command");
        };
        assert_eq!(args.reasoning_alias_field, expected, "{value}");
    }

    let error = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--reasoning-alias-field",
        "other",
    ])
    .expect_err("unknown reasoning alias field must be rejected");
    let text = error.to_string();
    assert!(text.contains("--reasoning-alias-field"), "{text}");
    assert!(text.contains("none, reasoning"), "{text}");
}

#[test]
fn serve_n_parallel_and_parallel_aliases_resolve_identically() {
    let primary = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--n-parallel", "2"])
        .expect("--n-parallel must parse on `mlxcel serve`");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };

    let aliased = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--parallel", "2"])
        .expect("--parallel alias must parse on `mlxcel serve`");
    let Commands::Serve(aliased_args) = aliased.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        primary_args.n_parallel, aliased_args.n_parallel,
        "--n-parallel and its --parallel alias must resolve to the same slot count"
    );
    assert_eq!(primary_args.n_parallel, 2);
}

#[test]
fn serve_n_predict_and_predict_aliases_resolve_identically() {
    let primary = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--n-predict", "64"])
        .expect("--n-predict must parse on `mlxcel serve`");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };

    let aliased = Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--predict", "64"])
        .expect("--predict alias must parse on `mlxcel serve`");
    let Commands::Serve(aliased_args) = aliased.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        primary_args.n_predict, aliased_args.n_predict,
        "--n-predict and its --predict alias must resolve to the same token cap"
    );
    assert_eq!(primary_args.n_predict, 64);
}

/// Every long name, alias, and short form on `mlxcel serve` must be distinct.
///
/// clap detects a duplicate itself, but only behind a `debug_assert` in
/// `Command::_build_self`, and `[profile.test-fast]` inherits `release`, so
/// `debug-assertions` is off in the profile this repository verifies with. A
/// `visible_alias` that collided with an existing flag would therefore be
/// silently last-wins rather than a panic, and issue #1109 added four of them.
/// Asserting uniqueness directly keeps the guard in every profile.
#[test]
fn serve_flag_names_and_aliases_are_unique() {
    let command = Cli::command();
    let serve = command
        .find_subcommand("serve")
        .expect("`mlxcel serve` subcommand exists");

    let mut longs: Vec<String> = Vec::new();
    let mut shorts: Vec<char> = Vec::new();
    for arg in serve.get_arguments() {
        if let Some(long) = arg.get_long() {
            longs.push(long.to_string());
        }
        if let Some(aliases) = arg.get_all_aliases() {
            longs.extend(aliases.into_iter().map(str::to_string));
        }
        if let Some(short) = arg.get_short() {
            shorts.push(short);
        }
        if let Some(aliases) = arg.get_all_short_aliases() {
            shorts.extend(aliases);
        }
    }

    let duplicate_longs = duplicates(&longs);
    assert!(
        duplicate_longs.is_empty(),
        "`mlxcel serve` declares these long names more than once: {duplicate_longs:?}. \
         clap resolves a duplicate silently in this profile, so the second definition \
         would shadow the first with no error."
    );

    let duplicate_shorts = duplicates(&shorts);
    assert!(
        duplicate_shorts.is_empty(),
        "`mlxcel serve` declares these short forms more than once: {duplicate_shorts:?}"
    );
}

/// Values appearing more than once in `items`, sorted and deduplicated.
fn duplicates<T: Clone + Ord>(items: &[T]) -> Vec<T> {
    let mut sorted = items.to_vec();
    sorted.sort();
    let mut repeated: Vec<T> = sorted
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0].clone())
        .collect();
    repeated.dedup();
    repeated
}

/// `--adapter` / `--lora` was already symmetric on this binary before issue
/// #1109; the divergence was on `mlxcel-server`, which accepted only `--lora`.
/// Pinning both directions keeps the whole table covered from both ends rather
/// than only the end that had to change.
#[test]
fn serve_adapter_and_lora_aliases_resolve_identically() {
    let primary = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--adapter",
        "lora/foo",
    ])
    .expect("--adapter must parse on `mlxcel serve`");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };

    let aliased =
        Cli::try_parse_from(["mlxcel", "serve", "-m", "models/foo", "--lora", "lora/foo"])
            .expect("--lora alias must parse on `mlxcel serve`");
    let Commands::Serve(aliased_args) = aliased.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        primary_args.adapter, aliased_args.adapter,
        "--adapter and its --lora alias must resolve to the same adapter path"
    );
    assert_eq!(primary_args.adapter, Some("lora/foo".to_string()));
}

#[test]
fn serve_dry_sequence_breaker_singular_and_plural_aliases_resolve_identically() {
    let primary = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--dry-sequence-breaker",
        "a,b",
    ])
    .expect("the singular --dry-sequence-breaker must parse on `mlxcel serve`");
    let Commands::Serve(primary_args) = primary.command else {
        panic!("expected serve command");
    };

    // The plural was this binary's only spelling before issue #1109 made the
    // singular primary, so it has to keep parsing.
    let aliased = Cli::try_parse_from([
        "mlxcel",
        "serve",
        "-m",
        "models/foo",
        "--dry-sequence-breakers",
        "a,b",
    ])
    .expect("the plural --dry-sequence-breakers alias must parse on `mlxcel serve`");
    let Commands::Serve(aliased_args) = aliased.command else {
        panic!("expected serve command");
    };

    assert_eq!(
        primary_args.dry_sequence_breakers, aliased_args.dry_sequence_breakers,
        "both DRY breaker spellings must resolve to the same breaker list"
    );
    assert_eq!(
        primary_args.dry_sequence_breakers,
        vec!["a".to_string(), "b".to_string()]
    );
}

#[test]
fn list_command_parses_to_list() {
    let cli = Cli::try_parse_from(["mlxcel", "list"]).expect("bare `list` must parse");
    let Commands::List(args) = cli.command else {
        panic!("expected list command");
    };
    assert!(
        args.models_dir.is_none(),
        "bare `list` must default --models-dir to None"
    );
}

#[test]
fn list_ls_alias_parses_to_list() {
    let cli = Cli::try_parse_from(["mlxcel", "ls"]).expect("`ls` alias must parse");
    assert!(
        matches!(cli.command, Commands::List(_)),
        "`ls` alias must map to the List command"
    );
}

#[test]
fn list_command_accepts_models_dir() {
    let cli = Cli::try_parse_from(["mlxcel", "list", "--models-dir", "/tmp/x"])
        .expect("`list --models-dir` must parse");
    let Commands::List(args) = cli.command else {
        panic!("expected list command");
    };
    assert_eq!(
        args.models_dir,
        Some(std::path::PathBuf::from("/tmp/x")),
        "--models-dir must be captured on the List command"
    );
}

#[test]
fn list_command_defaults_output_flags() {
    use crate::commands::models::SortKey;
    let cli = Cli::try_parse_from(["mlxcel", "list"]).expect("bare `list` must parse");
    let Commands::List(args) = cli.command else {
        panic!("expected list command");
    };
    assert!(!args.json, "--json must default off");
    assert!(!args.quiet, "--quiet must default off");
    assert!(!args.verbose, "--verbose must default off");
    assert_eq!(args.sort, SortKey::Name, "--sort must default to name");
}

#[test]
fn list_command_parses_output_flags_and_sort() {
    use crate::commands::models::SortKey;
    let cli = Cli::try_parse_from(["mlxcel", "list", "-v", "--sort", "size"])
        .expect("`list -v --sort size` must parse");
    let Commands::List(args) = cli.command else {
        panic!("expected list command");
    };
    assert!(args.verbose, "-v must set verbose");
    assert_eq!(args.sort, SortKey::Size, "--sort size must parse");

    let cli = Cli::try_parse_from(["mlxcel", "list", "--sort", "modified"])
        .expect("`--sort modified` must parse");
    let Commands::List(args) = cli.command else {
        panic!("expected list command");
    };
    assert_eq!(args.sort, SortKey::Modified);
}

#[test]
fn list_command_quiet_short_flag() {
    let cli = Cli::try_parse_from(["mlxcel", "list", "-q"]).expect("`list -q` must parse");
    let Commands::List(args) = cli.command else {
        panic!("expected list command");
    };
    assert!(args.quiet, "-q must set quiet");
}

#[test]
fn list_command_rejects_conflicting_output_modes() {
    // --json is mutually exclusive with --quiet and --verbose; --quiet with -v.
    assert!(
        Cli::try_parse_from(["mlxcel", "list", "--json", "--quiet"]).is_err(),
        "--json and --quiet must conflict"
    );
    assert!(
        Cli::try_parse_from(["mlxcel", "list", "--json", "--verbose"]).is_err(),
        "--json and --verbose must conflict"
    );
    assert!(
        Cli::try_parse_from(["mlxcel", "list", "--quiet", "--verbose"]).is_err(),
        "--quiet and --verbose must conflict"
    );
    // --sort is allowed alongside any single mode.
    assert!(
        Cli::try_parse_from(["mlxcel", "list", "--json", "--sort", "size"]).is_ok(),
        "--sort must be allowed with --json"
    );
}

#[test]
fn list_command_rejects_unknown_sort() {
    assert!(
        Cli::try_parse_from(["mlxcel", "list", "--sort", "bogus"]).is_err(),
        "an unknown --sort value must be rejected"
    );
}

#[test]
fn list_command_rejects_removed_local_flag() {
    // The `--local` flag was removed (issue #138): local is now the default,
    // so clap must reject it as an unknown argument. This pins the removal so
    // the flag cannot silently return.
    assert!(
        Cli::try_parse_from(["mlxcel", "list", "--local"]).is_err(),
        "the removed `--local` flag must be rejected as an unknown argument"
    );
}

#[test]
fn arch_command_parses_to_arch() {
    let cli = Cli::try_parse_from(["mlxcel", "arch"]).expect("`arch` must parse");
    assert!(
        matches!(cli.command, Commands::Arch(_)),
        "`arch` must map to the Arch command"
    );
}

#[test]
fn arch_supported_alias_parses_to_arch() {
    let cli = Cli::try_parse_from(["mlxcel", "supported"]).expect("`supported` alias must parse");
    assert!(
        matches!(cli.command, Commands::Arch(_)),
        "`supported` alias must map to the Arch command"
    );
}

// ── Space-separated negative values on `mlxcel serve` (issue #1459) ───
//
// b10621 accepts `llama-server --seed -1`, and `mlxcel serve` rejected the
// space form on all 122 of its value-taking long options with "unexpected
// argument '-1' found" until `allow_negative_numbers` landed on the
// `ServeArgs` command attributes. The `mlxcel-server` half of this lives in
// the test module of `src/bin/mlx_server.rs`; both binaries need their own
// coverage because they are separate clap commands built from separate
// structs.

/// Long options that decline a space-separated value of ANY sign because they
/// declare `require_equals`, so `--opt -1` failing on them is not a
/// negative-number defect. Both are mlxcel-only cache knobs with no b10621
/// counterpart, so no llama-server command line reaches them.
const SERVE_REQUIRE_EQUALS_LONGS: [&str; 2] = ["--apc-enabled", "--prompt-cache-enabled"];

/// `mlxcel serve -m models/foo <rest...>`. `serve` requires `--model`, so
/// every case carries it; the helper keeps the argv under test readable.
fn serve_argv(rest: &[&str]) -> Vec<String> {
    ["mlxcel", "serve", "-m", "models/foo"]
        .iter()
        .chain(rest.iter())
        .map(|token| (*token).to_owned())
        .collect()
}

/// The `ServeArgs` produced by `mlxcel serve -m models/foo <rest...>`.
fn parse_serve_args(rest: &[&str]) -> super::ServeArgs {
    let cli = Cli::try_parse_from(serve_argv(rest)).expect("`mlxcel serve` args should parse");
    match cli.command {
        Commands::Serve(args) => args,
        other => panic!("expected the serve command, got {other:?}"),
    }
}

/// The clap error from `mlxcel serve -m models/foo <rest...>`, which must fail.
fn serve_parse_error(rest: &[&str], expectation: &str) -> clap::Error {
    Cli::try_parse_from(serve_argv(rest)).expect_err(expectation)
}

#[test]
fn serve_accepts_a_space_separated_negative_number_on_every_value_taking_option() {
    let mut command = Cli::command();
    command.build();
    let serve = command
        .get_subcommands()
        .find(|sub| sub.get_name() == "serve")
        .expect("the serve subcommand must exist")
        .clone();

    let options: Vec<(String, bool)> = serve
        .get_arguments()
        .filter(|arg| arg.get_num_args().is_some_and(|r| r.takes_values()))
        .filter_map(|arg| {
            arg.get_long()
                .map(|long| (format!("--{long}"), arg.is_require_equals_set()))
        })
        .collect();

    let mut swept = 0usize;
    let mut exempt: Vec<String> = Vec::new();
    let mut rejected: Vec<String> = Vec::new();
    for (long, require_equals) in &options {
        if *require_equals {
            // Prove the exemption rather than assume it: an option that
            // requires `=` must decline a space-separated positive value too,
            // otherwise a real negative-number rejection could hide here.
            assert!(
                Cli::try_parse_from(serve_argv(&[long, "1"])).is_err(),
                "{long} declares require_equals but bound a space-separated positive value; \
                 the #1459 sweep exemption no longer describes it"
            );
            exempt.push(long.clone());
            continue;
        }
        swept += 1;
        let unexpected_argument = Cli::try_parse_from(serve_argv(&[long, "-1"]))
            .is_err_and(|err| err.kind() == clap::error::ErrorKind::UnknownArgument);
        if unexpected_argument {
            rejected.push(long.clone());
        }
    }

    assert!(
        rejected.is_empty(),
        "these `mlxcel serve` options still reject a space-separated negative value that \
         b10621 accepts: {rejected:?}"
    );
    exempt.sort();
    assert_eq!(
        exempt, SERVE_REQUIRE_EQUALS_LONGS,
        "the set of options that take no space-separated value has changed. If a b10621 option \
         gained require_equals, that is a compatibility regression; if a new mlxcel-only knob \
         gained it, extend SERVE_REQUIRE_EQUALS_LONGS deliberately"
    );
    // 122 value-taking long options at the time of the fix, minus the two
    // require_equals knobs. The floor guards against a sweep that silently
    // stops enumerating.
    assert!(
        swept >= 100,
        "only {swept} value-taking long options were swept; the enumeration has collapsed"
    );
}

#[test]
fn serve_negative_seed_predict_and_reasoning_budget_parse_in_space_form() {
    for rest in [&["--seed", "-1"][..], &["-s", "-1"][..]] {
        assert_eq!(
            parse_serve_args(rest).seed,
            -1,
            "{rest:?} must bind -1 to the seed"
        );
    }
    for rest in [&["--n-predict", "-1"][..], &["--predict", "-1"][..]] {
        assert_eq!(
            parse_serve_args(rest).n_predict,
            -1,
            "{rest:?} must bind -1 to the predict budget"
        );
    }
    assert_eq!(
        parse_serve_args(&["--reasoning-budget", "-1"]).reasoning_budget,
        -1
    );

    // -1 is also the default for all three, so a distinct negative is what
    // proves the token was consumed as the value rather than dropped.
    assert_eq!(parse_serve_args(&["--seed", "-7"]).seed, -7);
    assert_eq!(parse_serve_args(&["--n-predict", "-7"]).n_predict, -7);
    assert_eq!(
        parse_serve_args(&["--reasoning-budget", "-7"]).reasoning_budget,
        -7
    );
}

#[test]
fn serve_accepts_a_negative_float_in_space_form() {
    let args = parse_serve_args(&["--presence-penalty", "-1.5"]);
    assert!(
        (args.presence_penalty - (-1.5)).abs() < f32::EPSILON,
        "--presence-penalty -1.5 must bind -1.5, got {}",
        args.presence_penalty
    );
}

#[test]
fn serve_equals_form_negative_values_still_parse() {
    assert_eq!(parse_serve_args(&["--seed=-7"]).seed, -7);
    assert_eq!(parse_serve_args(&["--n-predict=-7"]).n_predict, -7);
    assert_eq!(parse_serve_args(&["--predict=-7"]).n_predict, -7);
    assert_eq!(
        parse_serve_args(&["--reasoning-budget=-7"]).reasoning_budget,
        -7
    );
}

#[test]
fn serve_rejects_a_negative_value_in_the_value_parser_for_a_non_negative_option() {
    for (long, rest) in [
        ("--port", &["--port", "-1"][..]),
        ("--ctx-size", &["--ctx-size", "-1"][..]),
        // Custom unit-interval parser rather than a clap numeric range, so it
        // exercises the other rejection path.
        (
            "--diffusion-threshold",
            &["--diffusion-threshold", "-1"][..],
        ),
    ] {
        let err = serve_parse_error(
            rest,
            "a negative value outside the option's domain must be rejected",
        );
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ValueValidation,
            "{long} must fail in its value parser, not as an unknown argument"
        );
        let text = err.to_string();
        assert!(
            text.contains(long) && text.contains("-1"),
            "the {long} rejection must name the option and the offending value, got: {text}"
        );
        assert!(
            !text.contains("unexpected argument"),
            "{long} must no longer report the value as an unexpected argument, got: {text}"
        );
    }
}

#[test]
fn serve_sends_a_negative_parallel_to_the_value_parser_not_the_argv_parser() {
    // b10621 documents a negative in `--parallel`'s help text while mlxcel's
    // field is `usize`. Command-wide `allow_negative_numbers` makes `-1` a
    // candidate value here, so the type parser is what must reject it, with a
    // message that names the option and the value.
    let err = serve_parse_error(
        &["--parallel", "-1"],
        "--parallel is usize and must reject a negative slot count",
    );
    assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    let text = err.to_string();
    assert!(
        text.contains("parallel") && text.contains("-1"),
        "the --parallel rejection must name the option and the value, got: {text}"
    );
    assert!(
        !text.contains("unexpected argument"),
        "--parallel -1 must no longer be an argv-parser error, got: {text}"
    );
}

#[test]
fn serve_still_rejects_a_negative_number_with_no_option_awaiting_a_value() {
    for rest in [
        &["-1"][..],
        // The seed consumes the first negative; the second has nothing pending
        // and must not be silently absorbed.
        &["--seed", "-1", "-2"][..],
    ] {
        let err = serve_parse_error(
            rest,
            "a stray negative number must not be silently consumed",
        );
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::UnknownArgument,
            "{rest:?} must report the stray negative number"
        );
    }
}

#[test]
fn serve_reports_a_mistyped_flag_after_an_option_instead_of_swallowing_it() {
    // This is the whole reason the fix is `allow_negative_numbers` and not
    // `allow_hyphen_values`: the latter takes any `-`-leading token as the
    // pending option's value, so this typo would bind `--moldel` to the seed
    // and never be reported.
    let err = serve_parse_error(
        &["--seed", "--moldel", "foo"],
        "a mistyped flag must not be swallowed as the seed value",
    );
    assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    assert!(
        err.to_string().contains("--moldel"),
        "the diagnostic must name the mistyped flag, got: {err}"
    );
}

#[test]
fn negative_numbers_stay_rejected_on_sibling_subcommands() {
    // `allow_negative_numbers` is a per-command setting applied to `serve`
    // only, so the sibling subcommands keep clap's default posture. Pinning it
    // here records that the blast radius of #1459 is the server surface, not
    // every `mlxcel` subcommand.
    let err = Cli::try_parse_from(["mlxcel", "generate", "--max-tokens", "-1"])
        .expect_err("`generate --max-tokens -1` must keep the default clap rejection");
    assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
}

#[test]
fn serve_spec_draft_spellings_resolve_to_the_draft_controls() {
    // b10621 canonical spellings (#1433) on `mlxcel serve`, mirroring the
    // mlxcel-server assertions in src/bin/mlx_server.rs.
    let canonical = parse_serve_args(&["--spec-draft-model", "models/draft"]);
    assert_eq!(
        canonical.draft_model,
        Some(std::path::PathBuf::from("models/draft"))
    );
    for spelling in ["--spec-draft-n-max", "--draft-n", "--draft", "--draft-max"] {
        let parsed = parse_serve_args(&[spelling, "24"]);
        assert_eq!(
            parsed.draft_max, 24,
            "{spelling} must set the draft-token cap"
        );
    }
}
