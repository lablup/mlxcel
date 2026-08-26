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
    assert_eq!(
        primary_args.adapter,
        Some(std::path::PathBuf::from("lora/foo"))
    );
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
