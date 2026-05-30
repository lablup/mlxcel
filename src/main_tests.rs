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

use clap::Parser;

use super::{
    Cli, Commands, FAMILY_ORDER, PipelineParallelOptions, TensorParallelOptions,
    write_supported_models,
};

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

// Issue #371 (A4): CLI argument-parsing tests for the `--surgery <FILE>`
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
    // load path bit-exact with pre-#371 main (acceptance criterion (e)).
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
