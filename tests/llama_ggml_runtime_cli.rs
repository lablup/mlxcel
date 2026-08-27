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

//! End-to-end behavior of the b10621 GGML runtime options on both server
//! binaries (issue #1445).
//!
//! `src/cli/ggml_compat_args_tests.rs` covers the classification itself. This
//! file covers what only the built binaries can answer: that every upstream
//! spelling actually parses, that an unsupported value stops startup with the
//! diagnostic the issue asks for, that an inert value does not, and that none
//! of it reaches the operator-facing `--help`.
//!
//! Every invocation fails (or prints help) before a weight is read, so the
//! file needs no checkpoint and no network.

use std::process::{Command, Output};

mod common;
use common::resolve_repo_binary;

/// Every `LLAMA_ARG_*` this file's flags bind, cleared before each run so a
/// developer's shell cannot change the outcome.
const CLEARED: [&str; 20] = [
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_CACHE_TYPE_K",
    "LLAMA_ARG_CACHE_TYPE_V",
    "LLAMA_ARG_N_GPU_LAYERS",
    "LLAMA_ARG_FLASH_ATTN",
    "LLAMA_ARG_DEVICE",
    "LLAMA_ARG_SPLIT_MODE",
    "LLAMA_ARG_TENSOR_SPLIT",
    "LLAMA_ARG_MAIN_GPU",
    "LLAMA_ARG_RPC",
    "LLAMA_ARG_NUMA",
    "LLAMA_ARG_THREADS",
    "LLAMA_ARG_LOAD_MODE",
    "LLAMA_ARG_MLOCK",
    "LLAMA_ARG_MMAP",
    "LLAMA_ARG_NO_MMAP",
    "LLAMA_ARG_CPU_MOE",
    "LLAMA_ARG_N_CPU_MOE",
    "LLAMA_ARG_FIT",
    "LLAMA_ARG_DEFRAG_THOLD",
];

fn run(bin: &str, args: &[&str], env: &[(&str, &str)]) -> Output {
    let (path, resolution) = resolve_repo_binary(bin);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in CLEARED {
        cmd.env_remove(key);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin} from {path:?}: {e}\n{resolution}"))
}

const ENTRY_POINTS: [(&str, &[&str]); 2] = [("mlxcel", &["serve"]), ("mlxcel-server", &[])];

/// Combined stdout+stderr of a run that must have failed.
fn expect_failure(entry: (&str, &[&str]), extra: &[&str]) -> String {
    let (bin, lead) = entry;
    let mut args: Vec<&str> = lead.to_vec();
    args.extend_from_slice(extra);
    let out = run(bin, &args, &[]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "{bin} {args:?} unexpectedly succeeded; output: {text}"
    );
    text
}

// ── every upstream spelling parses ──────────────────────────────────────────

/// Every b10621 GGML runtime spelling and, for value-taking ones, a value.
/// Used to prove the option reaches mlxcel's own classification rather than
/// clap's "unexpected argument".
const UPSTREAM_INVOCATIONS: &[&[&str]] = &[
    &["--backend-sampling"],
    &["--check-tensors"],
    &["--cpu-mask", "ff"],
    &["--cpu-mask-batch", "0f"],
    &["--cpu-moe"],
    &["--cpu-range", "0-7"],
    &["--cpu-range-batch", "0-3"],
    &["--cpu-strict", "1"],
    &["--cpu-strict-batch", "1"],
    &["--defrag-thold", "0.1"],
    &["--device", "CUDA0"],
    &["--direct-io"],
    &["--no-direct-io"],
    &["--fit", "on"],
    &["--fit-ctx", "8192"],
    &["--fit-target", "1024"],
    &["--flash-attn", "auto"],
    &["--gpu-layers", "all"],
    &["--n-gpu-layers", "all"],
    &["--kv-offload"],
    &["--no-kv-offload"],
    &["--list-devices"],
    &["--load-mode", "auto"],
    &["--main-gpu", "0"],
    &["--mlock"],
    &["--mmap"],
    &["--no-mmap"],
    &["--n-cpu-moe", "0"],
    &["--no-host"],
    &["--numa", "distribute"],
    &["--op-offload"],
    &["--no-op-offload"],
    &["--override-kv", "a=bool:false"],
    &["--override-tensor", "blk=CPU"],
    &["--perf"],
    &["--no-perf"],
    &["--poll", "50"],
    &["--poll-batch", "50"],
    &["--prio", "0"],
    &["--prio-batch", "0"],
    &["--repack"],
    &["--no-repack"],
    &["--rpc", "host:50052"],
    &["--split-mode", "none"],
    &["--tensor-split", "3,1"],
    &["--threads", "-1"],
    &["--threads-batch", "-1"],
];

#[test]
fn every_b10621_ggml_spelling_reaches_mlxcels_own_classification() {
    // Before #1445 thirty-seven of these were unknown arguments, so a
    // llama-server command line died on clap rather than on anything
    // explaining the boundary.
    for invocation in UPSTREAM_INVOCATIONS {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, invocation);
            assert!(
                !text.contains("unexpected argument"),
                "{} {invocation:?}: must not reach clap as an unknown token: {text}",
                entry.0
            );
            assert!(
                !text.contains("--help"),
                "{} {invocation:?}: must not render the help screen: {text}",
                entry.0
            );
        }
    }
}

// ── inert values do not stop startup for their own sake ─────────────────────

#[test]
fn inert_values_fail_only_for_the_missing_model() {
    // Each of these describes something mlxcel already does, so the only
    // complaint left must be the absent `-m`.
    for invocation in [
        &["--backend-sampling"][..],
        &["--split-mode", "none"][..],
        &["--threads", "-1"][..],
        &["--cpu-strict", "0"][..],
        &["--poll", "50"][..],
        &["--prio", "0"][..],
        &["--flash-attn", "on"][..],
        &["--flash-attn", "auto"][..],
        &["--gpu-layers", "all"][..],
        &["--main-gpu", "0"][..],
        &["--load-mode", "mmap"][..],
        &["--mmap"][..],
        &["--no-direct-io"][..],
        &["--repack"][..],
        &["--no-repack"][..],
        &["--kv-offload"][..],
        &["--op-offload"][..],
        &["--no-perf"][..],
        &["--n-cpu-moe", "0"][..],
        &["--fit", "off"][..],
        &["--cache-type-k", "f16", "--cache-type-v", "f16"][..],
    ] {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, invocation);
            assert!(
                text.contains("--model/-m is required"),
                "{} {invocation:?} is inert, so the only complaint must be the missing model: {text}",
                entry.0
            );
        }
    }
}

// ── unsupported values stop startup with the required diagnostic ────────────

#[test]
fn unsupported_values_are_rejected_with_option_value_limitation_and_alternative() {
    // The issue's acceptance criterion for diagnostics, checked on the real
    // binaries: the requested option, the value, the platform limitation, and
    // the supported mlxcel alternative.
    for (invocation, option, value, alternative) in [
        (
            &["--split-mode", "row"][..],
            "--split-mode",
            "row",
            "docs/distributed.md",
        ),
        (
            &["--tensor-split", "3,1"][..],
            "--tensor-split",
            "3,1",
            "docs/distributed.md",
        ),
        (
            &["--rpc", "host:50052"][..],
            "--rpc",
            "host:50052",
            "--node-role",
        ),
        (
            &["--numa", "distribute"][..],
            "--numa",
            "distribute",
            "no mlxcel equivalent",
        ),
        (
            &["--threads", "8"][..],
            "--threads",
            "8",
            "no mlxcel equivalent",
        ),
        (
            &["--cpu-mask", "ff"][..],
            "--cpu-mask",
            "ff",
            "no mlxcel equivalent",
        ),
        (&["--mlock"][..], "--mlock", "--mlock", "MLXCEL_WIRED_LIMIT"),
        (
            &["--no-mmap"][..],
            "--no-mmap",
            "--no-mmap",
            "no mlxcel equivalent",
        ),
        (
            &["--flash-attn", "off"][..],
            "--flash-attn",
            "off",
            "no mlxcel equivalent",
        ),
        (
            &["--device", "CUDA0"][..],
            "--device",
            "CUDA0",
            "MLXCEL_DEVICE",
        ),
        (&["--perf"][..], "--perf", "--perf", "--metrics"),
        (&["--fit", "on"][..], "--fit", "on", "--estimate-memory"),
        (
            &["--cpu-moe"][..],
            "--cpu-moe",
            "--cpu-moe",
            "no mlxcel equivalent",
        ),
        (
            &["--no-kv-offload"][..],
            "--no-kv-offload",
            "--no-kv-offload",
            "--cache-type-k",
        ),
    ] {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, invocation);
            assert!(
                text.contains(option),
                "{} {invocation:?}: diagnostic must name the option: {text}",
                entry.0
            );
            assert!(
                text.contains(value),
                "{} {invocation:?}: diagnostic must quote the value: {text}",
                entry.0
            );
            assert!(
                text.contains("is not supported"),
                "{} {invocation:?}: diagnostic must say so plainly: {text}",
                entry.0
            );
            assert!(
                text.contains(alternative),
                "{} {invocation:?}: diagnostic must name {alternative}: {text}",
                entry.0
            );
        }
    }
}

#[test]
fn a_partial_gpu_layer_budget_is_rejected_and_a_full_one_is_not() {
    // `--gpu-layers` is the one option whose classification needs the model:
    // only the layer count separates a full offload (inert, mlxcel always runs
    // every layer on the accelerator) from a partial one. It is therefore
    // checked after the reference resolves, which is why this case needs a
    // checkpoint directory while every other one in this file does not.
    let tmp = tempfile::tempdir().expect("tempdir");
    let model = tmp.path().join("checkpoint");
    std::fs::create_dir_all(&model).expect("model dir");
    std::fs::write(model.join("config.json"), br#"{"num_hidden_layers": 32}"#).expect("config");
    let model = model.to_str().expect("utf-8");

    for entry in ENTRY_POINTS {
        let text = expect_failure(entry, &["-m", model, "--gpu-layers", "20"]);
        assert!(text.contains("--gpu-layers 20"), "{}: {text}", entry.0);
        assert!(text.contains("partial"), "{}: {text}", entry.0);
        assert!(text.contains("--gpu-layers all"), "{}: {text}", entry.0);
    }

    // A budget covering the whole model is inert, so the run gets past the
    // classification and fails later on the empty checkpoint instead.
    for entry in ENTRY_POINTS {
        for value in ["32", "999", "-1", "auto", "all"] {
            let text = expect_failure(entry, &["-m", model, "--gpu-layers", value]);
            assert!(
                !text.contains("is not supported"),
                "{} --gpu-layers {value} covers the 32-layer model and must be inert: {text}",
                entry.0
            );
        }
    }
}

#[test]
fn a_ggml_kv_cache_quantizer_is_rejected_with_the_mlxcel_vocabulary() {
    // The KV cache type is resolved after the model reference, so this needs a
    // path the resolver accepts. An existing directory is returned verbatim
    // (the resolver's step 1), and the run stops on the cache type before any
    // weight is read.
    let tmp = tempfile::tempdir().expect("tempdir");
    let model = tmp.path().join("checkpoint");
    std::fs::create_dir_all(&model).expect("model dir");
    std::fs::write(model.join("config.json"), br#"{"num_hidden_layers": 32}"#).expect("config");
    let model = model.to_str().expect("utf-8");

    for value in ["q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"] {
        for entry in ENTRY_POINTS {
            let text = expect_failure(entry, &["-m", model, "--cache-type-k", value]);
            assert!(
                text.contains(value) && text.contains("GGML KV cache quantizer"),
                "{}: --cache-type-k {value} must be refused as a different quantizer: {text}",
                entry.0
            );
            assert!(
                text.contains("docs/turbo-kv-cache.md"),
                "{}: must point at the KV reference: {text}",
                entry.0
            );
        }
    }
}

// ── environment bindings use b10621's vocabulary ────────────────────────────

#[test]
fn a_value_less_flag_fires_from_the_environment_only_on_the_truthy_set() {
    for (value, must_reject) in [
        ("1", true),
        ("on", true),
        ("true", true),
        ("enabled", true),
        ("0", false),
        ("", false),
        ("yes", false),
        ("TRUE", false),
    ] {
        for (bin, lead) in ENTRY_POINTS {
            let mut args: Vec<&str> = lead.to_vec();
            args.push("--offline");
            let out = run(bin, &args, &[("LLAMA_ARG_CPU_MOE", value)]);
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let rejected = text.contains("--cpu-moe");
            assert_eq!(
                rejected, must_reject,
                "{bin}: LLAMA_ARG_CPU_MOE={value:?} rejected={rejected}, expected {must_reject}: {text}"
            );
        }
    }
}

#[test]
fn a_bool_pair_environment_value_reaches_the_classifier() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--offline");
        // `LLAMA_ARG_MMAP=0` means the negative half, which mlxcel cannot do.
        let out = run(bin, &args, &[("LLAMA_ARG_MMAP", "0")]);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            text.contains("--no-mmap"),
            "{bin}: LLAMA_ARG_MMAP=0 must reach the --no-mmap rejection: {text}"
        );
    }
}

#[test]
fn an_unparseable_bool_pair_environment_value_stops_startup_like_upstream() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--offline");
        let out = run(bin, &args, &[("LLAMA_ARG_PERF", "sometimes")]);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!out.status.success(), "{bin}: expected failure: {text}");
        assert!(
            text.contains("LLAMA_ARG_PERF") && text.contains("sometimes"),
            "{bin}: must name the variable and its value: {text}"
        );
    }
}

// ── the operator surface stays mlxcel's ─────────────────────────────────────

#[test]
fn no_ggml_runtime_option_appears_in_help() {
    // Advertising these would imply a GGML backend mlxcel does not have. The
    // mlxcel-native options they are confused with must still be documented.
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--help");
        let out = run(bin, &args, &[]);
        let help = String::from_utf8_lossy(&out.stdout);
        for hidden in [
            "--backend-sampling",
            "--check-tensors",
            "--cpu-mask",
            "--cpu-moe",
            "--defrag-thold",
            "--device ",
            "--gpu-layers",
            "--list-devices",
            "--load-mode",
            "--main-gpu",
            "--mlock",
            "--numa",
            "--override-kv",
            "--rpc",
            "--split-mode",
            "--tensor-split",
            "--threads ",
        ] {
            assert!(
                !help.contains(hidden),
                "{bin} --help advertises {hidden}, which mlxcel always rejects or ignores"
            );
        }
        for visible in ["--cache-type-k", "--cache-type-v", "--kv-cache-mode"] {
            assert!(
                help.contains(visible),
                "{bin} --help must still document the mlxcel-native {visible}"
            );
        }
    }
}
