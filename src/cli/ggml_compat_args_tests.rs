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

//! Unit tests for the b10621 GGML runtime compatibility group (issue #1445).

use super::*;

fn args() -> GgmlCompatArgs {
    GgmlCompatArgs::default()
}

/// The rejection for `args`, or `None` when everything in it is inert.
/// 32 layers is a plausible model, used wherever `--gpu-layers` is involved.
fn reject_for(args: GgmlCompatArgs) -> Option<GgmlCompatRejection> {
    args.rejection(Some(32))
}

fn message(args: GgmlCompatArgs) -> String {
    reject_for(args)
        .map(|r| r.to_string())
        .unwrap_or_else(|| panic!("expected a rejection"))
}

// ── nothing supplied is always inert ────────────────────────────────────────

#[test]
fn an_empty_argument_set_is_inert() {
    assert_eq!(args().rejection(Some(32)), None);
    assert_eq!(args().rejection(None), None);
    args().ensure_inert(None).expect("nothing requested");
}

// ── inert values are accepted ───────────────────────────────────────────────

#[test]
fn values_matching_what_mlxcel_already_does_are_inert() {
    let inert = GgmlCompatArgs {
        // mlxcel samples on the accelerator, memory-maps its shards, keeps
        // every layer and the KV cache on the device, and runs one device.
        backend_sampling: true,
        mmap: true,
        no_direct_io: true,
        repack: true,
        no_repack: true,
        kv_offload: true,
        op_offload: true,
        no_perf: true,
        flash_attn: Some("on".to_owned()),
        gpu_layers: Some("all".to_owned()),
        main_gpu: Some("0".to_owned()),
        split_mode: Some("none".to_owned()),
        load_mode: Some("auto".to_owned()),
        n_cpu_moe: Some("0".to_owned()),
        threads: Some("-1".to_owned()),
        threads_batch: Some("-1".to_owned()),
        cpu_strict: Some("0".to_owned()),
        cpu_strict_batch: Some("0".to_owned()),
        poll: Some("50".to_owned()),
        poll_batch: Some("50".to_owned()),
        prio: Some("0".to_owned()),
        prio_batch: Some("0".to_owned()),
        fit: Some("off".to_owned()),
        ..args()
    };
    assert_eq!(
        reject_for(inert),
        None,
        "every value here describes what mlxcel already does"
    );
}

#[test]
fn a_blank_environment_value_is_treated_as_absent() {
    // `LLAMA_ARG_DEVICE=` inherited from a llama.cpp deployment must not stop
    // the server; b10621 tests these fields with `.empty()` too.
    // `--numa` is excluded on purpose: b10621 reads an empty value there as
    // `distribute`, which is covered by its own test below.
    let blank = GgmlCompatArgs {
        device: Some(String::new()),
        rpc: Some("   ".to_owned()),
        tensor_split: Some(String::new()),
        override_kv: Some(String::new()),
        fit_target: Some("  ".to_owned()),
        ..args()
    };
    assert_eq!(reject_for(blank), None);
}

#[test]
fn both_flash_attention_positions_that_mlxcel_satisfies_are_inert() {
    for value in ["on", "enabled", "true", "1", "auto", "-1"] {
        assert_eq!(
            reject_for(GgmlCompatArgs {
                flash_attn: Some(value.to_owned()),
                ..args()
            }),
            None,
            "--flash-attn {value} must be inert"
        );
    }
}

// ── observable values are rejected ──────────────────────────────────────────

#[test]
fn turning_flash_attention_off_is_rejected() {
    for value in ["off", "disabled", "false", "0"] {
        let text = message(GgmlCompatArgs {
            flash_attn: Some(value.to_owned()),
            ..args()
        });
        assert!(text.contains("--flash-attn"), "{text}");
        assert!(text.contains(value), "{text}");
    }
}

#[test]
fn an_unknown_flash_attention_value_reports_the_upstream_vocabulary() {
    let text = message(GgmlCompatArgs {
        flash_attn: Some("maybe".to_owned()),
        ..args()
    });
    assert!(text.contains("`on`, `off`, or `auto`"), "{text}");
}

#[test]
fn a_partial_gpu_layer_budget_is_rejected_and_a_full_one_is_not() {
    // 32-layer model: 20 layers means 12 on the CPU, which mlxcel cannot do.
    let text = message(GgmlCompatArgs {
        gpu_layers: Some("20".to_owned()),
        ..args()
    });
    assert!(text.contains("--gpu-layers 20"), "{text}");
    assert!(text.contains("partial"), "{text}");
    assert!(text.contains("--gpu-layers all"), "{text}");

    for value in ["32", "99", "999", "-1", "auto", "all"] {
        assert_eq!(
            reject_for(GgmlCompatArgs {
                gpu_layers: Some(value.to_owned()),
                ..args()
            }),
            None,
            "--gpu-layers {value} covers the whole 32-layer model"
        );
    }
}

#[test]
fn an_unknown_layer_count_makes_every_non_negative_budget_unsupported() {
    // Guessing would be worse than refusing: without the model's layer count
    // there is no way to know whether `--gpu-layers 40` covers it.
    let rejection = GgmlCompatArgs {
        gpu_layers: Some("40".to_owned()),
        ..args()
    }
    .rejection(None);
    assert!(rejection.is_some());
    // `auto` / `all` / negative still need no layer count.
    for value in ["auto", "all", "-1"] {
        assert_eq!(
            GgmlCompatArgs {
                gpu_layers: Some(value.to_owned()),
                ..args()
            }
            .rejection(None),
            None,
            "--gpu-layers {value}"
        );
    }
}

#[test]
fn a_non_numeric_gpu_layer_value_reports_the_upstream_vocabulary() {
    let text = message(GgmlCompatArgs {
        gpu_layers: Some("most".to_owned()),
        ..args()
    });
    assert!(text.contains("`auto`, or `all`"), "{text}");
}

#[test]
fn every_multi_device_request_points_at_mlxcels_own_parallelism() {
    for args_with in [
        GgmlCompatArgs {
            split_mode: Some("layer".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            tensor_split: Some("3,1".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            main_gpu: Some("1".to_owned()),
            ..args()
        },
    ] {
        let text = message(args_with);
        assert!(
            text.contains("docs/distributed.md"),
            "a multi-device request must point at mlxcel's own parallelism: {text}"
        );
    }
}

#[test]
fn an_rpc_server_list_points_at_mlxcels_own_transport() {
    let text = message(GgmlCompatArgs {
        rpc: Some("host:50052".to_owned()),
        ..args()
    });
    assert!(text.contains("host:50052"), "{text}");
    assert!(text.contains("--node-role"), "{text}");
}

#[test]
fn every_cpu_thread_pool_knob_rejects_a_non_default_value() {
    for (build, option, value) in [
        (
            GgmlCompatArgs {
                threads: Some("8".to_owned()),
                ..args()
            },
            "--threads",
            "8",
        ),
        (
            GgmlCompatArgs {
                threads_batch: Some("4".to_owned()),
                ..args()
            },
            "--threads-batch",
            "4",
        ),
        (
            GgmlCompatArgs {
                cpu_strict: Some("1".to_owned()),
                ..args()
            },
            "--cpu-strict",
            "1",
        ),
        (
            GgmlCompatArgs {
                cpu_strict_batch: Some("1".to_owned()),
                ..args()
            },
            "--cpu-strict-batch",
            "1",
        ),
        (
            GgmlCompatArgs {
                poll: Some("0".to_owned()),
                ..args()
            },
            "--poll",
            "0",
        ),
        (
            GgmlCompatArgs {
                poll_batch: Some("0".to_owned()),
                ..args()
            },
            "--poll-batch",
            "0",
        ),
        (
            GgmlCompatArgs {
                prio: Some("3".to_owned()),
                ..args()
            },
            "--prio",
            "3",
        ),
        (
            GgmlCompatArgs {
                prio_batch: Some("2".to_owned()),
                ..args()
            },
            "--prio-batch",
            "2",
        ),
        (
            GgmlCompatArgs {
                cpu_mask: Some("ff".to_owned()),
                ..args()
            },
            "--cpu-mask",
            "ff",
        ),
        (
            GgmlCompatArgs {
                cpu_mask_batch: Some("0f".to_owned()),
                ..args()
            },
            "--cpu-mask-batch",
            "0f",
        ),
        (
            GgmlCompatArgs {
                cpu_range: Some("0-7".to_owned()),
                ..args()
            },
            "--cpu-range",
            "0-7",
        ),
        (
            GgmlCompatArgs {
                cpu_range_batch: Some("0-3".to_owned()),
                ..args()
            },
            "--cpu-range-batch",
            "0-3",
        ),
        (
            GgmlCompatArgs {
                numa: Some("distribute".to_owned()),
                ..args()
            },
            "--numa",
            "distribute",
        ),
    ] {
        let text = message(build);
        assert!(text.contains(option), "{option}: {text}");
        assert!(text.contains(value), "{option}: {text}");
    }
}

#[test]
fn every_memory_and_loading_request_mlxcel_cannot_honour_is_rejected() {
    for (build, marker) in [
        (
            GgmlCompatArgs {
                mlock: true,
                ..args()
            },
            "MLXCEL_WIRED_LIMIT",
        ),
        (
            GgmlCompatArgs {
                no_mmap: true,
                ..args()
            },
            "--no-mmap",
        ),
        (
            GgmlCompatArgs {
                direct_io: true,
                ..args()
            },
            "--direct-io",
        ),
        (
            GgmlCompatArgs {
                load_mode: Some("mlock".to_owned()),
                ..args()
            },
            "MLXCEL_WIRED_LIMIT",
        ),
        (
            GgmlCompatArgs {
                check_tensors: true,
                ..args()
            },
            "`inspect <model>`",
        ),
    ] {
        let text = message(build);
        assert!(text.contains(marker), "expected {marker} in: {text}");
    }
}

#[test]
fn every_expert_and_offload_request_is_rejected() {
    for build in [
        GgmlCompatArgs {
            cpu_moe: true,
            ..args()
        },
        GgmlCompatArgs {
            n_cpu_moe: Some("4".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            no_op_offload: true,
            ..args()
        },
        GgmlCompatArgs {
            no_kv_offload: true,
            ..args()
        },
        GgmlCompatArgs {
            no_host: true,
            ..args()
        },
    ] {
        assert!(reject_for(build).is_some());
    }
}

#[test]
fn gguf_metadata_overrides_point_at_the_config_file() {
    let text = message(GgmlCompatArgs {
        override_kv: Some("tokenizer.ggml.add_bos_token=bool:false".to_owned()),
        ..args()
    });
    assert!(text.contains("config.json"), "{text}");
    assert!(text.contains("SafeTensors"), "{text}");
}

#[test]
fn context_fitting_points_at_the_memory_estimator() {
    for build in [
        GgmlCompatArgs {
            fit: Some("on".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            fit_ctx: Some("8192".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            fit_target: Some("1024".to_owned()),
            ..args()
        },
    ] {
        let text = message(build);
        assert!(text.contains("--estimate-memory"), "{text}");
    }
}

#[test]
fn enabling_libllama_timers_points_at_the_http_counters() {
    let text = message(GgmlCompatArgs {
        perf: true,
        ..args()
    });
    assert!(text.contains("--metrics"), "{text}");
}

#[test]
fn a_defrag_threshold_is_inert_because_it_is_inert_upstream_too() {
    // b10621's handler is literally `GGML_UNUSED(params); GGML_UNUSED(value);`
    // plus a deprecation warning, so NO value of `--defrag-thold` changes
    // anything upstream. Rejecting it would refuse a command line that starts
    // there, which is the opposite of what the issue asks for.
    for value in ["0.1", "-1", "0", "0.5"] {
        assert_eq!(
            reject_for(GgmlCompatArgs {
                defrag_thold: Some(value.to_owned()),
                ..args()
            }),
            None,
            "--defrag-thold {value} is a no-op upstream and must be inert here"
        );
    }
}

#[test]
fn a_device_list_points_at_the_mlxcel_device_variable() {
    let text = message(GgmlCompatArgs {
        device: Some("CUDA0,CUDA1".to_owned()),
        ..args()
    });
    assert!(text.contains("MLXCEL_DEVICE"), "{text}");
}

#[test]
fn listing_devices_points_at_inspect() {
    let text = message(GgmlCompatArgs {
        list_devices: true,
        ..args()
    });
    assert!(text.contains("`inspect <model>`"), "{text}");
}

// ── diagnostic shape ────────────────────────────────────────────────────────

#[test]
fn every_diagnostic_names_the_option_the_value_and_a_limitation() {
    // The issue's acceptance criterion: "Include the requested option, value,
    // platform limitation, and supported mlxcel alternative in diagnostics."
    let cases = [
        GgmlCompatArgs {
            numa: Some("isolate".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            rpc: Some("h:1".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            mlock: true,
            ..args()
        },
        GgmlCompatArgs {
            gpu_layers: Some("1".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            split_mode: Some("row".to_owned()),
            ..args()
        },
    ];
    for build in cases {
        let rejection = reject_for(build).expect("a rejection");
        let text = rejection.to_string();
        assert!(text.starts_with(rejection.option), "{text}");
        assert!(text.contains(&rejection.value), "{text}");
        assert!(text.contains("is not supported: "), "{text}");
        assert!(
            text.contains("Use instead: ") || text.contains("no mlxcel equivalent"),
            "every diagnostic must offer an alternative or say there is none: {text}"
        );
        for line in text.lines() {
            assert!(
                !line.trim().contains("   "),
                "diagnostic line carries collapsed indentation: {line:?}"
            );
        }
    }
}

// ── environment vocabulary matches b10621 ───────────────────────────────────

fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let _guard = crate::test_support::env_lock::env_lock();
    let saved: Vec<(String, Option<String>)> = pairs
        .iter()
        .map(|(k, _)| ((*k).to_owned(), std::env::var(k).ok()))
        .collect();
    unsafe {
        for (key, value) in pairs {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    let out = body();
    unsafe {
        for (key, value) in &saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    out
}

#[test]
fn a_value_less_flag_fires_only_on_b10621s_truthy_set() {
    for value in ["on", "enabled", "true", "1"] {
        assert!(
            with_env(&[("LLAMA_ARG_CPU_MOE", Some(value))], || env_flag(
                "LLAMA_ARG_CPU_MOE"
            )),
            "LLAMA_ARG_CPU_MOE={value} must enable the flag"
        );
    }
    // Notably `yes`, `TRUE` and an empty value are NOT in b10621's set, and
    // clap's own boolish env parser would accept or reject them differently.
    for value in ["", "0", "false", "off", "yes", "TRUE", "On"] {
        assert!(
            !with_env(&[("LLAMA_ARG_CPU_MOE", Some(value))], || env_flag(
                "LLAMA_ARG_CPU_MOE"
            )),
            "LLAMA_ARG_CPU_MOE={value:?} must not enable the flag"
        );
    }
    assert!(!with_env(&[("LLAMA_ARG_CPU_MOE", None)], || env_flag(
        "LLAMA_ARG_CPU_MOE"
    )));
}

#[test]
fn a_bool_pair_reads_both_the_variable_and_its_no_alias() {
    let read = |pairs: &[(&str, Option<&str>)]| with_env(pairs, || env_bool_pair("LLAMA_ARG_MMAP"));
    assert_eq!(
        read(&[("LLAMA_ARG_MMAP", None), ("LLAMA_ARG_NO_MMAP", None)]),
        None
    );
    assert_eq!(
        read(&[("LLAMA_ARG_MMAP", Some("1")), ("LLAMA_ARG_NO_MMAP", None)]),
        Some(Ok(true))
    );
    assert_eq!(
        read(&[("LLAMA_ARG_MMAP", Some("0")), ("LLAMA_ARG_NO_MMAP", None)]),
        Some(Ok(false))
    );
    // The negative alias wins whatever the positive says, matching
    // `common_arg::get_value_from_env`.
    assert_eq!(
        read(&[
            ("LLAMA_ARG_MMAP", Some("1")),
            ("LLAMA_ARG_NO_MMAP", Some("")),
        ]),
        Some(Ok(false))
    );
    // b10621's `parse_bool_value` throws outside its two sets.
    assert_eq!(
        read(&[
            ("LLAMA_ARG_MMAP", Some("maybe")),
            ("LLAMA_ARG_NO_MMAP", None)
        ]),
        Some(Err("maybe".to_owned()))
    );
}

#[test]
fn env_bindings_reach_the_argument_set_and_are_then_classified() {
    let rejection = with_env(
        &[
            ("LLAMA_ARG_MMAP", Some("0")),
            ("LLAMA_ARG_NO_MMAP", None),
            ("LLAMA_ARG_CPU_MOE", None),
            ("LLAMA_ARG_NO_HOST", None),
            ("LLAMA_ARG_MLOCK", None),
            ("LLAMA_ARG_BACKEND_SAMPLING", None),
            ("LLAMA_ARG_DIO", None),
            ("LLAMA_ARG_NO_DIO", None),
            ("LLAMA_ARG_REPACK", None),
            ("LLAMA_ARG_NO_REPACK", None),
            ("LLAMA_ARG_PERF", None),
            ("LLAMA_ARG_NO_PERF", None),
            ("LLAMA_ARG_KV_OFFLOAD", None),
            ("LLAMA_ARG_NO_KV_OFFLOAD", None),
        ],
        || {
            let mut args = GgmlCompatArgs::default();
            args.apply_env_bindings().expect("recognized values");
            args.rejection(Some(32))
        },
    );
    let rejection = rejection.expect("LLAMA_ARG_MMAP=0 means --no-mmap, which is unsupported");
    assert_eq!(rejection.option, "--no-mmap");
}

#[test]
fn an_unparseable_bool_pair_value_is_reported_like_b10621_throws() {
    let result = with_env(
        &[
            ("LLAMA_ARG_PERF", Some("sometimes")),
            ("LLAMA_ARG_NO_PERF", None),
        ],
        || {
            let mut args = GgmlCompatArgs::default();
            args.apply_env_bindings()
        },
    );
    let (var, raw) = result.expect_err("b10621 throws on an unparseable boolean");
    assert_eq!(var, "LLAMA_ARG_PERF");
    assert_eq!(raw, "sometimes");
}

#[test]
fn an_explicit_command_line_flag_wins_over_the_environment() {
    let args = with_env(
        &[("LLAMA_ARG_MMAP", Some("0")), ("LLAMA_ARG_NO_MMAP", None)],
        || {
            let mut args = GgmlCompatArgs {
                mmap: true,
                ..GgmlCompatArgs::default()
            };
            args.apply_env_bindings().expect("recognized");
            args
        },
    );
    assert!(args.mmap && !args.no_mmap, "--mmap must survive the env");
}

// ── layer-count reader ──────────────────────────────────────────────────────

#[test]
fn the_layer_count_is_read_from_either_config_shape() {
    let tmp = tempfile::tempdir().unwrap();

    let flat = tmp.path().join("flat");
    std::fs::create_dir_all(&flat).unwrap();
    std::fs::write(flat.join("config.json"), br#"{"num_hidden_layers": 28}"#).unwrap();
    assert_eq!(read_model_layer_count(&flat), Some(28));

    let nested = tmp.path().join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(
        nested.join("config.json"),
        br#"{"text_config": {"num_hidden_layers": 40}}"#,
    )
    .unwrap();
    assert_eq!(read_model_layer_count(&nested), Some(40));

    let missing = tmp.path().join("missing");
    std::fs::create_dir_all(&missing).unwrap();
    assert_eq!(read_model_layer_count(&missing), None);
    std::fs::write(missing.join("config.json"), b"not json").unwrap();
    assert_eq!(read_model_layer_count(&missing), None);
}

// ── review follow-ups (issue #1445) ─────────────────────────────────────────

#[test]
fn load_mode_is_the_authority_over_the_deprecated_loading_flags() {
    // All four write one `params.load_mode` field upstream, applied in
    // command-line order, and b10621 deprecates the three in favour of
    // `--load-mode` while warning that combining them is last-wins. Rejecting
    // on a superseded half would refuse a command line that starts there.
    for deprecated in [
        GgmlCompatArgs {
            no_mmap: true,
            load_mode: Some("mmap".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            mlock: true,
            load_mode: Some("auto".to_owned()),
            ..args()
        },
        GgmlCompatArgs {
            direct_io: true,
            load_mode: Some("mmap".to_owned()),
            ..args()
        },
    ] {
        assert_eq!(
            reject_for(deprecated),
            None,
            "--load-mode supersedes the deprecated loading flag"
        );
    }
    // Without `--load-mode` the deprecated flags are classified as before.
    assert!(
        reject_for(GgmlCompatArgs {
            no_mmap: true,
            ..args()
        })
        .is_some()
    );
    // And an unsupported `--load-mode` is still reported, on its own name.
    let rejection = reject_for(GgmlCompatArgs {
        no_mmap: true,
        load_mode: Some("mlock".to_owned()),
        ..args()
    })
    .expect("mlock is unsupported");
    assert_eq!(rejection.option, "--load-mode");
}

#[test]
fn only_the_residency_load_modes_point_at_the_wired_memory_limit() {
    for (mode, wants_hint) in [
        ("mlock", true),
        ("mmap+mlock", true),
        ("none", false),
        ("dio", false),
    ] {
        let rejection = reject_for(GgmlCompatArgs {
            load_mode: Some(mode.to_owned()),
            ..args()
        })
        .expect("unsupported");
        assert_eq!(
            rejection.alternative.is_some(),
            wants_hint,
            "--load-mode {mode}: MLXCEL_WIRED_LIMIT is a residency control and is only \
             relevant to the residency modes"
        );
    }
}

#[test]
fn any_non_positive_thread_count_is_inert_because_upstream_treats_it_as_auto() {
    // `arg.cpp`: `if (n_threads <= 0) n_threads = hardware_concurrency();`, so
    // `0` and `-8` are the same request as the `-1` default.
    for value in ["-1", "0", "-8"] {
        for build in [
            GgmlCompatArgs {
                threads: Some(value.to_owned()),
                ..args()
            },
            GgmlCompatArgs {
                threads_batch: Some(value.to_owned()),
                ..args()
            },
        ] {
            assert_eq!(reject_for(build), None, "--threads {value} must be inert");
        }
    }
    assert!(
        reject_for(GgmlCompatArgs {
            threads: Some("8".to_owned()),
            ..args()
        })
        .is_some()
    );
}

#[test]
fn inert_numeric_values_are_compared_as_numbers_not_spellings() {
    for (build, what) in [
        (
            GgmlCompatArgs {
                main_gpu: Some(" 00 ".to_owned()),
                ..args()
            },
            "--main-gpu",
        ),
        (
            GgmlCompatArgs {
                n_cpu_moe: Some("+0".to_owned()),
                ..args()
            },
            "--n-cpu-moe",
        ),
        (
            GgmlCompatArgs {
                prio: Some("+0".to_owned()),
                ..args()
            },
            "--prio",
        ),
    ] {
        assert_eq!(
            reject_for(build),
            None,
            "{what} must compare its value as a number, as upstream's std::stoi does"
        );
    }
}

#[test]
fn an_empty_numa_value_is_b10621s_distribute_and_is_still_rejected() {
    // The one option in this shard where an empty value is not absent:
    // `arg.cpp` reads `--numa ""` as `distribute`.
    let rejection = reject_for(GgmlCompatArgs {
        numa: Some(String::new()),
        ..args()
    })
    .expect("an empty --numa is a distribute request");
    assert_eq!(rejection.option, "--numa");
    assert_eq!(rejection.value, "distribute");
}

#[test]
fn an_override_tensor_specification_is_rejected() {
    let text = message(GgmlCompatArgs {
        override_tensor: Some("blk\\.\\d+\\.ffn.*=CPU".to_owned()),
        ..args()
    });
    assert!(text.contains("--override-tensor"), "{text}");
    assert!(text.contains("GGML"), "{text}");
}

#[test]
fn the_layer_count_survives_every_config_spelling_in_tree() {
    // gpt2 / gpt_bigcode use `n_layer`; dbrx, falcon_ocr, jina_vlm and molmo
    // use `n_layers`; exaone uses `num_layers`. Reading only
    // `num_hidden_layers` here is how #927 reproduced five times, and it would
    // make every non-negative `--gpu-layers` unsupported on those families.
    let tmp = tempfile::tempdir().unwrap();
    for (index, key) in ["num_hidden_layers", "n_layers", "num_layers", "n_layer"]
        .iter()
        .enumerate()
    {
        let dir = tmp.path().join(format!("m{index}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!("{{\"{key}\": 24}}").as_bytes(),
        )
        .unwrap();
        assert_eq!(
            read_model_layer_count(&dir),
            Some(24),
            "{key} must be recognised"
        );
    }
}
