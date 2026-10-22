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

//! Bitwise run-to-run determinism of CUDA quantized matmul at prefill shapes.
//!
//! Issue #910: the qmm_sm80 kernel staged its output tile into shared memory
//! that aliases the A/B cp.async pipeline slots (SharedStorage is a union), and
//! the epilogue did not retire in-flight cp.async groups before overwriting
//! it, so every M >= 8 quantized matmul was bitwise non-deterministic at
//! temp 0 on every 4-bit CUDA model. Pre-fix the model test below failed on
//! 23/23 repeat iterations with an 8-token prefill on GB10 (sm_121); post-fix
//! it passes and compute-sanitizer racecheck reports 0 hazards.
//!
//! Issue #1558: this file then reported `ok` for as long as it existed without
//! ever running, because its checkpoint was missing everywhere and a missing
//! checkpoint was a silent early return. Two kinds of test live here now.
//!
//! * The kernel tests need no checkpoint. Each repeats one direct
//!   `quantized_matmul` on fixed synthetic operands at M in {8, 17, 64, 128,
//!   333} and requires byte-identical output. Each test pins one dispatch arm
//!   by its operands, following the `supports_*` predicates in MLX's
//!   `quantized/qmm/qmm.cu` (qmm_sm80 needs cc >= 8, N % 128 == 0, f16/bf16
//!   activations and transpose; everything else at M >= 8 falls to
//!   `qmm_naive`). They run in every `--features cuda` test run.
//! * The model test (`temp0_quantized_forward_is_bitwise_deterministic`) is
//!   `#[ignore]`, so a plain run reports it as `ignored` rather than as a pass
//!   it did not earn. `make verify-test-cuda-qmm-determinism` selects it with
//!   `--include-ignored` and sets `MLXCEL_REQUIRE_MODELS=1`, under which a
//!   checkpoint that resolves nowhere is a failure, not a skip.
//!
//! Checkpoint: the first of `llama-3.2-1b-instruct-4bit` (the name most of
//! `tests/` uses) and `llama-3.2-1b-4bit` (the name on the GB10 host) that
//! resolves under the roots `tests/common/mod.rs` searches, independent of the
//! working directory. `MLXCEL_TEST_QMM_MODEL_DIR` names a directory explicitly;
//! an override that does not exist always fails, with or without
//! `MLXCEL_REQUIRE_MODELS`, because a mistyped path is an operator error and
//! not a missing checkpoint. `MLXCEL_TEST_QMM_REPEATS` raises the per-shape
//! repeat count of the kernel tests (default 10).
//!
//! Every quantized projection of Llama-3.2-1B has N divisible by 128, so on
//! sm_80 and later the model test exercises `qmm_sm80` at prefill, not
//! `qmm_naive`. The `qmm_naive_*` kernel tests are what cover the naive kernel
//! there.
//!
//! ```sh
//! make verify-test-cuda-qmm-determinism
//! ```
//!
//! On a host that serializes GPU jobs behind `gpu-lock`, build without the
//! lock and run only the test binary under it, since a cargo build holds the
//! lock through its compiler daemon:
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --test cuda_qmm_determinism --no-run
//! gpu-lock run --tag qmm-determinism -- env MLX_ENABLE_TF32=0 MLXCEL_REQUIRE_MODELS=1 \
//!     target/test-fast/deps/cuda_qmm_determinism-<hash> --include-ignored --test-threads=1
//! ```
//!
//! `MLXCEL_TRACE_ARCH=1` prints the quantized-matmul path of the first call in
//! the process and `MLXCEL_TRACE_QMM_TILE=1` prints every distinct `qmm_naive`
//! kernel, including its `MmaPath`, so running one test per process with
//! `--exact` shows which kernel that test actually reached.

#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Mutex;

use mlxcel::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Checkpoint names tried in order; see the module docs.
const DEFAULT_MODEL_NAMES: &[&str] = &["llama-3.2-1b-instruct-4bit", "llama-3.2-1b-4bit"];

/// Prefill row counts for the kernel tests. 8 is the smallest M that leaves
/// qmv; 17 and 333 leave a partial last M tile, so the row predicates run.
const KERNEL_ROWS: &[i32] = &[8, 17, 64, 128, 333];
const DEFAULT_KERNEL_REPEATS: usize = 10;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// Serializes the tests in this binary. MLX's CUDA first-use paths (JIT module
/// load, graph capture) are not safe against concurrent test threads, and a
/// plain `cargo test --test cuda_qmm_determinism` runs them in parallel.
static GPU: Mutex<()> = Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Byte-level hash of an array's contents after forcing evaluation.
fn hash_array(arr: &MlxArray) -> u64 {
    let f = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&f);
    let bytes = mlxcel_core::array_to_raw_bytes(&f);
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Resolve the model test's checkpoint, or `None` to skip.
fn resolve_model_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MLXCEL_TEST_QMM_MODEL_DIR").filter(|v| !v.is_empty()) {
        let dir = PathBuf::from(dir);
        assert!(
            dir.exists(),
            "MLXCEL_TEST_QMM_MODEL_DIR={} does not exist. An explicit override never \
             skips (issue #1558); unset it to use the default checkpoint lookup.",
            dir.display()
        );
        return Some(dir);
    }
    // Panics under MLXCEL_REQUIRE_MODELS when no name resolves.
    let dir = common::repo_model_dir_any(DEFAULT_MODEL_NAMES);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!(
            "skipping cuda_qmm_determinism: none of {DEFAULT_MODEL_NAMES:?} is present; set \
             MLXCEL_REQUIRE_MODELS=1 to make this a failure"
        );
        None
    }
}

/// Repeats an identical prefill + fixed-token decode sequence with fresh KV
/// caches and asserts that the raw logits of every step are byte-identical
/// across iterations. The prefill length must be >= 8 so the quantized
/// matmuls take a tiled qmm kernel (M * B < 8 dispatches to qmv).
#[test]
#[ignore = "needs a 4-bit Llama-3.2-1B checkpoint; run `make verify-test-cuda-qmm-determinism`"]
fn temp0_quantized_forward_is_bitwise_deterministic() {
    let _gpu = gpu_guard();
    let Some(model_dir) = resolve_model_dir() else {
        return;
    };
    eprintln!("cuda_qmm_determinism: model {}", model_dir.display());

    const ITERS: usize = 10;
    const PREFILL_LEN: usize = 64;
    const DECODE_STEPS: usize = 8;

    let (model, _) = mlxcel::load_model(&model_dir).expect("load model");

    // Fixed, sampling-free input schedule so every iteration performs the
    // identical computation regardless of what the logits contain.
    let prompt: Vec<i32> = (0..PREFILL_LEN)
        .map(|i| 100 + (i as i32 * 37) % 900)
        .collect();

    let mut reference: Option<Vec<u64>> = None;
    for iter in 0..ITERS {
        let mut caches: Vec<KVCache> = model.make_caches();
        let mut step_hashes = Vec::with_capacity(1 + DECODE_STEPS);

        let input = mlxcel_core::from_slice_i32(&prompt, &[1, PREFILL_LEN as i32]);
        let logits = model.forward(&input, &mut caches, None);
        step_hashes.push(hash_array(&logits));

        for s in 0..DECODE_STEPS {
            let tok = [500 + (s as i32 * 13) % 400];
            let input = mlxcel_core::from_slice_i32(&tok, &[1, 1]);
            let logits = model.forward(&input, &mut caches, None);
            step_hashes.push(hash_array(&logits));
        }
        mlxcel_core::synchronize_default();

        match &reference {
            None => reference = Some(step_hashes),
            Some(reference) => {
                let first_bad = step_hashes
                    .iter()
                    .zip(reference.iter())
                    .position(|(a, b)| a != b);
                assert_eq!(
                    &step_hashes, reference,
                    "iteration {iter} produced different logits than iteration 0 \
                     (first divergent step: {first_bad:?}, 0 = prefill); quantized \
                     prefill is non-deterministic (issues #910, #1558)"
                );
            }
        }
    }
    eprintln!(
        "cuda_qmm_determinism: {ITERS} iterations of prefill + {DECODE_STEPS} decode steps, \
         byte-identical logits at every step"
    );
}

/// Deterministic pseudo-random values in [-0.5, 0.5), so the operands are the
/// same in every process and on every machine.
fn fixed_f32(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state = seed;
    let values: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let out = mlxcel_core::from_slice_f32(&values, shape);
    mlxcel_core::eval(&out);
    out
}

fn kernel_repeats() -> usize {
    std::env::var("MLXCEL_TEST_QMM_REPEATS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(DEFAULT_KERNEL_REPEATS)
}

/// Repeat one affine 4-bit, group-64 `quantized_matmul` per row count in
/// `KERNEL_ROWS` and require one distinct output per shape.
///
/// `transpose = true` is the layout every quantized linear layer uses: the
/// weight is `[n, k]` and `k` must be a multiple of 64. `transpose = false`
/// quantizes a `[k, n]` weight along `n` and accepts any `k`, which is the only
/// way to reach `qmm_naive`'s k-residue path.
fn assert_quantized_matmul_is_repeatable(
    label: &str,
    x_dtype: i32,
    n: i32,
    k: i32,
    transpose: bool,
) {
    let _gpu = gpu_guard();
    let repeats = kernel_repeats();
    let w_shape = if transpose { [n, k] } else { [k, n] };
    let w_full = fixed_f32(&w_shape, 0x1558_0001);
    let qw = mlxcel_core::quantize_weights(&w_full, GROUP_SIZE, BITS);
    let w = mlxcel_core::quantized_weights_w(&qw);
    let scales = mlxcel_core::quantized_weights_scales(&qw);
    let biases = mlxcel_core::quantized_weights_biases(&qw);
    mlxcel_core::eval(&w);
    mlxcel_core::eval(&scales);
    mlxcel_core::eval(&biases);

    let mut failures = Vec::new();
    for &m in KERNEL_ROWS {
        let x = mlxcel_core::astype(&fixed_f32(&[m, k], 0x1558_0100 + m as u64), x_dtype);
        mlxcel_core::eval(&x);

        let mut outputs: BTreeMap<u64, usize> = BTreeMap::new();
        for _ in 0..repeats {
            // SAFETY: `biases` is a live affine bias array for `w` and outlives
            // the call.
            let out = unsafe {
                mlxcel_core::quantized_matmul(
                    &x,
                    &w,
                    &scales,
                    &*biases as *const MlxArray,
                    transpose,
                    GROUP_SIZE,
                    BITS,
                    "affine",
                )
            };
            mlxcel_core::eval(&out);
            assert_eq!(mlxcel_core::array_shape(&out), vec![m, n]);
            let mut h = DefaultHasher::new();
            mlxcel_core::array_to_raw_bytes(&out).hash(&mut h);
            *outputs.entry(h.finish()).or_insert(0) += 1;
        }
        eprintln!(
            "[qmm-determinism] {label}: M={m} N={n} K={k} transpose={transpose}: {} distinct \
             output(s) over {repeats} calls",
            outputs.len()
        );
        if outputs.len() != 1 {
            failures.push(format!(
                "M={m}: counts {:?}",
                outputs.values().collect::<Vec<_>>()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: identical quantized_matmul calls gave different bytes ({}); quantized \
         prefill is non-deterministic on this device (issue #1558)",
        failures.join("; ")
    );
}

/// bf16 activations, N % 128 == 0, transpose: `qmm_sm80` on sm_80 and later,
/// `qmm_naive` below. The arm every 4-bit dense model's prefill takes on GB10.
#[test]
fn qmm_sm80_prefill_shapes_are_bitwise_repeatable() {
    assert_quantized_matmul_is_repeatable(
        "qmm_sm80 (bf16, N=256)",
        dtype::BFLOAT16,
        256,
        512,
        true,
    );
}

/// f16 activations with N % 128 != 0: `qmm_sm80` declines, so `qmm_naive`
/// runs on its tensor-core arm (`MmaPath::Sm80` on sm_80 and later,
/// `MmaPath::Sm70` on sm_70). N = 200 also leaves a partial N tile.
#[test]
fn qmm_naive_tensor_core_arm_is_bitwise_repeatable() {
    assert_quantized_matmul_is_repeatable(
        "qmm_naive tensor-core arm (f16, N=200)",
        dtype::FLOAT16,
        200,
        512,
        true,
    );
}

/// f32 activations: `qmm_sm80` takes only f16/bf16 and the naive kernel has no
/// f32 tensor-core atom, so this is `MmaPath::Scalar` (`UniversalFMA`) on every
/// architecture. When #1558 was reported, sm_70 ran this same arm for f16 and
/// bf16 too (the Volta tensor-core arm came later, in #1678), so this is the
/// closest sm_80+ stand-in for that run: same mainloop, same atom, f32
/// elements instead of 16-bit ones.
#[test]
fn qmm_naive_scalar_arm_is_bitwise_repeatable() {
    assert_quantized_matmul_is_repeatable(
        "qmm_naive scalar arm (f32, N=256)",
        dtype::FLOAT32,
        256,
        512,
        true,
    );
}

/// `transpose = false` with K = 200, not a multiple of the 64-wide K tile:
/// `qmm_naive` with `HasKResidue`, the path #1558 asked to check first. No
/// quantized linear layer takes it; it is reachable only this way.
#[test]
fn qmm_naive_k_residue_path_is_bitwise_repeatable() {
    assert_quantized_matmul_is_repeatable(
        "qmm_naive k-residue (f16, transpose=false, K=200)",
        dtype::FLOAT16,
        256,
        200,
        false,
    );
}
