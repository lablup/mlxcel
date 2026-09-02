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

//! Test-only helpers shared by the decoder-backbone embedding family tests:
//! a deterministic weight generator, small array utilities, and the
//! soft-skipping checkpoint lookup the real-checkpoint gates use.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use safetensors::tensor::{Dtype as SafeTensorDtype, View};

/// Serialize every test that builds an embedding model or runs a forward
/// pass, across both decoder-backbone family test modules.
///
/// `EmbeddingModel` is documented as single-thread and the product honors
/// that through the embedding worker, so this hazard is test-side only.
/// libtest runs one thread per logical CPU, and two concurrent MLX forward
/// passes in one process interfere in two observed ways on this tree: a CUDA
/// graph capture aborts the process outright (`cudaStreamEndCapture ...
/// operation failed due to a previous error during capture`), and results
/// drift, with two byte-identical rows of one batch scoring cosine 0.999912
/// instead of 1.0. A gate number read from an unguarded parallel run is not
/// evidence of anything.
///
/// The lock is process-wide rather than per-module so the EmbeddingGemma and
/// Qwen3-Embedding gates also serialize against each other. A poisoned lock
/// is recovered rather than propagated: one panicking test must fail alone,
/// not cascade into every later test in the module.
///
/// It cannot serialize against MLX work in *other* modules, which is why
/// every gate this repository defines (`make verify-test`,
/// `make verify-test-cuda`, `make test-fast`) passes `--test-threads=1`. A
/// raw multi-threaded `cargo test` over a wide filter is not a supported way
/// to run this suite on either backend.
///
/// One thing it does serialize across modules is the process-global default
/// device, by holding `mlxcel_core::streams::lock_default_device` for the
/// test's duration. That is the same lock the modules that move the default
/// device take, so a gate here cannot be measuring a checkpoint while
/// `vision::merge` or `multimodal::host_preprocessor` holds the device on the
/// CPU. Without it those spans overlap under the parallel `cargo test --lib`
/// in `scripts/run_quality_gate.sh`, and the reranker scores below would come
/// from the CPU backend with nothing saying so (issue #1421).
///
/// The guard also checks that MLX's process-wide default device is still the
/// GPU when a GPU backend exists. Other test modules move the default device
/// to the CPU for their own reasons, and libtest runs modules in name order,
/// so a gate that sorted after a module that never moved it back silently
/// measured the CPU backend: under `make verify-test-cuda` the
/// `rerank::real_checkpoint_tests` gates scored a 4-bit Qwen3 reranker at
/// 0.35 instead of 0.99 and produced NaN image scores from a bf16 Qwen3-VL
/// reranker, while the same tests passed in isolation. PR #1420 repaired that
/// here with an unconditional `set_default_device(true)`, which hid the leak
/// rather than removing it. The leaking modules now hold a
/// `mlxcel_core::streams::DefaultDeviceGuard` for each test's duration, so
/// the pin is reduced to this assertion and a future leak fails loudly at the
/// first gate after it, naming the cause (issue #1421). Every
/// real-checkpoint number this repository records is a GPU number, and the
/// assertion is what makes that explicit instead of depending on module
/// order.
///
/// It is a plain `assert!`, not a `debug_assert!`: every gate runs under
/// `--profile test-fast`, which inherits `release` and compiles debug
/// assertions out, and this helper is test-only code where the check costs
/// nothing. The check reads `mlxcel_core::default_device_is_gpu()` against
/// `mlxcel_core::gpu_backend_available()`, the backend answer that does not
/// move with the default device (the old `is_gpu_available()` did, which is
/// why a guarded pin never fired). An explicit `MLXCEL_DEVICE=cpu` is not a
/// leak: `initialize_runtime()` moves the default device to the CPU on
/// purpose under it, so the check is skipped for that request and the gates
/// measure the CPU the operator asked for.
///
/// A `DefaultDeviceGuard` that another test is holding *right now* is not a
/// leak either, so the check is skipped while any guard is alive
/// (`mlxcel_core::streams::default_device_guards_held`). The default-device
/// lock above already excludes every in-tree guard, so this is the backstop
/// for a guard taken without that lock; it costs the gates nothing, since
/// they all pass `--test-threads=1`, where no guard can be alive while this
/// test runs and a real leak still fails here.
pub(crate) fn mlx_test_guard() -> MlxTestGuard {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    let serial = GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Taken after the serial lock and never in the other order, so the two
    // locks cannot deadlock against a module that takes only this one.
    let device = mlxcel_core::streams::lock_default_device();
    assert!(
        mlxcel_core::default_device_is_gpu()
            || !mlxcel_core::gpu_backend_available()
            || crate::execution::runtime::cpu_override_requested()
            || mlxcel_core::streams::default_device_guards_held() > 0,
        "MLX's default device is the CPU although a GPU backend is available, MLXCEL_DEVICE=cpu is not set, and no DefaultDeviceGuard is held: an earlier test moved the default device and never restored it. Hold a mlxcel_core::streams::DefaultDeviceGuard for the test's duration instead of calling set_default_device directly."
    );
    MlxTestGuard {
        _device: device,
        _serial: serial,
    }
}

/// What [`mlx_test_guard`] returns: the process-wide MLX serialization lock
/// and the default-device lock, both released when the test's binding drops.
/// Fields drop in declaration order, which is immaterial here because neither
/// lock is taken while the other is being released anywhere in this crate.
#[must_use = "the locks are released when the guard drops; bind it to a named local, not `_`"]
pub(crate) struct MlxTestGuard {
    _device: mlxcel_core::streams::DefaultDeviceLock,
    _serial: MutexGuard<'static, ()>,
}

/// Deterministic xorshift64* generator.
///
/// Synthetic model weights must be reproducible across runs and across the
/// two models a differential test compares, so nothing here draws from a
/// seeded-by-time source.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[-scale, scale)`.
    pub(crate) fn next_f32(&mut self, scale: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * scale
    }

    /// `count` uniform values in `[-scale, scale)`.
    pub(crate) fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_f32(scale)).collect()
    }

    /// A dense f32 tensor of the given shape.
    pub(crate) fn tensor(&mut self, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
        let count: i32 = shape.iter().product();
        mlxcel_core::from_slice_f32(&self.values(count as usize, scale), shape)
    }

    /// Insert a dense f32 tensor under `key`.
    pub(crate) fn insert(&mut self, weights: &mut WeightMap, key: &str, shape: &[i32], scale: f32) {
        weights.insert(key.to_string(), self.tensor(shape, scale));
    }
}

/// `[B, L]` int32 token ids.
pub(crate) fn ids_array(ids: &[i32], batch: i32, length: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(ids, &[batch, length])
}

/// Read an array back as a flat row-major `Vec<f32>`.
pub(crate) fn to_vec(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(array)
}

/// Largest absolute element-wise difference between two equally sized slices.
pub(crate) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared slices must have equal length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Render an expected failure, including the anyhow context chain.
///
/// Neither the family models nor [`crate::embeddings::EmbeddingOutput`]
/// implement `Debug` (they hold raw MLX handles), so `unwrap_err` is not
/// available on their results.
pub(crate) fn err_string<T>(result: anyhow::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => format!("{error:#}"),
    }
}

/// Cosine similarity of two equally sized vectors.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// A shaped f32 tensor for the safetensors writer below.
struct F32Tensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &F32Tensor {
    fn dtype(&self) -> SafeTensorDtype {
        SafeTensorDtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// A fresh empty directory under the system temp dir.
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("mlxcel_embedding_test_{name}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a shaped multi-tensor f32 safetensors file.
///
/// Used to materialize a synthetic checkpoint directory on disk so a family
/// loader can be exercised through its real `load(dir)` path, subfolder
/// module layout included, rather than only through an in-memory weight map.
pub(crate) fn write_f32_safetensors(path: &Path, tensors: &[(String, Vec<i32>, Vec<f32>)]) {
    let owned: Vec<(String, F32Tensor)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            (
                name.clone(),
                F32Tensor {
                    shape: shape.iter().map(|&d| d as usize).collect(),
                    data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
            )
        })
        .collect();
    let views: HashMap<&str, &F32Tensor> = owned
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    safetensors::serialize_to_file(views, None, path).unwrap();
}

/// Locate a downloaded checkpoint: the mlxcel store, then the HuggingFace
/// cache, then `<repo>/models/<name>`. `None` skips the calling gate, the
/// same convention `src/embeddings/real_checkpoint_tests.rs` and
/// `tests/*_parity.rs` follow.
pub(crate) fn local_checkpoint(repo_id: &str) -> Option<PathBuf> {
    let candidates = [
        crate::downloader::model_dir(repo_id),
        crate::downloader::hf_cache_snapshot(repo_id, None),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join(crate::downloader::repo_basename(repo_id)),
        ),
    ];
    let found = candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("config.json").is_file());
    if found.is_none() {
        eprintln!(
            "skipping real-checkpoint gate: {repo_id} not present (mlxcel download {repo_id})"
        );
    }
    found
}
