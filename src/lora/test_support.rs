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

//! On-disk adapter fixtures shared by the LoRA unit tests.
//!
//! The loader's entry points (`apply_lora_adapters`, `stage_runtime_adapters`)
//! read an adapter *directory*, not a [`WeightMap`], so the tests that cover
//! adapter acceptance have to write real `adapter_config.json` and
//! `adapters.safetensors` files. This module holds that writer once so the
//! fused-path tests and the runtime-path tests do not each carry a copy.
//!
//! Used by: `loader_tests.rs`, `runtime_tests.rs`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A safetensors view over an owned little-endian f32 buffer.
///
/// The impl is on `&OwnedTensor` because `serialize_to_file` is handed a
/// `&HashMap`, which yields borrowed values.
pub(crate) struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for &OwnedTensor {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        safetensors::tensor::Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        self.data.as_slice().into()
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// An all-ones `[rows, cols]` f32 tensor, so a fused sum stays predictable.
pub(crate) fn ones_tensor(rows: usize, cols: usize) -> OwnedTensor {
    let mut data = Vec::with_capacity(rows * cols * 4);
    for _ in 0..rows * cols {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    OwnedTensor {
        shape: vec![rows, cols],
        data,
    }
}

/// A unique temporary directory for one test case.
///
/// The PID disambiguates parallel `cargo test` processes and the counter
/// disambiguates cases inside one process, so two tests cannot collide on the
/// same nanosecond timestamp.
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after the unix epoch")
        .as_nanos();
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mlxcel_lora_{name}_{pid}_{nanos}_{seq}"))
}

/// Write an adapter directory holding `tensors` and an `adapter_config.json`
/// that declares `fine_tune_type`.
///
/// The tensor map is passed in rather than derived from a layer name so a test
/// can write exactly the malformed shape it is about: an unpaired half, an
/// unknown leaf, or a pair whose layer path no base weight answers to.
pub(crate) fn write_adapter_dir(
    dir: &Path,
    fine_tune_type: &str,
    rank: usize,
    tensors: HashMap<String, OwnedTensor>,
) {
    std::fs::create_dir_all(dir).expect("create adapter dir");
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"fine_tune_type": "{fine_tune_type}", "lora_parameters": {{"rank": {rank}, "scale": 1.0}}}}"#
        ),
    )
    .expect("write adapter config");
    safetensors::serialize_to_file(&tensors, None, &dir.join("adapters.safetensors"))
        .expect("write adapter safetensors");
}

/// An mlx-lm-convention pair for `layer`: `lora_a` is `[in, rank]`, `lora_b` is
/// `[rank, out]`, both all ones.
pub(crate) fn adapter_pair_tensors(
    layer: &str,
    in_features: usize,
    rank: usize,
    out_features: usize,
) -> HashMap<String, OwnedTensor> {
    let mut tensors = HashMap::new();
    tensors.insert(format!("{layer}.lora_a"), ones_tensor(in_features, rank));
    tensors.insert(format!("{layer}.lora_b"), ones_tensor(rank, out_features));
    tensors
}
