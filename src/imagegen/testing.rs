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

//! Test-only stub image family.
//!
//! Registered through an explicit [`ImageFamily`] table (never the shipping
//! [`super::registry::FAMILIES`]) so the CLI helpers and the server routes can
//! be driven end to end without a real model.

use std::path::Path;

use anyhow::Result;
use serde_json::json;

use super::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};
use super::registry::ImageFamily;

/// A deterministic fake: pixel `(x, y, c)` depends on the seed and position,
/// so two seeds differ and two runs with one seed match.
pub(crate) struct StubModel {
    pub family: &'static str,
    pub channels: u8,
    pub edit: bool,
}

impl ImageGenerationModel for StubModel {
    fn family(&self) -> &'static str {
        self.family
    }

    fn supports(&self, task: ImageTask) -> bool {
        task == ImageTask::Generate || self.edit
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        ImageSamplingDefaults::new(4, 1.0).expect("valid defaults")
    }

    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        Ok((0..req.n)
            .map(|i| {
                let seed = req.seed_for(i);
                let mut pixels = Vec::with_capacity(req.width as usize * req.height as usize * 4);
                for y in 0..req.height {
                    for x in 0..req.width {
                        for c in 0..self.channels {
                            let v =
                                (u64::from(x) + 3 * u64::from(y) + 7 * u64::from(c) + seed) % 251;
                            // Alpha is deliberately not opaque, so a writer that
                            // drops it is caught.
                            pixels.push(if self.channels == 4 && c == 3 {
                                128
                            } else {
                                v as u8
                            });
                        }
                    }
                }
                GeneratedImage {
                    pixels,
                    width: req.width,
                    height: req.height,
                    channels: self.channels,
                    seed,
                    revised_prompt: Some(format!("revised: {}", req.prompt)),
                }
            })
            .collect())
    }
}

fn load_rgb(_: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(StubModel {
        family: "flux2",
        channels: 3,
        edit: true,
    }))
}

fn load_rgba(_: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(StubModel {
        family: "qwen_image",
        channels: 4,
        edit: false,
    }))
}

/// `flux2` yields RGB (generate and edit); `qwen_image` yields RGBA
/// (generate only).
pub(crate) static STUB_FAMILIES: [ImageFamily; 2] = [
    ImageFamily {
        name: "flux2",
        load: load_rgb,
    },
    ImageFamily {
        name: "qwen_image",
        load: load_rgba,
    },
];

/// Write a `model_index.json` the probe classifies as `class_name`'s family.
pub(crate) fn write_model_index(dir: &Path, class_name: &str) {
    std::fs::write(
        dir.join("model_index.json"),
        json!({ "_class_name": class_name }).to_string(),
    )
    .expect("write model_index.json");
}

/// FNV-1a over the bytes of `s`; keys [`det`].
pub(crate) fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for b in s.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// `offset + scale * v[i]` for a closed-form pseudo-random sequence `v` keyed by
/// `name`. The Python fixture generator
/// (`tests/fixtures/generate_imagegen_fixtures.py`) implements the same
/// function, so weights need not be stored next to their upstream outputs.
pub(crate) fn det(
    name: &str,
    shape: &[i32],
    scale: f64,
    offset: f64,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let phase = f64::from(fnv1a32(name) % 6283) / 1000.0;
    let values: Vec<f32> = (0..n)
        .map(|i| {
            let i = f64::from(i);
            let v = ((0.37 * i + phase).sin() + 0.5 * (1.13 * i + 2.0 * phase).cos()) / 1.5;
            (offset + scale * v) as f32
        })
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for OwnedTensor {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::BF16
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

/// Write `weights` to `path` as bf16 safetensors.
pub(crate) fn write_bf16_safetensors(path: &Path, weights: &mlxcel_core::weights::WeightMap) {
    let tensors: std::collections::HashMap<String, OwnedTensor> = weights
        .iter()
        .map(|(name, array)| {
            let bf16 = mlxcel_core::astype(array, mlxcel_core::dtype::BFLOAT16);
            mlxcel_core::eval(&bf16);
            let shape = mlxcel_core::array_shape(&bf16)
                .iter()
                .map(|d| *d as usize)
                .collect();
            let data = mlxcel_core::array_to_raw_bytes(&bf16).to_vec();
            (name.clone(), OwnedTensor { shape, data })
        })
        .collect();
    safetensors::serialize_to_file(tensors, None, path).expect("write safetensors");
}

/// Environment variable naming the file a [`with_forced_attention_chunking`]
/// child writes its result to.
const CHUNK_CHILD_OUT: &str = "MLXCEL_IMAGEGEN_CHUNK_CHILD_OUT";

/// Runs `compute` in this process and in a fresh run of the test `test_path`
/// (its full `module::path::name`) with `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=1`,
/// returning `(default budget, forced chunking)`.
///
/// The budget is read once per process through a `OnceLock`, so the forced arm
/// has to be another process. Inside that child this returns `None` after
/// writing the result for the parent; the caller returns on `None`. On a
/// non-CUDA build the chunk gate never fires and both arms run one SDPA.
pub(crate) fn with_forced_attention_chunking(
    test_path: &str,
    compute: impl Fn() -> Vec<f32>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let to_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    if let Some(out) = std::env::var_os(CHUNK_CHILD_OUT) {
        std::fs::write(out, to_bytes(&compute())).expect("write the chunked result");
        return None;
    }
    let default_budget = compute();
    let out = std::env::temp_dir().join(format!(
        "mlxcel-chunk-{}-{:08x}.f32",
        std::process::id(),
        fnv1a32(test_path)
    ));
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", test_path, "--test-threads=1"])
        .env(CHUNK_CHILD_OUT, &out)
        .env("MLXCEL_ATTENTION_CHUNK_BUDGET_MB", "1")
        .output()
        .expect("spawn the forced-chunking run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "forced-chunking run failed:\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out).expect("read the chunked result");
    let _ = std::fs::remove_file(&out);
    let chunked = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Some((default_budget, chunked))
}
