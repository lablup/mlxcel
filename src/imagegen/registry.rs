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

//! Registry of image-generation families.
//!
//! [`FAMILIES`] maps a probe family name (see [`super::probe::IMAGE_FAMILIES`])
//! to a constructor. It is empty in the runtime issue; each model issue adds
//! one row:
//!
//! ```ignore
//! ImageFamily { name: "flux2", load: crate::imagegen::flux2::load },
//! ```
//!
//! A family the probe recognizes but this table does not list is reported as
//! unsupported by this build rather than as an unknown checkpoint.

use std::path::Path;

use anyhow::{Result, bail};

use super::pipeline::ImageGenerationModel;
use super::probe::detect_image_family;

/// Builds a model from a checkpoint directory, on the calling thread.
pub type FamilyLoader = fn(&Path) -> Result<Box<dyn ImageGenerationModel>>;

/// One row of the family table.
#[derive(Clone, Copy)]
pub struct ImageFamily {
    /// Probe family name.
    pub name: &'static str,
    pub load: FamilyLoader,
}

/// Families this build can run; each model issue adds its row.
pub static FAMILIES: &[ImageFamily] = &[
    ImageFamily {
        name: "flux2",
        load: super::flux2::load,
    },
    ImageFamily {
        name: "z_image",
        load: super::z_image::load,
    },
    ImageFamily {
        name: "bonsai",
        load: super::bonsai::load,
    },
    ImageFamily {
        name: "ernie_image",
        load: super::ernie_image::load,
    },
    ImageFamily {
        name: "mage_flow",
        load: super::mage_flow::load,
    },
    ImageFamily {
        name: "qwen_image",
        load: super::qwen_image::load,
    },
    ImageFamily {
        name: "ideogram4",
        load: super::ideogram4::load,
    },
    ImageFamily {
        name: "llada_image",
        load: super::llada_image::load,
    },
];

/// Classify the checkpoint at `path` and build its model.
///
/// MLX work is thread-affine: call this on the thread that will run the model.
pub fn load_image_model(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    load_image_model_from(path, FAMILIES)
}

/// Classify the checkpoint at `path` and find its row in `families`.
///
/// Reads only small JSON files, so it needs no MLX runtime; the CLI calls it
/// before initializing the device so a wrong checkpoint fails immediately.
pub fn resolve_family<'a>(path: &Path, families: &'a [ImageFamily]) -> Result<&'a ImageFamily> {
    let Some(family) = detect_image_family(path)? else {
        bail!(
            "no image generation family recognized in {}",
            path.display()
        );
    };
    families.iter().find(|f| f.name == family).ok_or_else(|| {
        anyhow::anyhow!("image generation family {family} is not supported by this build")
    })
}

/// [`load_image_model`] against an explicit table, so tests can register a
/// stub family without a `cfg(test)` row in the shipping table.
pub fn load_image_model_from(
    path: &Path,
    families: &[ImageFamily],
) -> Result<Box<dyn ImageGenerationModel>> {
    (resolve_family(path, families)?.load)(path)
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
