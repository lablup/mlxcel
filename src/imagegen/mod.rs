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

//! Image generation runtime.
//!
//! Model-independent pieces shared by every text-to-image and image-editing
//! family: the request and result types, the family probe, the flow-match
//! scheduler, latent packing, the FLUX.2 VAE, the Z-Image DiT blocks, PNG
//! output, and the family registry. A model issue adds its own transformer
//! and text conditioner and registers one row in [`registry::FAMILIES`].
//!
//! The surfaces on top of this module are `mlxcel generate-image` and the
//! `POST /v1/images/generations` and `POST /v1/images/edits` server routes.
//! See `docs/image-generation.md`.

pub mod bonsai;
pub mod dit_blocks;
pub mod flux2;
pub mod latent;
pub mod output;
pub mod pipeline;
pub mod probe;
pub mod qwen3_text_encoder;
pub mod registry;
pub mod run;
pub mod scheduler;
pub mod vae;

#[cfg(test)]
pub(crate) mod testing;

pub use output::{default_output_path, encode_png, indexed_output_path, to_b64_png};
pub use pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};
pub use probe::{IMAGE_FAMILIES, detect_image_family, image_checkpoint_redirect_error};
pub use qwen3_text_encoder::Qwen3TextEncoder;
pub use registry::{FAMILIES, ImageFamily, load_image_model, resolve_family};
pub use scheduler::{DynamicShift, FlowMatchEulerDiscrete};
pub use vae::{Flux2Vae, Flux2VaeConfig};
