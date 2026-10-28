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

//! The FLUX.2 VAE.
//!
//! A port of `Flux2VAE` and its encoder and decoder in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/flux2/vae>,
//! loaded from `<model>/vae/`. Tiled decode (`flux2/tiling.py`) is out of scope.
//!
//! Used by: ErnieImage, LLaDAImage, ZImage, Ideogram4 (and FLUX.2 itself). The conv and
//! group-norm layers and the conv-layout probe are also used by MageFlow.

mod config;
mod layers;
mod model;

pub use config::Flux2VaeConfig;
pub(crate) use layers::{Conv2d, GroupNorm};
pub use model::Flux2Vae;
pub(crate) use model::{ConvLayout, probe_conv_layout};
