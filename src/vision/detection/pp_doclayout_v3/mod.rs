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

//! PP-DocLayoutV3 document-layout detection with model reading order.

mod aifi;
mod backbone;
mod config_limits;
mod decoder;
mod deformable;
#[cfg(test)]
mod diagnostics;
mod encoder;
mod layers;
mod mask;
mod mask_box;
mod model;
mod order;
mod postprocess;
mod predictor;
mod processor;
mod sanitize;
mod weights;

pub use config::{HgNetV2Config, PpDocLayoutV3Config};
pub use model::{PpDocLayoutV3Model, PpDocLayoutV3Output};
pub use postprocess::{decode_detections, decode_reading_order, denormalize_box};
pub use predictor::{DEFAULT_THRESHOLD, PpDocLayoutV3Predictor};
pub use processor::{PpDocLayoutV3Processor, ProcessedImage, ProcessorConfig};
pub use sanitize::{needs_sanitize, sanitize};

pub mod config;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "real_tests.rs"]
mod real_tests;
