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

//! Object-detection vision models.
//!
//! Detection models output bounding boxes (`l, t, r, b, label, confidence`)
//! rather than a token stream, so they sit outside the text/VLM generation flow
//! (`LanguageModel`, the generate loop) and are exposed through the `detect`
//! CLI subcommand instead.
//!
//! Note: this module is for object *detection* and is distinct from
//! `crate::models::detection`, which performs config-driven model-*type*
//! classification. The naming overlap is unfortunate; the model-type module is
//! the older one and renaming it is deferred to a follow-up so concurrent work
//! on that file is not disrupted.

pub mod rt_detr_v2;

pub use rt_detr_v2::{
    Detection, DetectionResult, RtDetrV2Config, RtDetrV2Model, RtDetrV2Predictor,
};
