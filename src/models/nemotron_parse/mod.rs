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

//! Nemotron-Parse (`nemotron_parse`): NVIDIA's document OCR / layout model.
//!
//! A C-RADIO ViT-H/16 tower reads a white-padded 2048x1664 page
//! (128x104 patches plus 8 prefix tokens), a compression neck shrinks the
//! patch grid 4x horizontally and appends one summary row (3329 encoder
//! states), and a 10-layer pre-norm mBART decoder writes markdown annotated
//! with `<x_..><y_..>` box tokens and `<class_..>` tags.
//!
//! It is encoder-decoder, so it runs on the same seq2seq plumbing as
//! Florence-2: the attention sublayers, the dual self/cross KV cache and the
//! causal-mask helper come from `crate::models::florence2::layers`, the CLI
//! drives it through its own early-exit driver, and the server serves it on
//! a dedicated single-stream worker.
//!
//! References (shipped with `nvidia/NVIDIA-Nemotron-Parse-2.0`):
//! `hf_nemotron_parse_modeling.py`, `hf_nemotron_parse_processor.py`, and
//! the C-RADIO code in `nvidia/C-RADIOv2-H`.

mod checkpoint;
mod config;
mod decoder;
mod encoder;
mod model;
mod neck;
mod processor;
mod runtime;

pub use config::{NemotronParseConfig, NemotronParseTextConfig, NemotronParseVisionConfig};
pub use model::{
    NemotronParseGeneration, NemotronParseModel, NemotronParseSeqCache, apply_repetition_penalty,
};
pub use processor::{
    DEFAULT_TASK_PROMPT, NemotronParseImageProcessor, seed_from_prompt_ids, seed_ids,
};
pub use runtime::{NemotronParseRunOutput, NemotronParseVlmModel};

#[cfg(test)]
#[path = "nemotron_parse_tests.rs"]
mod nemotron_parse_tests;

#[cfg(test)]
#[path = "nemotron_parse_model_tests.rs"]
mod nemotron_parse_model_tests;
