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

//! Audio modality root.
//!
//! mlxcel hosts more than one audio family. The legacy entries
//! (`config`, `encoder`, `feature_extractor`, `attention`) are the
//! Gemma 4 USM-style implementation; sibling families add their own
//! sub-namespaces (currently `nemotron_h_nano_omni`). The Gemma 4
//! modules are kept at the top level for backwards-compatibility with
//! the existing VLM wiring; new families should land under their own
//! submodule from the start.
//!
//! Used by:
//! - Gemma 4 VLM (audio modality, top-level files)
//! - Nemotron H Nano Omni VLM (audio modality, [`nemotron_h_nano_omni`])

mod attention;
pub mod config;
pub mod encoder;
pub mod feature_extractor;
pub mod gemma3n;
pub mod nemotron_h_nano_omni;
pub mod phi4mm;
pub mod qwen3_omni_moe;
pub mod wav_writer;
pub mod whisper_mel;

pub use config::AudioConfig;
pub use encoder::AudioEncoder;
pub(crate) use encoder::audio_probe_dump as encoder_probe_dump;
pub use feature_extractor::{
    AudioFeatureExtractor, AudioFeatureExtractorConfig, compute_audio_num_tokens, load_wav_file,
    load_wav_from_bytes,
};
pub use wav_writer::encode_wav_pcm16;
