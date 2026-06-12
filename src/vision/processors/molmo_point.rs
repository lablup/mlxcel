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

//! Molmo-Point Image Processor
//!
//! The image processing pipeline for Molmo-Point is identical to Molmo2:
//! multi-scale overlapping crop preprocessing with the same parameters.
//! This module re-exports the Molmo2 processor directly.
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/image_processing.py

/// Molmo-Point uses the same image processor as Molmo2.
/// Re-export for clarity in the loading code.
pub type MolmoPointProcessor = super::molmo2::Molmo2Processor;
pub type MolmoPointProcessorOutput = super::molmo2::Molmo2ProcessorOutput;
