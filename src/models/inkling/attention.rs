// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! Compatibility exports for Inkling attention tests.
//!
//! Runtime target and MTP layers both use the implementation in
//! `mlxcel_core::inkling_layer`; keeping these exports avoids duplicating the
//! numerical helpers in the model crate.

#[cfg(test)]
pub(crate) use mlxcel_core::inkling_layer::{
    InklingShortConv, banded_additive_mask, log_scaling_tau,
};
