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

//! Validated dimensions for MoGe3 sparse tensors.

use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseShape {
    batch: i64,
    height: i64,
    width: i64,
    depth: i64,
}

impl SparseShape {
    pub fn new(batch: i64, height: i64, width: i64, depth: i64) -> Result<Self> {
        ensure!(
            batch > 0 && height > 0 && width > 0 && depth > 0,
            "MoGe3 sparse shape dimensions must be positive"
        );
        ensure!(
            [batch, height, width, depth]
                .into_iter()
                .all(|v| v <= i64::from(i32::MAX)),
            "MoGe3 sparse shape dimensions must fit int32 coordinates"
        );
        let total = batch
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .and_then(|v| v.checked_mul(depth))
            .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse key space overflow"))?;
        ensure!(total < i64::MAX, "MoGe3 sparse key space is too large");
        Ok(Self {
            batch,
            height,
            width,
            depth,
        })
    }

    pub fn batch(self) -> i64 {
        self.batch
    }
    pub fn height(self) -> i64 {
        self.height
    }
    pub fn width(self) -> i64 {
        self.width
    }
    pub fn depth(self) -> i64 {
        self.depth
    }
    pub(super) fn pooled(self) -> Self {
        Self {
            batch: self.batch,
            height: (self.height + 1) / 2,
            width: (self.width + 1) / 2,
            depth: (self.depth + 1) / 2,
        }
    }
}
