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

//! Checked temporary-storage formulas paired with the sparse kernel scopes.
use super::sparse::GATHER_CHUNK_BYTES;
use anyhow::{Result, ensure};

pub(super) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse phase budget overflow"))
    })
}
pub(super) fn bytes(rows: usize, width: usize) -> Result<usize> {
    rows.checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse phase budget overflow"))
}
pub(super) fn features(rows: usize, channels: usize) -> Result<usize> {
    bytes(rows, bytes(channels, 4)?)
}

/// Includes raw/safe indices, validity, gathered AND masked values, and output.
/// Pool additionally needs sort-position indices and sum/division temporaries.
pub(super) fn chunk_rows(
    channels: usize,
    output: usize,
    itemsize: usize,
    members: usize,
) -> Result<usize> {
    let member = sum([bytes(bytes(channels, itemsize)?, 2)?, 32])?;
    let row = sum([bytes(members, member)?, bytes(bytes(output, itemsize)?, 3)?])?;
    ensure!(
        row > 0 && row <= GATHER_CHUNK_BYTES,
        "MoGe3 sparse chunk row requires {row} bytes"
    );
    Ok(GATHER_CHUNK_BYTES / row)
}

/// One residual source + owned convolution input + accumulated output. The
/// input is dropped before concatenation, so concat replaces that allocation.
/// Native layer norm statistics and validation fit in the separate 16R term.
pub(super) fn residual(rows: usize, channels: usize) -> Result<usize> {
    sum([
        bytes(features(rows, channels)?, 3)?,
        bytes(rows, 16)?,
        bytes(rows, 16)?.max(GATHER_CHUNK_BYTES),
    ])
}

/// Pool sort metadata fits 128R; gather retains order/parent (8R), first/counts
/// and output coords (24S), input features/coords, chunks, and one bounded chunk.
/// Before concat only parent/output coords survive alongside input and chunks.
pub(super) fn pool(rows: usize, channels: usize) -> Result<usize> {
    let f = features(rows, channels)?;
    let sort = sum([f, bytes(rows, 144)?])?;
    let gather = sum([bytes(f, 2)?, bytes(rows, 48)?, GATHER_CHUNK_BYTES])?;
    let concat = sum([bytes(f, 3)?, bytes(rows, 36)?])?;
    Ok(sort.max(gather).max(concat))
}
pub(super) fn projection(
    rows: usize,
    pooled: usize,
    channels: usize,
    next: usize,
) -> Result<usize> {
    sum([
        features(rows, channels)?,
        bytes(rows, 20)?,
        bytes(pooled, 16)?,
        features(pooled, channels)?,
        features(pooled, next)?,
    ])
}

/// Compatibility API retains its caller's input and eventually returns a map
/// copy. The copy is deferred until owned convolution temporaries are gone.
pub(super) fn public_conv(
    rows: usize,
    input: usize,
    output: usize,
    itemsize: usize,
) -> Result<usize> {
    let input = bytes(bytes(rows, input)?, itemsize)?;
    let output = bytes(bytes(rows, output)?, itemsize)?;
    let map = bytes(rows, 108)?;
    let coords = bytes(rows, 16)?;
    let gather = sum([bytes(input, 2)?, output, map, coords, GATHER_CHUNK_BYTES])?;
    let concat = sum([input, bytes(output, 2)?, map, coords])?;
    let returned_map = sum([input, output, bytes(map, 2)?, coords])?;
    let validation = sum([input, map, coords, bytes(rows, 81)?])?;
    Ok(gather.max(concat).max(returned_map).max(validation))
}

pub(super) fn fusion(rows: usize, channels: usize, encoder: usize) -> Result<usize> {
    sum([
        bytes(features(rows, channels)?, 6)?,
        features(rows, encoder)?,
        bytes(rows, 64)?,
    ])
}
pub(super) fn upsample(
    source: usize,
    source_channels: usize,
    target: usize,
    channels: usize,
) -> Result<usize> {
    let live = sum([
        features(source, source_channels)?,
        bytes(source, 16)?,
        features(source, channels)?,
        bytes(features(target, channels)?, 2)?,
        bytes(target, 16)?,
    ])?;
    let validation = sum([
        features(source, source_channels)?,
        bytes(source, 16)?,
        bytes(target, 96)?,
    ])?;
    Ok(live.max(validation))
}
