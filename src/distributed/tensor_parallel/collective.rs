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

//! Collective communication primitives for tensor parallelism.
//!
//! This module provides the core collective operations needed for TP inference:
//!
//! - [`all_reduce_sum`] — element-wise sum across all ranks (ring all-reduce)
//! - [`all_gather`] — gather sharded outputs into a full tensor
//! - [`reduce_scatter`] — scatter-reduce for bandwidth-efficient communication
//!
//! All operations work on raw byte buffers with explicit [`TensorDtype`] for
//! type-safe element arithmetic. The ring topology ensures near-optimal
//! bandwidth utilization: each rank transfers `2 * (N-1)/N * tensor_size`
//! bytes total.
//!
//! Used by: tensor_parallel forward pass (row-parallel all-reduce, vocab-parallel all-gather)

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::distributed::tensor_protocol::TensorDtype;

/// Configuration for collective operations.
#[derive(Debug, Clone)]
pub struct CollectiveConfig {
    /// This rank's index in the collective group (0-based).
    pub rank: usize,
    /// Total number of ranks participating.
    pub world_size: usize,
    /// Chunk size in bytes for pipelined transfers. Larger chunks amortize
    /// latency but increase memory pressure. Default: 1 MiB.
    pub chunk_size: usize,
}

impl CollectiveConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            bail!("world_size must be >= 1");
        }
        if self.rank >= self.world_size {
            bail!(
                "rank {} out of range for world_size {}",
                self.rank,
                self.world_size
            );
        }
        if self.chunk_size == 0 {
            bail!("chunk_size must be > 0");
        }
        Ok(())
    }
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            chunk_size: 1024 * 1024, // 1 MiB
        }
    }
}

/// Ring topology helper for computing send/receive neighbors.
///
/// In a ring of N ranks numbered 0..N-1, rank `r` sends to `(r+1) % N`
/// and receives from `(r+N-1) % N` in each step. Over `N-1` steps every
/// rank has communicated with every other rank exactly once.
#[derive(Debug, Clone)]
pub struct RingTopology {
    pub rank: usize,
    pub world_size: usize,
}

impl RingTopology {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    /// The peer this rank sends to in the ring.
    #[inline]
    pub fn send_to(&self) -> usize {
        (self.rank + 1) % self.world_size
    }

    /// The peer this rank receives from in the ring.
    #[inline]
    pub fn recv_from(&self) -> usize {
        (self.rank + self.world_size - 1) % self.world_size
    }

    /// The chunk index this rank "owns" initially (for reduce-scatter).
    #[inline]
    pub fn initial_chunk(&self) -> usize {
        self.rank
    }

    /// The chunk index this rank should send during step `step` of reduce-scatter.
    #[inline]
    pub fn reduce_scatter_send_chunk(&self, step: usize) -> usize {
        // In step s, rank r sends chunk (r - s) mod N
        (self.rank + self.world_size - step) % self.world_size
    }

    /// The chunk index this rank receives during step `step` of reduce-scatter.
    #[inline]
    pub fn reduce_scatter_recv_chunk(&self, step: usize) -> usize {
        // In step s, rank r receives chunk (r - s - 1) mod N
        (self.rank + self.world_size - step - 1) % self.world_size
    }

    /// The chunk index this rank should send during step `step` of all-gather.
    #[inline]
    pub fn all_gather_send_chunk(&self, step: usize) -> usize {
        // In step s, rank r sends chunk (r - s + 1) mod N
        (self.rank + self.world_size - step + 1) % self.world_size
    }

    /// The chunk index this rank receives during step `step` of all-gather.
    #[inline]
    pub fn all_gather_recv_chunk(&self, step: usize) -> usize {
        // In step s, rank r receives chunk (r - s) mod N
        (self.rank + self.world_size - step) % self.world_size
    }
}

/// A group of ranks participating in a collective operation.
///
/// Holds the per-rank exchange function. The `exchange_fn` is called during
/// each ring step: it sends a chunk to the next rank and receives a chunk
/// from the previous rank. This abstraction decouples the algorithm from
/// the transport layer.
pub struct CollectiveGroup {
    pub config: CollectiveConfig,
    /// Exchange function: `(send_to_rank, send_data) -> received_data`.
    ///
    /// The caller must ensure that this function performs a paired send/recv
    /// with the appropriate peer. For in-process testing, this is backed by
    /// channels; for real clusters, by the Transport trait.
    exchange_fn: Arc<dyn Fn(usize, Vec<u8>) -> Result<Vec<u8>> + Send + Sync>,
}

impl std::fmt::Debug for CollectiveGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollectiveGroup")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CollectiveGroup {
    /// Create a new collective group with a synchronous exchange function.
    pub fn new(
        config: CollectiveConfig,
        exchange_fn: Arc<dyn Fn(usize, Vec<u8>) -> Result<Vec<u8>> + Send + Sync>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            exchange_fn,
        })
    }

    /// Convenience: create a single-rank group that performs no communication.
    pub fn single_rank() -> Result<Self> {
        Self::new(
            CollectiveConfig::default(),
            Arc::new(|_, _| bail!("single-rank group should not exchange")),
        )
    }

    /// Perform a send/recv exchange with the ring neighbor.
    fn exchange(&self, send_to_rank: usize, data: Vec<u8>) -> Result<Vec<u8>> {
        (self.exchange_fn)(send_to_rank, data)
    }

    pub fn rank(&self) -> usize {
        self.config.rank
    }

    pub fn world_size(&self) -> usize {
        self.config.world_size
    }
}

// ---------------------------------------------------------------------------
// Element-wise arithmetic helpers
// ---------------------------------------------------------------------------

/// Add `src` elements into `dst` in-place, interpreting both buffers as the
/// given dtype. Buffers must have the same length.
fn elementwise_add_inplace(dst: &mut [u8], src: &[u8], dtype: TensorDtype) -> Result<()> {
    if dst.len() != src.len() {
        bail!(
            "buffer length mismatch: dst={} src={}",
            dst.len(),
            src.len()
        );
    }
    let elem_size = dtype.element_size();
    if elem_size == 0 {
        bail!("elementwise_add not supported for sub-byte dtypes");
    }
    if !dst.len().is_multiple_of(elem_size) {
        bail!(
            "buffer length {} not aligned to element size {elem_size}",
            dst.len()
        );
    }

    match dtype {
        TensorDtype::Float32 => add_inplace_f32(dst, src),
        TensorDtype::Float16 => add_inplace_f16(dst, src),
        TensorDtype::BFloat16 => add_inplace_bf16(dst, src),
        TensorDtype::Int32 => add_inplace_i32(dst, src),
        TensorDtype::Int16 => add_inplace_i16(dst, src),
        TensorDtype::Int8 => add_inplace_i8(dst, src),
        other => bail!("elementwise_add not supported for dtype {other}"),
    }
    Ok(())
}

/// Add f32 elements in-place.
fn add_inplace_f32(dst: &mut [u8], src: &[u8]) {
    for i in (0..dst.len()).step_by(4) {
        let a = f32::from_le_bytes([dst[i], dst[i + 1], dst[i + 2], dst[i + 3]]);
        let b = f32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
        let sum = a + b;
        dst[i..i + 4].copy_from_slice(&sum.to_le_bytes());
    }
}

/// Add float16 elements in-place by promoting to f32.
///
/// IEEE 754 half-precision: sign(1) | exponent(5) | mantissa(10).
fn add_inplace_f16(dst: &mut [u8], src: &[u8]) {
    for i in (0..dst.len()).step_by(2) {
        let a = f16_to_f32(u16::from_le_bytes([dst[i], dst[i + 1]]));
        let b = f16_to_f32(u16::from_le_bytes([src[i], src[i + 1]]));
        let sum = f32_to_f16(a + b);
        dst[i..i + 2].copy_from_slice(&sum.to_le_bytes());
    }
}

/// Add bfloat16 elements in-place by promoting to f32.
///
/// BFloat16: sign(1) | exponent(8) | mantissa(7). Same exponent range as f32.
fn add_inplace_bf16(dst: &mut [u8], src: &[u8]) {
    for i in (0..dst.len()).step_by(2) {
        let a = bf16_to_f32(u16::from_le_bytes([dst[i], dst[i + 1]]));
        let b = bf16_to_f32(u16::from_le_bytes([src[i], src[i + 1]]));
        let sum = f32_to_bf16(a + b);
        dst[i..i + 2].copy_from_slice(&sum.to_le_bytes());
    }
}

/// Add i32 elements in-place.
fn add_inplace_i32(dst: &mut [u8], src: &[u8]) {
    for i in (0..dst.len()).step_by(4) {
        let a = i32::from_le_bytes([dst[i], dst[i + 1], dst[i + 2], dst[i + 3]]);
        let b = i32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
        let sum = a.wrapping_add(b);
        dst[i..i + 4].copy_from_slice(&sum.to_le_bytes());
    }
}

/// Add i16 elements in-place.
fn add_inplace_i16(dst: &mut [u8], src: &[u8]) {
    for i in (0..dst.len()).step_by(2) {
        let a = i16::from_le_bytes([dst[i], dst[i + 1]]);
        let b = i16::from_le_bytes([src[i], src[i + 1]]);
        let sum = a.wrapping_add(b);
        dst[i..i + 2].copy_from_slice(&sum.to_le_bytes());
    }
}

/// Add i8 elements in-place.
fn add_inplace_i8(dst: &mut [u8], src: &[u8]) {
    for i in 0..dst.len() {
        let a = dst[i] as i8;
        let b = src[i] as i8;
        dst[i] = a.wrapping_add(b) as u8;
    }
}

// ---------------------------------------------------------------------------
// IEEE 754 half-precision (float16) conversion helpers
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (float16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            // +/- zero
            f32::from_bits(sign << 31)
        } else {
            // Denormalized: convert to normalized f32
            let mut m = mantissa;
            let mut e: i32 = -14; // f16 denorm exponent bias
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // remove implicit leading 1
            let f32_exp = ((e + 127) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
    } else {
        // Normalized: rebias from f16 bias (15) to f32 bias (127).
        let f32_exp = exp + 112; // (exp - 15 + 127) = (exp + 112)
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

/// Convert f32 to IEEE 754 half-precision (float16) bits with
/// round-to-nearest-even semantics.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or f32 denorm -> f16 zero (preserving sign)
        sign << 15
    } else if exp == 0xFF {
        // Inf or NaN
        if mantissa == 0 {
            (sign << 15) | 0x7C00 // Inf
        } else {
            (sign << 15) | 0x7C00 | ((mantissa >> 13) as u16).max(1) // NaN
        }
    } else if exp > 142 {
        // Overflow -> Inf (f32 exponent 143 = f16 exponent 31 = Inf)
        (sign << 15) | 0x7C00
    } else if exp < 103 {
        // Too small even for f16 denorm -> zero
        sign << 15
    } else if exp < 113 {
        // Denormalized f16 range: exponent maps to f16 exp=0.
        // Shift mantissa right by (113 - exp), adding the implicit leading 1.
        let shift = (113 - exp) as u32;
        let full_mantissa = 0x800000u32 | mantissa;
        let total_shift = shift + 13;
        let denorm_mantissa = (full_mantissa >> total_shift) as u16;
        // Round-to-nearest-even for the bits we are discarding.
        let round_bit = (full_mantissa >> (total_shift - 1)) & 1;
        let sticky_mask = (1u32 << (total_shift - 1)) - 1;
        let sticky = if (full_mantissa & sticky_mask) != 0 {
            1u32
        } else {
            0
        };
        let mut result = (sign << 15) | denorm_mantissa;
        if round_bit == 1 && (sticky == 1 || denorm_mantissa & 1 == 1) {
            result += 1; // may promote to smallest normal, which is correct
        }
        result
    } else {
        let new_exp = (exp - 112) as u16;
        // Round-to-nearest-even: examine the 13 bits being truncated.
        let round_bit = (mantissa >> 12) & 1;
        let sticky = mantissa & 0xFFF;
        let new_mantissa = (mantissa >> 13) as u16;
        let mut result = (sign << 15) | (new_exp << 10) | new_mantissa;
        if round_bit == 1 && (sticky != 0 || new_mantissa & 1 == 1) {
            result += 1; // carry into exponent is fine (produces next float16)
        }
        result
    }
}

/// Convert bfloat16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    // BF16 is simply the upper 16 bits of a float32.
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bfloat16 bits with round-to-nearest-even semantics.
fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round-to-nearest-even: examine bit 15 (round) and bits 0-14 (sticky).
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let truncated = (bits >> 16) as u16;
    if round_bit == 1 && (sticky != 0 || truncated & 1 == 1) {
        truncated + 1 // carry is safe; wraps correctly for NaN/Inf boundary
    } else {
        truncated
    }
}

// ---------------------------------------------------------------------------
// Chunk helpers
// ---------------------------------------------------------------------------

/// Split a buffer into `n_chunks` pieces. The last chunk may be smaller if
/// the element count is not evenly divisible.
fn split_into_chunks(data: &[u8], n_chunks: usize, elem_size: usize) -> Result<Vec<Vec<u8>>> {
    if n_chunks == 0 {
        bail!("n_chunks must be > 0");
    }
    let total_elems = data.len() / elem_size;
    let base_elems = total_elems / n_chunks;
    let remainder = total_elems % n_chunks;

    let mut chunks = Vec::with_capacity(n_chunks);
    let mut offset = 0;
    for i in 0..n_chunks {
        let elems = base_elems + if i < remainder { 1 } else { 0 };
        let byte_len = elems * elem_size;
        chunks.push(data[offset..offset + byte_len].to_vec());
        offset += byte_len;
    }
    Ok(chunks)
}

/// Reassemble chunks into a contiguous buffer.
fn reassemble_chunks(chunks: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(total);
    for c in chunks {
        buf.extend_from_slice(c);
    }
    buf
}

// ---------------------------------------------------------------------------
// Public collective operations
// ---------------------------------------------------------------------------

/// All-reduce (sum) the data buffer across all ranks in the group.
///
/// Uses a ring all-reduce algorithm consisting of two phases:
/// 1. **Reduce-scatter**: Each rank accumulates partial sums so that after
///    `N-1` steps, chunk `i` is fully reduced on rank `i`.
/// 2. **All-gather**: The fully-reduced chunks are propagated around the ring
///    so every rank ends up with the complete reduced result.
///
/// The operation is performed **in-place**: on return, `data` contains the
/// element-wise sum of the original data from all ranks.
///
/// # Errors
///
/// Returns an error if the dtype is unsupported, the buffer length is not
/// aligned to the element size, or the exchange function fails.
pub fn all_reduce_sum(
    data: &mut Vec<u8>,
    dtype: TensorDtype,
    group: &CollectiveGroup,
) -> Result<()> {
    let world_size = group.world_size();
    // Single-rank: no-op.
    if world_size == 1 {
        return Ok(());
    }

    let elem_size = dtype.element_size();
    if elem_size == 0 {
        bail!("all_reduce_sum does not support sub-byte dtypes (e.g., int4)");
    }
    if !data.len().is_multiple_of(elem_size) {
        bail!(
            "buffer length {} is not aligned to element size {}",
            data.len(),
            elem_size
        );
    }

    let ring = RingTopology::new(group.rank(), world_size);
    let send_to = ring.send_to();

    // Phase 1: Reduce-scatter
    let mut chunks = split_into_chunks(data, world_size, elem_size)
        .context("splitting buffer for reduce-scatter")?;

    for step in 0..world_size - 1 {
        let send_idx = ring.reduce_scatter_send_chunk(step);
        let recv_idx = ring.reduce_scatter_recv_chunk(step);

        let send_data = chunks[send_idx].clone();
        let recv_data = group
            .exchange(send_to, send_data)
            .with_context(|| format!("reduce-scatter step {step}"))?;

        // Accumulate received data into the recv chunk.
        elementwise_add_inplace(&mut chunks[recv_idx], &recv_data, dtype)
            .with_context(|| format!("reduce-scatter accumulate step {step}"))?;
    }

    // Phase 2: All-gather
    for step in 0..world_size - 1 {
        let send_idx = ring.all_gather_send_chunk(step);
        let recv_idx = ring.all_gather_recv_chunk(step);

        let send_data = chunks[send_idx].clone();
        let recv_data = group
            .exchange(send_to, send_data)
            .with_context(|| format!("all-gather step {step}"))?;

        // Replace the chunk with the fully-reduced version.
        chunks[recv_idx] = recv_data;
    }

    // Reassemble into the output buffer.
    let result = reassemble_chunks(&chunks);
    data.clear();
    data.extend_from_slice(&result);
    Ok(())
}

/// Gather sharded data from all ranks into a full tensor.
///
/// Each rank contributes its `shard` (a contiguous piece of the full tensor).
/// On return, every rank holds the concatenation of all shards in rank order.
///
/// Uses a ring all-gather: over `N-1` steps, each rank forwards the chunk it
/// received in the previous step to the next rank.
///
/// # Returns
///
/// The fully gathered buffer (concatenation of all shards in rank order).
pub fn all_gather(shard: &[u8], dtype: TensorDtype, group: &CollectiveGroup) -> Result<Vec<u8>> {
    let world_size = group.world_size();
    if world_size == 1 {
        return Ok(shard.to_vec());
    }

    let elem_size = dtype.element_size();
    if elem_size == 0 {
        bail!("all_gather does not support sub-byte dtypes");
    }

    let ring = RingTopology::new(group.rank(), world_size);
    let send_to = ring.send_to();

    // Initialize chunks: only our rank's slot is populated.
    let mut chunks: Vec<Option<Vec<u8>>> = (0..world_size).map(|_| None).collect();
    chunks[group.rank()] = Some(shard.to_vec());

    // The data we send in the first step is our own shard.
    let mut send_data = shard.to_vec();

    for step in 0..world_size - 1 {
        let recv_data = group
            .exchange(send_to, send_data)
            .with_context(|| format!("all_gather step {step}"))?;

        // Determine which rank's data we just received.
        // In step s, we receive data from rank (rank - s - 1) mod N.
        let source_rank = (group.rank() + world_size - step - 1) % world_size;
        chunks[source_rank] = Some(recv_data.clone());

        // Forward what we received in the next step.
        send_data = recv_data;
    }

    // Reassemble in rank order.
    let mut result = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let data = chunk
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("all_gather: missing chunk for rank {i}"))?;
        result.extend_from_slice(data);
    }
    Ok(result)
}

/// Reduce-scatter: reduce (sum) across all ranks, then each rank keeps only
/// its portion (1/N-th) of the result.
///
/// This is equivalent to an all-reduce followed by keeping only the local
/// shard, but is more bandwidth-efficient because each rank only needs the
/// final result for its chunk.
///
/// # Returns
///
/// The reduced shard for this rank (approximately `data.len() / world_size` bytes).
pub fn reduce_scatter(data: &[u8], dtype: TensorDtype, group: &CollectiveGroup) -> Result<Vec<u8>> {
    let world_size = group.world_size();
    if world_size == 1 {
        return Ok(data.to_vec());
    }

    let elem_size = dtype.element_size();
    if elem_size == 0 {
        bail!("reduce_scatter does not support sub-byte dtypes");
    }
    if !data.len().is_multiple_of(elem_size) {
        bail!(
            "buffer length {} is not aligned to element size {}",
            data.len(),
            elem_size
        );
    }

    let ring = RingTopology::new(group.rank(), world_size);
    let send_to = ring.send_to();

    let mut chunks = split_into_chunks(data, world_size, elem_size)
        .context("splitting buffer for reduce_scatter")?;

    // Reduce-scatter phase only (no all-gather).
    for step in 0..world_size - 1 {
        let send_idx = ring.reduce_scatter_send_chunk(step);
        let recv_idx = ring.reduce_scatter_recv_chunk(step);

        let send_data = chunks[send_idx].clone();
        let recv_data = group
            .exchange(send_to, send_data)
            .with_context(|| format!("reduce_scatter step {step}"))?;

        elementwise_add_inplace(&mut chunks[recv_idx], &recv_data, dtype)
            .with_context(|| format!("reduce_scatter accumulate step {step}"))?;
    }

    // After N-1 reduce-scatter steps, rank r's fully reduced chunk is at
    // index (r + 1) % N due to the ring rotation pattern. Return it as
    // rank r's shard (standard MPI reduce-scatter semantics: rank r holds
    // the r-th portion of the reduced result).
    let reduced_idx = (group.rank() + 1) % world_size;
    Ok(chunks[reduced_idx].clone())
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

/// Result of a collective operation benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the operation benchmarked.
    pub operation: String,
    /// Data type used.
    pub dtype: TensorDtype,
    /// Total tensor size in bytes (per rank input).
    pub tensor_bytes: usize,
    /// Number of ranks.
    pub world_size: usize,
    /// Elapsed wall-clock time in microseconds.
    pub elapsed_us: u64,
    /// Achieved bandwidth in bytes/second (based on algorithm data volume).
    pub bandwidth_bytes_per_sec: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size_label = if self.tensor_bytes >= 1024 * 1024 {
            format!("{} MiB", self.tensor_bytes / (1024 * 1024))
        } else if self.tensor_bytes >= 1024 {
            format!("{} KiB", self.tensor_bytes / 1024)
        } else {
            format!("{} B", self.tensor_bytes)
        };
        let bw_label = if self.bandwidth_bytes_per_sec >= 1e9 {
            format!("{:.2} GB/s", self.bandwidth_bytes_per_sec / 1e9)
        } else if self.bandwidth_bytes_per_sec >= 1e6 {
            format!("{:.2} MB/s", self.bandwidth_bytes_per_sec / 1e6)
        } else {
            format!("{:.2} KB/s", self.bandwidth_bytes_per_sec / 1e3)
        };
        write!(
            f,
            "{}: {} x {} ranks, {:.0} us, {}",
            self.operation, size_label, self.world_size, self.elapsed_us, bw_label
        )
    }
}

/// Compute the theoretical data volume transferred per rank for ring
/// all-reduce: `2 * (N-1)/N * tensor_size`.
pub fn ring_allreduce_data_volume(tensor_bytes: usize, world_size: usize) -> f64 {
    if world_size <= 1 {
        return 0.0;
    }
    2.0 * (world_size as f64 - 1.0) / world_size as f64 * tensor_bytes as f64
}

#[cfg(test)]
#[path = "collective_tests.rs"]
mod tests;
