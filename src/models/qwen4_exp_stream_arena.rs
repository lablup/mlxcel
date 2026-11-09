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

//! The resident slot tensors of one layer and the routing step that fills
//! them (issue #2882).
//!
//! Each plane is a page-aligned host allocation wrapped as an MLX array
//! without a copy, so a `pread` into a slot is the only movement of the bytes.
//! The GPU reads the same memory: a slot may only be rewritten once every
//! command that reads it has finished, which `pending` enforces.

use std::alloc::Layout;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::ExpertStream;
use super::slots::{LayerSlots, split_expert_groups, split_rows};
use super::source::{ExpertSource, PLANE_SUFFIXES};
use crate::models::qwen3_next::SwitchGLU;

/// Metal wraps memory without copying only on page boundaries; 16 KiB is the
/// Apple Silicon page, and a multiple of the 4 KiB page elsewhere.
const ARENA_PAGE: usize = 16 * 1024;
/// `SwitchGLU::forward_with_sort`'s sorted-path threshold, in routed experts.
const SORTED_MIN_ROWS: usize = 64;
/// MLX's `B / E >= 4` condition for its sorted gathered matmul.
const SORTED_KERNEL_RATIO: usize = 4;
/// Concurrent expert reads while filling one window.
const READ_THREADS: usize = 4;

/// A page-aligned allocation an MLX array borrows. Untouched pages cost
/// nothing until their slot is first filled.
struct PlaneBuffer {
    ptr: *mut u8,
    layout: Layout,
    row_bytes: usize,
}

// SAFETY: the pointer is owned here, and writers touch disjoint slot rows
// while the arena's lock is held and no command reads those rows.
unsafe impl Send for PlaneBuffer {}
unsafe impl Sync for PlaneBuffer {}

impl PlaneBuffer {
    fn new(slots: usize, row_bytes: usize) -> Result<Self, String> {
        let bytes = slots * row_bytes;
        if bytes == 0 || !bytes.is_multiple_of(ARENA_PAGE) {
            return Err(format!(
                "expert arena plane of {bytes} bytes is not a whole number of pages"
            ));
        }
        let layout = Layout::from_size_align(bytes, ARENA_PAGE).map_err(|e| e.to_string())?;
        // SAFETY: `layout` has a non-zero size; the result is checked for null.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(format!(
                "cannot allocate {bytes} bytes for the expert arena"
            ));
        }
        Ok(Self {
            ptr,
            layout,
            row_bytes,
        })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `layout.size()` bytes for the life of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    /// The bytes of one slot. Callers hold distinct slots per thread.
    #[allow(clippy::mut_from_ref)]
    fn slot(&self, slot: u32) -> &mut [u8] {
        let start = slot as usize * self.row_bytes;
        debug_assert!(start + self.row_bytes <= self.layout.size());
        // SAFETY: in bounds; distinct slots are disjoint rows, and the window
        // being filled is the only writer under the layer lock.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), self.row_bytes) }
    }
}

impl Drop for PlaneBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

struct LayerState {
    slots: LayerSlots,
    /// The last output that read this layer's slots and may still be queued.
    pending: Option<UniquePtr<MlxArray>>,
}

/// One layer's arena.
pub(crate) struct LayerStream {
    layer: usize,
    stream: Arc<ExpertStream>,
    /// Declared before `buffers`: the arrays wrapping them must drop first.
    state: Mutex<LayerState>,
    buffers: Vec<PlaneBuffer>,
}

impl LayerStream {
    /// Allocate the layer's planes and return them with the slot tensors,
    /// keyed the way `SwitchGLU::from_weights` reads them under `prefix`.
    pub(crate) fn new(
        stream: Arc<ExpertStream>,
        layer: usize,
        prefix: &str,
    ) -> Result<(Self, WeightMap), String> {
        let num_slots = stream.slots[layer];
        let mut buffers = Vec::with_capacity(PLANE_SUFFIXES.len());
        let mut weights = WeightMap::new();
        for (plane, suffix) in stream.source.layers[layer].iter().zip(PLANE_SUFFIXES) {
            let buffer = PlaneBuffer::new(num_slots, plane.row_bytes as usize)?;
            let shape = [num_slots as i32, plane.shape[0], plane.shape[1]];
            let array = mlxcel_core::from_bytes_nocopy(buffer.bytes(), &shape, plane.dtype);
            // Metal refuses a no-copy wrap it cannot honour and MLX then falls
            // back to a copy, which would silently detach the arena from the
            // tensors the kernels read.
            let address = mlxcel_core::array_data_address(&array);
            if address != buffer.ptr as usize {
                return Err(format!(
                    "expert arena: MLX copied the {suffix} plane instead of wrapping it"
                ));
            }
            weights.insert(format!("{prefix}.switch_mlp.{suffix}"), array);
            buffers.push(buffer);
        }
        let num_experts = stream.source.num_experts;
        let this = Self {
            layer,
            stream,
            state: Mutex::new(LayerState {
                slots: LayerSlots::new(num_experts, num_slots),
                pending: None,
            }),
            buffers,
        };
        Ok((this, weights))
    }

    /// Run the routed experts of `topk` through the arena.
    ///
    /// This is the layer's host synchronisation point: `topk` is evaluated,
    /// the window's misses are read, and the gather runs on slot indices.
    pub(crate) fn forward(
        &self,
        experts: &SwitchGLU,
        x: &MlxArray,
        topk: &MlxArray,
        allow_sort: bool,
    ) -> UniquePtr<MlxArray> {
        let stats = &self.stream.stats;
        let synced = Instant::now();
        let topk = mlxcel_core::astype(topk, mlxcel_core::dtype::UINT32);
        mlxcel_core::eval(&topk);
        stats
            .sync_ns
            .fetch_add(synced.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let ids: Vec<u32> = mlxcel_core::array_to_raw_bytes(&topk)
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
            .collect();
        let shape = mlxcel_core::array_shape(&topk);
        let (rows, k) = (shape[0] as usize, shape[1] as usize);

        let mut state = self.state.lock().expect("expert arena lock poisoned");
        if allow_sort && rows * k >= SORTED_MIN_ROWS {
            return self.forward_sorted(&mut state, experts, x, &topk, rows, k);
        }
        let ranges = split_rows(&ids, k, state.slots.capacity());
        let segmented = ranges.len() > 1;
        let hidden = *mlxcel_core::array_shape(x).last().expect("input rank");
        let mut output: Option<UniquePtr<MlxArray>> = None;
        for range in ranges {
            let window = &ids[range.start * k..range.end * k];
            let mapping = state
                .slots
                .map_window(window)
                .expect("split_rows keeps every range within capacity");
            stats.windows.fetch_add(1, Ordering::Relaxed);
            stats.hits.fetch_add(mapping.hits as u64, Ordering::Relaxed);
            stats
                .misses
                .fetch_add(mapping.misses.len() as u64, Ordering::Relaxed);
            if !mapping.misses.is_empty() {
                if let Some(pending) = state.pending.take() {
                    mlxcel_core::eval(&pending);
                }
                self.fill(&mapping.misses);
            }

            let bytes: Vec<u8> = mapping
                .slots
                .iter()
                .flat_map(|slot| slot.to_le_bytes())
                .collect();
            let indices = mlxcel_core::from_bytes(
                &bytes,
                &[(range.end - range.start) as i32, k as i32],
                mlxcel_core::dtype::UINT32,
            );
            let piece = if range.start == 0 && range.end == rows {
                experts.forward_with_sort(x, &indices, allow_sort)
            } else {
                let x_piece =
                    mlxcel_core::slice(x, &[range.start as i32, 0], &[range.end as i32, hidden]);
                experts.forward_with_sort(&x_piece, &indices, allow_sort)
            };
            if segmented {
                // The next range may evict this one's slots.
                mlxcel_core::eval(&piece);
            }
            output = Some(match output {
                None => piece,
                Some(previous) => mlxcel_core::concatenate(&previous, &piece, 0),
            });
        }
        let output = output.expect("a forward has at least one row");
        state.pending = Some(mlxcel_core::array_handle_clone(&output));
        output
    }

    /// The sorted prefill path, reproducing the resident sort.
    ///
    /// Rows are ordered by expert exactly as the resident path orders them. A
    /// window with more distinct experts than slots is cut between experts,
    /// never inside one: cutting by token rows would split an expert's rows
    /// over several matmuls and change the kernel batching, which alters the
    /// low bits of the result.
    ///
    /// The kernel choice also follows the resident path. MLX takes its sorted
    /// matmul when `rows / num_experts >= 4` over the tensor it is given, and
    /// a slot tensor has fewer experts than the checkpoint, so the arena
    /// asks for the sorted kernel only when the resident tensor would get it.
    fn forward_sorted(
        &self,
        state: &mut LayerState,
        experts: &SwitchGLU,
        x: &MlxArray,
        topk: &MlxArray,
        rows: usize,
        k: usize,
    ) -> UniquePtr<MlxArray> {
        let x_expanded = mlxcel_core::expand_dims(&mlxcel_core::expand_dims(x, -2), -3);
        let (sorted_x, sorted_ids, inverse) = experts.gather_sort(&x_expanded, topk);
        mlxcel_core::eval(&sorted_ids);
        let ids: Vec<u32> = mlxcel_core::array_to_raw_bytes(&sorted_ids)
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
            .collect();

        let resident_sorted = rows * k / self.stream.source.num_experts >= SORTED_KERNEL_RATIO;
        let batches = split_expert_groups(&ids, state.slots.capacity());
        let segmented = batches.len() > 1;
        let stats = &self.stream.stats;
        let mut output: Option<UniquePtr<MlxArray>> = None;
        let mut positions: Vec<i32> = Vec::with_capacity(ids.len());
        for batch in batches {
            let window = &ids[batch.clone()];
            let mapping = state
                .slots
                .map_window(window)
                .expect("split_expert_groups keeps every batch within capacity");
            stats.windows.fetch_add(1, Ordering::Relaxed);
            stats.hits.fetch_add(mapping.hits as u64, Ordering::Relaxed);
            stats
                .misses
                .fetch_add(mapping.misses.len() as u64, Ordering::Relaxed);
            if !mapping.misses.is_empty() {
                if let Some(pending) = state.pending.take() {
                    mlxcel_core::eval(&pending);
                }
                self.fill(&mapping.misses);
            }

            // The sorted kernel needs ascending indices, and slot numbers are
            // not ordered like expert numbers. Reorder the groups by slot,
            // keeping each group's rows in place and in order.
            let mut order: Vec<usize> = (0..window.len()).collect();
            order.sort_by_key(|&index| mapping.slots[index]);
            let batch_positions: Vec<i32> = order
                .iter()
                .map(|&index| (batch.start + index) as i32)
                .collect();
            let slot_bytes: Vec<u8> = order
                .iter()
                .flat_map(|&index| mapping.slots[index].to_le_bytes())
                .collect();
            let take = mlxcel_core::from_slice_i32(&batch_positions, &[window.len() as i32]);
            let batch_x = mlxcel_core::take(&sorted_x, &take, 0);
            let indices = mlxcel_core::from_bytes(
                &slot_bytes,
                &[window.len() as i32],
                mlxcel_core::dtype::UINT32,
            );
            let piece = experts.forward_presorted(&batch_x, &indices, resident_sorted);
            if segmented {
                mlxcel_core::eval(&piece);
            }
            positions.extend_from_slice(&batch_positions);
            output = Some(match output {
                None => piece,
                Some(previous) => mlxcel_core::concatenate(&previous, &piece, 0),
            });
        }
        let output = output.expect("a forward has at least one row");
        state.pending = Some(mlxcel_core::array_handle_clone(&output));

        // Back to the resident's sorted order, then the resident's unsort.
        let mut back = vec![0i32; positions.len()];
        for (at, position) in positions.iter().enumerate() {
            back[*position as usize] = at as i32;
        }
        let back = mlxcel_core::from_slice_i32(&back, &[positions.len() as i32]);
        let sorted_output = mlxcel_core::take(&output, &back, 0);
        experts.scatter_unsort(&sorted_output, &inverse, &[rows as i32, k as i32])
    }

    /// Read every miss into its slot, several at a time.
    fn fill(&self, misses: &[(u32, u32)]) {
        let started = Instant::now();
        let threads = READ_THREADS.min(misses.len());
        let (buffers, source, layer) = (&self.buffers, &self.stream.source, self.layer);
        let bytes: u64 = std::thread::scope(|scope| {
            let handles: Vec<_> = misses
                .chunks(misses.len().div_ceil(threads))
                .map(|chunk| scope.spawn(move || read_chunk(buffers, source, layer, chunk)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("expert read thread panicked"))
                .sum()
        });
        let stats = &self.stream.stats;
        stats.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        stats
            .io_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Read one thread's share of a window's misses.
fn read_chunk(
    buffers: &[PlaneBuffer],
    source: &ExpertSource,
    layer: usize,
    misses: &[(u32, u32)],
) -> u64 {
    let mut total = 0;
    for &(expert, slot) in misses {
        let mut dest: Vec<&mut [u8]> = buffers.iter().map(|buffer| buffer.slot(slot)).collect();
        total += source
            .read_expert(layer, expert as usize, &mut dest)
            .unwrap_or_else(|error| {
                panic!("expert streaming cannot continue: layer {layer} expert {expert}: {error}")
            });
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_into_a_slot_is_seen_through_the_wrapped_array() {
        const ROW: usize = ARENA_PAGE;
        let buffer = PlaneBuffer::new(4, ROW).unwrap();
        for slot in 0..4 {
            buffer.slot(slot).fill(0);
        }
        let array = mlxcel_core::from_bytes_nocopy(
            buffer.bytes(),
            &[4, ROW as i32],
            mlxcel_core::dtype::UINT8,
        );
        assert_eq!(
            mlxcel_core::array_data_address(&array),
            buffer.ptr as usize,
            "MLX must wrap a page-aligned plane, not copy it"
        );
        mlxcel_core::eval(&array);
        buffer.slot(2)[7] = 42;
        let row = mlxcel_core::slice(&array, &[2, 7], &[3, 8]);
        mlxcel_core::eval(&row);
        assert_eq!(mlxcel_core::array_to_raw_bytes(&row), vec![42]);
    }
}
