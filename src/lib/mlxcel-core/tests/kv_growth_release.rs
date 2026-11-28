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

//! A chunked-prefill piece that grows every layer's dense FP16 `KVCache` must
//! not leave each layer's superseded buffer parked in MLX's allocator cache
//! until the piece ends (issue #3092). These cases read process-wide MLX
//! memory, so they live in their own test binary: in the lib suite, tests on
//! other threads allocate while they measure.

use cxx::UniquePtr;
use mlxcel_core::MlxArray;
use mlxcel_core::cache::KVCache;
use std::sync::Mutex;
use std::time::Duration;

/// The cases share the process's MLX memory counters; run them one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

const LAYERS: usize = 12;
const HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
/// Rows each layer holds before the measured piece; a whole number of the
/// cache's 256-row steps, so the buffer is exactly this long.
const HELD: i32 = 4096;
const PIECE: i32 = 2048;
/// One layer's K plus V buffer before the piece.
const OLD_LAYER_BYTES: u64 = 2 * HEADS as u64 * HELD as u64 * HEAD_DIM as u64 * 2;

fn rows(n: i32, value: f32) -> UniquePtr<MlxArray> {
    let ones = mlxcel_core::ones(&[1, HEADS, n, HEAD_DIM], mlxcel_core::dtype::FLOAT16);
    mlxcel_core::multiply(
        &ones,
        &mlxcel_core::full_f32(&[1], value, mlxcel_core::dtype::FLOAT16),
    )
}

/// The CPU time a model spends building one layer's graph, during which the
/// previous layer's queued copy runs on the GPU.
fn build_layer_graph() {
    std::thread::sleep(Duration::from_millis(5));
}

/// Runs one lazy piece over `LAYERS` layers, each appending `PIECE` rows
/// through `append` to a layer already holding `HELD` evaluated rows, and
/// returns the allocator-cache bytes once the piece's graph has run, before
/// anything flushes them (as the engine's per-piece flush would).
fn parked_after_piece<C>(
    mut layers: Vec<C>,
    append: impl Fn(&mut C, UniquePtr<MlxArray>, UniquePtr<MlxArray>) -> UniquePtr<MlxArray>,
) -> u64 {
    // Materialize the piece's K/V first, so the only buffers the piece's
    // graph frees are the caches' own.
    let mut inputs: Vec<_> = (0..layers.len())
        .map(|_| {
            let (k, v) = (rows(PIECE, 2.0), rows(PIECE, 3.0));
            mlxcel_core::eval(&k);
            mlxcel_core::eval(&v);
            (k, v)
        })
        .collect();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();
    let mut total: Option<UniquePtr<MlxArray>> = None;
    for layer in &mut layers {
        build_layer_graph();
        let (k, v) = inputs.pop().expect("one input per layer");
        let out = append(layer, k, v);
        total = Some(match total.take() {
            Some(t) => mlxcel_core::add(&t, &out),
            None => out,
        });
    }
    let total = total.expect("at least one layer");
    mlxcel_core::eval(&total);
    mlxcel_core::synchronize_default();
    let parked = mlxcel_core::memory::cache_memory();
    drop(total);
    drop(layers);
    mlxcel_core::clear_memory_cache();
    parked
}

/// A consumer that reads the last cached row of K and V: enough to force the
/// append, small enough that its own temporaries do not swamp the measure.
fn sum_pair(k: &MlxArray, v: &MlxArray) -> UniquePtr<MlxArray> {
    let last = |a: &MlxArray| {
        let shape = mlxcel_core::array_shape(a);
        let start = [0, 0, shape[2] - 1, 0];
        mlxcel_core::sum_all(&mlxcel_core::slice(a, &start, &shape))
    };
    mlxcel_core::add(&last(k), &last(v))
}

/// The piece through `KVCache`, whose FP16 growth releases eagerly.
fn releasing_piece() -> u64 {
    let layers: Vec<KVCache> = (0..LAYERS)
        .map(|_| {
            let mut cache = KVCache::new();
            let (k, v) = cache.update_and_fetch(rows(HELD, 1.0), rows(HELD, 1.0));
            mlxcel_core::eval(&sum_pair(&k, &v));
            mlxcel_core::eval(cache.keys.as_ref().expect("filled"));
            mlxcel_core::eval(cache.values.as_ref().expect("filled"));
            cache
        })
        .collect();
    parked_after_piece(layers, |cache, k, v| {
        let (k, v) = cache.update_and_fetch(k, v);
        sum_pair(&k, &v)
    })
}

/// The same piece with the lazy growth `KVCache` used before #3092:
/// concatenate the old buffer with a zero block inside the graph, then write
/// the rows.
fn lazy_piece() -> u64 {
    let layers: Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> = (0..LAYERS)
        .map(|_| {
            let (k, v) = (rows(HELD, 1.0), rows(HELD, 1.0));
            mlxcel_core::eval(&k);
            mlxcel_core::eval(&v);
            (k, v)
        })
        .collect();
    parked_after_piece(layers, |(keys, values), k, v| {
        let zeros =
            || mlxcel_core::zeros(&[1, HEADS, PIECE, HEAD_DIM], mlxcel_core::dtype::FLOAT16);
        let grown_k = mlxcel_core::concatenate(keys, &zeros(), 2);
        let grown_v = mlxcel_core::concatenate(values, &zeros(), 2);
        let end = [1, HEADS, HELD + PIECE, HEAD_DIM];
        *keys = mlxcel_core::slice_update(&grown_k, &k, &[0, 0, HELD, 0], &end);
        *values = mlxcel_core::slice_update(&grown_v, &v, &[0, 0, HELD, 0], &end);
        sum_pair(keys, values)
    })
}

/// Both pieces park the K/V inputs they consumed; only the lazy one also parks
/// each layer's old buffer. Measured on M5 Max: 151 MB against 403 MB, with
/// 16.8 MB of old buffer per layer.
#[test]
fn chunked_piece_growth_does_not_park_every_layers_old_buffer() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let lazy = lazy_piece();
    let releasing = releasing_piece();
    assert!(
        releasing + OLD_LAYER_BYTES * (LAYERS as u64) / 2 <= lazy,
        "the releasing piece parked {releasing} bytes against {lazy} for the lazy one; \
         one layer's old buffer is {OLD_LAYER_BYTES}, {LAYERS} layers"
    );
}

/// The control: the lazy piece parks nearly every layer's old buffer, so the
/// comparison above measures the release rather than noise.
#[test]
fn lazy_growth_parks_every_layers_old_buffer() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let parked = lazy_piece();
    assert!(
        parked >= OLD_LAYER_BYTES * (LAYERS as u64 - 2),
        "the lazy piece parked only {parked} bytes; one layer's old buffer is \
         {OLD_LAYER_BYTES}, {LAYERS} layers"
    );
}
