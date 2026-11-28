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

//! Which fused decode-MoE launches cut the CUDA graph they are encoded into
//! (issue #3387).
//!
//! MLX's CUDA encoder counts every input of a kernel at its full
//! `data_size()` and commits the graph once the count passes the device's
//! byte budget, 25 << 20 elements on DGX Spark (sm_121). The fused decode-MoE
//! kernels take a layer's whole expert stacks and read the experts their index
//! names, 8 of 128 on Qwen3-30B-A3B, so counted whole each of the two launches
//! cut the graph on its own. The `custom_kernel.cpp` CUDA overlay charges a
//! stack for the experts the index names instead. These tests hold that from
//! the encoder's side, through its graph-launch counter:
//!
//! 1. the arithmetic behind the charge, at its boundaries (no row, one row,
//!    every row, more than every row, no rows to choose from, and `u64`
//!    extremes), without a GPU launch;
//! 2. stacks over the budget with a small index: the gate/up launch and the
//!    down launch that follows it land in one graph, and the output is the
//!    bytes the same experts give from a stack under the budget;
//! 3. the other side of the rule: an index that names every expert reads the
//!    stacks whole, so the launches still cut the graph, and so does an index
//!    that names enough experts to be over the budget on its own.
//!
//! The second and third need a device whose byte budget the test stacks
//! exceed. That is DGX Spark at MLX's default; on a device with a larger
//! budget they skip, because nothing there cuts a graph for either reason.
//! Set `MLX_MAX_MB_PER_BUFFER=25` to run them on such a device.
//!
//! The launch counter is process-wide. Every count below is the minimum over
//! several repetitions, so graphs another test thread launches meanwhile can
//! only be noticed as a failure to reach the minimum, never as a false pass.
//! The CUDA gate runs single threaded.
//!
//! Run on CUDA (GB10 etc.):
//!   cargo test -p mlxcel-core --profile test-fast --features cuda \
//!     cuda_graph_commit -- --test-threads=1

use super::*;
use crate::cuda_graph_budget::CudaGraphCommitStats;
use crate::fused_moe_parity_tests::random_quantized_expert_stack;

/// Activation width, expert width, and expert count of the stacks under test.
/// Each packed weight stack is `EXPERTS * DFF * DIN / 8` = 33.5M elements at 4
/// bits, over the 26.2M-element budget of DGX Spark on its own.
const DIN: i32 = 2048;
const DFF: i32 = 1024;
const EXPERTS: i32 = 128;
/// Distinct experts the stacks are tiled from. Expert `e` of a tiled stack is
/// expert `e % BASE_EXPERTS` of the base stack, which is what lets the output
/// of an over-budget launch be compared with an under-budget one.
const BASE_EXPERTS: i32 = 16;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// Weight, scale and bias stacks of one projection.
struct Stack {
    w: UniquePtr<MlxArray>,
    s: UniquePtr<MlxArray>,
    b: UniquePtr<MlxArray>,
}

impl Stack {
    fn random(rows: i32, cols: i32) -> Self {
        let (w, s, b) = random_quantized_expert_stack(BASE_EXPERTS, rows, cols, GROUP_SIZE, BITS);
        Self { w, s, b }
    }

    /// The stack repeated along the expert axis until it holds `EXPERTS`.
    fn tiled(&self) -> Self {
        let reps = [EXPERTS / BASE_EXPERTS, 1, 1];
        let out = Self {
            w: tile(&self.w, &reps),
            s: tile(&self.s, &reps),
            b: tile(&self.b, &reps),
        };
        eval(&out.w);
        eval(&out.s);
        eval(&out.b);
        out
    }

    fn elements(&self) -> u64 {
        (array_size(&self.w) + array_size(&self.s) + array_size(&self.b)) as u64
    }
}

/// The three projections of one MoE layer.
struct Layer {
    gate: Stack,
    up: Stack,
    down: Stack,
}

impl Layer {
    fn random() -> Self {
        Self {
            gate: Stack::random(DFF, DIN),
            up: Stack::random(DFF, DIN),
            down: Stack::random(DIN, DFF),
        }
    }

    fn tiled(&self) -> Self {
        Self {
            gate: self.gate.tiled(),
            up: self.up.tiled(),
            down: self.down.tiled(),
        }
    }
}

type Launcher = fn(
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    &MlxArray,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
) -> Result<UniquePtr<MlxArray>, cxx::Exception>;

/// One lazy fused MoE call over `layer` for the experts in `indices`.
fn fused(launch: Launcher, layer: &Layer, x: &MlxArray, indices: &[u32]) -> UniquePtr<MlxArray> {
    let k = indices.len() as i32;
    let idx = from_slice_u32(indices, &[k]);
    let scores: Vec<f32> = (0..k).map(|i| 1.0 / (2.0 + i as f32)).collect();
    let scores = astype(&from_slice_f32(&scores, &[k]), dtype::BFLOAT16);
    launch(
        x,
        &idx,
        &layer.gate.w,
        &layer.gate.s,
        &layer.gate.b,
        &layer.up.w,
        &layer.up.s,
        &layer.up.b,
        &layer.down.w,
        &layer.down.s,
        &layer.down.b,
        &scores,
        DIN,
        DFF,
        k,
        BITS,
        BITS,
        GROUP_SIZE,
    )
    .expect("the CUDA backend has both fused MoE kernel ports")
}

/// Graphs one eval of `build()` launches: the minimum over a few repetitions,
/// with the stream drained before each so nothing encoded earlier is counted.
fn graphs_launched_by(build: impl Fn() -> UniquePtr<MlxArray>) -> (u64, Vec<u8>) {
    let count = || {
        CudaGraphCommitStats::snapshot()
            .expect("the caller checked for a CUDA device")
            .graphs_launched
    };
    let mut fewest = u64::MAX;
    let mut bytes = Vec::new();
    for _ in 0..5 {
        synchronize_default();
        let before = count();
        let out = build();
        eval(&out);
        synchronize_default();
        fewest = fewest.min(count() - before);
        bytes = array_to_raw_bytes(&out);
    }
    (fewest, bytes)
}

/// The encoder's counters when the device's byte budget is one the test
/// stacks exceed, `None` (skip) otherwise.
fn stats_or_skip(layer: &Layer) -> Option<CudaGraphCommitStats> {
    let stats = CudaGraphCommitStats::snapshot()?;
    let cut = stats.elements_that_cut_a_graph();
    if layer.down.elements() < cut || layer.gate.elements() + layer.up.elements() < cut {
        eprintln!(
            "skipping: this device's graph byte budget ({} << 20 elements) is larger than the \
             test stacks; set MLX_MAX_MB_PER_BUFFER=25 to run",
            stats.max_mb
        );
        return None;
    }
    Some(stats)
}

fn cuda_or_skip() -> bool {
    if CudaGraphCommitStats::snapshot().is_none() {
        eprintln!("skipping: no CUDA device");
        return false;
    }
    true
}

fn random_x(seed: u64) -> UniquePtr<MlxArray> {
    random_seed(seed);
    let x_f32 = unsafe { random_normal(&[DIN], dtype::FLOAT32, std::ptr::null()) };
    let x = astype(&x_f32, dtype::BFLOAT16);
    eval(&x);
    x
}

/// The charge for a partly read input, at the edges of every argument. This is
/// the encoder's own function (`counted_input_elements` in the `device.cpp`
/// overlay), not a copy of it.
#[test]
fn a_partly_read_input_is_charged_for_the_rows_read() {
    if !cuda_or_skip() {
        return;
    }
    let charged = cuda_graph_counted_input_elements;
    // A 128-expert stack of 33,554,432 elements, the test stacks' own size.
    let stack = 128 * 1024 * 256u64;
    assert_eq!(charged(stack, 0, 128), 0, "no row read, nothing charged");
    assert_eq!(charged(stack, 1, 128), stack / 128, "one row");
    assert_eq!(charged(stack, 8, 128), stack / 16, "8 of 128 experts");
    assert_eq!(charged(stack, 127, 128), stack / 128 * 127, "all but one");
    assert_eq!(
        charged(stack, 128, 128),
        stack,
        "every row is the whole input"
    );
    assert_eq!(
        charged(stack, 129, 128),
        stack,
        "an index may repeat a row; the charge never exceeds the input"
    );
    assert_eq!(charged(stack, u64::MAX, 128), stack);
    // An input read whole, and one left out, as the overlay spells them.
    assert_eq!(charged(stack, 1, 1), stack);
    assert_eq!(charged(stack, 0, 1), 0);
    // No leading rows to choose from: charged whole, whatever the index says.
    for rows_read in [0, 1, u64::MAX] {
        assert_eq!(charged(stack, rows_read, 0), stack);
    }
    // An empty input costs nothing, and an input smaller than its row count
    // rounds down to nothing instead of dividing by a zero row size.
    assert_eq!(charged(0, 3, 7), 0);
    assert_eq!(charged(0, 0, 0), 0);
    assert_eq!(charged(5, 3, 7), 0);
    // Extremes: the quotient is taken before the product, so nothing wraps.
    assert_eq!(charged(u64::MAX, 1, 2), u64::MAX / 2);
    assert_eq!(charged(u64::MAX, u64::MAX - 1, u64::MAX), u64::MAX - 1);
    assert_eq!(charged(u64::MAX, u64::MAX, u64::MAX), u64::MAX);
    assert_eq!(charged(u64::MAX, 1, u64::MAX), 1);
    assert_eq!(charged(u64::MAX, 0, u64::MAX), 0);
    // Never more than the input, never decreasing in the rows read, and short
    // of the exact share by less than the rows read.
    for data_size in [1u64, 127, 128, 129, 1000, stack, stack + 77] {
        for rows in [1u64, 2, 7, 128, 4096] {
            let mut previous = 0;
            for rows_read in 0..=rows + 1 {
                let got = charged(data_size, rows_read, rows);
                assert!(got <= data_size, "{data_size} {rows_read}/{rows}");
                assert!(got >= previous, "{data_size} {rows_read}/{rows}");
                let exact = (u128::from(data_size) * u128::from(rows_read.min(rows))
                    / u128::from(rows)) as u64;
                assert!(
                    got <= exact && exact - got <= rows_read,
                    "{data_size} {rows_read}/{rows}"
                );
                previous = got;
            }
        }
    }
}

/// Stacks over the byte budget, an index naming a few experts: the gate/up
/// launch and the down launch that follows it stay in one graph. Counted
/// whole, each of them cut it (three graphs for this eval on DGX Spark before
/// the change). The output is byte-for-byte what the same experts give from
/// the 16-expert stack, which is under the budget.
#[test]
fn moe_expert_stacks_over_the_byte_budget_do_not_cut_the_graph() {
    if !cuda_or_skip() {
        return;
    }
    let base = Layer::random();
    let layer = base.tiled();
    if stats_or_skip(&layer).is_none() {
        return;
    }
    let x = random_x(0x3387);
    for (name, launch) in [
        ("swiglu", fused_moe_expert_kernel as Launcher),
        ("geglu", fused_moe_geglu_kernel as Launcher),
    ] {
        for indices in [vec![77u32], vec![3, 40, 41, 64, 90, 101, 126, 127]] {
            let (graphs, bytes) = graphs_launched_by(|| fused(launch, &layer, &x, &indices));
            assert_eq!(
                graphs,
                1,
                "{name}, {} of {EXPERTS} experts: the two launches must share one graph",
                indices.len()
            );
            let folded: Vec<u32> = indices.iter().map(|e| e % BASE_EXPERTS as u32).collect();
            let want = fused(launch, &base, &x, &folded);
            eval(&want);
            assert_eq!(
                bytes,
                array_to_raw_bytes(&want),
                "{name}, {} experts: the charge must not change the output",
                indices.len()
            );
        }
    }
}

/// The rule's other side. An index that names every expert reads the stacks
/// whole, so they are charged whole and each launch still cuts the graph, and
/// an index naming half of them is charged for more than the budget too.
#[test]
fn an_index_over_the_byte_budget_still_cuts_the_graph() {
    if !cuda_or_skip() {
        return;
    }
    let layer = Layer::random().tiled();
    let Some(stats) = stats_or_skip(&layer) else {
        return;
    };
    let x = random_x(0x3388);
    let every: Vec<u32> = (0..EXPERTS as u32).collect();
    let (graphs, _) = graphs_launched_by(|| fused(fused_moe_expert_kernel, &layer, &x, &every));
    assert!(
        graphs >= 3,
        "every expert named: both launches read their stacks whole and must cut the graph, got \
         {graphs} graph(s)"
    );
    // Half the experts: the gate and up stacks alone are charged for more than
    // the budget, so the gate/up launch cuts the graph.
    let half: Vec<u32> = (0..EXPERTS as u32).step_by(2).collect();
    let charged = cuda_graph_counted_input_elements(
        layer.gate.elements() + layer.up.elements(),
        half.len() as u64,
        EXPERTS as u64,
    );
    assert!(charged >= stats.elements_that_cut_a_graph());
    let (graphs, _) = graphs_launched_by(|| fused(fused_moe_expert_kernel, &layer, &x, &half));
    assert!(
        graphs >= 2,
        "half the experts named: the gate/up launch is over the budget, got {graphs} graph(s)"
    );
}
