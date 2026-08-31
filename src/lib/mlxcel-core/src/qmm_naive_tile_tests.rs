//! [#1541] Host-side unit tests for the `qmm_naive` CTA tile selector.
//!
//! `qmm_naive` sized its N tile with `enough_smem = sm80 && itemsize <= 2 &&
//! group_size <= 64`, where `sm80` is `compute_capability_major() >= 8`. Two
//! different things sat under that one name. `itemsize <= 2 && group_size <=
//! 64` is a shared-memory rule, and #1541 replaced it with a comparison
//! against the device's real per-block budget. `sm80` is not: a V100 has 96 KB
//! per block through the opt-in against the 24 KB the widest eligible tile
//! needs, so shared memory was never what excluded pre-Ampere parts. It
//! selects the MMA atom, and it appears here as `tensor_core_mma`.
//!
//! The measurement in `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`
//! found upstream's exclusion correct and its stated reason wrong, so the
//! selected tile is the same tile upstream selects on every architecture. That
//! is what these tests pin, and it is a stronger claim than the acceptance
//! criterion #1541 was given: not "sm_80 and later are unchanged" but "every
//! architecture is unchanged", settled by enumeration over every reachable
//! `(itemsize, group_size, m)` combination and every real per-block budget.
//!
//! Epic #1536 deferred the sm_80+ half of that to a GB10 host. It is not a
//! hardware question. The selector is host code and a pure function of five
//! arguments, so it is answered here, on the shipped function itself through
//! the C shim in `cpp/qmm_naive_tile_probe.cpp` rather than on a Rust
//! restatement of it. No GPU, no CUDA toolkit and no `cuda` feature are
//! needed, so these run in every CI job and on every developer machine.

// `qmm_naive_tile.h` through `cpp/qmm_naive_tile_probe.cpp`. Pure integer
// arithmetic; it touches no CUDA API and no device.
//
// `Tile` mirrors `MlxcelQmmNaiveTile` in that file field for field. The flags
// are `i32` rather than `bool` so the ABI does not depend on how either
// language sizes a bool.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tile {
    tile_m: i32,
    tile_n: i32,
    tile_k: i32,
    smem_bytes: i64,
    needs_smem_opt_in: i32,
    fits: i32,
}

impl Tile {
    fn needs_smem_opt_in(self) -> bool {
        self.needs_smem_opt_in != 0
    }

    fn fits(self) -> bool {
        self.fits != 0
    }

    fn shape(self) -> (i32, i32, i32) {
        (self.tile_m, self.tile_n, self.tile_k)
    }
}

unsafe extern "C" {
    fn mlxcel_qmm_naive_choose_tile(
        itemsize: i32,
        m: i32,
        group_size: i32,
        smem_budget_bytes: i64,
        tensor_core_mma: i32,
        forced_tile_n: i32,
    ) -> Tile;
    fn mlxcel_qmm_naive_smem_reserve_bytes() -> i64;
    fn mlxcel_qmm_naive_smem_opt_in_free_bytes() -> i64;
}

fn choose(itemsize: i32, m: i32, group_size: i32, budget: i64, tensor_core_mma: bool) -> Tile {
    choose_forced(itemsize, m, group_size, budget, tensor_core_mma, 0)
}

fn choose_forced(
    itemsize: i32,
    m: i32,
    group_size: i32,
    budget: i64,
    tensor_core_mma: bool,
    forced_tile_n: i32,
) -> Tile {
    unsafe {
        mlxcel_qmm_naive_choose_tile(
            itemsize,
            m,
            group_size,
            budget,
            i32::from(tensor_core_mma),
            forced_tile_n,
        )
    }
}

/// The tile shape upstream MLX selects, transcribed from
/// `mlx/backend/cuda/quantized/qmm/qmm_naive.cu:17-23` at pin `9a795735`:
///
/// ```text
/// bool enough_smem = sm80 && itemsize <= 2 && group_size <= 64;
/// int tile_m = std::max(16, std::min(64, next_power_of_2(m)));
/// int tile_n = enough_smem ? 128 : 64;
/// int tile_k = std::max(64, group_size);
/// ```
///
/// This is the code #1541 replaces, kept here as the reference the replacement
/// is compared against. It is deliberately a transcription and not a call into
/// the new selector, so the comparison has two independent sides.
fn upstream_tile(itemsize: i32, m: i32, group_size: i32, sm80: bool) -> (i32, i32, i32) {
    let enough_smem = sm80 && itemsize <= 2 && group_size <= 64;
    let tile_m = 16.max(64.min(next_power_of_2(m)));
    let tile_n = if enough_smem { 128 } else { 64 };
    let tile_k = 64.max(group_size);
    (tile_m, tile_n, tile_k)
}

/// `mlx::core::next_power_of_2`, which upstream computes as
/// `pow(2, ceil(log2(n)))` in double precision.
fn next_power_of_2(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut p = 1i32;
    while p < n {
        p <<= 1;
    }
    p
}

/// Every `itemsize` `qmm_naive` can be called with. `x` is the activation, so
/// 2 covers f16 and bf16 and 4 covers f32; nothing else reaches this kernel.
const ITEMSIZES: [i32; 2] = [2, 4];

/// Every group size MLX accepts. `mlx/ops.cpp:5073` rejects anything else:
/// "The supported group sizes are 32, 64, and 128."
const GROUP_SIZES: [i32; 3] = [32, 64, 128];

/// Per-block opt-in shared-memory budgets, by architecture.
const BUDGET_SM70: i64 = 98304; // Volta, 96 KB, measured on this host
const BUDGET_SM75: i64 = 65536; // Turing, 64 KB
const BUDGET_SM86: i64 = 101376; // Ampere consumer, Ada, Blackwell consumer
const BUDGET_SM80: i64 = 166912; // A100 / Orin, 163 KB
const BUDGET_SM90: i64 = 232448; // Hopper / Blackwell datacenter, 227 KB

/// Every budget above, plus the 48 KB floor `qmm_naive.cu` falls back to when
/// the device attribute query fails.
const ALL_BUDGETS: [i64; 6] = [
    48 * 1024,
    BUDGET_SM75,
    BUDGET_SM70,
    BUDGET_SM86,
    BUDGET_SM80,
    BUDGET_SM90,
];

/// An `m` sweep that hits every reachable `tile_m` and both sides of each
/// clamp boundary. `tile_m` is `clamp(next_power_of_2(m), 16, 64)`, so the
/// interesting points are around 16, 32 and 64 and then everything above.
fn m_sweep() -> Vec<i32> {
    let mut ms: Vec<i32> = (1..=70).collect();
    ms.extend([
        96, 127, 128, 129, 255, 256, 512, 600, 671, 1024, 1283, 1906, 2048, 4096, 8192, 65536,
    ]);
    ms
}

/// The acceptance criterion, stated as a test and widened past what it asked
/// for: the tile #1541 selects is the tile upstream selected, on **every**
/// architecture, for every `(itemsize, group_size, m)` combination.
///
/// There is no sampling and no tolerance here. `itemsize` and `group_size` have
/// finitely many reachable values, `tile_m` is a clamp with three outcomes, and
/// the budget axis is enumerated over every real per-block figure plus the
/// query-failure floor, so this is exhaustive over the selector's input space
/// rather than a spot check.
///
/// Epic #1536's GB10 handoff lists "the tiler's sm_80+ selection provably
/// identical for every `(itemsize, group_size, m)` combination" as deferred to
/// Ampere-or-later hardware. This closes it without any.
#[test]
fn the_selected_tile_is_upstreams_on_every_architecture() {
    for &budget in &ALL_BUDGETS {
        for &tensor_core_mma in &[false, true] {
            for &itemsize in &ITEMSIZES {
                for &group_size in &GROUP_SIZES {
                    for m in m_sweep() {
                        let got = choose(itemsize, m, group_size, budget, tensor_core_mma);
                        let want = upstream_tile(itemsize, m, group_size, tensor_core_mma);
                        assert_eq!(
                            got.shape(),
                            want,
                            "budget {budget}, tensor_core_mma {tensor_core_mma}, itemsize \
                             {itemsize}, group_size {group_size}, m {m}: the selected tile must \
                             equal upstream's"
                        );
                    }
                }
            }
        }
    }
}

/// `fits` tracks the budget rather than being decorative, and the one
/// combination that fails is one upstream already fails at launch.
///
/// Kept separate from the shape assertion because it is a different claim: the
/// shapes agreeing says nothing about whether the budget term could refuse one.
/// f32 at group size 128 needs 64 KB, which is exactly Turing's whole per-block
/// budget, so the reserve pushes it over and this selector refuses it. Upstream
/// launches it there today with no opt-in and fails inside the driver.
#[test]
fn the_fit_flag_tracks_the_budget() {
    let reserve = unsafe { mlxcel_qmm_naive_smem_reserve_bytes() };
    for &budget in &ALL_BUDGETS {
        for &tensor_core_mma in &[false, true] {
            for &itemsize in &ITEMSIZES {
                for &group_size in &GROUP_SIZES {
                    for m in m_sweep() {
                        let got = choose(itemsize, m, group_size, budget, tensor_core_mma);
                        assert_eq!(
                            got.fits(),
                            got.smem_bytes + reserve <= budget,
                            "budget {budget}, itemsize {itemsize}, group_size {group_size}, m {m}"
                        );
                    }
                }
            }
        }
    }

    // Named instances of both outcomes, so an edit that makes `fits` constant
    // fails here rather than passing the sweep above vacuously.
    assert!(choose(4, 2048, 128, BUDGET_SM70, false).fits());
    assert!(!choose(4, 2048, 128, BUDGET_SM75, false).fits());
}

/// The widest tile a V100 can be handed is nowhere near its budget.
///
/// This pins the headroom the issue was built on with numbers rather than a
/// claim, and it is why the architecture term in upstream's predicate could not
/// have been a shared-memory term: 24 KB out of 96 KB, and under the 48 KB a
/// launch gets without opting in at all.
#[test]
fn the_widest_volta_tile_is_24_kb_and_needs_no_opt_in() {
    // The production shape for every checkpoint in epic #1536: bf16
    // activations, group size 64, a prefill-sized m. Forced wide, because the
    // measured selection keeps the narrow tile on the pre-Ampere FMA path.
    let tile = choose_forced(2, 2048, 64, BUDGET_SM70, false, 128);
    assert_eq!(tile.shape(), (64, 128, 64));
    assert_eq!(tile.smem_bytes, 24576);
    assert!(!tile.needs_smem_opt_in());
    assert!(tile.fits());

    // Against 96 KB of budget, and against the 48 KB available without any
    // opt-in.
    assert!(tile.smem_bytes * 4 <= BUDGET_SM70);
    assert!(tile.smem_bytes < unsafe { mlxcel_qmm_naive_smem_opt_in_free_bytes() });
}

/// f32 activations at group size 128 select a 64 KB tile, which is the one
/// case upstream `qmm_naive` reaches that needs the dynamic shared-memory
/// opt-in. Without it the launch fails inside the driver on every architecture
/// from Volta on, which is why #1541 adds the `cuFuncSetAttribute` call.
#[test]
fn the_f32_group128_tile_crosses_the_opt_in_ceiling() {
    let ceiling = unsafe { mlxcel_qmm_naive_smem_opt_in_free_bytes() };
    for &budget in &[BUDGET_SM70, BUDGET_SM86, BUDGET_SM80, BUDGET_SM90] {
        for &tensor_core_mma in &[false, true] {
            let tile = choose(4, 2048, 128, budget, tensor_core_mma);
            assert_eq!(tile.shape(), (64, 64, 128));
            assert_eq!(tile.smem_bytes, 65536);
            assert!(
                tile.smem_bytes > ceiling,
                "budget {budget}: this tile is the reason the opt-in call exists"
            );
            assert!(tile.needs_smem_opt_in(), "budget {budget}");
            assert!(tile.fits(), "budget {budget}");
        }
    }
}

/// A tile that does not fit is rejected at selection time.
///
/// Turing is the real instance rather than a contrived one: its opt-in budget
/// is 64 KB, so the 64 KB f32 group-128 tile above does not fit it. `fits` goes
/// false and `qmm_naive.cu` throws a message naming the tile, the requirement
/// and the budget, instead of handing the driver a launch it cannot honour.
#[test]
fn an_oversized_tile_is_rejected_rather_than_launched() {
    let tile = choose(4, 2048, 128, BUDGET_SM75, false);
    assert_eq!(tile.smem_bytes, 65536);
    assert!(
        !tile.fits(),
        "65536 bytes plus the reserve must not fit Turing's 65536-byte budget"
    );

    // And a budget below anything real rejects even the narrowest tile, which
    // is the failure path for a device attribute query that returns nonsense.
    assert!(!choose(2, 1, 32, 4096, false).fits());
}

/// The reserve is applied at the boundary rather than approximately. A budget
/// one byte short of `smem + reserve` must reject.
#[test]
fn the_reserve_is_enforced_exactly() {
    let reserve = unsafe { mlxcel_qmm_naive_smem_reserve_bytes() };
    assert!(reserve > 0);

    // bf16, group 64, m >= 64: the 128-wide tile needs 24576 bytes, and on the
    // tensor-core path it is the one the selector reaches for first.
    let need = 24576i64;
    let exact = choose(2, 2048, 64, need + reserve, true);
    assert_eq!(
        exact.tile_n, 128,
        "exactly enough budget takes the wide tile"
    );
    assert!(exact.fits());

    let short = choose(2, 2048, 64, need + reserve - 1, true);
    assert_eq!(
        short.tile_n, 64,
        "one byte short of the wide tile falls back to the narrow one"
    );
    assert!(short.fits());
}

/// `MLXCEL_QMM_NAIVE_TILE_N` pins the N tile so one build can measure both
/// widths, which is how #1541's sweep and its bitwise parity test work. It
/// overrides the selection but not the fit check, so a pinned tile that does
/// not fit is still rejected rather than launched.
#[test]
fn the_forced_tile_n_override_pins_the_width_but_not_the_fit_check() {
    for &forced in &[64i32, 128] {
        // Forcing works on the FMA path too, where the selector would never
        // choose the wide tile. That is what makes it useful for a sweep.
        let tile = choose_forced(2, 2048, 64, BUDGET_SM70, false, forced);
        assert_eq!(tile.tile_n, forced);
        assert!(tile.fits());
    }

    // 128 wide at f32 and group size 128 needs 96 KB exactly, which does not
    // leave room for the reserve inside Volta's 96 KB budget.
    let tile = choose_forced(4, 2048, 128, BUDGET_SM70, false, 128);
    assert_eq!(tile.smem_bytes, 98304);
    assert_eq!(tile.smem_bytes, BUDGET_SM70);
    assert!(!tile.fits(), "a forced tile still has to fit");

    // Anything other than 64 or 128 is not an override.
    for &bogus in &[0i32, 1, 32, 96, 256, -1] {
        let tile = choose_forced(2, 2048, 64, BUDGET_SM70, false, bogus);
        assert_eq!(tile.tile_n, 64, "bogus override {bogus} must be ignored");
    }
}

/// The shared-memory model is the product form the CuTe layouts reduce to.
/// `qmm_naive.cu` recomputes it from the real layouts and refuses to launch if
/// the two disagree, so this pins the arithmetic that check compares against.
#[test]
fn the_smem_model_is_itemsize_times_tile_k_times_tile_m_plus_tile_n() {
    for &itemsize in &ITEMSIZES {
        for &group_size in &GROUP_SIZES {
            for m in m_sweep() {
                for &forced in &[64i32, 128] {
                    let t = choose_forced(itemsize, m, group_size, BUDGET_SM90, true, forced);
                    let expected = i64::from(itemsize)
                        * i64::from(t.tile_k)
                        * (i64::from(t.tile_m) + i64::from(t.tile_n));
                    assert_eq!(t.smem_bytes, expected);
                }
            }
        }
    }
}
