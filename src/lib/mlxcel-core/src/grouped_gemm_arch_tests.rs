//! [#1544] The CUTLASS architecture tag the grouped GEMM selects, and the
//! numbers that path produces on the device it selects it for.
//!
//! `grouped_gemm_unaligned.cu` mapped every part below compute capability 8.0
//! to `cutlass::arch::Sm75`. That tag names Turing. A Tesla V100 is compute
//! capability 7.0, one generation below Turing, and has no `m16n8k8` MMA; the
//! dispatch described hardware the device does not have on every Volta part
//! that reached it.
//!
//! Two separate claims have to hold after the retag, and they need different
//! kinds of evidence:
//!
//! 1. **The decision is right, and only the pre-Ampere arm moved.** That is a
//!    pure function of the compute capability major version, so it is settled
//!    here by enumeration over every architecture, through the C shim in
//!    `cpp/grouped_gemm_arch_probe.cpp` rather than a Rust restatement of the
//!    mapping. No GPU, no CUDA toolkit and no `cuda` feature are needed, which
//!    is what lets the "sm_80 and later are untouched" half of #1544 be
//!    answered on a host with no Ampere-or-later part in it. Epic #1536 has no
//!    such host, so the alternative was deferring it to GB10.
//!
//! 2. **The path produces correct numbers on the device.** The retag is device
//!    codegen neutral (`docs/benchmark_results/grouped-gemm-arch-v100-2026-08-31.md`
//!    records 51 device symbols with byte-identical bodies either way), so this
//!    is not really a test of the retag. It is the check the issue asked for and
//!    nobody had made: the grouped GEMM is what every non-quantized MoE
//!    checkpoint routes its experts through, on a tag that named the wrong
//!    architecture, and the one test that touched it asserted the output shape
//!    and never looked at a value. A CUTLASS configuration compiled for an
//!    architecture it does not match can return plausible numbers that are
//!    wrong, and greedy text generation will not reveal it.
//!
//! This module carries claim 1, which needs no GPU. Claim 2 lives next door in
//! `grouped_gemm_numeric_tests.rs`, which does.

// `grouped_gemm_arch.h` through `cpp/grouped_gemm_arch_probe.cpp`. Pure
// integer arithmetic; it touches no CUDA API and no device.
unsafe extern "C" {
    fn mlxcel_grouped_gemm_arch_for(compute_capability_major: i32) -> i32;
}

fn arch_for(compute_capability_major: i32) -> i32 {
    unsafe { mlxcel_grouped_gemm_arch_for(compute_capability_major) }
}

// The tag values, which `mlxcel::GroupedGemmArch` defines to be the compute
// capability each one names.
const SM70: i32 = 70;
const SM80: i32 = 80;
const SM90: i32 = 90;

// What the dispatch did before #1544, restated here so the tests can say
// exactly which inputs changed and which did not. This is the only place a
// restatement of the old mapping appears, and it exists to be compared
// against, never to stand in for the shipped function.
fn arch_before_1544(compute_capability_major: i32) -> i32 {
    if compute_capability_major < 8 {
        75
    } else if compute_capability_major == 8 {
        SM80
    } else {
        SM90
    }
}

/// Volta must not be tagged as Turing. This is the defect #1544 exists for.
///
/// A compute capability 7.0 part has the `8x8x4` HMMA shape and nothing else;
/// `cutlass::arch::Sm75` names the `m16n8k8` MMA Turing introduced, which does
/// not exist on the part.
#[test]
fn volta_selects_sm70_not_sm75() {
    assert_eq!(
        arch_for(7),
        SM70,
        "compute capability 7.x must select cutlass::arch::Sm70; \
         Sm75 names an MMA shape a Volta part does not have"
    );
    assert_ne!(arch_for(7), 75, "the pre-Ampere arm still names Turing");
}

/// The sm_80 and later arms are untouched, settled by enumeration rather than
/// deferred to hardware.
///
/// This is #1544's "zero change on sm_80+" acceptance criterion. The decision
/// is a pure function of one integer, so every input can be tried, and the
/// range here runs well past any capability that exists: Ampere at 8, Hopper
/// at 9, Blackwell at 10 through 12 (GB10 is compute capability 12.1), and on
/// into values no part has.
#[test]
fn only_the_pre_ampere_arm_changed() {
    for major in 0..=32 {
        let before = arch_before_1544(major);
        let after = arch_for(major);
        if major < 8 {
            assert_eq!(
                before, 75,
                "the pre-#1544 mapping is misstated for compute capability {major}"
            );
            assert_eq!(
                after, SM70,
                "compute capability {major} is pre-Ampere and must select Sm70"
            );
        } else {
            assert_eq!(
                before, after,
                "compute capability {major} changed tag across #1544; the sm_80 \
                 and later arms were supposed to be untouched"
            );
        }
    }
}

/// The whole mapping, written out.
///
/// A table rather than a restatement of the branch structure, so a future edit
/// that reorganises the branches still has to agree with an explicit list of
/// what each architecture gets.
#[test]
fn arch_mapping_table() {
    let cases: &[(i32, i32, &str)] = &[
        (5, SM70, "Maxwell, below MLX's CUDA support floor"),
        (6, SM70, "Pascal, below MLX's CUDA support floor"),
        (7, SM70, "Volta and Turing"),
        (8, SM80, "Ampere and Ada"),
        (9, SM90, "Hopper"),
        (10, SM90, "Blackwell datacenter"),
        (12, SM90, "Blackwell consumer, GB10 is 12.1"),
    ];
    for &(major, want, what) in cases {
        assert_eq!(
            arch_for(major),
            want,
            "compute capability {major} ({what}) selected the wrong tag"
        );
    }
}

/// Turing shares the pre-Ampere arm with Volta, deliberately.
///
/// `grouped_gemm_arch.h` records why: the configuration the arm selects is
/// `OpClassSimt` with `InstructionShape<1, 1, 1>`, so CUTLASS never reads an
/// MMA atom off the tag and erases it. Compiling the translation unit with a
/// separate `Sm70` arm alongside the `Sm75` one emits the identical set of 51
/// device symbols and a byte identical SASS dump, at a cost of 26 duplicated
/// host instantiations and 194,704 bytes of object. The `static_assert` pair in
/// `grouped_gemm_unaligned.cu` fails the build if that configuration ever
/// stops being SIMT, which is the condition under which this test's
/// expectation would have to change.
#[test]
fn turing_shares_the_pre_ampere_arm() {
    assert_eq!(arch_for(7), SM70);
}
