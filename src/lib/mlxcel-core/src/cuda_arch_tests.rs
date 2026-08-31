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

//! Unit tests for the CUDA architecture-list parser and the coverage /
//! mismatch predicate (issue #1537).
//!
//! None of these touch a GPU: they drive the pure functions with the exact
//! architecture lists the release workflow builds and the exact compute
//! capabilities the project targets, so the mismatch rule is pinned on hosts
//! that have neither card.

use crate::cuda_arch::{
    ArchCoverage, ArchVariant, CudaArchEntry, CudaArchMismatch, arch_list_coverage,
    compiled_cuda_architectures, cuda_arch_mismatch, cuda_arch_startup_summary,
    cuda_compute_capability, entry_coverage, parse_cuda_arch_entry, parse_cuda_arch_list,
};

/// The two lists `.github/workflows/release.yml` builds, and the CI pin.
const RELEASE_AARCH64: &str = "90a;100;121";
const RELEASE_X86_64: &str = "80;86;89;90a;100;120";

/// Compute capabilities of the machines this project actually runs on.
const V100: (u32, u32) = (7, 0);
const A100: (u32, u32) = (8, 0);
const H100: (u32, u32) = (9, 0);
const GB10: (u32, u32) = (12, 1);

// ── Parsing ───────────────────────────────────────────────────────────────────

#[test]
fn parses_plain_entry_into_major_and_minor() {
    let entry = parse_cuda_arch_entry("86").expect("86 is a valid entry");
    assert_eq!(entry.major, 8);
    assert_eq!(entry.minor, 6);
    assert_eq!(entry.variant, ArchVariant::Generic);
    assert!(entry.emits_cubin);
    assert!(entry.emits_ptx);
}

#[test]
fn parses_three_digit_entry_with_single_digit_minor() {
    // The minor version is the last digit, so 121 is 12.1 and not 1.21.
    let entry = parse_cuda_arch_entry("121").expect("121 is a valid entry");
    assert_eq!((entry.major, entry.minor), GB10);
    let entry = parse_cuda_arch_entry("100").expect("100 is a valid entry");
    assert_eq!((entry.major, entry.minor), (10, 0));
}

#[test]
fn parses_arch_and_family_suffixes() {
    let hopper = parse_cuda_arch_entry("90a").expect("90a is a valid entry");
    assert_eq!((hopper.major, hopper.minor), H100);
    assert_eq!(hopper.variant, ArchVariant::ArchSpecific);

    let family = parse_cuda_arch_entry("100f").expect("100f is a valid entry");
    assert_eq!((family.major, family.minor), (10, 0));
    assert_eq!(family.variant, ArchVariant::FamilySpecific);
}

#[test]
fn parses_real_and_virtual_qualifiers() {
    let real = parse_cuda_arch_entry("80-real").expect("80-real is a valid entry");
    assert!(real.emits_cubin);
    assert!(!real.emits_ptx);

    let virt = parse_cuda_arch_entry("80-virtual").expect("80-virtual is a valid entry");
    assert!(!virt.emits_cubin);
    assert!(virt.emits_ptx);

    let both = parse_cuda_arch_entry("90a-real").expect("90a-real is a valid entry");
    assert_eq!(both.variant, ArchVariant::ArchSpecific);
    assert!(both.emits_cubin);
    assert!(!both.emits_ptx);
}

#[test]
fn drops_entries_the_parser_does_not_recognise() {
    for junk in [
        "",
        "   ",
        "native",
        "all-major",
        "8",
        "sm_80",
        "8x",
        "-real",
    ] {
        assert!(
            parse_cuda_arch_entry(junk).is_none(),
            "{junk:?} must not parse as an architecture entry"
        );
    }
}

#[test]
fn parses_the_release_matrices_completely() {
    let aarch64 = parse_cuda_arch_list(RELEASE_AARCH64);
    assert_eq!(aarch64.len(), 3);
    assert_eq!(aarch64[0].variant, ArchVariant::ArchSpecific);

    let x86_64 = parse_cuda_arch_list(RELEASE_X86_64);
    assert_eq!(x86_64.len(), 6);
    // Round-tripping through Display keeps the list readable in the error
    // message and proves nothing was silently reinterpreted.
    let rendered: Vec<String> = x86_64.iter().map(ToString::to_string).collect();
    assert_eq!(rendered.join(";"), RELEASE_X86_64);
}

#[test]
fn accepts_a_comma_separated_list_too() {
    assert_eq!(parse_cuda_arch_list("80,86,89").len(), 3);
    assert_eq!(parse_cuda_arch_list(" 80 ; 86 ").len(), 2);
}

// ── Coverage: exact match ─────────────────────────────────────────────────────

#[test]
fn exact_match_is_covered_by_a_cubin() {
    // The V100 build this repository does by auto-detection.
    assert_eq!(arch_list_coverage("70", V100), Some(ArchCoverage::Cubin));
    // GB10 against the aarch64 release matrix, which names 121 explicitly.
    assert_eq!(
        arch_list_coverage(RELEASE_AARCH64, GB10),
        Some(ArchCoverage::Cubin)
    );
}

#[test]
fn cubin_carries_forward_across_minor_revisions_but_not_backward() {
    // sm_120 code is binary compatible with a 12.1 device, per CUDA's
    // minor-revision rule.
    assert_eq!(arch_list_coverage("120", GB10), Some(ArchCoverage::Cubin));
    // The reverse does not hold: a 12.0 device cannot run sm_121 code, and
    // compute_121 PTX is newer than the device, so nothing covers it.
    assert_eq!(arch_list_coverage("121", (12, 0)), None);
}

#[test]
fn cubin_does_not_carry_backward_to_an_older_minor() {
    // 8.6 code on an 8.0 device: same major but a newer minor, so no cubin,
    // and the PTX is likewise too new to JIT backward.
    assert_eq!(arch_list_coverage("86", A100), None);
}

#[test]
fn cubin_does_not_carry_across_major_versions() {
    // A 10.0 cubin does not load on a 9.0 device, and compute_100 PTX is newer
    // than the device so there is nothing to JIT either.
    assert_eq!(arch_list_coverage("100", H100), None);
    // The other direction is PTX-only: 8.0 has no cubin for a 10.0 device but
    // its PTX JITs forward.
    assert_eq!(arch_list_coverage("80", (10, 0)), Some(ArchCoverage::Ptx));
}

// ── Coverage: PTX forward compatibility ───────────────────────────────────────

#[test]
fn older_generic_entry_covers_a_newer_device_by_ptx_jit() {
    // compute_80 PTX JITs forward onto Hopper and Blackwell.
    assert_eq!(arch_list_coverage("80", H100), Some(ArchCoverage::Ptx));
    assert_eq!(arch_list_coverage("80", GB10), Some(ArchCoverage::Ptx));
    // But an x86_64 release binary is not JIT-compatible downward onto a V100.
    assert_eq!(arch_list_coverage("80", V100), None);
}

#[test]
fn a_real_only_entry_offers_no_ptx_to_jit() {
    // -real strips the PTX, so forward compatibility disappears with it.
    assert_eq!(arch_list_coverage("80-real", H100), None);
    assert_eq!(
        arch_list_coverage("80-virtual", H100),
        Some(ArchCoverage::Ptx)
    );
    // On the exact device a -virtual entry still only offers JIT.
    assert_eq!(
        arch_list_coverage("80-virtual", A100),
        Some(ArchCoverage::Ptx)
    );
}

#[test]
fn cubin_wins_over_ptx_when_the_list_offers_both() {
    // 80 would JIT onto a 8.6 device, but 86 has the cubin; report the better.
    assert_eq!(
        arch_list_coverage("80;86", (8, 6)),
        Some(ArchCoverage::Cubin)
    );
}

// ── Coverage: `a` and `f` suffixed entries ────────────────────────────────────

#[test]
fn arch_specific_entry_covers_only_its_exact_target() {
    let hopper = parse_cuda_arch_entry("90a").expect("90a is a valid entry");
    assert_eq!(entry_coverage(hopper, H100), Some(ArchCoverage::Cubin));
    // Neither backward nor forward: 90a code objects use Hopper-only
    // instructions, so a 90a-only binary is stranded everywhere else. This is
    // why the nvidia-smi-less build.rs fallback to `90a` is a trap worth
    // naming at startup.
    assert_eq!(entry_coverage(hopper, V100), None);
    assert_eq!(entry_coverage(hopper, GB10), None);
    assert_eq!(entry_coverage(hopper, (9, 1)), None);
    assert_eq!(arch_list_coverage("90a", GB10), None);
}

#[test]
fn family_specific_entry_covers_forward_within_its_major_only() {
    let family = parse_cuda_arch_entry("100f").expect("100f is a valid entry");
    assert_eq!(entry_coverage(family, (10, 0)), Some(ArchCoverage::Cubin));
    assert_eq!(entry_coverage(family, (10, 3)), Some(ArchCoverage::Cubin));
    assert_eq!(entry_coverage(family, GB10), None);
    assert_eq!(entry_coverage(family, H100), None);
}

// ── Coverage: the no-match case the mismatch check exists for ─────────────────

#[test]
fn x86_64_release_matrix_does_not_cover_a_v100() {
    // The exact case in issue #1537: every published x86_64 archive starts at
    // 80, so none of its cubins loads on a 7.0 card and none of its PTX JITs
    // backward. Without the check this is an opaque CUDA load failure at the
    // first kernel launch.
    assert_eq!(arch_list_coverage(RELEASE_X86_64, V100), None);
    assert_eq!(arch_list_coverage(RELEASE_AARCH64, V100), None);
}

#[test]
fn x86_64_release_matrix_covers_the_devices_it_claims() {
    for device in [A100, (8, 6), (8, 9), H100, (10, 0), (12, 0)] {
        assert_eq!(
            arch_list_coverage(RELEASE_X86_64, device),
            Some(ArchCoverage::Cubin),
            "the x86_64 release matrix should ship a cubin for {device:?}"
        );
    }
}

#[test]
fn an_empty_or_unparseable_list_covers_nothing() {
    // The caller, not this predicate, decides that "nothing parsed" means
    // "unknown"; see the mismatch check. Here the answer is simply None.
    assert_eq!(arch_list_coverage("", V100), None);
    assert_eq!(arch_list_coverage("native", V100), None);
}

// ── Entry construction is stable ──────────────────────────────────────────────

#[test]
fn entry_display_round_trips_through_the_parser() {
    for spelling in ["70", "90a", "100f", "80-real", "89-virtual", "121"] {
        let entry = parse_cuda_arch_entry(spelling).expect("valid entry");
        assert_eq!(entry.to_string(), spelling);
        assert_eq!(parse_cuda_arch_entry(&entry.to_string()), Some(entry));
    }
}

#[test]
fn constructed_entries_follow_the_same_rules_as_parsed_ones() {
    let entry = CudaArchEntry {
        major: 7,
        minor: 0,
        variant: ArchVariant::Generic,
        emits_cubin: true,
        emits_ptx: true,
    };
    assert_eq!(entry_coverage(entry, V100), Some(ArchCoverage::Cubin));
    assert_eq!(entry_coverage(entry, A100), Some(ArchCoverage::Ptx));
}

// ── This build, on this host ──────────────────────────────────────────────────
//
// The three tests below are the only ones here that read real build and device
// state. They still run everywhere: off CUDA the recorded list is empty and the
// capability is `None`, which each test asserts explicitly rather than skipping.

#[test]
fn the_build_records_the_architecture_list_it_compiled_for() {
    let compiled = compiled_cuda_architectures();
    if cfg!(feature = "cuda") {
        assert!(
            !compiled.is_empty(),
            "a CUDA build must record MLX_CUDA_ARCHITECTURES; build.rs emits it as              MLXCEL_CUDA_ARCHITECTURES"
        );
        assert!(
            !parse_cuda_arch_list(compiled).is_empty(),
            "the recorded architecture list {compiled:?} parsed to nothing, so the mismatch              check would silently disable itself"
        );
    } else {
        assert!(
            compiled.is_empty(),
            "a non-CUDA build compiles no device code and must record no architecture list,              got {compiled:?}"
        );
    }
}

#[test]
fn the_capability_probe_agrees_with_the_backend() {
    let capability = cuda_compute_capability();
    if cfg!(feature = "cuda") {
        // A CUDA build on a host with no visible device legitimately reports
        // `None`; what must never happen is a nonsense pair.
        if let Some((major, minor)) = capability {
            assert!(
                (1..=99).contains(&major) && minor < 10,
                "implausible compute capability {major}.{minor} from the device probe"
            );
        }
    } else {
        assert_eq!(
            capability, None,
            "a non-CUDA build has no compute capability to report"
        );
    }
    // The probe is cached, so a second call must not disagree with the first.
    assert_eq!(cuda_compute_capability(), capability);
    // The startup summary exists exactly when a capability does.
    assert_eq!(cuda_arch_startup_summary().is_some(), capability.is_some());
}

#[test]
fn this_build_can_run_on_the_host_running_its_tests() {
    // A binary that cannot run on its own build host is the failure this module
    // exists to name (`build.rs` falls back to `90a` when nvidia-smi is absent,
    // which produces exactly that). If it ever happens here, the suite should
    // say so rather than pass and let the first kernel launch discover it.
    assert_eq!(
        cuda_arch_mismatch(),
        None,
        "compiled for [{}], running on {:?}",
        compiled_cuda_architectures(),
        cuda_compute_capability()
    );
}

// ── The named error ───────────────────────────────────────────────────────────

#[test]
fn the_mismatch_message_names_both_sides_and_a_working_rebuild() {
    let mismatch = CudaArchMismatch {
        device: V100,
        compiled: RELEASE_X86_64,
    };
    let message = mismatch.to_string();
    // The whole point of the error: an operator reading it knows what they have
    // and what the binary has, without decoding a CUDA status code.
    assert!(message.contains(RELEASE_X86_64), "{message}");
    assert!(message.contains("7.0"), "{message}");
    assert!(message.contains("sm_70"), "{message}");
    assert!(
        message.contains("MLX_CUDA_ARCHITECTURES=70"),
        "the message should name the rebuild that fixes it: {message}"
    );
    assert!(
        message.contains("MLXCEL_DEVICE=cpu"),
        "the message should name the CPU escape hatch: {message}"
    );
}

#[test]
fn the_suggested_rebuild_carries_the_hopper_suffix() {
    // `90` and `90a` are not interchangeable: MLX only compiles its dedicated
    // Hopper quantized kernel when the list says `90a`, so a suggestion that
    // dropped the suffix would rebuild into a slower binary.
    let suggest = |device| {
        CudaArchMismatch {
            device,
            compiled: RELEASE_X86_64,
        }
        .suggested_architecture()
    };
    assert_eq!(suggest(V100), "70");
    assert_eq!(suggest(A100), "80");
    assert_eq!(suggest(H100), "90a");
    assert_eq!(suggest(GB10), "121a");
}
