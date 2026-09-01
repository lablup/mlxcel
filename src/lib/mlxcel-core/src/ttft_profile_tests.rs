//! [#1545] Host-side unit tests for the pre-first-token phase breakdown.
//!
//! `MLXCEL_PROFILE_TTFT` reports the phases of the path that produces the
//! first token. The measurement it exists for is on a GPU, but the arithmetic
//! that decides whether the named phases account for the reported prefill is
//! not, so it is pinned here: no GPU, no CUDA toolkit and no `cuda` feature
//! needed, which means these run in every CI job.
//!
//! What matters about the line is that a reader can check it. `setup` is
//! measured before the prefill clock starts and every other phase inside it,
//! so `in_prefill_ns` is the sum that has to reconcile against the reported
//! prefill, and the printed `residual` is what is left once it does. Getting
//! that boundary wrong is exactly the mistake that made the audit's ~13 s and
//! the baseline record's 24.94 s look irreconcilable.

use crate::generate::TtftPhases;

fn phases() -> TtftPhases {
    TtftPhases {
        setup_ns: 1_000_000,
        build_ns: 2_000_000,
        sample_ns: 3_000_000,
        eval_ns: 4_000_000,
        post_ns: 5_000_000,
    }
}

#[test]
fn in_prefill_excludes_setup() {
    // setup is measured before `prefill_start`, so it must not be counted
    // against the reported prefill time.
    assert_eq!(phases().in_prefill_ns(), 14_000_000);
}

#[test]
fn total_includes_setup() {
    assert_eq!(phases().total_ns(), 15_000_000);
}

#[test]
fn default_phases_are_all_zero() {
    let empty = TtftPhases::default();
    assert_eq!(empty.in_prefill_ns(), 0);
    assert_eq!(empty.total_ns(), 0);
}

#[test]
fn residual_is_the_unattributed_remainder() {
    // A prefill wall clock 6 ms longer than the named phases leaves 6 ms
    // unattributed, and the line has to say so rather than round it away.
    let line = phases().format_line(3, 20_000_000);
    assert!(line.contains("residual=6.00ms"), "{line}");
    assert!(line.contains("prefill=20.00ms"), "{line}");
}

#[test]
fn residual_saturates_instead_of_wrapping() {
    // The phases are nested inside the prefill clock, so a prefill smaller
    // than their sum means the timers disagree. Unsigned wrap would print a
    // 584-year residual; saturation prints zero and keeps the line readable.
    let line = phases().format_line(3, 1_000_000);
    assert!(line.contains("residual=0.00ms"), "{line}");
}

#[test]
fn line_names_every_phase_and_the_prompt_length() {
    let line = phases().format_line(54, 14_000_000);
    for field in [
        "[TTFT]",
        "prompt=54 tok",
        "setup=1.00ms",
        "build=2.00ms",
        "sample=3.00ms",
        "eval=4.00ms",
        "post=5.00ms",
    ] {
        assert!(line.contains(field), "missing {field} in {line}");
    }
}
