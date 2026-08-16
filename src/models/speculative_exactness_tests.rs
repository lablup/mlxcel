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

//! Unit coverage for the block-vs-chain exactness gate's decision logic.
//!
//! The probe itself needs a Metal device and a checkpoint; the comparison
//! and the memo do not, and they are where a wrong answer would silently
//! re-open the contract. Everything here runs on any host.

use super::{BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate};
use std::sync::atomic::{AtomicUsize, Ordering};

fn key(block_size: u32) -> ProbeKey {
    ProbeKey {
        block_size,
        hidden_size: 5120,
        num_hidden_layers: 64,
    }
}

#[test]
fn identical_arms_compare_equal() {
    let block = vec![vec![1u8, 2, 3], vec![4, 5, 6]];
    let chain = block.clone();
    assert_eq!(
        compare_block_against_chain(&block, &chain),
        BlockChainExactness::Equal
    );
}

#[test]
fn a_single_differing_byte_is_a_divergence_with_the_first_position() {
    let block = vec![vec![1u8, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    let mut chain = block.clone();
    chain[1][2] = 0xff;
    chain[2][0] = 0xff;
    match compare_block_against_chain(&block, &chain) {
        BlockChainExactness::Diverges {
            position,
            differing_bytes,
            total_bytes,
        } => {
            assert_eq!(position, 1, "must report the FIRST diverging position");
            assert_eq!(differing_bytes, 1);
            assert_eq!(total_bytes, 3);
        }
        other => panic!("expected a divergence, got {other:?}"),
    }
}

#[test]
fn an_empty_or_ragged_comparison_is_not_a_pass() {
    // The failure this guards: a probe that produced nothing must never be
    // indistinguishable from a probe that produced agreement.
    assert!(!compare_block_against_chain(&[], &[]).is_equal());
    let block = vec![vec![1u8, 2, 3]];
    let short = vec![vec![1u8, 2]];
    assert!(!compare_block_against_chain(&block, &short).is_equal());
    let two = vec![vec![1u8], vec![2u8]];
    assert!(!compare_block_against_chain(&block, &two).is_equal());
}

#[test]
fn not_run_is_a_decline() {
    assert!(!BlockChainExactness::NotRun("no metal").is_equal());
}

#[test]
fn the_gate_runs_the_probe_once_per_key_and_caches_the_decision() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    // A block width no other test in this process uses, so the shared memo
    // cannot cross-contaminate.
    let k = key(9001);
    let probe = || {
        CALLS.fetch_add(1, Ordering::SeqCst);
        BlockChainExactness::Equal
    };
    assert!(mtp_exactness_gate(k, probe));
    assert!(mtp_exactness_gate(k, || {
        CALLS.fetch_add(1, Ordering::SeqCst);
        BlockChainExactness::Equal
    }));
    assert_eq!(
        CALLS.load(Ordering::SeqCst),
        1,
        "the second call must be served from the memo, not re-measured"
    );
}

#[test]
fn a_diverging_probe_declines_unless_the_override_is_set() {
    // `allow_inexact()` caches the env read for the process lifetime, so
    // this test asserts against whatever the ambient value is rather than
    // mutating the environment under other tests. Either way the gate's
    // decision must equal the override, never "equal by accident".
    let k = key(9002);
    let decision = mtp_exactness_gate(k, || BlockChainExactness::Diverges {
        position: 0,
        differing_bytes: 4,
        total_bytes: 1024,
    });
    assert_eq!(
        decision,
        super::allow_inexact(),
        "a diverging probe may only engage MTP when MLXCEL_MTP_ALLOW_INEXACT is set"
    );
}

#[test]
fn different_block_widths_are_measured_separately() {
    // The whole point of probing: byte-identity holds at one block width
    // and not at another on the same host (M1 Ultra passes at 3, fails at
    // 12), so the memo key must carry the width.
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let bump = || {
        CALLS.fetch_add(1, Ordering::SeqCst);
        BlockChainExactness::Equal
    };
    assert!(mtp_exactness_gate(key(9003), bump));
    assert!(mtp_exactness_gate(key(9004), bump));
    assert_eq!(CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn reason_strings_name_the_position_and_the_byte_counts() {
    let reason = BlockChainExactness::Diverges {
        position: 2,
        differing_bytes: 17,
        total_bytes: 496640,
    }
    .reason();
    assert!(reason.contains('2'), "{reason}");
    assert!(reason.contains("17"), "{reason}");
    assert!(reason.contains("496640"), "{reason}");
}
