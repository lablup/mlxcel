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

use std::net::{IpAddr, Ipv4Addr};

use super::*;

fn ip(last: u8) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(10, 0, 0, last))
}

fn ip6(s: &str) -> IpAddr {
    s.parse().unwrap()
}

#[test]
fn per_ip_limit_refuses_the_excess_and_releases_on_drop() {
    let limiter = PendingHandshakeLimiter::new(8, 2).unwrap();
    let a = limiter.try_acquire(ip(1)).unwrap();
    let b = limiter.try_acquire(ip(1)).unwrap();
    assert_eq!(
        limiter.try_acquire(ip(1)).unwrap_err(),
        PendingSlotRefusal::PerIp { ip: ip(1), max: 2 }
    );
    // Another source is unaffected.
    let c = limiter.try_acquire(ip(2)).unwrap();
    assert_eq!(limiter.pending_from(ip(1)), 2);
    assert_eq!(limiter.pending_total(), 3);
    drop(a);
    assert_eq!(limiter.pending_from(ip(1)), 1);
    let _d = limiter.try_acquire(ip(1)).unwrap();
    drop(b);
    drop(c);
    assert_eq!(limiter.pending_from(ip(2)), 0);
}

#[test]
fn global_limit_refuses_across_sources_and_does_not_leak_per_ip_counts() {
    let limiter = PendingHandshakeLimiter::new(2, 8).unwrap();
    let _a = limiter.try_acquire(ip(1)).unwrap();
    let _b = limiter.try_acquire(ip(2)).unwrap();
    assert_eq!(
        limiter.try_acquire(ip(3)).unwrap_err(),
        PendingSlotRefusal::Global { max: 2 }
    );
    // The refused source's per-IP entry was rolled back and removed.
    assert_eq!(limiter.pending_from(ip(3)), 0);
    assert_eq!(limiter.pending_total(), 2);
}

#[test]
fn ipv4_mapped_ipv6_counts_against_the_ipv4_key() {
    let limiter = PendingHandshakeLimiter::new(8, 1).unwrap();
    let mapped: IpAddr = "::ffff:10.0.0.1".parse().unwrap();
    let _a = limiter.try_acquire(mapped).unwrap();
    assert_eq!(
        limiter.try_acquire(ip(1)).unwrap_err(),
        PendingSlotRefusal::PerIp { ip: ip(1), max: 1 }
    );
}

/// A host that owns a /64 cannot get one pending slot per address: every
/// address inside the /64 draws from the same per-source budget, and the
/// refusal names the /64 key.
#[test]
fn ipv6_sources_in_one_64_share_the_per_source_limit() {
    let limiter = PendingHandshakeLimiter::new(64, 2).unwrap();
    let a = limiter.try_acquire(ip6("2001:db8:1:2::1")).unwrap();
    let _b = limiter.try_acquire(ip6("2001:db8:1:2::2")).unwrap();
    let refusal = limiter
        .try_acquire(ip6("2001:db8:1:2:ffff::3"))
        .unwrap_err();
    assert_eq!(
        refusal,
        PendingSlotRefusal::PerIp {
            ip: ip6("2001:db8:1:2::"),
            max: 2,
        }
    );
    assert_eq!(
        refusal.to_string(),
        "max pending handshakes from 2001:db8:1:2::/64 (2) reached"
    );
    assert_eq!(limiter.pending_from(ip6("2001:db8:1:2::9")), 2);
    // Dropping one slot frees the shared budget for any address in the /64.
    drop(a);
    let _c = limiter.try_acquire(ip6("2001:db8:1:2:ffff::3")).unwrap();
    assert_eq!(limiter.pending_from(ip6("2001:db8:1:2::")), 2);
}

#[test]
fn ipv6_sources_in_different_64s_are_counted_separately() {
    let limiter = PendingHandshakeLimiter::new(64, 1).unwrap();
    let _a = limiter.try_acquire(ip6("2001:db8:1:2::1")).unwrap();
    let _b = limiter.try_acquire(ip6("2001:db8:1:3::1")).unwrap();
    assert_eq!(limiter.pending_from(ip6("2001:db8:1:2::1")), 1);
    assert_eq!(limiter.pending_from(ip6("2001:db8:1:3::1")), 1);
    assert_eq!(
        limiter.try_acquire(ip6("2001:db8:1:3::2")).unwrap_err(),
        PendingSlotRefusal::PerIp {
            ip: ip6("2001:db8:1:3::"),
            max: 1,
        }
    );
    // IPv4 keys are unchanged and print without a prefix length.
    assert_eq!(pending_source_key(ip(1)), ip(1));
    assert_eq!(
        PendingSlotRefusal::PerIp { ip: ip(1), max: 1 }.to_string(),
        "max pending handshakes from 10.0.0.1 (1) reached"
    );
}

#[test]
fn zero_bounds_are_refused_at_construction() {
    for (total, per_ip) in [(0, 8), (64, 0), (0, 0)] {
        let err = PendingHandshakeLimiter::new(total, per_ip)
            .err()
            .unwrap_or_else(|| panic!("({total}, {per_ip}) must be refused"))
            .to_string();
        assert!(err.contains("at least 1"), "{err}");
    }
}
