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

#[test]
fn per_ip_limit_refuses_the_excess_and_releases_on_drop() {
    let limiter = PendingHandshakeLimiter::new(8, 2);
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
    let limiter = PendingHandshakeLimiter::new(2, 8);
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
    let limiter = PendingHandshakeLimiter::new(8, 1);
    let mapped: IpAddr = "::ffff:10.0.0.1".parse().unwrap();
    let _a = limiter.try_acquire(mapped).unwrap();
    assert_eq!(
        limiter.try_acquire(ip(1)).unwrap_err(),
        PendingSlotRefusal::PerIp { ip: ip(1), max: 1 }
    );
}
