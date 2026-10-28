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

//! The address predicate behind the media URL policy in `media_net`: which
//! resolved or literal addresses a remote media fetch may reach.
//!
//! Pure apart from reading the configured NAT64 prefixes. The IPv6
//! special-purpose blocks follow the IANA IPv6 Special-Purpose Address
//! Registry's "Globally Reachable" column, the same source nightly
//! `Ipv6Addr::is_global` mirrors (issues #1451, #2693, #2713).

use std::net::{IpAddr, Ipv4Addr};

use super::nat64::{self, Nat64Prefix};

/// True when `addr` is a public unicast address a media fetch may reach.
///
/// Everything else is refused: loopback, the unspecified address and the rest
/// of `0.0.0.0/8`, multicast and broadcast, IPv4 private and link-local ranges (which is what makes
/// `169.254.169.254` unreachable), carrier-grade NAT, the documentation and
/// benchmark ranges, any IPv4-mapped, IPv4-compatible, NAT64 (`64:ff9b::/96`)
/// or 6to4 (`2002::/16`) IPv6 form of a refused IPv4 address, and the whole
/// RFC 8215 local-use NAT64 block `64:ff9b:1::/48`. An address inside a prefix
/// listed in `MLXCEL_MEDIA_NAT64_PREFIXES` is judged by the IPv4 address it
/// embeds instead, which is how a DNS64 deployment on the local-use block or
/// on an operator network-specific prefix reaches public hosts (issue #2693).
///
/// Any other IPv6 address outside `2000::/3`, the only block IANA has
/// allocated for global unicast, is refused: unique-local, link-local,
/// site-local `fec0::/10`, IPv4-translated `::ffff:0:0:0/96`, discard-only
/// `100::/64`, dummy `100:0:0:1::/64`, SRv6 SIDs `5f00::/16` and the reserved
/// rest. Inside it, the `2001::/23` IETF protocol assignments (Teredo,
/// benchmarking and deprecated ORCHID among them) are refused except the
/// entries the IANA registry marks globally reachable, and so are the
/// documentation prefixes `2001:db8::/32` and `3fff::/20` (issue #2713).
#[must_use]
pub(crate) fn is_public_address(addr: IpAddr) -> bool {
    nat64::with_configured_prefixes(|prefixes| is_public_address_with(addr, prefixes))
}

/// [`is_public_address`] against an explicit NAT64 prefix list. Pure.
#[must_use]
pub(crate) fn is_public_address_with(addr: IpAddr, nat64: &[Nat64Prefix]) -> bool {
    match addr {
        IpAddr::V4(v4) => {
            {
                let [a, b, c, _] = v4.octets();
                // 100.64.0.0/10 carrier-grade NAT, 192.0.0.0/24 IETF protocol
                // assignments, 198.18.0.0/15 benchmarking, 192.0.2.0/24 /
                // 198.51.100.0/24 / 203.0.113.0/24 documentation, 240.0.0.0/4
                // reserved, 0.0.0.0/8 this network. `Ipv4Addr::is_global` and `is_documentation` cover
                // these on nightly only, so they are spelled out here, each at
                // its exact prefix so no ordinary public address is refused.
                // 0.0.0.0/8 "this network" (RFC 791) is refused whole, not just
                // `0.0.0.0`: a NAT64 prefix configured shorter than its real
                // length decodes a /96-synthesized address into `0.0.0.x`.
                let reserved = a == 0
                    || (a == 100 && (64..128).contains(&b))
                    || (a == 192 && b == 0 && c == 0)
                    || (a == 192 && b == 0 && c == 2)
                    || (a == 198 && (18..20).contains(&b))
                    || (a == 198 && b == 51 && c == 100)
                    || (a == 203 && b == 0 && c == 113)
                    || a >= 240;
                if reserved {
                    return false;
                }
            }
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast())
        }
        IpAddr::V6(v6) => {
            // The IPv6-native rules come first. `::1` and `::` both live inside
            // the deprecated IPv4-compatible `::/96` block, so converting to
            // IPv4 before testing them would turn `::1` into `0.0.0.1` and let
            // loopback through.
            if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
                return false;
            }
            // An IPv4 address written in IPv4-mapped form must be judged by its
            // IPv4 rules, or `::ffff:127.0.0.1` walks straight past them.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_public_address_with(IpAddr::V4(mapped), nat64);
            }
            let segments = v6.segments();
            // The rest of `::/96` is the deprecated IPv4-compatible form, which
            // no reachable public host uses; refuse the whole block rather than
            // reasoning about the embedded address.
            if segments[..6] == [0, 0, 0, 0, 0, 0] {
                return false;
            }
            // An operator-listed NAT64 prefix decides by the IPv4 address it
            // embeds; one with the reserved `u` octet set embeds none.
            if let Some(prefix) = nat64.iter().find(|prefix| prefix.contains(v6)) {
                return prefix
                    .embedded_ipv4(v6)
                    .is_some_and(|embedded| is_public_address_with(IpAddr::V4(embedded), nat64));
            }
            // A NAT64 translator on the well-known prefix (RFC 6052) forwards
            // `64:ff9b::a00:5` to `10.0.0.5`, and cloud NAT64 gateways do reach
            // private VPC addresses this way, so judge the embedded address.
            if segments[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
                return is_public_address_with(embedded_ipv4(segments[6], segments[7]), nat64);
            }
            // The RFC 8215 local-use block may carry its IPv4 address at any
            // RFC 6052 position, so unless a listed prefix said where, nothing
            // in it can be judged; it is not globally reachable either.
            if segments[..3] == [0x64, 0xff9b, 0x0001] {
                return false;
            }
            // IANA has allocated only 2000::/3 for global unicast (RFC 4291,
            // RFC 3587). What reaches this point outside it is unique-local
            // fc00::/7 (RFC 4193), link-local fe80::/10 (RFC 4291), site-local
            // fec0::/10 (RFC 3879), IPv4-translated ::ffff:0:0:0/96 (RFC 2765),
            // discard-only 100::/64 (RFC 6666), dummy 100:0:0:1::/64
            // (RFC 9780), SRv6 SIDs 5f00::/16 (RFC 9602) or reserved space.
            // A gate rather than a list, so the next block carved from
            // reserved space is refused too. Listed NAT64 prefixes and the
            // well-known prefix were decided above, so they still win.
            if segments[0] & 0xe000 != 0x2000 {
                return false;
            }
            // 6to4 (RFC 3056) embeds the IPv4 relay endpoint in bits 16..48.
            if segments[0] == 0x2002 {
                return is_public_address_with(embedded_ipv4(segments[1], segments[2]), nat64);
            }
            // 2001:db8::/32 documentation (RFC 3849), 3fff::/20 documentation
            // (RFC 9637).
            let documentation = (segments[0] == 0x2001 && segments[1] == 0x0db8)
                || (segments[0] == 0x3fff && segments[1] & 0xf000 == 0);
            !(ietf_protocol_assignment_not_global(segments) || documentation)
        }
    }
}

/// The IPv4 address carried in two IPv6 segments.
fn embedded_ipv4(high: u16, low: u16) -> IpAddr {
    let [a, b] = high.to_be_bytes();
    let [c, d] = low.to_be_bytes();
    IpAddr::V4(Ipv4Addr::new(a, b, c, d))
}

/// True when `segments` lies in the `2001::/23` IETF Protocol Assignments
/// block (RFC 2928) and is not one of its globally reachable entries.
///
/// Refused through this rule: Teredo `2001::/32` (RFC 4380, RFC 8190), whose
/// embedded server and obfuscated client IPv4 addresses are deliberately not
/// decoded, benchmarking `2001:2::/48` (RFC 5180, Errata 1752), the deprecated
/// ORCHID `2001:10::/28` (RFC 4843), and the unassigned rest of the block.
fn ietf_protocol_assignment_not_global(segments: [u16; 8]) -> bool {
    if segments[0] != 0x2001 || segments[1] >= 0x0200 {
        return false;
    }
    let globally_reachable =
        // 2001:1::1/128 Port Control Protocol anycast (RFC 7723),
        // 2001:1::2/128 TURN anycast (RFC 8155),
        // 2001:1::3/128 DNS-SD SRP anycast (RFC 9665).
        (segments[1] == 0x0001
            && segments[2..7] == [0, 0, 0, 0, 0]
            && (1..=3).contains(&segments[7]))
        // 2001:3::/32 AMT (RFC 7450).
        || segments[1] == 0x0003
        // 2001:4:112::/48 AS112-v6 (RFC 7535).
        || (segments[1] == 0x0004 && segments[2] == 0x0112)
        // 2001:20::/28 ORCHIDv2 (RFC 7343).
        || segments[1] & 0xfff0 == 0x0020
        // 2001:30::/28 Drone Remote ID Protocol Entity Tags (RFC 9374).
        || segments[1] & 0xfff0 == 0x0030;
    !globally_reachable
}
