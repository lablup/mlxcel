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

//! NAT64 prefix tests for the media address policy (issues #2693, #2728,
//! #2764), moved out of `media_net_tests.rs` to keep it under the file-size
//! limit.
//!
//! Like the address-policy tests, nothing here opens a socket or resolves a
//! name.

use super::*;

fn v6(raw: &str) -> std::net::Ipv6Addr {
    raw.parse().expect("IPv6 address parses")
}

fn prefixes(raw: &str) -> Vec<Nat64Prefix> {
    parse_media_nat64_prefixes(raw).expect("prefix list parses")
}

#[test]
fn local_use_nat64_forms_are_refused() {
    // RFC 8215 local-use NAT64 (`64:ff9b:1::/48`) is not globally reachable,
    // and with no prefix configured nothing says where its IPv4 address sits,
    // so `64:ff9b:1::a9fe:a9fe` must not reach 169.254.169.254 through a local
    // translator. The last one is a /96 synthesis of a public address, refused
    // all the same until the operator lists the prefix (issue #2693).
    for addr in [
        "64:ff9b:1::a9fe:a9fe",
        "64:ff9b:1::7f00:1",
        "64:ff9b:1:0:0:0:5db8:d822",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
        assert!(
            !is_public_address_with(parsed, &[]),
            "{addr} must be refused"
        );
    }
}

#[test]
fn a_listed_local_use_prefix_is_judged_by_its_embedded_address() {
    let listed = prefixes("64:ff9b:1::/96");
    for (addr, public) in [
        ("64:ff9b:1::5db8:d822", true),
        ("64:ff9b:1::a00:5", false),
        ("64:ff9b:1::a9fe:a9fe", false),
        // Inside the /48 but outside the listed /96: still refused.
        ("64:ff9b:1:1::5db8:d822", false),
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert_eq!(is_public_address_with(parsed, &listed), public, "{addr}");
    }
}

#[test]
fn the_rfc_6052_examples_decode_to_their_ipv4_address() {
    // RFC 6052 section 2.4: 192.0.2.33 under each prefix length.
    let expected = std::net::Ipv4Addr::new(192, 0, 2, 33);
    for (prefix, addr) in [
        ("2001:db8::/32", "2001:db8:c000:221::"),
        ("2001:db8:100::/40", "2001:db8:1c0:2:21::"),
        ("2001:db8:122::/48", "2001:db8:122:c000:2:2100::"),
        ("2001:db8:122:300::/56", "2001:db8:122:3c0:0:221::"),
        ("2001:db8:122:344::/64", "2001:db8:122:344:c0:2:2100:0"),
        ("2001:db8:122:344::/96", "2001:db8:122:344::192.0.2.33"),
    ] {
        // The examples use the documentation prefix, which `parse` refuses
        // as an operator prefix (issue #2764), so build them unchecked.
        let prefix = Nat64Prefix::parse_unchecked_for_test(prefix);
        assert!(prefix.contains(v6(addr)), "{prefix} contains {addr}");
        assert_eq!(
            prefix.embedded_ipv4(v6(addr)),
            Some(expected),
            "{addr} under {prefix}"
        );
    }
}

#[test]
fn a_nonzero_u_octet_embeds_nothing_and_is_refused() {
    let listed = prefixes("2001:db9:122:344::/64");
    // Octet 8 (the RFC 6052 `u` octet) is 0x01 here.
    let addr = v6("2001:db9:122:344:1c0:2:2100:0");
    assert_eq!(listed[0].embedded_ipv4(addr), None);
    assert!(!is_public_address_with(IpAddr::V6(addr), &listed));
}

#[test]
fn an_operator_prefix_is_decoded_instead_of_passing_as_public() {
    let listed = prefixes("2001:db9:64::/96");
    let metadata: IpAddr = "2001:db9:64::a9fe:a9fe".parse().expect("parses");
    let public: IpAddr = "2001:db9:64::808:808".parse().expect("parses");
    assert!(!is_public_address_with(metadata, &listed));
    assert!(is_public_address_with(public, &listed));
    // Unlisted, the same space is ordinary global unicast.
    assert!(is_public_address_with(metadata, &[]));
}

#[test]
fn the_well_known_prefix_and_6to4_still_decode_with_prefixes_listed() {
    let listed = prefixes("64:ff9b:1::/96, 2001:db9:64::/96");
    for (addr, public) in [
        ("64:ff9b::808:808", true),
        ("64:ff9b::a9fe:a9fe", false),
        ("2002:808:808::1", true),
        ("2002:a9fe:a9fe::", false),
        ("::1", false),
        ("::ffff:127.0.0.1", false),
        ("fd12:3456::1", false),
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert_eq!(is_public_address_with(parsed, &listed), public, "{addr}");
    }
}

#[test]
fn nat64_prefix_parsing_rejects_what_rfc_6052_does_not_define() {
    for bad in [
        "64:ff9b:1::/50",
        "64:ff9b:1::1/96",
        "not-an-ip/96",
        "64:ff9b:1::",
        "10.0.0.0/32",
        "64:ff9b:1::/abc",
        "2001:db8:122:344:100::/96",
    ] {
        assert!(Nat64Prefix::parse(bad).is_err(), "{bad} must be rejected");
    }
}

#[test]
fn non_global_nat64_prefixes_are_refused() {
    // A listed prefix outranks the `2000::/3` gate, so a link-local, unique-local,
    // site-local or other non-global prefix would let an address inside it pass
    // whenever it embeds a public IPv4 address (issue #2728). Every entry is
    // checked before failing, so the message lists all that still parse.
    let accepted: Vec<String> = [
        "fe80::/64",
        "fd00:64::/96",
        "fc00::/96",
        "fec0::/96",
        "64:ff9b:100::/40",
    ]
    .into_iter()
    .filter_map(|bad| match Nat64Prefix::parse(bad) {
        Ok(prefix) => Some(format!("{bad} parsed as {prefix}")),
        Err(err) => {
            assert!(err.contains(bad), "{err}");
            assert!(err.contains("2000::/3"), "{err}");
            None
        }
    })
    .collect();
    assert!(
        accepted.is_empty(),
        "non-global NAT64 prefixes were accepted: {accepted:?}"
    );
    for good in ["64:ff9b:1::/96", "64:ff9b:1:abcd::/64"] {
        assert!(Nat64Prefix::parse(good).is_ok(), "{good} must still parse");
    }
    let err = parse_media_nat64_prefixes("64:ff9b:1::/96, fd00:64::/96")
        .expect_err("startup refuses a unique-local translator prefix");
    assert!(err.contains(MEDIA_NAT64_PREFIXES_ENV), "{err}");
    assert!(err.contains("fd00:64::/96"), "{err}");
}

#[test]
fn special_purpose_nat64_prefixes_are_refused() {
    // Inside `2000::/3` but in IANA special-purpose blocks no translator is
    // numbered from: addresses there must stay refused rather than be judged
    // by the IPv4 address they would embed (issue #2764).
    for (bad, block) in [
        ("2001:db8:64::/96", "2001:db8::/32"),
        ("2001:0:1::/96", "2001::/23"),
        ("2002:c000:201::/96", "2002::/16"),
        ("3fff:64::/96", "3fff::/20"),
    ] {
        let err = match Nat64Prefix::parse(bad) {
            Ok(prefix) => panic!("{bad} parsed as {prefix}"),
            Err(err) => err,
        };
        assert!(err.contains(bad), "{err}");
        assert!(err.contains(block), "{bad}: {err}");
        let err = parse_media_nat64_prefixes(bad).expect_err("startup refuses it");
        assert!(err.contains(MEDIA_NAT64_PREFIXES_ENV), "{err}");
    }
}

#[test]
fn the_nat64_prefix_list_trims_and_skips_empty_entries() {
    assert_eq!(
        prefixes(" 64:ff9b:1::/96 , "),
        vec![Nat64Prefix::parse("64:ff9b:1::/96").expect("parses")]
    );
    assert!(prefixes("").is_empty());
    assert!(prefixes(" , ,").is_empty());
}

#[test]
fn a_bad_nat64_prefix_list_names_the_variable_and_the_entry() {
    // What startup reports before refusing to start.
    let err = parse_media_nat64_prefixes("64:ff9b:1::/96, 64:ff9b:1::/50")
        .expect_err("a disallowed length fails");
    assert!(err.contains(MEDIA_NAT64_PREFIXES_ENV), "{err}");
    assert!(err.contains("64:ff9b:1::/50"), "{err}");
}

#[test]
fn a_prefix_shorter_than_96_covering_the_well_known_prefix_is_rejected() {
    for bad in [
        "64:ff9b::/64",
        "64:ff9b::/32",
        "64:ff9b::/40",
        "64:ff9b::/48",
    ] {
        let err = Nat64Prefix::parse(bad).expect_err("must be rejected");
        assert!(err.contains(bad), "{err}");
    }
    assert!(Nat64Prefix::parse("64:ff9b::/96").is_ok());
    // Prefixes that do not cover the well-known block stay accepted.
    assert!(Nat64Prefix::parse("64:ff9b:1::/64").is_ok());
    assert!(Nat64Prefix::parse("2001:db9::/32").is_ok());
    let err = parse_media_nat64_prefixes("64:ff9b::/64").expect_err("list rejects it");
    assert!(err.contains(MEDIA_NAT64_PREFIXES_ENV), "{err}");
}
