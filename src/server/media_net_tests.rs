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

//! Address-policy tests for remote media URLs (issue #1451).
//!
//! Nothing here opens a socket or resolves a name: every case goes through the
//! synchronous shape check or the pure address predicate, so the suite is
//! deterministic on a machine with no network and cannot be steered by whatever
//! the local resolver happens to answer.

use super::*;

fn url(raw: &str) -> reqwest::Url {
    reqwest::Url::parse(raw).expect("test URL parses")
}

#[test]
fn loopback_and_unspecified_addresses_are_refused() {
    for addr in [
        "127.0.0.1",
        "127.4.5.6",
        "0.0.0.0",
        "::1",
        "::",
        "::ffff:127.0.0.1",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn the_cloud_metadata_addresses_are_refused() {
    // 169.254.169.254 is EC2 / GCE / Azure instance metadata, reachable from
    // inside the instance and the single highest-value SSRF target.
    for addr in ["169.254.169.254", "169.254.0.1", "fe80::1"] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn private_and_reserved_ranges_are_refused() {
    for addr in [
        "10.0.0.1",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "100.64.0.1",
        "192.0.0.1",
        "192.0.2.1",
        "198.18.0.1",
        "198.51.100.1",
        "203.0.113.1",
        "240.0.0.1",
        "255.255.255.255",
        "224.0.0.1",
        "fc00::1",
        "fd12:3456::1",
        "2001:db8::1",
        "ff02::1",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn ordinary_public_addresses_are_allowed() {
    // Over-blocking is a compatibility bug, so the neighbours of every
    // reserved prefix are asserted too.
    for addr in [
        "8.8.8.8",
        "1.1.1.1",
        "172.15.0.1",
        "172.32.0.1",
        "100.63.255.255",
        "100.128.0.1",
        "192.0.1.1",
        "192.0.3.1",
        "198.17.255.255",
        "198.20.0.1",
        "198.51.99.1",
        "198.51.101.1",
        "203.0.112.1",
        "203.0.114.1",
        "8.8.4.4",
        "2606:4700::1111",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",
        "2002:808:808::1",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(is_public_address(parsed), "{addr} must be allowed");
    }
}

#[test]
fn nat64_and_6to4_forms_of_a_refused_ipv4_address_are_refused() {
    // A NAT64 gateway translates `64:ff9b::/96` to the embedded IPv4 address,
    // so `64:ff9b::a9fe:a9fe` reaches the metadata service on a host whose
    // IPv6 route goes through one. 6to4 embeds its IPv4 endpoint the same way.
    for addr in [
        "64:ff9b::7f00:1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::a00:5",
        "64:ff9b::c0a8:101",
        "2002:7f00:1::1",
        "2002:a9fe:a9fe::",
        "2002:a00:5::1",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn numeric_ipv4_host_spellings_are_checked_as_literals() {
    // getaddrinfo would read `0x7f.1` or `2130706433` as 127.0.0.1. The URL
    // parser normalizes them to the dotted literal first, so they meet the
    // literal check instead of reaching a resolver as a name.
    for raw in [
        "http://0x7f.1/x.png",
        "http://2130706433/x.png",
        "http://0177.0.0.1/x.png",
        "http://127.1/x.png",
        "http://0xa9.0xfe.0xa9.0xfe/latest/meta-data/",
    ] {
        let parsed = url(raw);
        assert!(
            matches!(
                ensure_media_url_shape_with(&parsed, false),
                Err(MediaUrlError::BlockedAddress { .. })
            ),
            "{raw} parsed as host {:?} must be refused",
            parsed.host_str()
        );
    }
}

#[test]
fn a_non_http_scheme_is_refused() {
    for raw in [
        "file:///etc/passwd",
        "ftp://example.com/x.png",
        "gopher://x/",
    ] {
        let parsed = reqwest::Url::parse(raw).expect("test URL parses");
        assert!(
            matches!(
                ensure_media_url_shape_with(&parsed, false),
                Err(MediaUrlError::Scheme { .. })
            ),
            "{raw} must be refused"
        );
    }
}

#[test]
fn credentials_in_a_media_url_are_refused() {
    assert!(matches!(
        ensure_media_url_shape_with(&url("http://user:pass@example.com/a.png"), false),
        Err(MediaUrlError::Credentials)
    ));
    assert!(matches!(
        ensure_media_url_shape_with(&url("http://user@example.com/a.png"), false),
        Err(MediaUrlError::Credentials)
    ));
}

#[test]
fn an_ip_literal_host_is_checked_without_resolution() {
    for raw in [
        "http://127.0.0.1:9090/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:8080/x.png",
        "http://10.1.2.3/internal.png",
        "http://[::ffff:127.0.0.1]/x.png",
    ] {
        let parsed = url(raw);
        assert!(
            matches!(
                ensure_media_url_shape_with(&parsed, false),
                Err(MediaUrlError::BlockedAddress { .. })
            ),
            "{raw} must be refused"
        );
    }
    assert!(ensure_media_url_shape_with(&url("https://8.8.8.8/a.png"), false).is_ok());
}

#[test]
fn a_public_hostname_passes_the_shape_check() {
    // The shape check is name-agnostic; resolution happens in the async half.
    assert!(ensure_media_url_shape_with(&url("https://example.com/a.png"), false).is_ok());
    assert!(ensure_media_url_shape_with(&url("http://example.com:8080/a.png"), false).is_ok());
}

#[test]
fn a_blocked_peer_address_keeps_the_host_and_the_address_out_of_its_text() {
    // Issue #2644: the rendered text reaches API clients, so it must not name
    // the host, the address it resolved to, or the operator opt-in. Those stay
    // on the structured fields for the debug log.
    let addr: SocketAddr = "127.0.0.1:9090".parse().expect("socket address");
    let err = ensure_socket_addr_allowed_with("images.example.com", addr, false)
        .expect_err("loopback is refused");
    let rendered = err.to_string();
    // Spelled out rather than read from `MEDIA_HOST_REFUSED`, so the client
    // text is pinned and this file still compiles against the pre-#2644 type.
    assert_eq!(
        rendered,
        "media URL host is not allowed or could not be resolved"
    );
    for leaked in [
        "images.example.com",
        "127.0.0.1",
        ALLOW_PRIVATE_MEDIA_URLS_ENV,
    ] {
        assert!(!rendered.contains(leaked), "{leaked} leaked: {rendered}");
    }
    assert!(matches!(
        err,
        MediaUrlError::BlockedAddress { ref host, address }
            if host == "images.example.com" && address == addr.ip()
    ));
}

#[test]
fn a_blocked_host_and_an_unresolvable_host_render_the_same_text() {
    // A caller must not be able to tell "this internal name exists" from "this
    // name does not exist" by comparing the two responses.
    let blocked = MediaUrlError::BlockedAddress {
        host: "intranet-admin.internal".to_owned(),
        address: "10.0.0.7".parse().expect("address parses"),
    };
    let unresolvable = MediaUrlError::Unresolvable {
        host: "intranet-admin.internal".to_owned(),
    };
    assert_eq!(blocked.to_string(), unresolvable.to_string());
    assert!(!blocked.to_string().contains("intranet-admin"));
    assert!(!blocked.to_string().contains("10.0.0.7"));
}

#[test]
fn a_public_peer_address_passes() {
    let addr: SocketAddr = "93.184.216.34:443".parse().expect("socket address");
    assert!(ensure_socket_addr_allowed_with("example.com", addr, false).is_ok());
}

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
        let prefix = Nat64Prefix::parse(prefix).expect("prefix parses");
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
