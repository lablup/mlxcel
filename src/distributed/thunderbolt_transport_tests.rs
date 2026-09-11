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

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use super::*;
use crate::distributed::transport::TransportBackend;

#[test]
fn thunderbolt_parse_networksetup_output() {
    let output = r#"
Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:ff

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 11:22:33:44:55:66

Hardware Port: Thunderbolt 5 Bridge
Device: bridge1
Ethernet Address: 22:33:44:55:66:77
"#;

    assert_eq!(
        parse_networksetup_hardware_ports(output),
        vec!["bridge0".to_string(), "bridge1".to_string()]
    );
}

#[test]
fn thunderbolt_parse_ifconfig_addresses() {
    let output = r#"
bridge0: flags=8863<UP,BROADCAST,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet 169.254.91.10 netmask 0xffff0000 broadcast 169.254.255.255
    inet6 fe80::18f:42ff:fe3c:9a76%bridge0 prefixlen 64 scopeid 0xe
"#;

    assert_eq!(
        parse_ifconfig_addresses(output),
        vec![
            IpAddr::V4(Ipv4Addr::new(169, 254, 91, 10)),
            "fe80::18f:42ff:fe3c:9a76".parse().unwrap(),
        ]
    );
}

#[test]
fn thunderbolt_resolve_bind_address_from_specific_host() {
    let interfaces = vec![ThunderboltInterfaceInfo {
        interface: "bridge0".to_string(),
        addresses: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 91, 10))],
    }];
    let resolved = resolve_bind_address(
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(169, 254, 91, 10)), 9200),
        &interfaces,
    )
    .unwrap();

    assert_eq!(resolved.interface, "bridge0");
    assert_eq!(
        resolved.bind_address,
        "169.254.91.10:9200".parse::<SocketAddr>().unwrap()
    );
}

#[test]
fn thunderbolt_resolve_bind_address_from_unspecified_host() {
    let interfaces = vec![ThunderboltInterfaceInfo {
        interface: "bridge0".to_string(),
        addresses: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 91, 10))],
    }];
    let resolved =
        resolve_bind_address("0.0.0.0:9200".parse::<SocketAddr>().unwrap(), &interfaces).unwrap();

    assert_eq!(resolved.interface, "bridge0");
    assert_eq!(
        resolved.bind_address,
        "169.254.91.10:9200".parse::<SocketAddr>().unwrap()
    );
}

#[test]
fn thunderbolt_rejects_non_thunderbolt_bind_address() {
    let interfaces = vec![ThunderboltInterfaceInfo {
        interface: "bridge0".to_string(),
        addresses: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 91, 10))],
    }];
    let err = resolve_bind_address("127.0.0.1:9200".parse::<SocketAddr>().unwrap(), &interfaces)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not belong to a Thunderbolt Bridge interface")
    );
}

#[test]
fn thunderbolt_config_defaults() {
    let config = ThunderboltTransportConfig::default();
    assert_eq!(config.interface, "bridge0");
    assert_eq!(config.port, 9200);
    assert!(config.use_shared_memory);
    assert_eq!(config.max_transfer_size, 1024 * 1024 * 1024);
}

#[test]
fn thunderbolt_config_from_bind_address_uses_matching_interface() {
    let interfaces = vec![
        ThunderboltInterfaceInfo {
            interface: "bridge0".to_string(),
            addresses: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 91, 10))],
        },
        ThunderboltInterfaceInfo {
            interface: "bridge1".to_string(),
            addresses: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 92, 10))],
        },
    ];
    let resolved = resolve_bind_address(
        "169.254.92.10:9300".parse::<SocketAddr>().unwrap(),
        &interfaces,
    )
    .unwrap();

    let config = ThunderboltTransportConfig {
        interface: resolved.interface,
        port: resolved.bind_address.port(),
        ..Default::default()
    };

    assert_eq!(config.interface, "bridge1");
    assert_eq!(config.port, 9300);
}

#[test]
fn thunderbolt_backend_type_and_availability_query() {
    assert_eq!(TransportBackend::Thunderbolt.to_string(), "thunderbolt");
    let _ = ThunderboltTransport::is_available();
}
