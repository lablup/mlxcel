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

use super::*;

fn default_monitor() -> BackpressureMonitor {
    BackpressureMonitor::new(BackpressureConfig::default())
}

#[test]
fn load_level_ordering() {
    assert!(LoadLevel::Low < LoadLevel::Normal);
    assert!(LoadLevel::Normal < LoadLevel::High);
    assert!(LoadLevel::High < LoadLevel::Critical);
}

#[test]
fn load_level_display() {
    assert_eq!(format!("{}", LoadLevel::Low), "low");
    assert_eq!(format!("{}", LoadLevel::Normal), "normal");
    assert_eq!(format!("{}", LoadLevel::High), "high");
    assert_eq!(format!("{}", LoadLevel::Critical), "critical");
}

#[test]
fn backpressure_policy_display() {
    assert_eq!(format!("{}", BackpressurePolicy::Drop), "drop");
    assert_eq!(format!("{}", BackpressurePolicy::Block), "block");
    assert_eq!(format!("{}", BackpressurePolicy::Redirect), "redirect");
}

#[test]
fn update_from_metrics_low() {
    let monitor = default_monitor();
    monitor.update_from_metrics("node-0", 0, 0.1);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::Low));
    assert!(!monitor.is_under_pressure("node-0"));
    assert!(!monitor.is_critical("node-0"));
}

#[test]
fn update_from_metrics_normal() {
    let monitor = default_monitor();
    monitor.update_from_metrics("node-0", 3, 0.5);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::Normal));
    assert!(!monitor.is_under_pressure("node-0"));
}

#[test]
fn update_from_metrics_high() {
    let monitor = default_monitor();
    // Default high_watermark is 8.
    monitor.update_from_metrics("node-0", 10, 0.5);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::High));
    assert!(monitor.is_under_pressure("node-0"));
    assert!(!monitor.is_critical("node-0"));
}

#[test]
fn update_from_metrics_critical_by_requests() {
    let monitor = default_monitor();
    // Default critical_watermark is 16.
    monitor.update_from_metrics("node-0", 20, 0.5);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::Critical));
    assert!(monitor.is_critical("node-0"));
}

#[test]
fn update_from_metrics_critical_by_memory() {
    let monitor = default_monitor();
    // Default memory_critical_threshold is 0.95.
    monitor.update_from_metrics("node-0", 1, 0.96);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::Critical));
}

#[test]
fn update_from_metrics_high_by_memory() {
    let monitor = default_monitor();
    // Default memory_high_threshold is 0.80.
    monitor.update_from_metrics("node-0", 1, 0.85);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::High));
}

#[test]
fn record_signal() {
    let monitor = default_monitor();
    let signal = BackpressureSignal {
        node_id: "node-0".to_string(),
        load_level: LoadLevel::High,
        active_requests: 10,
        memory_utilization: 0.7,
        message: Some("test".to_string()),
    };
    monitor.record_signal(&signal);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::High));
}

#[test]
fn unknown_node_returns_none() {
    let monitor = default_monitor();
    assert_eq!(monitor.get_load_level("unknown"), None);
    assert!(!monitor.is_under_pressure("unknown"));
    assert!(!monitor.is_critical("unknown"));
}

#[test]
fn critical_nodes_list() {
    let monitor = default_monitor();
    monitor.update_from_metrics("node-0", 20, 0.5);
    monitor.update_from_metrics("node-1", 3, 0.3);
    monitor.update_from_metrics("node-2", 20, 0.5);

    let critical = monitor.critical_nodes();
    assert_eq!(critical.len(), 2);
    assert!(critical.contains(&"node-0".to_string()));
    assert!(critical.contains(&"node-2".to_string()));
}

#[test]
fn all_load_levels_snapshot() {
    let monitor = default_monitor();
    monitor.update_from_metrics("node-0", 0, 0.1);
    monitor.update_from_metrics("node-1", 20, 0.5);

    let levels = monitor.all_load_levels();
    assert_eq!(levels.len(), 2);
    assert_eq!(levels["node-0"], LoadLevel::Low);
    assert_eq!(levels["node-1"], LoadLevel::Critical);
}

#[test]
fn remove_node() {
    let monitor = default_monitor();
    monitor.update_from_metrics("node-0", 5, 0.5);
    assert!(monitor.get_load_level("node-0").is_some());

    monitor.remove_node("node-0");
    assert!(monitor.get_load_level("node-0").is_none());
}

#[test]
fn overflow_policy_default_is_redirect() {
    let monitor = default_monitor();
    assert_eq!(monitor.overflow_policy(), BackpressurePolicy::Redirect);
}

#[test]
fn custom_config_thresholds() {
    let config = BackpressureConfig {
        high_watermark: 4,
        critical_watermark: 8,
        memory_high_threshold: 0.5,
        memory_critical_threshold: 0.8,
        overflow_policy: BackpressurePolicy::Drop,
    };
    let monitor = BackpressureMonitor::new(config);

    monitor.update_from_metrics("node-0", 5, 0.3);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::High));

    monitor.update_from_metrics("node-0", 9, 0.3);
    assert_eq!(monitor.get_load_level("node-0"), Some(LoadLevel::Critical));

    assert_eq!(monitor.overflow_policy(), BackpressurePolicy::Drop);
}
