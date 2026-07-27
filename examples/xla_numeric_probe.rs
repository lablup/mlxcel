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

fn main() -> Result<(), String> {
    let device = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "local-task".to_string());
    let reports = mlxcel_xla::run_core_operator_probes(&device)?;
    let json = serde_json::to_string_pretty(&reports)
        .map_err(|error| format!("serialize numeric probe reports: {error}"))?;
    println!("{json}");
    let failed = reports
        .iter()
        .filter(|report| !report.passed())
        .map(|report| report.operation())
        .collect::<Vec<_>>();
    if !failed.is_empty() {
        return Err(format!(
            "canonical-decomposition probes diverged: {}",
            failed.join(", ")
        ));
    }
    if reports.iter().any(|report| report.production_qualified()) {
        return Err(
            "canonical-decomposition probes must never claim production qualification".to_string(),
        );
    }
    Ok(())
}
