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

#[test]
fn scheduler_modules_stay_below_documented_anti_pattern_threshold() {
    const MAX_LINES_WITHOUT_JUSTIFICATION: usize = 2_000;

    let batch_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/batch");
    let scheduler_dir = batch_dir.join("scheduler");
    let mut paths = vec![batch_dir.join("scheduler.rs")];
    let entries = std::fs::read_dir(&scheduler_dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", scheduler_dir.display()));
    let mut oversized = Vec::new();
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        paths.push(path);
    }

    for path in paths {
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
        let line_count = source.lines().count();
        if line_count > MAX_LINES_WITHOUT_JUSTIFICATION {
            oversized.push(format!("{} has {line_count} lines", path.display()));
        }
    }

    assert!(
        oversized.is_empty(),
        "scheduler module files must stay at or below {MAX_LINES_WITHOUT_JUSTIFICATION} lines unless docs/code-guidelines.md records a specific exception: {}",
        oversized.join(", ")
    );
}
