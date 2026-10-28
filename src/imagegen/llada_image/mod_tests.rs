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

use super::*;

#[test]
fn the_family_is_registered_and_probed() {
    assert!(
        crate::imagegen::registry::FAMILIES
            .iter()
            .any(|f| f.name == FAMILY)
    );
    assert!(crate::imagegen::probe::IMAGE_FAMILIES.contains(&FAMILY));
}

#[test]
fn loading_a_foreign_checkpoint_fails_by_name() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("model_index.json"),
        r#"{"_class_name": "Flux2Pipeline"}"#,
    )
    .unwrap();
    let err = load(dir.path())
        .err()
        .expect("not a LLaDA-Image checkpoint");
    assert!(
        format!("{err:#}").contains("Not a LLaDA-Image checkpoint"),
        "{err:#}"
    );
}
