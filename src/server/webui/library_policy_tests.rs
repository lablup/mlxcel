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

use super::{WebUiServerMode, project};

#[test]
fn capability_matrix_is_pure_and_fail_closed() {
    for mode in [
        WebUiServerMode::SingleModel,
        WebUiServerMode::ModelFree,
        WebUiServerMode::RouterPool,
    ] {
        for cache in [false, true] {
            for offline in [false, true] {
                for platform in [false, true] {
                    for download in [false, true] {
                        let value = project(mode, cache, download, offline, platform);
                        let enabled = mode != WebUiServerMode::SingleModel
                            && cache
                            && platform
                            && !(download && offline);
                        assert_eq!(value.state == "enabled", enabled);
                        assert_eq!(value.reason.is_none(), enabled);
                        assert_eq!(value.instructions.is_none(), enabled);
                        if mode == WebUiServerMode::SingleModel {
                            assert_eq!(value.state, "read_only");
                        }
                    }
                }
            }
        }
    }
}
