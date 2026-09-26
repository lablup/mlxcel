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

//! Check real Metal runtime controls, including a plain default-feature build.
//! Run: cargo run -p mlxcel-core --profile test-fast --example metal_runtime_switch_probe

fn main() {
    if !mlxcel_core::metal_is_available() {
        println!("Metal unavailable; runtime-control check skipped");
        return;
    }

    struct Restore {
        wide: bool,
        budget: i32,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            mlxcel_core::set_qmv_wide(self.wide);
            mlxcel_core::set_metal_mb_per_buffer_override(self.budget);
        }
    }
    let _restore = Restore {
        wide: mlxcel_core::qmv_wide_enabled(),
        budget: mlxcel_core::metal_mb_per_buffer_override(),
    };
    for enabled in [false, true] {
        mlxcel_core::set_qmv_wide(enabled);
        let observed = mlxcel_core::qmv_wide_enabled();
        println!("qmv_wide requested={enabled} observed={observed}");
        assert_eq!(
            observed, enabled,
            "Metal QMV runtime switch must not be a stub"
        );
    }
    mlxcel_core::set_metal_mb_per_buffer_override(17);
    assert_eq!(mlxcel_core::metal_mb_per_buffer_override(), 17);
    println!("Metal runtime controls passed");
}
