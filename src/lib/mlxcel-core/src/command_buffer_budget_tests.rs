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
use std::sync::Mutex;

// The override is one process-wide slot, so the tests that write it must not
// interleave with each other.
static SLOT: Mutex<()> = Mutex::new(());

#[cfg(feature = "metal")]
#[test]
fn override_round_trips_through_the_device_overlay() {
    let _lock = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    ffi::set_metal_mb_per_buffer_override(777);
    assert_eq!(ffi::metal_mb_per_buffer_override(), 777);
    // Non-positive values mean "no override", not a zero budget: a zero cap
    // would commit a buffer after every op.
    ffi::set_metal_mb_per_buffer_override(-5);
    assert_eq!(ffi::metal_mb_per_buffer_override(), 0);
    ffi::set_metal_mb_per_buffer_override(0);
    assert_eq!(ffi::metal_mb_per_buffer_override(), 0);
}

#[cfg(feature = "metal")]
#[test]
fn guard_applies_the_budget_and_restores_what_it_found() {
    let _lock = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    ffi::set_metal_mb_per_buffer_override(0);
    {
        let outer = DecodeCommandBufferBudget::with_budget(Some(1000));
        assert!(outer.is_active());
        assert_eq!(ffi::metal_mb_per_buffer_override(), 1000);
        {
            let inner = DecodeCommandBufferBudget::with_budget(Some(400));
            assert!(inner.is_active());
            assert_eq!(ffi::metal_mb_per_buffer_override(), 400);
        }
        assert_eq!(ffi::metal_mb_per_buffer_override(), 1000);
    }
    assert_eq!(ffi::metal_mb_per_buffer_override(), 0);
}

#[test]
fn guard_without_a_budget_leaves_the_slot_alone() {
    let _lock = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    ffi::set_metal_mb_per_buffer_override(0);
    let before = ffi::metal_mb_per_buffer_override();
    {
        let guard = DecodeCommandBufferBudget::with_budget(None);
        assert!(!guard.is_active());
        assert_eq!(ffi::metal_mb_per_buffer_override(), before);
    }
    assert_eq!(ffi::metal_mb_per_buffer_override(), before);
}
