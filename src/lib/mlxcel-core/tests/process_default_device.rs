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

//! `set_process_default_device` reaches threads spawned after it, which is
//! what `--device cpu` relies on now that `set_default_device` moves only the
//! calling thread. Its own binary, because it changes the process default.

#[test]
fn the_process_default_reaches_later_threads_and_a_thread_override_wins() {
    if !mlxcel_core::gpu_backend_available() {
        return;
    }
    mlxcel_core::set_process_default_device(false);
    assert!(!mlxcel_core::default_device_is_gpu());
    let spawned = std::thread::spawn(mlxcel_core::default_device_is_gpu)
        .join()
        .expect("thread");
    assert!(
        !spawned,
        "a thread spawned after the CPU process default ran on the GPU"
    );

    let overridden = std::thread::spawn(|| {
        mlxcel_core::set_default_device(true);
        mlxcel_core::default_device_is_gpu()
    })
    .join()
    .expect("thread");
    assert!(
        overridden,
        "a thread's own default did not win over the process default"
    );

    mlxcel_core::set_process_default_device(true);
    assert!(mlxcel_core::default_device_is_gpu());
}
