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

#ifndef MLXCEL_XLA_LOCAL_TASK_H_
#define MLXCEL_XLA_LOCAL_TASK_H_

#include <stdbool.h>
#include <stdint.h>

#include <iree/runtime/api.h>

// Worker stack size of the programmatic local-task executor (issue #2790).
// IREE's flag default (128 KiB) is smaller than the static TLS of a process
// that has loaded the CUDA or ROCm runtime libraries, and glibc then refuses
// the worker threads with EINVAL ("thread creation failed with 22", #945).
#define XLA_LOCAL_TASK_WORKER_STACK_SIZE ((uint64_t)8 * 1024 * 1024)

// True when `device_uri` names IREE's local-task driver ("local-task" or
// "local-task://...").
bool xla_device_uri_is_local_task(const char* device_uri);

// Open the HAL device for `device_uri`.
//
// local-task is built here, without IREE's global flag registry: one
// executor with XLA_LOCAL_TASK_WORKER_STACK_SIZE worker stacks and either
// `local_task_workers` unpinned workers or, when it is 0, one worker per
// physical core of the calling thread's NUMA node. Every other URI goes
// through the instance's driver registry unchanged.
iree_status_t xla_open_device(iree_runtime_instance_t* instance,
                              const char* device_uri,
                              int32_t local_task_workers,
                              iree_hal_device_t** out_device);

#endif  // MLXCEL_XLA_LOCAL_TASK_H_
