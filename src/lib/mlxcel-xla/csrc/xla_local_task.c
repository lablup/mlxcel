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

// Programmatic IREE local-task device (issue #2790).
//
// `iree_runtime_instance_try_create_default_device("local-task")` reaches the
// local-task driver factory, which builds its executors from IREE's global
// flag registry (`iree_task_executors_create_from_flags`). Its default worker
// stack (128 KiB) cannot hold the static TLS of a process that has loaded
// CUDA (about 235 KB on GB10 with cuBLASLt, OpenBLAS and cuSOLVER), so
// `pthread_create` fails with EINVAL. Overriding the flag would mean parsing
// into that process-global registry, which the production shim never does.
// Instead this file mirrors the factory in
// `iree/hal/drivers/local_task/registration/driver_module.c` with an executor
// configured here.

// Same system allocator as the rest of the shim (see xla_iree.c, which
// defines `iree_xla_libc_ctl`); it must be set before the IREE headers.
#define IREE_ALLOCATOR_SYSTEM_CTL iree_xla_libc_ctl

#include "xla_local_task.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iree/async/util/proactor_pool.h>
#include <iree/base/threading/numa.h>
#include <iree/base/tooling/flags.h>
#include <iree/hal/drivers/local_task/task_driver.h>
#include <iree/hal/local/loaders/registration/init.h>
#include <iree/task/api.h>

// IREE's `--task_topology_max_group_count` default: the cap on workers when
// the count comes from the physical-core query.
#define XLA_LOCAL_TASK_MAX_GROUP_COUNT 64

bool xla_device_uri_is_local_task(const char* device_uri) {
  if (!device_uri) return false;
  static const char kName[] = "local-task";
  static const char kPrefix[] = "local-task://";
  return strcmp(device_uri, kName) == 0 ||
         strncmp(device_uri, kPrefix, sizeof(kPrefix) - 1) == 0;
}

// One executor over `local_task_workers` unpinned workers, or one worker per
// physical core of the current NUMA node when it is 0, each with an
// XLA_LOCAL_TASK_WORKER_STACK_SIZE stack.
static iree_status_t xla_local_task_create_executor(
    int32_t local_task_workers, iree_allocator_t host_allocator,
    iree_task_executor_t** out_executor) {
  *out_executor = NULL;
  if (local_task_workers < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local-task worker count must not be negative");
  }
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_stack_size = (iree_host_size_t)XLA_LOCAL_TASK_WORKER_STACK_SIZE;

  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  iree_status_t status = iree_ok_status();
  if (local_task_workers > 0) {
    iree_task_topology_initialize_from_group_count(
        (iree_host_size_t)local_task_workers, &topology);
  } else {
    status = iree_task_topology_initialize_from_physical_cores(
        iree_task_topology_query_current_node(),
        IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
        IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER,
        XLA_LOCAL_TASK_MAX_GROUP_COUNT, &topology);
  }
  if (iree_status_is_ok(status) && topology.group_count == 0) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "the local-task topology query found no cores");
  }
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create(options, &topology, host_allocator,
                                       out_executor);
  }
  iree_task_topology_deinitialize(&topology);
  return status;
}

// The local-task driver over one executor, built the way IREE's factory
// builds it but without consulting any flag. Every reference this creates is
// released before returning; the driver retains what it keeps.
static iree_status_t xla_local_task_create_driver(
    int32_t local_task_workers, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  *out_driver = NULL;
  iree_task_executor_t* executor = NULL;
  IREE_RETURN_IF_ERROR(xla_local_task_create_executor(
      local_task_workers, host_allocator, &executor));

  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  iree_status_t status = iree_hal_create_all_available_executable_loaders(
      /*plugin_manager=*/NULL, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
      host_allocator);

  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
                                            host_allocator, host_allocator,
                                            &device_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        iree_make_cstring_view("local-task"), &params, /*queue_count=*/1,
        &executor, loader_count, loaders, device_allocator, host_allocator,
        out_driver);
  }

  iree_hal_allocator_release(device_allocator);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_task_executor_release(executor);
  return status;
}

static iree_status_t xla_open_local_task_device(
    int32_t local_task_workers, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(
      xla_local_task_create_driver(local_task_workers, host_allocator, &driver));

  // The same proactor pool `iree_runtime_instance_try_create_default_device`
  // hands every driver; the device retains it.
  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), host_allocator,
      &proactor_pool);
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(driver, &create_params,
                                                   host_allocator, out_device);
  }
  iree_async_proactor_pool_release(proactor_pool);
  iree_hal_driver_release(driver);
  return status;
}

iree_status_t xla_open_device(iree_runtime_instance_t* instance,
                              const char* device_uri,
                              int32_t local_task_workers,
                              iree_hal_device_t** out_device) {
  *out_device = NULL;
  if (!xla_device_uri_is_local_task(device_uri)) {
    return iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view(device_uri), out_device);
  }
  iree_status_t status = xla_open_local_task_device(
      local_task_workers, iree_runtime_instance_host_allocator(instance),
      out_device);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(
        status,
        "[runtime-init] failed to build the local-task device (%" PRId32
        " requested workers, 0 = one per physical core; "
        "MLXCEL_XLA_LOCAL_TASK_WORKERS sets it)",
        local_task_workers);
  }
  return iree_ok_status();
}

// Test support for issue #2790: build the local-task executor exactly as
// `xla_open_device` does and report its worker count and the stack size it
// was given, plus the `--task_worker_stack_size` value IREE's global flag
// registry holds afterwards (-1 when the registry is compiled out or the
// flag is not listed). Returns 0 on success, the IREE status code otherwise.
int xla_local_task_probe(int32_t local_task_workers, uint64_t* out_stack_size,
                         uint64_t* out_worker_count,
                         int64_t* out_flag_stack_size) {
  if (!out_stack_size || !out_worker_count || !out_flag_stack_size) return 1;
  *out_stack_size = 0;
  *out_worker_count = 0;
  *out_flag_stack_size = -1;
  iree_task_executor_t* executor = NULL;
  iree_status_t status = xla_local_task_create_executor(
      local_task_workers, iree_allocator_system(), &executor);
  if (!iree_status_is_ok(status)) {
    int code = (int)iree_status_code(status);
    fputs("xla_local_task_probe failed: ", stderr);
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return code == 0 ? 1 : code;
  }
  *out_stack_size = XLA_LOCAL_TASK_WORKER_STACK_SIZE;
  *out_worker_count = (uint64_t)iree_task_executor_worker_count(executor);
  iree_task_executor_release(executor);

  char* dump = NULL;
  size_t dump_size = 0;
  FILE* file = open_memstream(&dump, &dump_size);
  if (file) {
    iree_flags_dump(IREE_FLAG_DUMP_MODE_DEFAULT, file);
    fclose(file);
    static const char kFlag[] = "--task_worker_stack_size=";
    const char* found = dump ? strstr(dump, kFlag) : NULL;
    if (found) {
      *out_flag_stack_size = strtoll(found + sizeof(kFlag) - 1, NULL, 10);
    }
  }
  free(dump);
  return 0;
}
