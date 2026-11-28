// Copyright © 2023-2026 Apple Inc.
// Patched by mlxcel: the default device is per thread.
// Modified from upstream MLX 9f806b3606eaa9f7c807238f6bd465185cd94d6d mlx/device.cpp
// Synced to upstream 9f806b36. The delta is the block marked `mlxcel:` (the
// default-device storage, `default_device`, `set_default_device`, and the added
// `mlxcel_set_process_default_device`); `is_available`, `device_count` and
// `device_info` are upstream verbatim.
//
// Why it exists. Upstream keeps one process-wide default device but gives
// every thread its own default stream per device (mlx/stream.cpp). An op built
// without an explicit stream takes the default stream of the default device,
// so one thread moving the default to the CPU silently moves every other
// thread's ops there too, mid-graph. mlxcel's test binaries run tests on
// parallel threads and several of them pin the CPU, which made unrelated GPU
// tests compare a CPU result with a GPU reference. Here `set_default_device`
// moves only the calling thread. A thread that never set one follows the
// process default, which `mlxcel_set_process_default_device` sets (the runtime
// does that once at startup for `--device cpu`). Refresh on a pin bump: take
// the new upstream file and re-apply the marked block. The ROCm overlay
// replaces this file with its own copy, which carries the same block.

#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>

#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"

namespace mlx::core {

// mlxcel: begin per-thread default device.
namespace {

Device initial_default_device() {
  return Device{gpu::is_available() ? Device::gpu : Device::cpu};
}

// The process default, packed as (type << 16) | index; -1 until set.
std::atomic<int> process_default_device{-1};

Device process_default() {
  int packed = process_default_device.load(std::memory_order_acquire);
  if (packed < 0) {
    return initial_default_device();
  }
  return Device{
      static_cast<Device::DeviceType>(packed >> 16), packed & 0xffff};
}

void check_settable(const Device& d, const char* caller) {
  if (!gpu::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        std::string("[") + caller +
        "] Cannot set gpu device without gpu backend.");
  }
}

thread_local std::optional<Device> thread_default_device;

} // namespace

Device& mutable_default_device() {
  // The returned reference is the calling thread's own slot, refreshed from
  // the override or the process default on every call.
  static thread_local Device view{Device::cpu};
  view = thread_default_device.value_or(process_default());
  return view;
}

const Device& default_device() {
  return mutable_default_device();
}

void set_default_device(const Device& d) {
  check_settable(d, "set_default_device");
  thread_default_device = d;
}

// Sets the default for every thread that has not set its own, and clears the
// calling thread's override so it follows the new default as well.
MLX_API void mlxcel_set_process_default_device(const Device& d) {
  check_settable(d, "mlxcel_set_process_default_device");
  process_default_device.store(
      (static_cast<int>(d.type) << 16) | (d.index & 0xffff),
      std::memory_order_release);
  thread_default_device.reset();
}
// mlxcel: end per-thread default device.

bool is_available(const Device& d) {
  switch (d.type) {
    case Device::cpu:
      return cpu::is_available() && (d.index < cpu::device_count());
    case Device::gpu:
      return gpu::is_available() && (d.index < gpu::device_count());
  }
  // appease compiler
  return false;
}

int device_count(Device::DeviceType type) {
  switch (type) {
    case Device::cpu:
      return cpu::device_count();
    case Device::gpu:
      return gpu::device_count();
  }
  // appease compiler
  return 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(const Device& d) {
  switch (d.type) {
    case Device::cpu:
      return cpu::device_info(d.index);
    case Device::gpu:
      return gpu::device_info(d.index);
  }
  // appease compiler
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;
  return empty;
}

} // namespace mlx::core
