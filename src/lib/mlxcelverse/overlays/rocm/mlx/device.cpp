// Copyright © 2023-2026 Apple Inc.
// mlxcel: the per-thread default-device block below is shared with
// overlays/core/mlx/device.cpp; keep the two identical apart from
// `initial_default_device`.

#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>

#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"

#ifdef MLX_USE_ROCM
#include "mlx/backend/rocm/rocm.h"
#endif

namespace mlx::core {

// mlxcel: begin per-thread default device.
namespace {

Device initial_default_device() {
  Device::DeviceType default_type = Device::cpu;
  if (gpu::is_available()) {
    default_type = Device::gpu;
  }
#ifdef MLX_USE_ROCM
  else if (rocm::is_available()) {
    default_type = Device::gpu; // ROCm devices use the generic gpu type
  }
#endif
  return Device{default_type};
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
#ifdef MLX_USE_ROCM
      return (gpu::is_available() || rocm::is_available()) &&
          (d.index < gpu::device_count());
#else
      return gpu::is_available() && (d.index < gpu::device_count());
#endif
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
