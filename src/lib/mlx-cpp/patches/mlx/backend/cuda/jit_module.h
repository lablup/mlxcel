// Copyright © 2025 Apple Inc.
//
// Modified by mlxcel (lablup/mlxcel#1566): serialize the per-kernel lazy
// configuration in JitModule::get_kernel_and_dims.
//
// Upstream stores each kernel as (CUfunction, configured_flag, max_block_dim)
// and fills the last two on first use with no lock, while publishing the flag
// *before* the dimension. A second thread that observes the flag already set
// skips the block and reads max_block_dim while it is still zero, which reaches
// get_launch_args as max_block_dim = 0 and divides by zero in the grid
// computation. The kernels_ map itself is safe: it is filled once in the
// constructor under the write lock in get_jit_module and never restructured.
// Only the two mapped fields race, so a per-module mutex around the lazy
// initialization is sufficient. Everything else is byte-identical upstream
// (pin 9a795735).

#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"

#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>

#include <cuda.h>
#include <fmt/format.h>

namespace mlx::core::cu {

class Device;

using KernelBuilderResult = std::tuple<
    /* precompiled */ bool,
    /* source code */ std::string,
    /* kernel names */ std::vector<std::string>>;
using KernelBuilder = std::function<KernelBuilderResult()>;

struct KernelArgs {
  void** args() {
    return args_.data();
  }

  void append(const array& a) {
    append(reinterpret_cast<CUdeviceptr>(gpu_ptr<void>(a)));
  }

  template <typename T>
  void append(T val) {
    storage_.emplace_back(val);
    append_ptr(&storage_.back());
  }

  template <typename T>
  void append(SmallVector<T> vec) {
    storage_.emplace_back(std::move(vec));
    append_ptr(std::get<SmallVector<T>>(storage_.back()).data());
  }

  template <typename T>
  void append(const std::vector<T>& vec) {
    append(SmallVector<T>(vec.begin(), vec.end()));
  }

  // Make sure the arg is copied to an array with size of NDIM.
  template <size_t NDIM = MAX_NDIM, typename T>
  void append_ndim(SmallVector<T> vec) {
    if (vec.size() > NDIM) {
      throw std::runtime_error(
          fmt::format("ndim can not be larger than {}.", NDIM));
    }
    vec.resize(NDIM);
    append(std::move(vec));
  }

  void append_ptr(const void* v) {
    args_.push_back(const_cast<void*>(v));
  }

 private:
  std::vector<void*> args_;

  // The cuGraphAddKernelNode API requires passing pointers to arguments so
  // store temporary values until the node is created.
  using Arg = std::variant<
      std::monostate,
      CUdeviceptr,
      bool,
      int32_t,
      uint32_t,
      int64_t,
      float,
      SmallVector<const void*>,
      SmallVector<int32_t>,
      SmallVector<int64_t>>;
  std::deque<Arg> storage_;
};

class JitModule {
 public:
  JitModule(
      Device& device,
      const std::string& module_name,
      const KernelBuilder& builder,
      bool use_disk_cache);
  ~JitModule();

  JitModule(const JitModule&) = delete;
  JitModule& operator=(const JitModule&) = delete;

  CUfunction get_kernel(
      const std::string& kernel_name,
      std::function<void(CUfunction)> configure_kernel = nullptr);
  std::pair<CUfunction, uint32_t> get_kernel_and_dims(
      const std::string& kernel_name,
      std::function<void(CUfunction)> configure_kernel = nullptr);

 private:
  CUmodule module_{nullptr};
  std::unordered_map<std::string, std::tuple<CUfunction, bool, uint32_t>>
      kernels_;
  // [mlxcel #1566] Guards the lazy (configured_flag, max_block_dim) fill in
  // get_kernel_and_dims. Held only around that initialization, which happens
  // once per kernel; steady-state launches take an uncontended lock.
  std::mutex kernels_mtx_;
};

JitModule& get_jit_module(
    Device& device,
    const std::string& name,
    const KernelBuilder& builder,
    bool use_disk_cache = true);

} // namespace mlx::core::cu
