// Copyright © 2025 Apple Inc.

// This file include utilities that are used by C++ code (i.e. .cpp files).

#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdio>

namespace mlx::core {

namespace rocm {
class Device;
}

struct Dtype;

// Throw exception if the HIP API does not succeed.
void check_rocblas_error(const char* name, rocblas_status err);
void check_hip_error(const char* name, hipError_t err);

// The macro version that prints the command that failed.
#define CHECK_ROCBLAS_ERROR(cmd) check_rocblas_error(#cmd, (cmd))
#define CHECK_HIP_ERROR(cmd) check_hip_error(#cmd, (cmd))

// Convert Dtype to HIP C++ types.
const char* dtype_to_hip_type(const Dtype& dtype);

// Base class for RAII managed HIP resources.
template <typename Handle, hipError_t (*Destroy)(Handle)>
class HipHandle {
 public:
  HipHandle(Handle handle = nullptr) : handle_(handle) {}

  HipHandle(HipHandle&& other) : handle_(other.handle_) {
    assert(this != &other);
    other.handle_ = nullptr;
  }

  ~HipHandle() {
    reset();
  }

  HipHandle(const HipHandle&) = delete;
  HipHandle& operator=(const HipHandle&) = delete;

  HipHandle& operator=(HipHandle&& other) {
    assert(this != &other);
    reset();
    std::swap(handle_, other.handle_);
    return *this;
  }

  void reset() {
    if (handle_ != nullptr) {
      // Destructor path: report, never throw. Once the device has faulted
      // every destroy returns the fault, and a throw from a destructor would
      // terminate the process while it is already failing or exiting
      // (lablup/mlxcel#1804). One line per handle type, not per handle.
      hipError_t status = Destroy(handle_);
      handle_ = nullptr;
      if (status != hipSuccess) {
        static bool reported = false;
        if (!reported) {
          reported = true;
          fprintf(
              stderr,
              "[mlx-rocm] releasing a HIP handle failed: %s\n",
              hipGetErrorString(status));
        }
      }
    }
  }

  operator Handle() const {
    return handle_;
  }

 protected:
  Handle handle_;
};

// Wrappers of HIP resources.
class HipGraph : public HipHandle<hipGraph_t, hipGraphDestroy> {
 public:
  using HipHandle::HipHandle;
  explicit HipGraph(rocm::Device& device);
  void end_capture(hipStream_t stream);
};

class HipGraphExec : public HipHandle<hipGraphExec_t, hipGraphExecDestroy> {
 public:
  void instantiate(hipGraph_t graph);
};

class HipStream : public HipHandle<hipStream_t, hipStreamDestroy> {
 public:
  explicit HipStream(rocm::Device& device);
};

} // namespace mlx::core
