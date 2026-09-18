// Copyright © 2025 Apple Inc.
//
// CUDA patch: do not throw from ~CudaHandle while the driver is unloading
//
// Modified from upstream MLX 81ba1c6a mlx/backend/cuda/cuda_utils.h
//
// Changes to ~CudaHandle():
//   - Release the handle directly instead of calling reset(), which throws.
//
// Submitted upstream as inureyes:fix/cuda-handle-destructor-no-throw; drop
// this overlay once it lands and the MLX pin moves past it. Rationale and
// measurements: docs/upstream/mlx-cuda-handle-destructor-throws-at-teardown.md
// (lablup/mlxcel#1422).

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace mlx::core {

// Throw exception if the cuda API does not succeed.
void check_cuda_error(const char* name, cudaError_t err);
void check_cuda_error(const char* name, CUresult err);

// The macro version that prints the command that failed.
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

// Base class for RAII managed CUDA resources.
template <typename Handle, cudaError_t (*Destroy)(Handle)>
class CudaHandle {
 public:
  CudaHandle(Handle handle = nullptr) : handle_(handle) {}

  CudaHandle(CudaHandle&& other) : handle_(other.handle_) {
    assert(this != &other);
    other.handle_ = nullptr;
  }

  ~CudaHandle() {
    // Skip if there was an error to avoid throwing in the destructors
    if (cudaPeekAtLastError() != cudaSuccess) {
      return;
    }
    // Not reset(): it throws via CHECK_CUDA_ERROR, and a throw escaping a
    // destructor terminates. At exit the runtime may already be unloading, so
    // Destroy fails and the handle can no longer be reclaimed or reported on.
    if (handle_ != nullptr) {
      Destroy(handle_);
      handle_ = nullptr;
    }
  }

  CudaHandle(const CudaHandle&) = delete;
  CudaHandle& operator=(const CudaHandle&) = delete;

  CudaHandle& operator=(CudaHandle&& other) {
    assert(this != &other);
    reset();
    std::swap(handle_, other.handle_);
    return *this;
  }

  void reset() {
    if (handle_ != nullptr) {
      CHECK_CUDA_ERROR(Destroy(handle_));
      handle_ = nullptr;
    }
  }

  Handle release() {
    Handle handle = handle_;
    handle_ = nullptr;
    return handle;
  }

  operator Handle() const {
    return handle_;
  }

 protected:
  Handle handle_;
};

namespace cu {
class Device;
}; // namespace cu

// Wrappers of CUDA resources.
class CudaGraph : public CudaHandle<cudaGraph_t, cudaGraphDestroy> {
 public:
  using CudaHandle::CudaHandle;
  explicit CudaGraph(cu::Device& device);
  void end_capture(cudaStream_t stream);
};

class CudaGraphExec : public CudaHandle<cudaGraphExec_t, cudaGraphExecDestroy> {
 public:
  void instantiate(cudaGraph_t graph);
};

class CudaStream : public CudaHandle<cudaStream_t, cudaStreamDestroy> {
 public:
  using CudaHandle::CudaHandle;
  explicit CudaStream(cu::Device& device);
};

} // namespace mlx::core
