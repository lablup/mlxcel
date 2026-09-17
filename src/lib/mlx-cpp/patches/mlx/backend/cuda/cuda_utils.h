// Copyright © 2025 Apple Inc.

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
    // LOCAL PATCH (lablup/mlxcel#1422): release without CHECK_CUDA_ERROR.
    //
    // `reset()` routes through CHECK_CUDA_ERROR, which throws. Throwing from a
    // destructor calls std::terminate, and at thread or process teardown the
    // CUDA runtime may already be unloading, so every Destroy fails with
    // cudaErrorCudartUnloading and the throw is guaranteed rather than
    // unlikely. That is the `Destroy(handle_) failed: driver shutting down`
    // SIGABRT seen on GB10 after a suite had already printed `test result: ok`.
    //
    // The existing cudaPeekAtLastError guard above does not cover it: driver
    // unloading is not a sticky per-context error, so the peek returns
    // cudaSuccess and reset() proceeds into the throw.
    //
    // The handle cannot be reclaimed once the driver is gone, so there is
    // nothing to recover and nothing to report. `reset()` keeps its checking
    // behavior for every non-destructor caller (`operator=`, explicit calls),
    // where throwing is legal and useful.
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
