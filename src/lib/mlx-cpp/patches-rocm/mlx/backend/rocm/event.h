// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/stream.h"

#include <atomic>
#include <memory>

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// RAII-managed move-only wrapper of hipEvent_t.
struct HipEventHandle : public HipHandle<hipEvent_t, hipEventDestroy> {
  HipEventHandle(int flags);
  int flags;
  // The HIP device the event was created on. A hipEvent is bound to its device:
  // recording it on a stream of a DIFFERENT device is invalid and on a
  // multi-GPU host hangs the queue. The pool must hand back an event from the
  // right device.
  int device{0};
};

// Wrapper of native HIP event. It can synchronize between GPU streams, or wait
// on GPU stream in CPU stream, but can not wait on CPU stream.
//
// Host waits and queries report the HIP status instead of assuming success:
// once the queue behind the event has faulted, hipEventQuery returns the fault
// (hipErrorIllegalAddress and friends) for the rest of the process, and a loop
// that only asks "is it hipSuccess yet" spins forever (lablup/mlxcel#1804).
class HipEvent {
 public:
  explicit HipEvent(int flags);
  ~HipEvent();

  HipEvent(HipEvent&&) = default;
  HipEvent& operator=(HipEvent&&) = default;

  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;

  // Block the host until the event completes. Returns hipSuccess, the status
  // the runtime reports once the stream has failed, or hipErrorLaunchTimeOut
  // when MLX_ROCM_GPU_WATCHDOG_SECS is set and the wait outlived it.
  [[nodiscard]] hipError_t wait();
  // Make |stream| wait for the event. Throws on failure.
  void wait(hipStream_t stream);
  // Record the event on |stream|. Throws on failure.
  void record(hipStream_t stream);

  // hipSuccess once the recorded kernels have completed, hipErrorNotReady while
  // they run, and the stream's failure status once it has failed. Note that
  // this method returns hipSuccess if record() has not been called.
  [[nodiscard]] hipError_t query() const;

 private:
  HipEventHandle event_;
};

// Event that can synchronize between CPU and GPU. It is much slower than
// HipEvent so the latter should always be preferred when possible.
//
// The GPU side signals it through a hipLaunchHostFunc callback, and a stream
// that has faulted never runs its callbacks, so a host waiter would otherwise
// block forever. wait() therefore polls the signaling stream's status while it
// waits, and signal() marks the event failed when the callback cannot even be
// queued (lablup/mlxcel#1804).
class AtomicEvent {
 public:
  AtomicEvent();

  // Block until the counter reaches |value|. Returns hipSuccess, the failure
  // status of the signaling stream once the signal can no longer arrive, or
  // hipErrorLaunchTimeOut when MLX_ROCM_GPU_WATCHDOG_SECS is set and the wait
  // outlived it. Only a GPU-stream signaler is polled: an event signaled from
  // a CPU stream has no HIP status to fail with.
  [[nodiscard]] hipError_t wait(uint64_t value);
  [[nodiscard]] hipError_t wait(hipStream_t stream, uint64_t value);
  // GPU-stream variant of wait(): commits |s| first and keeps the counter
  // alive until the stream drains. |s| must be a GPU stream.
  [[nodiscard]] hipError_t wait(Stream s, uint64_t value);
  void signal(uint64_t value);
  void signal(hipStream_t stream, uint64_t value);
  // GPU-stream variant of signal(). |s| must be a GPU stream.
  void signal(Stream s, uint64_t value);
  bool is_signaled(uint64_t value) const;
  // hipSuccess while the signal can still arrive, otherwise the failure status
  // of the signaling stream. Cheap when nothing has been signaled from a GPU
  // stream yet; one hipStreamQuery otherwise.
  [[nodiscard]] hipError_t status() const;
  uint64_t value() const;

  // State shared by every copy of the event. The counter lives in PINNED HOST
  // memory, not device memory. The GPU writes it and the CPU polls it (wait()).
  // Device memory, even fine-grained, is not reliably CPU-coherent on a
  // discrete GPU over a non-coherent link (e.g. an R9700 in a TB5 eGPU
  // enclosure), so the host poll would spin forever. Pinned host memory is the
  // canonical GPU->host signaling path and works on both the integrated APU
  // and a discrete dGPU.
  struct State {
    State();
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    std::atomic<uint64_t>* counter{nullptr};
    // The GPU stream the event was last signaled from, so a waiter can poll
    // whether that stream is still able to deliver the signal.
    std::atomic<hipStream_t> stream{nullptr};
    // hipSuccess, or the status that stopped the signal from being queued.
    std::atomic<int> error{static_cast<int>(hipSuccess)};
  };

 private:
  std::atomic<uint64_t>* atomic() const {
    return state_->counter;
  }

  std::shared_ptr<State> state_;
};

} // namespace mlx::core::rocm
