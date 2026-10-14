// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"

#include <stdexcept>

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  rocm::AtomicEvent event;
};

Fence::Fence(Stream s) {
  fence_ = std::shared_ptr<void>(
      new FenceImpl{0}, [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  // The fence is signaled by a host callback on the updating stream; if that
  // stream has failed the callback never runs, and the wait reports the
  // failure instead of blocking forever (lablup/mlxcel#1804). The throw
  // leaves eval_impl through its catch block like any primitive error.
  hipError_t st = fence->event.wait(fence->count);
  if (st != hipSuccess) {
    throw std::runtime_error(
        rocm::describe_device_error(st, "waiting for a stream fence"));
  }
}

void Fence::update(Stream s, const array&, bool cross_device) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  fence->event.signal(s, fence->count);
}

} // namespace mlx::core
