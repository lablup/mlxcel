// Copyright © 2025 Apple Inc.
//
// Modified by mlxcel (lablup/mlxcel#1564):
// - Weights are uploaded through a small, reused pool of four 64 MiB pinned
//   host staging slots per device instead of calling `hipHostMalloc` and
//   queuing a `hipLaunchHostFunc(..., hip_host_free_callback, ...)` per tensor.
//   The read into a slot overlaps the host-to-device copy out of the previous
//   one, per-tensor `hipHostMalloc`/`hipHostFree` KFD page-pinning churn is
//   gone, and pinned staging memory stays bounded at 256 MiB per device even
//   when `eval_all` materializes an entire checkpoint at once.
// - The synchronous pageable path is kept as the fallback: it is taken when
//   `MLXCEL_ROCM_PINNED_LOAD=0` (or `MLXCEL_CUDA_PINNED_LOAD=0`), or for the
//   rest of the process once a pinned slot allocation fails. An async H2D copy
//   from pageable memory is avoided because driver-internal staging over a
//   non-coherent link (TB5 eGPU) can stall the stream.
// - Process-wide counters (`load_staging.h`) report the host staging leg and
//   the copy back-pressure separately, matching the CUDA backend.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/load_staging.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

void swap_endianness_inplace(void* data, size_t size, size_t itemsize) {
  auto* bytes = reinterpret_cast<uint8_t*>(data);
  switch (itemsize) {
    case 2:
      swap_endianness<2>(bytes, size);
      break;
    case 4:
      swap_endianness<4>(bytes, size);
      break;
    case 8:
      swap_endianness<8>(bytes, size);
      break;
  }
}

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - start)
          .count());
}

struct Counters {
  std::atomic<uint64_t> tensors{0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> pinned_tensors{0};
  std::atomic<uint64_t> pageable_tensors{0};
  std::atomic<uint64_t> memcpy_calls{0};
  std::atomic<uint64_t> read_ns{0};
  std::atomic<uint64_t> wait_ns{0};
  std::atomic<uint64_t> total_ns{0};
};

Counters& counters() {
  static Counters c;
  return c;
}

// Four 64 MiB slots keep two reads and two copies in flight. 64 MiB is two of
// `ParallelFileReader`'s 32 MiB batches, so a full-slot read is split across
// its reader pool instead of running on one thread. A tensor larger than a slot
// is uploaded in slot-sized chunks.
constexpr size_t kSlotBytes = size_t(64) << 20;
constexpr int kSlotCount = 4;

bool pinned_load_enabled() {
  static bool enabled = [] {
    const char* v = std::getenv("MLXCEL_ROCM_PINNED_LOAD");
    if (!v) {
      v = std::getenv("MLXCEL_CUDA_PINNED_LOAD");
    }
    return !(v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
                   std::strcmp(v, "off") == 0));
  }();
  return enabled;
}

} // namespace

namespace mlx::core {

namespace {

// One pool per HIP device: an event can only be recorded on a stream of the
// device it was created on. Slots are allocated on first use and live for the
// rest of the process, like MLX's own allocator; they are deliberately never
// freed, so no destructor runs after the HIP runtime tears down at exit.
class StagingPool {
 public:
  bool ensure_ready() {
    if (state_ == State::Ready) {
      return true;
    }
    if (state_ == State::Unavailable) {
      return false;
    }
    for (int i = 0; i < kSlotCount; ++i) {
      Slot slot;
      const char* failed = nullptr;
      if (hipHostMalloc(&slot.host, kSlotBytes, hipHostMallocPortable) !=
              hipSuccess ||
          slot.host == nullptr) {
        failed = "hipHostMalloc";
      } else if (
          hipEventCreateWithFlags(&slot.done, hipEventDisableTiming) !=
          hipSuccess) {
        failed = "hipEventCreateWithFlags";
      }
      if (failed) {
        (void)hipGetLastError();
        if (slot.host) {
          (void)hipHostFree(slot.host);
        }
        release();
        state_ = State::Unavailable;
        std::cerr << "[mlxcel] pinned weight staging unavailable (" << failed
                  << " failed for " << (kSlotBytes >> 20)
                  << " MiB slots); loading weights through pageable memory."
                  << std::endl;
        return false;
      }
      slots_.push_back(slot);
    }
    state_ = State::Ready;
    return true;
  }

  void upload(
      io::Reader& reader,
      size_t offset,
      size_t nbytes,
      char* dst,
      hipStream_t stream,
      bool swap,
      size_t itemsize) {
    auto& c = counters();
    size_t done = 0;
    while (done < nbytes) {
      size_t n = std::min(kSlotBytes, nbytes - done);
      Slot& slot = slots_[next_];
      next_ = (next_ + 1) % slots_.size();
      if (slot.pending) {
        auto wait_start = Clock::now();
        CHECK_HIP_ERROR(hipEventSynchronize(slot.done));
        c.wait_ns += elapsed_ns(wait_start);
        slot.pending = false;
      }
      auto read_start = Clock::now();
      reader.read(static_cast<char*>(slot.host), n, offset + done);
      if (swap) {
        swap_endianness_inplace(slot.host, n / itemsize, itemsize);
      }
      c.read_ns += elapsed_ns(read_start);
      CHECK_HIP_ERROR(hipMemcpyAsync(
          dst + done, slot.host, n, hipMemcpyHostToDevice, stream));
      CHECK_HIP_ERROR(hipEventRecord(slot.done, stream));
      slot.pending = true;
      c.memcpy_calls += 1;
      done += n;
    }
  }

 private:
  enum class State { Unknown, Ready, Unavailable };
  struct Slot {
    void* host{nullptr};
    hipEvent_t done{nullptr};
    bool pending{false};
  };

  void release() {
    for (auto& slot : slots_) {
      (void)hipEventDestroy(slot.done);
      (void)hipHostFree(slot.host);
    }
    slots_.clear();
  }

  State state_{State::Unknown};
  std::vector<Slot> slots_;
  size_t next_{0};
};

struct StagingPools {
  std::mutex mutex;
  std::unordered_map<int, std::unique_ptr<StagingPool>> by_device;
};

StagingPools& staging_pools() {
  // Leaked on purpose, see `StagingPool`.
  static auto* pools = new StagingPools();
  return *pools;
}

} // namespace

namespace rocm {

LoadTransferStats load_transfer_stats() {
  auto& c = counters();
  return LoadTransferStats{
      c.tensors.load(),
      c.bytes.load(),
      c.pinned_tensors.load(),
      c.pageable_tensors.load(),
      c.memcpy_calls.load(),
      c.read_ns.load(),
      c.wait_ns.load(),
      c.total_ns.load(),
  };
}

} // namespace rocm

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto start = Clock::now();
  auto& c = counters();
  auto& encoder = rocm::get_command_encoder(stream());
  auto size = out.size();
  auto nbytes = size * out.itemsize();
  out.set_data(mlx::core::rocm::malloc_async(nbytes, encoder));
  c.tensors += 1;
  c.bytes += nbytes;

  if (nbytes > 0 && pinned_load_enabled()) {
    auto& pools = staging_pools();
    std::lock_guard<std::mutex> lock(pools.mutex);
    auto& pool = pools.by_device[encoder.device().hip_device()];
    if (!pool) {
      pool = std::make_unique<StagingPool>();
      encoder.device().make_current();
    }
    if (pool->ensure_ready()) {
      pool->upload(
          *reader_,
          offset_,
          nbytes,
          gpu_ptr<char>(out),
          encoder.stream(),
          swap_endianness_,
          out.itemsize());
      c.pinned_tensors += 1;
      c.total_ns += elapsed_ns(start);
      return;
    }
  }

  // Fallback: pageable staging buffer + synchronous H2D copy. An async H2D copy
  // from pageable memory is unreliable on a discrete GPU over a non-coherent
  // link (TB5 eGPU): driver-internal staging can stall the stream.
  auto read_start = Clock::now();
  auto out_ptr = malloc(nbytes);
  reader_->read(static_cast<char*>(out_ptr), nbytes, offset_);
  if (swap_endianness_) {
    swap_endianness_inplace(out_ptr, size, out.itemsize());
  }
  c.read_ns += elapsed_ns(read_start);
  if (nbytes > 0) {
    CHECK_HIP_ERROR(
        hipMemcpy(gpu_ptr<void>(out), out_ptr, nbytes, hipMemcpyHostToDevice));
    c.memcpy_calls += 1;
  }
  free(out_ptr);
  c.pageable_tensors += 1;
  c.total_ns += elapsed_ns(start);
}

} // namespace mlx::core
