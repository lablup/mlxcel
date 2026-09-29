// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <map>
#include <mutex>
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

///////////////////////////////////////////////////////////////////////////////
// HipEvent implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

// Manage cached hipEvent_t objects. Keyed by (device, flags): a hipEvent is
// bound to the device current when it was created, and recording it on another
// device's stream is invalid (hangs the queue on a multi-GPU host). So pool and
// hand back events per device, creating new ones on the CURRENT device.
struct HipEventPool {
  static HipEventHandle create(int flags) {
    int dev = 0;
    (void)hipGetDevice(&dev);
    auto& cache = cache_for(dev, flags);
    if (cache.empty()) {
      return HipEventHandle(flags); // created on the current device
    } else {
      HipEventHandle ret = std::move(cache.back());
      cache.pop_back();
      return ret;
    }
  }

  static void release(HipEventHandle event) {
    cache_for(event.device, event.flags).push_back(std::move(event));
  }

  static std::vector<HipEventHandle>& cache_for(int device, int flags) {
    static std::map<std::pair<int, int>, std::vector<HipEventHandle>> cache;
    return cache[{device, flags}];
  }
};

using Clock = std::chrono::steady_clock;

// Deadline for a host wait under MLX_ROCM_GPU_WATCHDOG_SECS, or the far future
// when the watchdog is off (the default).
Clock::time_point watchdog_deadline() {
  int secs = gpu_watchdog_seconds();
  if (secs <= 0) {
    return Clock::time_point::max();
  }
  return Clock::now() + std::chrono::seconds(secs);
}

} // namespace

HipEventHandle::HipEventHandle(int flags) : flags(flags) {
  CHECK_HIP_ERROR(hipEventCreateWithFlags(&handle_, flags));
  assert(handle_ != nullptr);
  (void)hipGetDevice(&device); // event is bound to the current device
}

HipEvent::HipEvent(int flags) : event_(HipEventPool::create(flags)) {}

HipEvent::~HipEvent() {
  HipEventPool::release(std::move(event_));
}

hipError_t HipEvent::wait() {
  // MLX_EVENT_BLOCKING: block (hipEventSynchronize) instead of CPU spin-polling.
  // The device is created with hipDeviceScheduleBlockingSync, so this sleeps the
  // thread rather than busy-waiting. On a bandwidth-bound APU decode the spin
  // poll steals shared-memory-controller bandwidth from the GPU, so blocking can
  // be a net win despite the wakeup latency. Default keeps the legacy spin.
  // A faulted stream fails the blocking wait promptly, so the watchdog only
  // applies to the polling mode.
  static const bool blocking = std::getenv("MLX_EVENT_BLOCKING") != nullptr;
  if (blocking) {
    return hipEventSynchronize(event_);
  }
  // Spin-wait with hipEventQuery instead of hipEventSynchronize.
  // Use progressive backoff to reduce hipEventQuery call overhead.
  // Anything other than hipErrorNotReady ends the wait: once the queue has
  // faulted the query returns the fault for the rest of the process, and the
  // old "until hipSuccess" loop spun forever on it (lablup/mlxcel#1804).
  const auto deadline = watchdog_deadline();
  for (uint32_t spins = 0;; spins++) {
    hipError_t status = hipEventQuery(event_);
    if (status != hipErrorNotReady) {
      return status;
    }
    if (spins < 100) {
      // Tight spin for fast completions
    } else if (spins < 1000) {
      _mm_pause(); // x86 pause hint (reduces power, avoids pipeline stall)
    } else {
      std::this_thread::yield();
      if ((spins & 1023) == 0 && Clock::now() > deadline) {
        return kWatchdogExpired;
      }
    }
  }
}

void HipEvent::wait(hipStream_t stream) {
  CHECK_HIP_ERROR(hipStreamWaitEvent(stream, event_, 0));
}

void HipEvent::record(hipStream_t stream) {
  CHECK_HIP_ERROR(hipEventRecord(event_, stream));
}

hipError_t HipEvent::query() const {
  return hipEventQuery(event_);
}

// Wraps HipEvent with a few features:
// 1. The class can be copied.
// 2. Add checks for waiting on un-recorded event.
class CopyableHipEvent {
 public:
  CopyableHipEvent()
      : event_(std::make_shared<HipEvent>(
            hipEventDisableTiming)) {}
            // Note: hipEventBlockingSync removed — on iGPU the blocking wait
            // contends with GPU for CPU resources. Polling is cheaper.

  [[nodiscard]] hipError_t wait() {
    check_recorded();
    return event_->wait();
  }

  void wait(hipStream_t stream) {
    check_recorded();
    event_->wait(stream);
  }

  void record(hipStream_t stream) {
    event_->record(stream);
    recorded_ = true;
  }

  // hipErrorNotReady until record() has been called, then the HIP status.
  [[nodiscard]] hipError_t query() const {
    return recorded_ ? event_->query() : hipErrorNotReady;
  }

 private:
  void check_recorded() const {
    if (!recorded_) {
      throw std::runtime_error(
          "Should not wait on a HipEvent before recording.");
    }
  }

  std::shared_ptr<HipEvent> event_;
  bool recorded_{false};
};

///////////////////////////////////////////////////////////////////////////////
// AtomicEvent implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

// Payload of the host callback that lands a signal. It owns a reference to the
// event state so the counter outlives every copy of the event that could be
// destroyed while the callback is still queued.
struct SignalPayload {
  std::shared_ptr<AtomicEvent::State> state;
  uint64_t value;
};

void signal_atomic_callback(void* data) {
  auto* payload = static_cast<SignalPayload*>(data);
  payload->state->counter->store(payload->value, std::memory_order_release);
  delete payload;
}

// Pool of 8-byte SIGNAL-MEMORY counters for AtomicEvent. The counter is used two
// ways: the GPU waits on it via hipStreamWaitValue64 (cross-stream Fence) and
// signals it via hipStreamWriteValue64, AND the CPU polls it (host wait()). HIP
// REQUIRES the pointer passed to hipStreamWaitValue64/WriteValue64 to be
// allocated with the hipMallocSignalMemory flag — plain hipHostMalloc memory is
// accepted by the call (returns success) but the GPU-side wait never observes the
// value, so on a discrete GPU the stream spins forever (busy 100%, mem-ctrl 0%).
// Signal memory is host-accessible, so the CPU poll works too. Recycle the blocks
// (hipFree blocks on a discrete GPU); intentionally leaked at process exit.
struct PinnedCounterPool {
  std::mutex m;
  std::vector<void*> free_list;

  void* acquire() {
    {
      std::lock_guard<std::mutex> lk(m);
      if (!free_list.empty()) {
        void* p = free_list.back();
        free_list.pop_back();
        return p;
      }
    }
    void* p = nullptr;
    CHECK_HIP_ERROR(hipHostMalloc(
        &p, sizeof(std::atomic<uint64_t>),
        hipHostMallocMapped | hipHostMallocCoherent));
    return p;
  }

  void release(void* p) {
    std::lock_guard<std::mutex> lk(m);
    free_list.push_back(p);
  }
};

PinnedCounterPool& counter_pool() {
  static PinnedCounterPool* pool = new PinnedCounterPool;
  return *pool;
}

} // namespace

AtomicEvent::State::State() {
  // Completion counter in pinned, device-mapped host memory: the GPU signals it
  // (hipStreamWriteValue64) and the CPU polls it (wait()). On a discrete GPU over
  // a non-coherent link (TB5 eGPU) device memory is NOT CPU-visible, so the
  // counter must live in host memory or the host poll spins forever. Drawn from a
  // recycling pool — see PinnedCounterPool (hipHostFree blocks on a discrete GPU).
  void* p = counter_pool().acquire();
  counter = static_cast<std::atomic<uint64_t>*>(p);
  counter->store(0, std::memory_order_release);
}

AtomicEvent::State::~State() {
  counter_pool().release(counter);
}

AtomicEvent::AtomicEvent() : state_(std::make_shared<State>()) {}

hipError_t AtomicEvent::status() const {
  auto recorded = static_cast<hipError_t>(
      state_->error.load(std::memory_order_acquire));
  if (recorded != hipSuccess) {
    return recorded;
  }
  hipStream_t stream = state_->stream.load(std::memory_order_acquire);
  if (stream == nullptr) {
    return hipSuccess; // nothing queued on a GPU stream yet
  }
  // hipSuccess (idle) and hipErrorNotReady (busy) both mean the callback can
  // still fire: it runs on the runtime's callback thread, slightly after the
  // stream itself reports idle.
  hipError_t queried = hipStreamQuery(stream);
  return queried == hipErrorNotReady ? hipSuccess : queried;
}

hipError_t AtomicEvent::wait(uint64_t value) {
  using namespace std::chrono_literals;
  auto* ac = atomic();
  static const bool blocking = std::getenv("MLX_EVENT_BLOCKING") != nullptr;
  const auto deadline = watchdog_deadline();
  // The counter is landed by a hipLaunchHostFunc callback, and a stream that
  // has faulted never runs its callbacks, so poll the signaling stream about
  // once a millisecond and give up once it reports a failure. The status query
  // is skipped entirely for an event nothing has signaled from a GPU stream.
  auto next_poll = Clock::now() + 1ms;
  for (uint32_t spins = 0; ac->load(std::memory_order_acquire) < value;
       spins++) {
    if (blocking) {
      // Sleep briefly instead of hot-yielding so the host poll of this pinned
      // counter doesn't contend with the GPU on the APU memory controller.
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    } else {
      std::this_thread::yield();
    }
    if ((spins & 63) != 0) {
      continue;
    }
    auto now = Clock::now();
    if (now < next_poll) {
      continue;
    }
    next_poll = now + 1ms;
    hipError_t st = status();
    if (st != hipSuccess) {
      // The callback may have landed the value between the loop test and the
      // status query; a landed value is a completed wait, whatever the stream
      // says afterwards.
      return ac->load(std::memory_order_acquire) >= value ? hipSuccess : st;
    }
    if (now > deadline) {
      return kWatchdogExpired;
    }
  }
  return hipSuccess;
}

hipError_t AtomicEvent::wait(hipStream_t stream, uint64_t value) {
  // Do NOT use hipStreamWaitValue64 on the host counter: it requires
  // hipMallocSignalMemory and silently never observes a plain pinned-host value,
  // wedging the queue. The counter is signaled by a host callback when the
  // producer stream reaches the signal point, so block the host here until the
  // value lands; subsequent work on this stream is correctly ordered after it.
  (void)stream;
  return wait(value);
}

hipError_t AtomicEvent::wait(Stream s, uint64_t value) {
  assert(s.device.type != mlx::core::Device::cpu);
  auto& encoder = get_command_encoder(s);
  encoder.commit();
  return wait(encoder.stream(), value);
}

void AtomicEvent::signal(uint64_t value) {
  atomic()->store(value, std::memory_order_release);
}

void AtomicEvent::signal(hipStream_t stream, uint64_t value) {
  // Signal the host-resident counter from the stream via a host callback that
  // fires when the stream reaches this point. We do NOT use hipStreamWriteValue64
  // here: it REQUIRES a hipMallocSignalMemory pointer, but returns success on a
  // plain pinned-host counter while never actually landing the write — so the
  // host poll (wait()) spins forever and the discrete-GPU queue wedges (busy
  // 100%, mem-ctrl 0%). The host callback always delivers the value correctly.
  state_->stream.store(stream, std::memory_order_release);
  auto* payload = new SignalPayload{state_, value};
  hipError_t st = hipLaunchHostFunc(stream, signal_atomic_callback, payload);
  if (st != hipSuccess) {
    delete payload;
    // The callback will never run, so no waiter would ever wake. Mark the
    // event failed first (waiters on other threads read it), then report.
    state_->error.store(static_cast<int>(st), std::memory_order_release);
    check_hip_error("hipLaunchHostFunc", st);
  }
}

void AtomicEvent::signal(Stream s, uint64_t value) {
  assert(s.device.type != mlx::core::Device::cpu);
  auto& encoder = get_command_encoder(s);
  encoder.commit();
  signal(encoder.stream(), value);
}

bool AtomicEvent::is_signaled(uint64_t value) const {
  return atomic()->load(std::memory_order_acquire) >= value;
}

uint64_t AtomicEvent::value() const {
  return atomic()->load(std::memory_order_acquire);
}

} // namespace rocm

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

struct EventImpl {
  std::unique_ptr<rocm::CopyableHipEvent> hip;
  std::unique_ptr<rocm::AtomicEvent> atomic;
  // Set when the event is "signaled" purely on the host because it was created
  // during HIP graph capture: recording a real completion event onto the
  // captured stream would bake it into the graph (never fires under capture),
  // so we skip the device record and treat the event as already satisfied. The
  // recorded compute kernels still execute on replay; completion is handled by
  // an explicit stream sync after hipGraphLaunch.
  bool host_signaled{false};
  // Error attached to the event (ml-explore/mlx#3742). The CPU scheduler
  // stores its stream's error here through Event::set_error, and this backend
  // stores the signaling GPU stream's encoder-owned Error when a wait or a
  // query finds that stream failed (lablup/mlxcel#1804). Event::wait() and
  // array::detach_event() throw it through Event::check_error().
  std::atomic<Error*> error{nullptr};

  bool is_created() const {
    return hip || atomic;
  }

  void ensure_created(Stream s, uint64_t signal_value) {
    if (is_created()) {
      return;
    }
    if (s.device == mlx::core::Device::cpu || signal_value > 1) {
      atomic = std::make_unique<rocm::AtomicEvent>();
    } else {
      // Bind the stream's device current before creating the hipEvent — it is
      // bound to whatever device is current at creation, and recording it on a
      // different device's stream hangs the queue on a multi-GPU host.
      (void)rocm::get_command_encoder(s);
      hip = std::make_unique<rocm::CopyableHipEvent>();
    }
  }
};

EventImpl& impl_of(const Event& event) {
  return event.cast<EventImpl>();
}

// Attach the failure of the stream that signals |event| to the event, keeping
// the earliest message. Never throws and never binds a device, so it is safe
// from const queries and on a device whose every HIP call now fails.
void poison_event(const Event& event, hipError_t status, const char* where) {
  Error& error = rocm::record_stream_error(event.stream(), status, where);
  const_cast<Event&>(event).set_error(error);
}

// poison_event(), then throw the error the way every other backend does:
// through Event::check_error(), which clears it. Falls back to a plain throw
// if a concurrent check consumed the message first.
[[noreturn]] void
fail_event(Event& event, hipError_t status, const char* where) {
  poison_event(event, status, where);
  event.check_error();
  throw std::runtime_error(rocm::describe_device_error(status, where));
}

constexpr const char* kWaitHost = "waiting on the host for a stream event";
constexpr const char* kWaitSignal =
    "waiting on the host for a stream's completion signal";

} // namespace

Event::Event(Stream s) : stream_(s) {
  event_ = std::shared_ptr<void>(
      new EventImpl(), [](void* ptr) { delete static_cast<EventImpl*>(ptr); });
}

void Event::wait() {
  check_error();
  auto& event = impl_of(*this);
  // Capture-created event: nothing to wait for on the host (see EventImpl).
  if (event.host_signaled) {
    return;
  }
  assert(event.is_created());
  if (event.hip) {
    assert(value() == 1);
    if (hipError_t st = event.hip->wait(); st != hipSuccess) {
      fail_event(*this, st, kWaitHost);
    }
  } else {
    if (hipError_t st = event.atomic->wait(value()); st != hipSuccess) {
      fail_event(*this, st, kWaitSignal);
    }
  }
  check_error();
}

void Event::wait(Stream s) {
  auto& event = impl_of(*this);
  if (event.host_signaled) {
    return;
  }
  assert(event.is_created());
  if (s.device == mlx::core::Device::cpu) {
    // Through the scheduler's wait_event so the CPU stream inherits the
    // event's error, as the Metal and CUDA backends do.
    scheduler::wait_event(s, *this, [value = value()](Event& self) {
      auto& ev = impl_of(self);
      hipError_t st = ev.hip ? ev.hip->wait() : ev.atomic->wait(value);
      if (st != hipSuccess) {
        fail_event(self, st, ev.hip ? kWaitHost : kWaitSignal);
      }
    });
    return;
  }
  auto& encoder = rocm::get_command_encoder(s);
  encoder.commit();
  if (event.hip) {
    assert(value() == 1);
    event.hip->wait(encoder.stream());
  } else if (hipError_t st = event.atomic->wait(encoder.stream(), value());
             st != hipSuccess) {
    fail_event(*this, st, kWaitSignal);
  }
}

void Event::signal(Stream s) {
  auto& event = impl_of(*this);
  event.ensure_created(s, value());
  if (event.hip) {
    assert(value() == 1);
    if (s.device == mlx::core::Device::cpu) {
      throw std::runtime_error("HipEvent can not be recorded on a CPU stream.");
    }
    auto& encoder = rocm::get_command_encoder(s);
    encoder.commit();
    event.hip->record(encoder.stream());
  } else if (s.device == mlx::core::Device::cpu) {
    // Through the scheduler's signal_event so a CPU stream that already
    // failed poisons the event before signaling it.
    scheduler::signal_event(s, *this, [value = value()](Event& self) {
      impl_of(self).atomic->signal(value);
    });
  } else {
    event.atomic->signal(s, value());
  }
}

std::atomic<Error*>& Event::error() {
  return impl_of(*this).error;
}

bool Event::is_signaled() const {
  auto& event = impl_of(*this);
  if (event.host_signaled) {
    return true;
  }
  if (!event.is_created()) {
    return false;
  }
  // A failed stream never signals. Report the event as signaled with its
  // error attached, so array::is_available() detaches it and throws instead
  // of leaving the array pending forever, and so a status read that never
  // waits (the bridge's stashed_launch_state) can see the failure.
  hipError_t status;
  const char* where;
  if (event.hip) {
    assert(value() == 1);
    status = event.hip->query();
    if (status == hipSuccess) {
      return true;
    }
    if (status == hipErrorNotReady) {
      return false;
    }
    where = "querying a stream event";
  } else {
    if (event.atomic->is_signaled(value())) {
      return true;
    }
    status = event.atomic->status();
    if (status == hipSuccess) {
      return false;
    }
    if (event.atomic->is_signaled(value())) {
      return true;
    }
    where = "querying a stream's completion signal";
  }
  poison_event(*this, status, where);
  return true;
}

} // namespace mlx::core
