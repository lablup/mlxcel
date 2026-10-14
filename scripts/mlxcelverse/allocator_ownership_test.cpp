// Copyright 2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Host-only contract test for the ROCm overlay's actual allocator::Data and
// malloc_async handoff. This does not exercise HIP execution or synchronization.
#include "mlx/allocator.h"

#include <cassert>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mlx::core {
class MockAllocator : public allocator::Allocator {
 public:
  allocator::Buffer malloc(size_t size) override {
    auto* ptr = new char[size ? size : 1];
    live.insert(ptr);
    ++allocations;
    return allocator::Buffer(ptr);
  }
  allocator::Buffer malloc_async(size_t size, int device, void* stream) {
    assert(device == 7 && stream == expected_stream);
    return malloc(size);
  }
  void free(allocator::Buffer buffer) override {
    assert(live.erase(buffer.ptr()) == 1);
    delete[] static_cast<char*>(buffer.ptr());
    ++releases;
  }
  size_t size(allocator::Buffer) const override {
    return 0;
  }
  std::set<void*> live;
  int allocations{0};
  int releases{0};
  void* expected_stream{this};
};

MockAllocator mock_allocator;
namespace allocator {
Allocator& allocator() {
  return mock_allocator;
}
} // namespace allocator

namespace rocm {
using hipStream_t = void*;
class CommandEncoder {
 public:
  struct Device {
    int hip_device() const {
      return 7;
    }
  };
  Device device() const {
    return {};
  }
  void* stream() const {
    return mock_allocator.expected_stream;
  }
};
MockAllocator& allocator() {
  return mock_allocator;
}
// Extracted verbatim from the overlay, not a duplicate implementation.
#include "rocm_malloc_async.inc"
} // namespace rocm
} // namespace mlx::core

int main() {
  using namespace mlx::core;
  using allocator::Data;
  static_assert(!std::is_copy_constructible_v<Data>);
  static_assert(!std::is_copy_assignable_v<Data>);
  static_assert(std::is_nothrow_move_constructible_v<Data>);
  rocm::CommandEncoder encoder;

  for (size_t size : {size_t{0}, size_t{32}}) {
    const int before = mock_allocator.releases;
    {
      auto original = rocm::malloc_async(size, encoder);
      assert(original.is_owned());
      auto buffer = original.buffer();
      Data moved(std::move(original));
      assert(!original.is_owned());
      assert(original.buffer().ptr() == nullptr);
      assert(moved.is_owned());
      assert(moved.buffer().ptr() == buffer.ptr());
      assert(mock_allocator.releases == before);
    }
    assert(mock_allocator.releases == before + 1);
  }

  // A borrowed slab shell must retain its custom release and never become
  // owned: upstream array::is_donatable requires Data::is_owned().
  int custom_releases = 0;
  int slab = 42;
  const int before_borrow = mock_allocator.releases;
  {
    Data borrowed(allocator::Buffer(&slab), [&](allocator::Buffer buffer) {
      assert(buffer.ptr() == &slab);
      ++custom_releases;
    });
    Data moved(std::move(borrowed));
    assert(!moved.is_owned());
    assert(custom_releases == 0);
  }
  assert(custom_releases == 1);
  assert(mock_allocator.releases == before_borrow);
  assert(slab == 42);

  // Model the encoder's shared ownership through a pending command batch.
  // Destroying the producer must not recycle storage before batch completion.
  std::vector<std::shared_ptr<Data>> pending;
  const int before_batch = mock_allocator.releases;
  {
    auto producer = std::make_shared<Data>(rocm::malloc_async(64, encoder));
    pending.push_back(producer);
  }
  assert(mock_allocator.releases == before_batch);
  pending.clear();
  assert(mock_allocator.releases == before_batch + 1);

  // Failure before handoff releases owned storage, including an empty buffer.
  for (size_t size : {size_t{0}, size_t{64}}) {
    const int before = mock_allocator.releases;
    try {
      auto data = rocm::malloc_async(size, encoder);
      throw std::runtime_error("handoff failed");
    } catch (const std::runtime_error&) {
    }
    assert(mock_allocator.releases == before + 1);
  }

  // Failure after enqueue leaves the pending batch's reference alive.
  const int before_failure = mock_allocator.releases;
  try {
    auto data = std::make_shared<Data>(rocm::malloc_async(64, encoder));
    pending.push_back(data);
    throw std::runtime_error("later enqueue failed");
  } catch (const std::runtime_error&) {
  }
  assert(mock_allocator.releases == before_failure);
  pending.clear();
  assert(mock_allocator.releases == before_failure + 1);
  assert(mock_allocator.live.empty());
  assert(mock_allocator.allocations == mock_allocator.releases);
}
