// Copyright © 2025 Apple Inc.

#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mlx::core::rocm {

// Cache capacity from an environment variable (lablup/mlxcel#2051). Unset
// gives the default with no output. The value must be a whole decimal integer
// from 1 to INT_MAX, parsed like MLX_ROCM_GPU_WATCHDOG_SECS in device.cpp;
// anything else (empty, trailing junk such as "8abc", overflow, zero or a
// negative value) prints one line to stderr naming the variable and the
// default, and the default is used. std::stoul used to take "-1" as SIZE_MAX
// and "8abc" as 8, accept "0" (whose first put() then popped an empty list),
// and throw a bare "stoul" for anything else.
inline size_t capacity_from_env(const char* env_var, size_t default_capacity) {
  const char* env = std::getenv(env_var);
  if (!env) {
    return default_capacity;
  }
  char* end = nullptr;
  errno = 0;
  long v = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || errno == ERANGE || v <= 0 ||
      v > std::numeric_limits<int>::max()) {
    std::fprintf(
        stderr,
        "[ROCm] ignoring invalid %s=\"%s\" (expected a positive integer); "
        "using the default %zu\n",
        env_var,
        env,
        default_capacity);
    return default_capacity;
  }
  return static_cast<size_t>(v);
}

// LRU cache with byte-based keys
template <typename Key, typename Value>
class LRUBytesKeyCache {
 public:
  LRUBytesKeyCache(const char* env_var, size_t default_capacity)
      : capacity_(capacity_from_env(env_var, default_capacity)) {
    // put() evicts while size() >= capacity_, which pops an empty list at 0.
    if (capacity_ == 0) {
      throw std::runtime_error("LRUCache requires capacity > 0.");
    }
  }

  std::optional<Value> get(const Key& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      return std::nullopt;
    }
    // Move to front (most recently used)
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
  }

  void put(const Key& key, const Value& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      // Update existing entry and move to front
      it->second->second = value;
      cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
      return;
    }

    // Evict if at capacity
    while (cache_list_.size() >= capacity_) {
      auto last = cache_list_.back();
      cache_map_.erase(last.first);
      cache_list_.pop_back();
    }

    // Insert new entry at front
    cache_list_.emplace_front(key, value);
    cache_map_[key] = cache_list_.begin();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_list_.clear();
    cache_map_.clear();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_list_.size();
  }

 private:
  size_t capacity_;
  std::list<std::pair<Key, Value>> cache_list_;
  std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator>
      cache_map_;
  mutable std::mutex mutex_;
};

// Simple LRU cache with size_t keys
template <typename Value>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::runtime_error("LRUCache requires capacity > 0.");
    }
  }

  std::optional<Value> get(size_t key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      return std::nullopt;
    }
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    return it->second->second;
  }

  void put(size_t key, const Value& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      it->second->second = value;
      cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
      return;
    }

    while (cache_list_.size() >= capacity_) {
      auto last = cache_list_.back();
      cache_map_.erase(last.first);
      cache_list_.pop_back();
    }

    cache_list_.emplace_front(key, value);
    cache_map_[key] = cache_list_.begin();
  }

 private:
  size_t capacity_;
  std::list<std::pair<size_t, Value>> cache_list_;
  std::unordered_map<
      size_t,
      typename std::list<std::pair<size_t, Value>>::iterator>
      cache_map_;
  mutable std::mutex mutex_;
};

} // namespace mlx::core::rocm
