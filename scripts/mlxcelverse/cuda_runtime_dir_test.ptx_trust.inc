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

// POSIX trust rules of the CUDA PTX cache directory (lablup/mlxcel#3503):
// ptx_cache_trust.h against real directories, with no CUDA. Included by
// cuda_runtime_dir_test.cpp inside its anonymous namespace; it uses that
// file's check() and fs alias. The CUDA child-process tests in
// tests/cuda_ptx_cache_trust.rs cover the same rules through a real kernel,
// but only on a host with a GPU, so this is where CI runs them.

using mlx::core::cu::kNoPtxCacheTrustedGroup;
using mlx::core::cu::make_private_directories_at;
using mlx::core::cu::open_ptx_cache_components;
using mlx::core::cu::ptx_cache_entry_ref;
using mlx::core::cu::PtxPinnedDir;
using mlx::core::cu::open_ptx_cache_below_trusted_root;
using mlx::core::cu::ptx_cache_component_refusal;
using mlx::core::cu::ptx_cache_dir_is_trusted;
using mlx::core::cu::PtxTrustedRoot;
using mlx::core::cu::resolve_ptx_trusted_root;

// A directory with exactly |mode| (mkdir is subject to the umask).
fs::path make_dir(const fs::path& path, mode_t mode) {
  if (::mkdir(path.c_str(), 0700) != 0 || ::chmod(path.c_str(), mode) != 0) {
    std::fprintf(stderr, "cannot create %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  return path;
}

mode_t mode_of(const fs::path& path) {
  struct stat st;
  return ::lstat(path.c_str(), &st) == 0 ? (st.st_mode & 07777) : 0;
}

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
      text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The one-component rule of the walk, on owners and modes no test host has.
void test_ptx_component_rule() {
  const uid_t me = 1000;
  const gid_t mine = 1000;
  const auto refused = [&](mode_t mode, uid_t uid, gid_t gid, std::int64_t trusted) {
    const char* refusal = ptx_cache_component_refusal(mode, uid, gid, me, trusted);
    return refusal == nullptr ? std::string() : std::string(refusal);
  };
  const std::string owner = "is not owned by the current user";
  const std::string shared = "is group- or other-writable";
  const std::int64_t none = kNoPtxCacheTrustedGroup;

  check(refused(0700, me, mine, none).empty(), "rule: private directory");
  check(refused(0755, 0, 0, none).empty(), "rule: root-owned directory");
  check(refused(0755, 2000, mine, none) == owner, "rule: another owner");
  check(refused(0755, 2000, mine, mine) == owner, "rule: another owner, trusted group");
  check(refused(01777, 0, 0, none).empty(), "rule: root-owned sticky (/tmp)");
  check(refused(01777, me, mine, none) == shared, "rule: sticky but not root-owned");
  check(refused(0770, me, mine, none) == shared, "rule: group write, no trusted group");
  check(refused(0770, me, mine, mine).empty(), "rule 1: group write by the trusted group");
  check(refused(0770, me, 2000, mine) == shared, "rule 1: group write by another group");
  check(refused(0777, me, mine, mine) == shared, "rule 1: other write is never excused");
  check(refused(0772, me, mine, mine) == shared, "rule 1: other write without group write");
  // A volume root with fsGroup: root-owned, setgid, group-writable.
  check(refused(02775, 0, mine, mine).empty(), "rule 1: fsGroup volume root");
  check(refused(02775, 0, mine, none) == shared, "rule 1: fsGroup volume root, no trusted group");
  check(refused(02775, 0, 2000, mine) == shared, "rule 1: volume root of another group");
  check(refused(0770, me, 0, 0).empty(), "rule 1: gid 0 can be the trusted group");
  check(refused(0770, me, 0, none) == shared, "rule 1: -1 never matches a gid");
}

// The walk over real directories.
void test_ptx_walk(const fs::path& base) {
  const uid_t euid = ::geteuid();
  const std::int64_t none = kNoPtxCacheTrustedGroup;
  const auto walk = [&](const fs::path& dir, uid_t as, std::int64_t trusted) {
    fs::path resolved;
    std::string reason;
    if (ptx_cache_dir_is_trusted(dir, as, trusted, resolved, reason)) {
      check(resolved == fs::canonical(dir), "walk: resolved is the canonical path");
      return std::string();
    }
    check(!reason.empty(), "walk: a refusal carries a reason");
    return reason;
  };
  const auto priv = make_dir(base / "walk-private", 0700);
  check(walk(priv, euid, none).empty(), "walk: private directory");
  if (euid != 0) {
    check(
        walk(priv, euid + 1, none) ==
            priv.string() + " is not owned by the current user",
        "walk: a directory of another user");
  }

  const auto shared = make_dir(base / "walk-group", 0770);
  struct stat st;
  check(::stat(shared.c_str(), &st) == 0, "walk: stat");
  const std::int64_t gid = st.st_gid;
  const std::string group_refusal =
      shared.string() + " is group- or other-writable";
  check(walk(shared, euid, none) == group_refusal, "walk: group-writable leaf");
  check(walk(shared, euid, gid).empty(), "walk rule 1: leaf of the trusted group");
  check(walk(shared, euid, gid + 1) == group_refusal, "walk rule 1: another gid");
  const auto under_group = make_dir(shared / "cache", 0700);
  check(walk(under_group, euid, none) == group_refusal, "walk: group-writable parent");
  check(walk(under_group, euid, gid).empty(), "walk rule 1: parent of the trusted group");

  const auto open = make_dir(base / "walk-open", 0777);
  const auto under_open = make_dir(open / "cache", 0700);
  const std::string open_refusal = open.string() + " is group- or other-writable";
  check(walk(under_open, euid, none) == open_refusal, "walk: other-writable parent");
  check(walk(under_open, euid, gid) == open_refusal, "walk rule 1: not for other write");

  check(
      ends_with(walk(base / "walk-missing", euid, none), ")") &&
          walk(base / "walk-missing", euid, none).find(" cannot be resolved (") !=
              std::string::npos,
      "walk: a missing directory");
}

// MLXCEL_CACHE_TRUSTED_ROOT validation and the split of the cache path.
void test_ptx_trusted_root(const fs::path& base) {
  const auto resolve = [](const fs::path& value, const fs::path& cache,
                          PtxTrustedRoot& out) {
    std::string ignored;
    const bool ok = resolve_ptx_trusted_root(value, cache, out, ignored);
    check(ok == ignored.empty() || value.empty(), "root: a reason exactly when ignored");
    return ignored;
  };
  const auto open = make_dir(base / "root-open", 0777);
  const auto other = make_dir(base / "root-other", 0700);
  const auto cache = open / "cuda-ptx" / "scope";
  PtxTrustedRoot out;

  check(resolve({}, cache, out).empty() && out.root.empty(), "root: unset");
  check(
      resolve("rel/root", cache, out) ==
          "must be an absolute path, got \"rel/root\"",
      "root: relative");
  const auto missing = base / "root-missing";
  check(
      resolve(missing, cache, out).rfind(missing.string() + " cannot be resolved (", 0) == 0,
      "root: missing");
  const auto file = base / "root-file";
  std::fclose(std::fopen(file.c_str(), "w"));
  check(
      resolve(file, cache, out) == file.string() + " is not a directory",
      "root: a regular file");
  check(
      resolve(other, cache, out) ==
          other.string() + " is not an ancestor of " + cache.string(),
      "root: not an ancestor");
  check(
      resolve(open, open, out) ==
          open.string() + " is not an ancestor of " + open.string(),
      "root: the cache directory itself");
  check(out.root.empty(), "root: nothing is set when ignored");

  check(resolve(open, cache, out).empty(), "root: a proper ancestor");
  check(out.root == open && out.below == fs::path("cuda-ptx") / "scope", "root: the split");
  check(
      resolve(open, open / "cuda-ptx" / "scope" / "", out).empty() &&
          out.below == fs::path("cuda-ptx") / "scope",
      "root: a trailing separator");
  check(
      resolve(open, open / "a" / ".." / "b", out).empty() && out.root == open &&
          out.below == "b",
      "root: dot-dot in the cache path is normalized");

  // The root spelled through a symlink above it is still the root.
  const auto alias = base / "root-alias";
  check(::symlink(open.c_str(), alias.c_str()) == 0, "root: symlink");
  check(
      resolve(alias, alias / "cuda-ptx", out).empty() && out.root == open &&
          out.below == "cuda-ptx",
      "root: reached through a symlink above it");

  // A cache path that enters the root's tree through a symlink elsewhere has
  // no prefix that resolves to the root.
  const auto deep = make_dir(open / "models", 0700);
  const auto door = base / "root-door";
  check(::symlink(deep.c_str(), door.c_str()) == 0, "root: symlink");
  check(
      resolve(open, door / "scope", out) ==
          open.string() + " is not an ancestor of " + (door / "scope").string(),
      "root: entered through a symlink from outside");
}

// The descent below a trusted root.
void test_ptx_descent(const fs::path& base) {
  const uid_t euid = ::geteuid();
  const auto descend = [&](const fs::path& root, const fs::path& below, uid_t as,
                           int* keep = nullptr) {
    std::string reason;
    const int fd = open_ptx_cache_below_trusted_root(root, below, as, reason);
    if (fd >= 0) {
      check(reason.empty(), "descent: no reason on success");
      if (keep != nullptr) {
        *keep = fd;
      } else {
        ::close(fd);
      }
      return std::string();
    }
    check(!reason.empty(), "descent: a refusal carries a reason");
    return reason;
  };
  // The root is other-writable and its parent chain is whatever the host
  // has: neither is judged.
  const auto open = make_dir(base / "descent-open", 0777);
  const fs::path below = fs::path("cuda-ptx") / "scope" / "sm121-nvrtc13.0";
  int pinned = -1;
  check(descend(open, below, euid, &pinned).empty(), "descent: creates the chain");
  check(
      mode_of(open / "cuda-ptx") == 0700 &&
          mode_of(open / "cuda-ptx" / "scope") == 0700 &&
          mode_of(open / below) == 0700,
      "descent: created components are 0700");
  check(descend(open, below, euid).empty(), "descent: an existing private chain");
  check(
      descend(open, below, euid + 1) ==
          (open / "cuda-ptx").string() + " is not owned by the current user",
      "descent: a component of another user");

  // The descriptor is the directory that was checked, wherever its path goes.
  check(
      ::rename((open / "cuda-ptx").c_str(), (open / "moved").c_str()) == 0,
      "descent: rename");
  const int file = ::openat(pinned, "entry.ptx", O_WRONLY | O_CREAT | O_EXCL, 0600);
  check(file >= 0, "descent: create through the descriptor");
  if (file >= 0) {
    ::close(file);
  }
  ::close(pinned);
  check(
      fs::exists(open / "moved" / "scope" / "sm121-nvrtc13.0" / "entry.ptx") &&
          !fs::exists(open / "cuda-ptx"),
      "descent: the descriptor follows the directory, not the path");

  // What someone who can write the root can put at the first component.
  const auto mine = make_dir(open / "models", 0700);
  make_dir(mine / "scope", 0700);
  check(::symlink(mine.c_str(), (open / "linked").c_str()) == 0, "descent: symlink");
  check(
      descend(open, fs::path("linked") / "scope", euid) ==
          (open / "linked").string() + " is a symbolic link",
      "descent: a symlink to another directory of this user");
  check(::symlink(open.c_str(), (open / "back").c_str()) == 0, "descent: symlink");
  check(
      descend(open, fs::path("back") / "models", euid) ==
          (open / "back").string() + " is a symbolic link",
      "descent: a symlink back to the root");
  check(
      ::symlink((open / "nowhere").c_str(), (open / "dangling").c_str()) == 0,
      "descent: symlink");
  check(
      descend(open, "dangling", euid) ==
          (open / "dangling").string() + " is a symbolic link",
      "descent: a dangling symlink is not created through");
  check(!fs::exists(open / "nowhere"), "descent: nothing was created through it");
  std::fclose(std::fopen((open / "file").c_str(), "w"));
  check(
      descend(open, "file", euid) == (open / "file").string() + " is not a directory",
      "descent: a regular file");

  // Below the root the rule is stricter than the walk's: no shared write,
  // whatever the group, and deeper components count too.
  make_dir(open / "group", 0770);
  check(
      descend(open, "group", euid) ==
          (open / "group").string() + " is group- or other-writable",
      "descent: a group-writable component");
  make_dir(mine / "world", 0707);
  check(
      descend(open, fs::path("models") / "world", euid) ==
          (mine / "world").string() + " is group- or other-writable",
      "descent: an other-writable deeper component");
  check(
      mode_of(open / "group") == 0770 && mode_of(mine / "world") == 0707,
      "descent: an existing component is never chmod-ed");

  if (euid != 0) {
    // Root does not count as an owner below the trusted root.
    check(
        descend("/usr", "share", euid) == "/usr/share is not owned by the current user",
        "descent: a root-owned component");
    const auto readonly = make_dir(base / "descent-readonly", 0555);
    check(
        descend(readonly, "cache", euid).rfind(
            (readonly / "cache").string() + " cannot be created (", 0) == 0,
        "descent: a root that cannot be written");
  }

  // The root and the directories above it must still not be symlinks.
  const auto alias = base / "descent-alias";
  check(::symlink(open.c_str(), alias.c_str()) == 0, "descent: symlink");
  check(
      descend(alias, "models", euid) == alias.string() + " is a symbolic link",
      "descent: a symlink at the root");
  check(
      descend(base / "descent-missing", "cache", euid) ==
          (base / "descent-missing").string() + " cannot be inspected",
      "descent: a missing root");
}

// The two halves as jit_module.cpp chains them: a symlink planted at the
// first component below the root is refused, not resolved away.
void test_ptx_planted_symlink(const fs::path& base) {
  const auto open = make_dir(base / "planted-open", 0777);
  const auto shaped = make_dir(open / "models", 0700);
  make_dir(shaped / "scope", 0700);
  check(
      ::symlink(shaped.c_str(), (open / "cuda-ptx").c_str()) == 0,
      "planted: symlink");
  PtxTrustedRoot root;
  std::string ignored;
  check(
      resolve_ptx_trusted_root(open, open / "cuda-ptx" / "scope", root, ignored) &&
          root.below == fs::path("cuda-ptx") / "scope",
      "planted: the split keeps the name as given");
  std::string reason;
  const int fd =
      open_ptx_cache_below_trusted_root(root.root, root.below, ::geteuid(), reason);
  check(fd < 0, "planted: the descent refuses");
  if (fd >= 0) {
    ::close(fd);
  }
  check(
      reason == (open / "cuda-ptx").string() + " is a symbolic link",
      "planted: the symlink is named");
}

// The cache directory's descriptor is kept and each device subdirectory is
// opened from it, so what happens to the path after the first descent does
// not decide where a later device subdirectory is.
void test_ptx_device_dir_from_held_descriptor(const fs::path& base) {
  const uid_t euid = ::geteuid();
  const auto open = make_dir(base / "held-open", 0777);
  std::string reason;
  const int cache_fd = open_ptx_cache_below_trusted_root(
      open, fs::path("cuda-ptx") / "scope", euid, reason);
  check(cache_fd >= 0, "held: the cache directory opens");
  if (cache_fd < 0) {
    return;
  }
  const auto cache = open / "cuda-ptx" / "scope";
  // Another directory of this user, shaped like the cache, takes its place.
  const auto foreign = make_dir(open / "models", 0700);
  make_dir(foreign / "scope", 0700);
  make_dir(foreign / "scope" / "sm80-nvrtc12.0", 0700);
  std::fclose(std::fopen(
      (foreign / "scope" / "sm80-nvrtc12.0" / "planted.ptx").c_str(), "w"));
  check(
      ::rename((open / "cuda-ptx").c_str(), (open / "real").c_str()) == 0 &&
          ::rename(foreign.c_str(), (open / "cuda-ptx").c_str()) == 0,
      "held: swap the first component");
  const int device_fd =
      open_ptx_cache_components(cache_fd, cache, "sm80-nvrtc12.0", euid, reason);
  check(device_fd >= 0, "held: the device subdirectory opens");
  struct stat st;
  check(
      device_fd >= 0 && ::fstatat(device_fd, "planted.ptx", &st, 0) != 0 &&
          fs::is_directory(open / "real" / "scope" / "sm80-nvrtc12.0"),
      "held: it is the checked directory's child, not the swapped-in one");
  check(
      ::fcntl(cache_fd, F_GETFD) != -1,
      "held: the starting descriptor stays open");
  check(
      open_ptx_cache_components(cache_fd, cache, fs::path(), euid, reason) < 0,
      "held: no names is not a directory to open");
  make_dir(open / "real" / "scope" / "shared", 0770);
  check(
      open_ptx_cache_components(cache_fd, cache, "shared", euid, reason) < 0 &&
          reason == (cache / "shared").string() + " is group- or other-writable",
      "held: the rule applies to names opened from a held descriptor");

  // Entries of a pinned directory are reached through its descriptor.
  const auto device = cache / "sm80-nvrtc12.0";
  const std::vector<PtxPinnedDir> pinned = {{device.string() + "/", device_fd}};
  const auto direct = ptx_cache_entry_ref(pinned, device / "mod.ptx");
  check(
      direct.dir_fd == device_fd && direct.name == "mod.ptx",
      "ref: an entry of the pinned directory");
  const auto nested = ptx_cache_entry_ref(pinned, device / "long" / "tail.ptx");
  check(
      nested.dir_fd == device_fd && nested.name == "long/tail.ptx",
      "ref: an entry in a nested directory");
  const auto sibling =
      ptx_cache_entry_ref(pinned, cache / "sm80-nvrtc12.0x" / "mod.ptx");
  check(
      sibling.dir_fd == AT_FDCWD &&
          sibling.name == (cache / "sm80-nvrtc12.0x" / "mod.ptx").string(),
      "ref: a sibling whose name extends the pinned one goes by path");
  check(
      ptx_cache_entry_ref(pinned, device).dir_fd == AT_FDCWD,
      "ref: the pinned directory itself goes by path");
  check(
      ptx_cache_entry_ref({}, device / "mod.ptx").dir_fd == AT_FDCWD,
      "ref: nothing pinned, everything goes by path");

  // Nested directories for a long module name are made below the descriptor,
  // which is the real directory, wherever the path leads now.
  check(
      make_private_directories_at(device_fd, fs::path("long") / "longer") &&
          make_private_directories_at(device_fd, fs::path("long") / "longer"),
      "at: nested directories, and again when they exist");
  const auto made = open / "real" / "scope" / "sm80-nvrtc12.0" / "long";
  check(
      mode_of(made) == 0700 && mode_of(made / "longer") == 0700 &&
          !fs::exists(open / "cuda-ptx" / "scope" / "sm80-nvrtc12.0" / "long"),
      "at: created 0700 in the pinned directory");
  std::fclose(std::fopen((made / "file").c_str(), "w"));
  check(
      !make_private_directories_at(device_fd, fs::path("long") / "file" / "x"),
      "at: a regular file in the way");
  if (device_fd >= 0) {
    ::close(device_fd);
  }
  ::close(cache_fd);
}

void test_ptx_cache_trust() {
  test_ptx_component_rule();
  std::error_code error;
  auto pattern =
      (fs::canonical(fs::temp_directory_path()) / "mlxcel-ptx-trust.XXXXXX").string();
  if (::mkdtemp(pattern.data()) == nullptr) {
    std::fprintf(stderr, "cannot create a temporary directory\n");
    std::exit(EXIT_FAILURE);
  }
  const fs::path base = pattern;
  test_ptx_walk(base);
  test_ptx_trusted_root(base);
  test_ptx_descent(base);
  test_ptx_planted_symlink(base);
  test_ptx_device_dir_from_held_descriptor(base);
  // Make everything removable again.
  for (auto it = fs::recursive_directory_iterator(base, error);
       !error && it != fs::recursive_directory_iterator();
       it.increment(error)) {
    if (it->is_directory(error) && !it->is_symlink(error)) {
      ::chmod(it->path().c_str(), 0700);
    }
  }
  fs::remove_all(base, error);
}
