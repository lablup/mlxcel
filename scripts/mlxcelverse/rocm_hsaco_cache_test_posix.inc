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

// The POSIX cases of rocm_hsaco_cache_test.cpp: real directories and files.

void write_file(const fs::path& path, const std::string& contents) {
  std::ofstream file(path, std::ios::binary);
  file << contents;
}

// Plant a complete entry for kModule, as an attacker would.
void plant_entry(const fs::path& dir, const std::string& source) {
  fs::create_directories(dir);
  write_file(dir / (std::string(kModule) + ".hsaco"), "PLANTED");
  write_file(dir / (std::string(kModule) + ".txt"), "kernel\tmangled\n");
  write_file(dir / (std::string(kModule) + ".hip"), source);
}

unsigned mode_of(const fs::path& path) {
  struct stat st;
  return ::lstat(path.c_str(), &st) == 0 ? (st.st_mode & 07777) : 0;
}

fs::path private_dir(const fs::path& path, unsigned mode = 0700) {
  fs::create_directories(path);
  ::chmod(path.c_str(), mode);
  return path;
}

bool mentions(const HsacoCacheDir& got, const char* text) {
  return got.refusal.find(text) != std::string::npos;
}

// Under the private base: refused directories, nothing created or loaded.
void test_refused_directories(const fs::path& base) {
  const auto euid = cache_current_euid();
  const std::string source = "src";

  // Control: the same planted entry in a private directory does load, so the
  // refusals below come from the trust check rather than from the test setup.
  {
    const auto over = private_dir(base / "control");
    private_dir(over / kArch);
    plant_entry(over / kArch, source);
    ::chmod((over / kArch).c_str(), 0700);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(!got.dir.empty(), "control dir is used: " + got.refusal);
    check(got.refusal.empty(), "control dir has no refusal");
    check(blob_loaded(got.dir, &source), "control blob is loaded");
  }

  const struct {
    const char* name;
    unsigned leaf_mode;
    unsigned parent_mode;
  } modes[] = {
      {"world-writable leaf", 0777, 0700},
      {"group-writable leaf", 0770, 0700},
      {"world-writable override root", 0700, 0777},
      {"group-writable override root", 0700, 0770},
      {"sticky world-writable leaf (not root-owned)", 01777, 0700},
  };
  int index = 0;
  for (const auto& c : modes) {
    if (euid == 0 && (c.leaf_mode & 01000) != 0) {
      continue; // root-owned and sticky is the /tmp case, which is trusted
    }
    const auto over = private_dir(base / ("mode" + std::to_string(index++)));
    const auto leaf = private_dir(over / kArch);
    plant_entry(leaf, source);
    ::chmod(leaf.c_str(), c.leaf_mode);
    ::chmod(over.c_str(), c.parent_mode);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    const std::string what = c.name;
    check(got.dir.empty(), what + ": refused");
    check(
        mentions(got, "group- or other-writable"),
        what + ": names the mode, got: " + got.refusal);
    check(!blob_loaded(got.dir, &source), what + ": no blob loaded");
    check(mode_of(leaf) == c.leaf_mode, what + ": leaf mode not changed");
    check(mode_of(over) == c.parent_mode, what + ": parent mode not changed");
    ::chmod(leaf.c_str(), 0700);
    ::chmod(over.c_str(), 0700);
  }

  // A pre-created directory owned by another user. Owner rules are the real
  // walk with real lstat results: only the identity asked about differs.
  {
    const auto over = private_dir(base / "foreign");
    const auto leaf = private_dir(over / kArch);
    plant_entry(leaf, source);
    std::uint32_t asking = euid + 1;
    if (euid == 0) {
      // Root-owned components are trusted, so as root the directory itself
      // has to change hands.
      check(::chown(leaf.c_str(), 65534, 65534) == 0, "chown as root");
      asking = 0;
    }
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, asking);
    check(got.dir.empty(), "foreign owner: refused");
    check(
        mentions(got, "is not owned by the current user"),
        "foreign owner: names the owner, got: " + got.refusal);
    check(!blob_loaded(got.dir, &source), "foreign owner: no blob loaded");
    if (euid == 0) {
      check(::chown(leaf.c_str(), 0, 0) == 0, "chown back as root");
    }
  }

  // A symlink at a component only mlxcel creates is refused whatever it points
  // at, and a blob behind it is not loaded (make_owned_directory's rule).
  for (unsigned target_mode : {0777u, 0700u}) {
    const std::string tag = std::to_string(target_mode);
    const auto target = private_dir(base / ("link-target" + tag));
    private_dir(target / kArch);
    plant_entry(target / kArch, source);
    ::chmod((target / kArch).c_str(), target_mode);
    const auto over = private_dir(base / ("link-over" + tag));
    fs::create_directory_symlink(target / kArch, over / kArch);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "a symlink at <arch>: refused, mode " + tag);
    check(
        mentions(got, "is a symbolic link"),
        "a symlink at <arch>: says so, got: " + got.refusal);
    check(mode_of(target / kArch) == target_mode, "the target is not changed");
    ::chmod((target / kArch).c_str(), 0700);
  }

  // The directory the user names may be a symlink, judged by its target.
  {
    const auto target = private_dir(base / "named-target");
    const auto link = base / "named-link";
    fs::create_directory_symlink(target, link);
    const auto got =
        resolve_hsaco_cache_dir({link, {}, {}}, kVersion, kArch, euid);
    check(got.dir == fs::canonical(target) / kArch, "a named symlink is followed");
  }

  // A regular file where the cache directory should be.
  {
    const auto over = private_dir(base / "file-leaf");
    write_file(over / kArch, "not a directory");
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "a file at the leaf: refused");
    check(
        mentions(got, "is not a directory"),
        "a file at the leaf: says so, got: " + got.refusal);
  }

  // A refused run creates nothing under the refused ancestor.
  {
    const auto over = private_dir(base / "no-leftovers", 0777);
    const auto missing = over / "named" / "dirs";
    const auto got =
        resolve_hsaco_cache_dir({missing, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "a shared ancestor of a missing path: refused");
    check(!fs::exists(over / "named"), "a refused run leaves no directories");
    ::chmod(over.c_str(), 0700);
  }
}

// The per-user default, under a umask that would make everything group-writable.
void test_default_location(const fs::path& base) {
  const auto euid = cache_current_euid();
  const mode_t old_umask = ::umask(002);
  const std::string source = "src";

  // Fresh: every component it creates is 0700, the path is under the home.
  {
    const auto home = private_dir(base / "home-fresh");
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    const auto expected = fs::canonical(home) / ".cache" / "mlxcel" / "hsaco" /
        kVersion / kArch;
    check(got.dir == expected, "default path: " + got.dir.string());
    check(got.refusal.empty(), "default path has no refusal: " + got.refusal);
    for (auto p = expected; p != fs::canonical(home); p = p.parent_path()) {
      check(mode_of(p) == 0700, "created private: " + p.string());
    }
  }

  // Directories an earlier release made under this umask are repaired, the
  // ones mlxcel manages only: the root, hsaco, <version> and <arch>.
  {
    const auto home = private_dir(base / "home-repair");
    const auto dot_cache = private_dir(home / ".cache", 0755);
    const auto root = private_dir(dot_cache / "mlxcel", 0775);
    const auto hsaco = private_dir(root / "hsaco", 0775);
    const auto ver = private_dir(hsaco / kVersion, 02775);
    const auto arch = private_dir(ver / kArch, 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(!got.dir.empty(), "repaired default is used: " + got.refusal);
    check(mode_of(root) == 0755, "root repaired");
    check(mode_of(hsaco) == 0755, "hsaco repaired");
    check(mode_of(ver) == 02755, "version repaired, other bits kept");
    check(mode_of(arch) == 0755, "arch repaired");
  }

  // Other-write is not repaired on any managed component, and is refused.
  for (const char* which : {"root", "hsaco", "version", "arch"}) {
    const std::string tag = which;
    const auto home = private_dir(base / ("home-world-" + tag));
    const auto root = private_dir(home / ".cache" / "mlxcel");
    ::chmod((home / ".cache").c_str(), 0755);
    const auto hsaco = private_dir(root / "hsaco");
    const auto ver = private_dir(hsaco / kVersion);
    const auto arch = private_dir(ver / kArch);
    plant_entry(arch, source);
    const fs::path& shared = tag == "root" ? root
        : tag == "hsaco"                   ? hsaco
        : tag == "version"                 ? ver
                                           : arch;
    ::chmod(shared.c_str(), 0777);
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(got.dir.empty(), "world-writable " + tag + ": refused");
    check(mentions(got, "group- or other-writable"), "says why: " + got.refusal);
    check(mode_of(shared) == 0777, "world-writable " + tag + ": not repaired");
    check(!blob_loaded(got.dir, &source), "world-writable " + tag + ": no blob");
    ::chmod(shared.c_str(), 0700);
  }

  // Nothing above the root is touched, so a group-writable ~/.cache is refused,
  // and nothing is created under it.
  {
    const auto home = private_dir(base / "home-shared-cache");
    const auto dot_cache = private_dir(home / ".cache", 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(got.dir.empty(), "group-writable ~/.cache: refused");
    check(mode_of(dot_cache) == 0775, "~/.cache not changed");
    check(!fs::exists(dot_cache / "mlxcel"), "nothing created under ~/.cache");
  }

  // A group-writable ~/.cache also stops the repair of what lies below it.
  {
    const auto home = private_dir(base / "home-no-repair");
    const auto dot_cache = private_dir(home / ".cache", 0775);
    const auto root = private_dir(dot_cache / "mlxcel", 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(got.dir.empty(), "ancestor refused: refused");
    check(mode_of(root) == 0775, "no repair below a refused ancestor");
  }

  // An explicit MLXCEL_CACHE_DIR is not repaired, and nothing below a refused
  // one is repaired or created.
  {
    const auto root = private_dir(base / "explicit-root", 0775);
    const auto hsaco = private_dir(root / "hsaco", 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, root, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "group-writable MLXCEL_CACHE_DIR: refused");
    check(mode_of(root) == 0775, "MLXCEL_CACHE_DIR not changed");
    check(mode_of(hsaco) == 0775, "hsaco under a refused root not repaired");
    check(!fs::exists(hsaco / kVersion), "nothing created under a refused root");
  }

  // Under a private MLXCEL_CACHE_DIR, what mlxcel keeps is repaired.
  {
    const auto root = private_dir(base / "explicit-private");
    const auto hsaco = private_dir(root / "hsaco", 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, root, {}}, kVersion, kArch, euid);
    check(!got.dir.empty(), "private MLXCEL_CACHE_DIR: " + got.refusal);
    check(mode_of(hsaco) == 0755, "hsaco under a private root repaired");
  }

  // An explicit override is never repaired, not even its <arch>.
  {
    const auto over = private_dir(base / "explicit-over");
    const auto leaf = private_dir(over / kArch, 0775);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "group-writable override: refused");
    check(mode_of(leaf) == 0775, "override not changed");
  }

  // A `..` creates no stray directory.
  {
    const auto root = private_dir(base / "dotdot");
    const auto got = resolve_hsaco_cache_dir(
        {root / "missing" / ".." / "cache", {}, {}}, kVersion, kArch, euid);
    check(!got.dir.empty(), "`..` in the override: " + got.refusal);
    check(!fs::exists(root / "missing"), "`..` left no stray directory");
  }

  // An intermediate symlink stops the walk, whatever it points at, and nothing
  // is chmodded through it: not the link's target, not what lies below it.
  {
    const auto root = private_dir(base / "intermediate-root");
    const auto elsewhere = private_dir(base / "intermediate-elsewhere", 0775);
    const auto below = private_dir(elsewhere / kVersion / kArch, 0775);
    ::chmod((elsewhere / kVersion).c_str(), 0775);
    fs::create_directory_symlink(elsewhere, root / "hsaco");
    const auto got =
        resolve_hsaco_cache_dir({{}, root, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "an intermediate symlink: refused");
    check(mentions(got, "is a symbolic link"), "says so: " + got.refusal);
    check(mode_of(elsewhere) == 0775, "the link's target is not chmodded");
    check(mode_of(elsewhere / kVersion) == 0775, "below the link: not chmodded");
    check(mode_of(below) == 0775, "the leaf below the link: not chmodded");
  }
  {
    // The same one level down, with a private target.
    const auto root = private_dir(base / "intermediate-root2");
    const auto hsaco = private_dir(root / "hsaco");
    const auto elsewhere = private_dir(base / "intermediate-elsewhere2");
    private_dir(elsewhere / kArch);
    plant_entry(elsewhere / kArch, source);
    fs::create_directory_symlink(elsewhere, hsaco / kVersion);
    const auto got =
        resolve_hsaco_cache_dir({{}, root, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "a <version> link to a private target: refused");
    check(mentions(got, "is a symbolic link"), "says so: " + got.refusal);
  }

  // ~/.cache/mlxcel may be a symlink (to a larger disk), judged by its target,
  // which is not repaired.
  {
    const auto home = private_dir(base / "home-root-link");
    private_dir(home / ".cache", 0755);
    const auto wide = private_dir(base / "root-link-wide", 0775);
    fs::create_directory_symlink(wide, home / ".cache" / "mlxcel");
    auto got = resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(got.dir.empty(), "a root link to a shared target: refused");
    check(mode_of(wide) == 0775, "the root link's target is not repaired");
    ::chmod(wide.c_str(), 0700);
    got = resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(!got.dir.empty(), "a root link to a private target: " + got.refusal);
  }

  ::umask(old_umask);
}

// Entries inside a trusted directory.
void test_entries(const fs::path& base) {
  const auto euid = cache_current_euid();
  const mode_t old_umask = ::umask(002);
  const std::string source = "src";
  const auto dir = private_dir(base / "entries");
  const auto entry = [&](const char* ext) {
    return dir / (std::string(kModule) + ext);
  };

  // Written entries are private whatever the umask, and read back.
  const std::vector<std::pair<std::string, std::string>> names = {{"k", "m"}};
  write_cached_hsaco(dir, kModule, "BLOB", names, source);
  for (const char* ext : {".hsaco", ".txt", ".hip"}) {
    check(mode_of(entry(ext)) == 0600, std::string(ext) + " is 0600");
  }
  check(blob_loaded(dir, &source), "a private entry loads");

  // Any shared-write entry is a miss: the blob, the map and the source.
  for (const char* ext : {".hsaco", ".txt", ".hip"}) {
    ::chmod(entry(ext).c_str(), 0664);
    check(!blob_loaded(dir, &source), std::string(ext) + " group-writable: miss");
    ::chmod(entry(ext).c_str(), 0666);
    check(!blob_loaded(dir, &source), std::string(ext) + " world-writable: miss");
    ::chmod(entry(ext).c_str(), 0600);
  }
  check(blob_loaded(dir, &source), "restored entry loads again");

  // An entry owned by someone else is a miss (euid is the only input).
  {
    std::string contents;
    check(
        !read_trusted_cache_entry(
            entry(".hsaco"), contents, euid == 0 ? 1 : euid + 1) ||
            euid == 0,
        "an entry owned by another user is a miss");
    check(
        read_trusted_cache_entry(entry(".hsaco"), contents, euid),
        "the owner's entry reads");
    check(contents == "BLOB", "the entry's contents");
  }

  // A symlink or a FIFO at the entry's name is a miss and does not block.
  {
    const auto outside = base / "outside.hsaco";
    write_file(outside, "PLANTED");
    ::chmod(outside.c_str(), 0600);
    fs::remove(entry(".hsaco"));
    fs::create_symlink(outside, entry(".hsaco"));
    check(!blob_loaded(dir, &source), "a symlinked entry: miss");

    // A publish replaces the link instead of writing through it.
    write_cached_hsaco(dir, kModule, "BLOB", names, source);
    check(!fs::is_symlink(entry(".hsaco")), "a publish replaces the symlink");
    std::string outside_after;
    {
      std::ifstream file(outside, std::ios::binary);
      std::getline(file, outside_after);
    }
    check(outside_after == "PLANTED", "a publish does not write through it");
    check(blob_loaded(dir, &source), "the replacement loads");

    fs::remove(entry(".hsaco"));
    check(::mkfifo(entry(".hsaco").c_str(), 0600) == 0, "mkfifo");
    check(!blob_loaded(dir, &source), "a FIFO entry: miss, no hang");
    fs::remove(entry(".hsaco"));
  }

  // An older, wider entry is replaced by the next publish, not kept.
  {
    write_file(entry(".hsaco"), "OLD");
    ::chmod(entry(".hsaco").c_str(), 0664);
    write_cached_hsaco(dir, kModule, "BLOB", names, source);
    check(mode_of(entry(".hsaco")) == 0600, "a publish narrows the entry");
    check(blob_loaded(dir, &source), "the republished entry loads");
  }

  // No publish temporaries are left behind.
  for (const auto& e : fs::directory_iterator(dir)) {
    check(
        e.path().filename().string().rfind(".hsaco-tmp-", 0) != 0,
        "stray temporary: " + e.path().string());
  }

  // Orphans at the next temporary names, one of them a symlink, are what a later
  // process with this pid finds after a crash. The publish moves past them and
  // leaves them in place, and writes through nothing.
  {
    const auto orphans = private_dir(base / "orphans");
    const auto name = [&](std::uint64_t counter) {
      return orphans /
          (".hsaco-tmp-" + std::to_string(::getpid()) + "-" +
           std::to_string(counter) + ".hsaco");
    };
    const std::uint64_t next = cache_publish_counter().load();
    write_file(name(next), "orphan");
    write_file(name(next + 1), "orphan");
    fs::create_symlink(base / "orphan-target", name(next + 2));
    check(
        publish_cache_entry(orphans / "real.hsaco", "BLOB", 4),
        "a publish moves past orphans");
    check(!fs::exists(base / "orphan-target"), "no link was written through");
    check(fs::exists(name(next)), "the orphan is left in place");
    std::string contents;
    check(
        read_trusted_cache_entry(orphans / "real.hsaco", contents, euid) &&
            contents == "BLOB",
        "the published entry reads back");

    // Every name taken: the publish gives up within its bound, publishing nothing.
    const std::uint64_t from = cache_publish_counter().load();
    for (std::uint64_t i = 0; i < 64; ++i) {
      write_file(name(from + i), "orphan");
    }
    check(
        !publish_cache_entry(orphans / "other.hsaco", "BLOB", 4),
        "a publish gives up when no name is free");
    check(!fs::exists(orphans / "other.hsaco"), "nothing was published");
  }
  ::umask(old_umask);
}

void test_platform(const fs::path& base) {
  test_refused_directories(base);
  test_default_location(base);
  test_entries(base);
}
