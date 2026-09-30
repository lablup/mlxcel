#!/usr/bin/env python3
"""Maintenance tooling for the mlxcelverse ROCm overlay (issue #1813).

The overlay in src/lib/mlx-cpp/patches-rocm/ has two moving upstreams: the MLX
pin (ml-explore/mlx) and the fork it vendors its backend from
(NripeshN/mlx@rocm-support). `UPSTREAM` records the three commits that define
it: the fork commit, the fork's merge base with upstream MLX, and the MLX pin
the core files were retargeted to. Every subcommand works from those records.

Subcommands:

  records   Offline and fast; run by `make verify-rocm-overlay`. Checks that
            the records agree with each other and with the tree: UPSTREAM is
            well formed and its pin is the build's pin, the README counts are
            the real ones, LOCAL_FIXES.md is numbered without gaps, and
            CORE_RESIDUAL.diff was generated for the current commits and the
            current content of every core overlay file.
  drift     Needs the three recorded commits (fetched into a cache). Rebuilds
            each core file as "MLX pin + the fork's own change to that file"
            with a 3-way merge and diffs the overlay against it. Whatever is
            left is the overlay's residual: changes that are neither upstream
            nor the fork's. It must match CORE_RESIDUAL.diff, where every file
            carries a note naming its LOCAL_FIXES.md item. Also checks that
            every backend file that differs from the fork, and every fork hook
            the overlay does not carry, is named in LOCAL_FIXES.md.
  sync      Moves the overlay to a new fork commit: 3-way merges each backend
            file (old fork, overlay, new fork) and each core file ("pin + old
            fork delta", overlay, "pin + new fork delta"), carries local-only
            files, and updates UPSTREAM. `--check` syncs into a scratch copy
            and compares it with the committed overlay byte for byte.
  retarget  Moves the core files to a new MLX pin by 3-way merging each one
            (old pin, overlay, new pin) and updates UPSTREAM.
  export-tree
            Writes the MLX tree at a commit with the ROCm files laid over it,
            for scripts/mlxcelverse/check_api_drift.sh.
  api-report
            Summarizes a build log and `nm` output into compile errors and
            undefined mlx::core symbols, for check_api_drift.sh.

Git objects come from a cache repository (default
~/.cache/mlxcel/mlxcelverse/mlx.git, override with --git-dir or
MLXCELVERSE_GIT_DIR). It is created and fetched on demand unless --no-fetch is
given. Nothing is ever pushed.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OVERLAY = REPO_ROOT / "src" / "lib" / "mlx-cpp" / "patches-rocm"
PIN_SCRIPT = REPO_ROOT / "scripts" / "ci" / "mlx_pinned_commit.sh"

# Files at the overlay root that are records, never copied into MLX.
RECORDS = ("README.md", "UPSTREAM", "LOCAL_FIXES.md", "CORE_RESIDUAL.diff")
RESIDUAL_FILE = "CORE_RESIDUAL.diff"
# Files a desktop drops into any directory it shows (Finder on macOS, where
# `make verify` runs). Neither overlay files nor records; CMake copies them
# into the MLX tree harmlessly, so they must not fail the records check or
# count as core files.
IGNORED_NAMES = (".DS_Store",)
BACKEND_PREFIX = "mlx/backend/rocm/"
UPSTREAM_KEYS = (
    "source",
    "branch",
    "commit",
    "fork_upstream_merge_base",
    "retargeted_to_mlx_pin",
)
SHA_KEYS = ("commit", "fork_upstream_merge_base", "retargeted_to_mlx_pin")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

DEFAULT_FORK_URL = "https://github.com/NripeshN/mlx.git"
DEFAULT_FORK_BRANCH = "rocm-support"
DEFAULT_MLX_URL = "https://github.com/ml-explore/mlx.git"
DEFAULT_MLX_BRANCH = "main"

NOTE_TODO = "TODO: name the LOCAL_FIXES.md item this residual implements"


class ToolError(Exception):
    """A failure with a message for the user; exits 2."""


# --------------------------------------------------------------------------
# Small helpers


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def text(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogateescape")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unified_diff(old: bytes, new: bytes, path: str) -> str:
    """A `diff -u` style patch from old to new, stable across hosts."""
    a = text(old).splitlines(keepends=True)
    b = text(new).splitlines(keepends=True)
    out = []
    for line in difflib.unified_diff(a, b, f"a/{path}", f"b/{path}", n=3):
        if line.startswith(("---", "+++")):
            out.append(line.rstrip("\n") + "\n")
            continue
        if not line.endswith("\n"):
            out.append(line + "\n\\ No newline at end of file\n")
        else:
            out.append(line)
    return "".join(out)


def normalize_patch(patch: str) -> str:
    """Drop hunk line numbers so a pure line shift is not a residual change."""
    return re.sub(r"(?m)^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", "@@", patch)


def safe_dest(root: Path, rel: str) -> Path:
    """`root / rel` for a path read from a git tree, refusing anything that
    would land outside `root`. Tree entries come from fetched objects (the fork
    or a --fork-url), and git stores and lists a `..` entry verbatim, so a
    hostile tree could otherwise make sync or export-tree write, or delete, a
    file anywhere the user can. Rejects absolute paths, empty, `.`, `..` and
    `.git` components, and a destination that resolves outside `root` through
    a symlink already on disk (export-tree extracts MLX, symlinks included)."""
    parts = rel.split("/")
    if (
        not rel
        or rel.startswith("/")
        or "\\" in rel
        or "\0" in rel
        or any(part in ("", ".", "..") or part.lower() == ".git" for part in parts)
    ):
        raise ToolError(f"refusing unsafe path {rel!r} from a git tree")
    dest = root / rel
    base = root.resolve()
    resolved = dest.resolve()
    if resolved != base and base not in resolved.parents:
        raise ToolError(f"refusing {rel!r}: it resolves to {resolved}, outside {base}")
    return dest


def patch_stats(patch: str) -> tuple[int, int]:
    plus = minus = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            plus += 1
        elif line.startswith("-"):
            minus += 1
    return plus, minus


# --------------------------------------------------------------------------
# Overlay model


@dataclass
class Overlay:
    root: Path

    def files(self) -> list[str]:
        """Every overlay file (not records), as paths relative to the root."""
        out = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root).as_posix()
            if rel in RECORDS or p.name in IGNORED_NAMES:
                continue
            out.append(rel)
        return out

    def core_files(self) -> list[str]:
        return [f for f in self.files() if not f.startswith(BACKEND_PREFIX)]

    def backend_files(self) -> list[str]:
        return [f for f in self.files() if f.startswith(BACKEND_PREFIX)]

    def read(self, rel: str) -> bytes:
        return (self.root / rel).read_bytes()

    def upstream(self) -> dict[str, str]:
        return parse_upstream((self.root / "UPSTREAM").read_text())


def parse_upstream(content: str) -> dict[str, str]:
    values = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ToolError(f"UPSTREAM: malformed line {line!r}")
        values[key.strip()] = value.strip()
    return values


def update_upstream(content: str, changes: dict[str, str]) -> str:
    out = []
    seen = set()
    for line in content.splitlines(keepends=True):
        key, sep, _ = line.partition(":")
        k = key.strip()
        if sep and k in changes:
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{k}: {changes[k]}{nl}")
            seen.add(k)
        else:
            out.append(line)
    missing = set(changes) - seen
    if missing:
        raise ToolError(f"UPSTREAM: missing keys {sorted(missing)}")
    return "".join(out)


def pinned_commit() -> str:
    """The build's MLX pin, from the single source of truth."""
    proc = subprocess.run(
        ["bash", str(PIN_SCRIPT)], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise ToolError(f"mlx_pinned_commit.sh failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------
# Residual record


@dataclass
class Residual:
    pin: str = ""
    fork: str = ""
    base: str = ""
    hashes: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    body: str = ""  # the concatenated per-file patches


RESIDUAL_PREAMBLE = """\
# CORE_RESIDUAL.diff: what the ROCm overlay's MLX core files carry beyond the
# MLX pin plus the fork's own change to each file.
#
# Generated by `scripts/mlxcelverse/rocm_overlay.py drift --write`; do not
# edit the patches by hand. For each core overlay file the tool rebuilds
# "pin + fork delta" (a 3-way merge of the fork commit into the pin, using the
# fork's merge base with upstream as the base; conflict markers stay in) and
# diffs the overlay against it. A file that matches has no entry. Every
# entry needs a `# note <path>:` line naming the LOCAL_FIXES.md item that
# explains it; `drift --write` keeps existing notes. `rocm_overlay.py records`
# (make verify-rocm-overlay) fails when the commits below are not the ones in
# UPSTREAM or a core file's hash no longer matches, so any change to a core
# overlay, a pin bump or a fork sync has to re-run `drift` and review this
# file. That review is the line-by-line comparison a count of +/- lines cannot
# replace: an overlay that silently drops part of an upstream change shows up
# here as a new hunk, even if it was wrong before the bump.
"""


def parse_residual(content: str) -> Residual:
    res = Residual()
    lines = content.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].startswith("--- "):
        line = lines[i].rstrip("\n")
        i += 1
        if not line.startswith("#"):
            if line.strip():
                raise ToolError(f"{RESIDUAL_FILE}: unexpected header line {line!r}")
            continue
        body = line[1:].strip()
        for key, attr in (("pin:", "pin"), ("fork:", "fork"), ("fork_upstream_merge_base:", "base")):
            if body.startswith(key):
                setattr(res, attr, body[len(key):].strip())
        if body.startswith("overlay-sha256:"):
            parts = body[len("overlay-sha256:"):].split()
            if len(parts) != 2:
                raise ToolError(f"{RESIDUAL_FILE}: malformed hash line {line!r}")
            res.hashes[parts[1]] = parts[0]
        elif body.startswith("note "):
            path, sep, note = body[len("note "):].partition(":")
            if not sep:
                raise ToolError(f"{RESIDUAL_FILE}: malformed note line {line!r}")
            res.notes[path.strip()] = note.strip()
    res.body = "".join(lines[i:])
    return res


def residual_files(body: str) -> list[str]:
    return [
        m.group(1)
        for m in re.finditer(r"(?m)^\+\+\+ b/(\S+)$", body)
    ]


def render_residual(res: Residual) -> str:
    out = [RESIDUAL_PREAMBLE, "#\n"]
    out.append(f"# pin: {res.pin}\n")
    out.append(f"# fork: {res.fork}\n")
    out.append(f"# fork_upstream_merge_base: {res.base}\n")
    for path in sorted(res.hashes):
        out.append(f"# overlay-sha256: {res.hashes[path]} {path}\n")
    for path in residual_files(res.body):
        out.append(f"# note {path}: {res.notes.get(path, NOTE_TODO)}\n")
    out.append("\n")
    out.append(res.body)
    return "".join(out)


# --------------------------------------------------------------------------
# Git access


class Git:
    def __init__(self, git_dir: Path, fetch: bool):
        self.git_dir = git_dir
        self.fetch_enabled = fetch

    def run(self, *args: str, check: bool = True, input: bytes | None = None) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", f"--git-dir={self.git_dir}", *args],
            capture_output=True,
            input=input,
            check=False,
        )
        if check and proc.returncode != 0:
            raise ToolError(
                f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}"
            )
        return proc

    def ensure_repo(self, fork_url: str, mlx_url: str) -> None:
        if (self.git_dir / "HEAD").exists():
            return
        if not self.fetch_enabled:
            raise ToolError(
                f"no git cache at {self.git_dir}; run without --no-fetch once or pass --git-dir"
            )
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "--bare", str(self.git_dir)], check=True)
        self.run("remote", "add", "fork", fork_url)
        self.run("remote", "add", "upstream", mlx_url)

    def has_commit(self, sha: str) -> bool:
        return self.run("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0

    def fetch(self, remote: str, *refspecs: str) -> None:
        if not self.fetch_enabled:
            raise ToolError(f"missing objects and --no-fetch given (wanted {refspecs} from {remote})")
        for spec in refspecs:
            # A value starting with "-" would be parsed as a git option
            # (for example --upload-pack=...), so refuse it outright.
            if spec.startswith("-"):
                raise ToolError(f"refusing to fetch {spec!r}: not a commit or ref")
        print(f"fetching {' '.join(refspecs)} from {remote} ...", file=sys.stderr)
        self.run("fetch", "-q", "--no-tags", remote, *refspecs)

    def ensure_commit(self, sha: str, remote: str) -> None:
        if not self.has_commit(sha):
            self.fetch(remote, sha)
            if not self.has_commit(sha):
                raise ToolError(f"commit {sha} not found on {remote}")

    def resolve(self, rev: str) -> str:
        return text(self.run("rev-parse", "--verify", f"{rev}^{{commit}}").stdout).strip()

    def show(self, sha: str, path: str) -> bytes | None:
        proc = self.run("cat-file", "blob", f"{sha}:{path}", check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout

    # -z: names come back raw, not C-quoted (core.quotePath quotes non-ASCII
    # and control characters), so they can be looked up and validated as is.
    def ls_tree(self, sha: str, prefix: str) -> list[str]:
        out = self.run("ls-tree", "-r", "-z", "--name-only", sha, "--", prefix).stdout
        return [name for name in text(out).split("\0") if name]

    def changed_paths(self, a: str, b: str) -> list[str]:
        out = self.run("diff", "-z", "--name-only", "--no-renames", a, b).stdout
        return [name for name in text(out).split("\0") if name]

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.run("merge-base", "--is-ancestor", a, b, check=False).returncode == 0

    def merge_base(self, a: str, b: str) -> str:
        return text(self.run("merge-base", a, b).stdout).strip()


def merge3(ours: bytes, base: bytes, theirs: bytes, labels: tuple[str, str, str]) -> tuple[bytes, int]:
    """`git merge-file -p`: returns the merged content and the conflict count."""
    if base == theirs:
        return ours, 0
    if base == ours:
        return theirs, 0
    with tempfile.TemporaryDirectory(prefix="mlxcelverse-merge-") as tmp:
        paths = []
        for name, data in (("ours", ours), ("base", base), ("theirs", theirs)):
            p = Path(tmp) / name
            p.write_bytes(data)
            paths.append(str(p))
        proc = subprocess.run(
            ["git", "merge-file", "-p", "-L", labels[0], "-L", labels[1], "-L", labels[2], *paths],
            capture_output=True,
            check=False,
        )
    if proc.returncode < 0 or proc.returncode > 127:
        raise ToolError(f"git merge-file failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout, proc.returncode


def reconstruct(git: Git, pin: str, base: str, fork: str, path: str) -> tuple[bytes, int]:
    """The pin's copy of `path` with the fork's own change to it merged in."""
    pin_b = git.show(pin, path)
    base_b = git.show(base, path)
    fork_b = git.show(fork, path)
    if fork_b is None:
        # The fork does not have the file; the reconstruction is the pin's.
        return pin_b or b"", 0
    return merge3(pin_b or b"", base_b or b"", fork_b, ("pin", "fork-merge-base", "fork"))


# --------------------------------------------------------------------------
# Shared context


@dataclass
class Context:
    overlay: Overlay
    git: Git | None
    fork_url: str
    mlx_url: str


def make_context(args: argparse.Namespace, need_git: bool) -> Context:
    overlay = Overlay(Path(args.overlay_dir).resolve())
    if not (overlay.root / "UPSTREAM").is_file():
        raise ToolError(f"{overlay.root} has no UPSTREAM record")
    git = None
    if need_git:
        git_dir = Path(
            args.git_dir
            or os.environ.get("MLXCELVERSE_GIT_DIR")
            or Path.home() / ".cache" / "mlxcel" / "mlxcelverse" / "mlx.git"
        )
        git = Git(git_dir, fetch=not args.no_fetch)
        git.ensure_repo(args.fork_url, args.mlx_url)
    return Context(overlay, git, args.fork_url, args.mlx_url)


def ensure_recorded_commits(ctx: Context, up: dict[str, str]) -> None:
    assert ctx.git is not None
    ctx.git.ensure_commit(up["commit"], "fork")
    ctx.git.ensure_commit(up["fork_upstream_merge_base"], "fork")
    ctx.git.ensure_commit(up["retargeted_to_mlx_pin"], "upstream")


def local_fixes_mentions(overlay: Overlay, path: str) -> bool:
    """True if LOCAL_FIXES.md names `path`: by full path, by backend-relative
    path when that has a directory (`quantized/qmm.hip`), or by bare file name
    when no other overlay file has that name. A short name only counts on its
    own, not as the tail of a longer path, so `mlx/device.cpp` does not name
    `mlx/backend/rocm/device.cpp` and the top-level `CMakeLists.txt` does not
    name `mlx/backend/rocm/CMakeLists.txt`."""
    content = (overlay.root / "LOCAL_FIXES.md").read_text()
    if re.search(rf"(?<![\w.-]){re.escape(path)}(?![\w-])", content):
        return True
    name = Path(path).name
    short = []
    if path.startswith(BACKEND_PREFIX):
        rel = path[len(BACKEND_PREFIX):]
        if "/" in rel:
            short.append(rel)
    if not any(Path(f).name == name and f != path for f in overlay.files()):
        short.append(name)
    for candidate in short:
        if re.search(rf"(?<![\w./-]){re.escape(candidate)}(?![\w-])", content):
            return True
    return False


# --------------------------------------------------------------------------
# records


def cmd_records(args: argparse.Namespace) -> int:
    overlay = Overlay(Path(args.overlay_dir).resolve())
    errors: list[str] = []
    root = overlay.root

    # Stray files at the overlay root are neither copied nor records.
    for p in sorted(root.iterdir()):
        name = p.name
        if name in RECORDS or name in IGNORED_NAMES or name in ("CMakeLists.txt", "mlx"):
            continue
        errors.append(
            f"{name}: not a record ({', '.join(RECORDS)}) and not an overlay path (CMakeLists.txt, mlx/); "
            "CMake would silently ignore it"
        )
    for rec in RECORDS:
        if not (root / rec).is_file():
            errors.append(f"{rec}: missing")
    if errors:
        return report("records", errors)

    # UPSTREAM.
    up = overlay.upstream()
    for key in UPSTREAM_KEYS:
        if key not in up:
            errors.append(f"UPSTREAM: missing `{key}`")
    for key in SHA_KEYS:
        if key in up and not SHA_RE.match(up[key]):
            errors.append(f"UPSTREAM: `{key}` is {up[key]!r}, not a full 40-character lowercase SHA")
    pin = args.pin or pinned_commit()
    if up.get("retargeted_to_mlx_pin") and up["retargeted_to_mlx_pin"] != pin:
        errors.append(
            f"UPSTREAM: retargeted_to_mlx_pin is {up['retargeted_to_mlx_pin']} but the MLX pin is {pin}. "
            "A pin bump has to retarget the ROCm overlay too: run "
            "`scripts/mlxcelverse/rocm_overlay.py retarget`, then `drift` (docs/mlxcelverse/rocm-overlay.md)."
        )

    # README counts.
    readme = (root / "README.md").read_text()
    backend = overlay.backend_files()
    core = overlay.core_files()
    m = re.search(r"`mlx/backend/rocm/\*\*`: the ROCm backend \((\d+) files\)", readme)
    if not m:
        errors.append("README.md: no '`mlx/backend/rocm/**`: the ROCm backend (N files)' line")
    elif int(m.group(1)) != len(backend):
        errors.append(f"README.md: says the backend has {m.group(1)} files; it has {len(backend)}")
    m = re.search(r"(\d+) MLX core files", readme)
    if not m:
        errors.append("README.md: no 'N MLX core files' statement")
    elif int(m.group(1)) != len(core):
        errors.append(f"README.md: says {m.group(1)} MLX core files; the overlay has {len(core)}")
    for path in core:
        # Every core file must be listed, in the README's brace-list form or plainly.
        name = Path(path).name
        stem = Path(path).stem
        if name not in readme and stem not in readme:
            errors.append(f"README.md: core file {path} is not listed")

    # LOCAL_FIXES numbering.
    fixes = (root / "LOCAL_FIXES.md").read_text()
    numbers = [int(n) for n in re.findall(r"(?m)^(\d+)\. \*\*", fixes)]
    if not numbers:
        errors.append("LOCAL_FIXES.md: no numbered entries")
    elif sorted(numbers) != list(range(1, len(numbers) + 1)):
        # Entries are grouped by section (core, kernels, build), so a new
        # kernel fix can sit above an older build fix; what matters is that
        # every number is used exactly once with no gap.
        errors.append(
            f"LOCAL_FIXES.md: entries are numbered {numbers}; expected each of 1..{len(numbers)} exactly once"
        )

    # Residual record.
    res = parse_residual((root / RESIDUAL_FILE).read_text())
    for attr, key in (("pin", "retargeted_to_mlx_pin"), ("fork", "commit"), ("base", "fork_upstream_merge_base")):
        if getattr(res, attr) != up.get(key):
            errors.append(
                f"{RESIDUAL_FILE}: generated for {attr} {getattr(res, attr) or '(none)'} but UPSTREAM has "
                f"{key} {up.get(key)}; re-run `rocm_overlay.py drift`, review, then `drift --write`"
            )
    if set(res.hashes) != set(core):
        missing = sorted(set(core) - set(res.hashes))
        extra = sorted(set(res.hashes) - set(core))
        errors.append(f"{RESIDUAL_FILE}: core file set differs (not hashed: {missing}, no longer core: {extra})")
    for path in core:
        if path in res.hashes and res.hashes[path] != sha256(overlay.read(path)):
            errors.append(
                f"{RESIDUAL_FILE}: {path} changed since the residual was generated; run "
                "`rocm_overlay.py drift`, review the residual, then `drift --write`"
            )
    files = residual_files(res.body)
    for path in files:
        if path not in core:
            errors.append(f"{RESIDUAL_FILE}: residual for {path}, which is not a core overlay file")
        note = res.notes.get(path, "")
        if not note or note.startswith("TODO"):
            errors.append(f"{RESIDUAL_FILE}: {path} has a residual but no note naming its LOCAL_FIXES.md item")
    for path in res.notes:
        if path not in files:
            errors.append(f"{RESIDUAL_FILE}: note for {path}, which has no residual")

    if not errors:
        print(
            f"[rocm-overlay] records OK: {len(backend)} backend + {len(core)} core files, "
            f"{len(numbers)} LOCAL_FIXES entries, {len(files)} core files with a noted residual, pin {pin[:8]}"
        )
    return report("records", errors)


def report(what: str, errors: list[str]) -> int:
    for e in errors:
        print(f"[rocm-overlay] {what}: {e}", file=sys.stderr)
    return 1 if errors else 0


# --------------------------------------------------------------------------
# drift


def compute_residual(ctx: Context, up: dict[str, str]) -> tuple[str, dict[str, int]]:
    assert ctx.git is not None
    patches = []
    conflicts = {}
    for path in ctx.overlay.core_files():
        recon, n = reconstruct(
            ctx.git, up["retargeted_to_mlx_pin"], up["fork_upstream_merge_base"], up["commit"], path
        )
        if n:
            conflicts[path] = n
        patch = unified_diff(recon, ctx.overlay.read(path), path)
        if patch:
            patches.append(patch)
    return "".join(patches), conflicts


def cmd_drift(args: argparse.Namespace) -> int:
    ctx = make_context(args, need_git=True)
    assert ctx.git is not None
    up = ctx.overlay.upstream()
    ensure_recorded_commits(ctx, up)
    errors: list[str] = []
    overlay = ctx.overlay
    fork, base, pin = up["commit"], up["fork_upstream_merge_base"], up["retargeted_to_mlx_pin"]

    if not ctx.git.is_ancestor(base, fork):
        errors.append(f"UPSTREAM: fork_upstream_merge_base {base} is not an ancestor of commit {fork}")

    # Core files.
    body, conflicts = compute_residual(ctx, up)
    record_path = overlay.root / RESIDUAL_FILE
    old = parse_residual(record_path.read_text()) if record_path.is_file() else Residual()
    new = Residual(
        pin=pin,
        fork=fork,
        base=base,
        hashes={p: sha256(overlay.read(p)) for p in overlay.core_files()},
        notes={p: n for p, n in old.notes.items() if p in residual_files(body)},
        body=body,
    )
    print(f"[rocm-overlay] core files: {len(overlay.core_files())}, reconstructed from pin {pin[:8]} + fork {fork[:8]} (merge base {base[:8]})")
    per_file = {}
    for chunk in re.split(r"(?m)^(?=--- a/)", body):
        if chunk.strip():
            m = re.search(r"(?m)^\+\+\+ b/(\S+)$", chunk)
            if m:
                per_file[m.group(1)] = chunk
    for path in overlay.core_files():
        if path in per_file:
            plus, minus = patch_stats(per_file[path])
            tag = f", {conflicts[path]} merge conflict(s) resolved by the overlay" if path in conflicts else ""
            print(f"  residual  {path}: +{plus} -{minus}{tag}")
    changed = normalize_patch(new.body) != normalize_patch(old.body)
    if args.show:
        sys.stdout.write(new.body)
    if changed:
        diff = "".join(
            difflib.unified_diff(
                old.body.splitlines(keepends=True),
                new.body.splitlines(keepends=True),
                f"{RESIDUAL_FILE} (recorded)",
                f"{RESIDUAL_FILE} (now)",
                n=2,
            )
        )
        print("[rocm-overlay] the core residual changed; review every hunk below:")
        sys.stdout.write(diff)
        if not args.write:
            errors.append(
                f"core residual differs from {RESIDUAL_FILE}; if every change is intended, note it in LOCAL_FIXES.md and re-run with --write"
            )
    stale_header = (old.pin, old.fork, old.base, old.hashes) != (new.pin, new.fork, new.base, new.hashes)
    if not changed and stale_header and not args.write:
        errors.append(f"{RESIDUAL_FILE} header is stale (commits or file hashes); the residual itself is unchanged, re-run with --write")

    # Backend files against the fork.
    fork_backend = set(ctx.git.ls_tree(fork, BACKEND_PREFIX))
    ours_backend = set(overlay.backend_files())
    modified, local_only = [], []
    for path in sorted(ours_backend):
        fb = ctx.git.show(fork, path) if path in fork_backend else None
        if fb is None:
            local_only.append(path)
        elif fb != overlay.read(path):
            modified.append(path)
    dropped = sorted(fork_backend - ours_backend)
    print(f"[rocm-overlay] backend: {len(ours_backend)} files, {len(modified)} modified from the fork, {len(local_only)} local-only, {len(dropped)} dropped")
    for path in modified + local_only + dropped:
        if not local_fixes_mentions(overlay, path):
            errors.append(f"{path} differs from the fork (modified, local-only or dropped) but LOCAL_FIXES.md does not name it")

    # Fork hooks outside the backend that the overlay does not carry.
    core = set(overlay.core_files())
    uncarried = [
        p
        for p in ctx.git.changed_paths(base, fork)
        if (p.startswith("mlx/") or p == "CMakeLists.txt")
        and not p.startswith(BACKEND_PREFIX)
        and p not in core
    ]
    for path in uncarried:
        if not local_fixes_mentions(overlay, path):
            errors.append(f"the fork changes {path}, which the overlay does not carry, and LOCAL_FIXES.md does not say why")
    for path in sorted(core):
        if ctx.git.show(fork, path) == ctx.git.show(base, path) and path not in per_file:
            print(f"  note: the fork does not change {path} and neither does the overlay; the overlay copy may be droppable")

    if args.write:
        record_path.write_text(render_residual(new))
        print(f"[rocm-overlay] wrote {record_path}")
        todo = [p for p in residual_files(new.body) if p not in new.notes]
        for path in todo:
            print(f"[rocm-overlay] add a note for {path} in {RESIDUAL_FILE} (it has a TODO now)", file=sys.stderr)
    if not errors:
        print("[rocm-overlay] drift OK")
    return report("drift", errors)


# --------------------------------------------------------------------------
# sync


def copy_overlay(src: Path, dst: Path) -> None:
    if dst.exists():
        raise ToolError(f"{dst} already exists")
    shutil.copytree(src, dst)


def cmd_sync(args: argparse.Namespace) -> int:
    ctx = make_context(args, need_git=True)
    git = ctx.git
    assert git is not None
    up = ctx.overlay.upstream()
    ensure_recorded_commits(ctx, up)
    f1, b1, pin = up["commit"], up["fork_upstream_merge_base"], up["retargeted_to_mlx_pin"]

    if args.fork_commit:
        if SHA_RE.match(args.fork_commit):
            git.ensure_commit(args.fork_commit, "fork")
            f2 = args.fork_commit
        else:
            git.fetch("fork", f"+refs/heads/{args.fork_commit}:refs/remotes/fork/{args.fork_commit}")
            f2 = git.resolve(f"refs/remotes/fork/{args.fork_commit}")
    else:
        branch = up.get("branch", DEFAULT_FORK_BRANCH)
        if git.fetch_enabled:
            git.fetch("fork", f"+refs/heads/{branch}:refs/remotes/fork/{branch}")
        f2 = git.resolve(f"refs/remotes/fork/{branch}")

    if args.fork_merge_base:
        b2 = args.fork_merge_base
        git.ensure_commit(b2, "upstream")
    elif f2 == f1:
        b2 = b1
    else:
        if git.fetch_enabled:
            git.fetch("upstream", f"+refs/heads/{DEFAULT_MLX_BRANCH}:refs/remotes/upstream/{DEFAULT_MLX_BRANCH}")
        b2 = git.merge_base(f2, f"refs/remotes/upstream/{DEFAULT_MLX_BRANCH}")
    if not git.is_ancestor(b2, pin) and not args.allow_fork_ahead:
        raise ToolError(
            f"the fork at {f2[:8]} has merged upstream MLX up to {b2[:8]}, which the pin {pin[:8]} does not contain. "
            "Bump the MLX pin to include it first (retarget), or pass --allow-fork-ahead to merge anyway."
        )

    if args.check:
        tmp = Path(tempfile.mkdtemp(prefix="mlxcelverse-sync-"))
        out_root = tmp / "patches-rocm"
    elif args.out:
        out_root = Path(args.out).resolve()
    else:
        out_root = ctx.overlay.root
    if out_root != ctx.overlay.root:
        copy_overlay(ctx.overlay.root, out_root)
    out = Overlay(out_root)
    conflicts: list[str] = []
    notes: list[str] = []
    changed: list[str] = []

    def write(rel: str, data: bytes) -> None:
        dest = safe_dest(out.root, rel)
        if dest.is_file() and dest.read_bytes() == data:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        changed.append(rel)

    # Core files: "pin + old fork delta" -> "pin + new fork delta", onto the overlay.
    for path in ctx.overlay.core_files():
        old_recon, _ = reconstruct(git, pin, b1, f1, path)
        new_recon, _ = reconstruct(git, pin, b2, f2, path)
        merged, n = merge3(
            ctx.overlay.read(path), old_recon, new_recon, ("overlay", "pin+fork-old", "pin+fork-new")
        )
        write(path, merged)
        if n:
            conflicts.append(f"{path}: {n} conflict(s) (core file)")
        if git.show(f2, path) == git.show(b2, path):
            notes.append(f"{path}: the new fork commit no longer changes this core file; check whether the overlay still needs it")

    # Backend files.
    old_fork = set(git.ls_tree(f1, BACKEND_PREFIX))
    new_fork = set(git.ls_tree(f2, BACKEND_PREFIX))
    ours = set(ctx.overlay.backend_files())
    for path in sorted(ours | old_fork | new_fork):
        in_ours, in_old, in_new = path in ours, path in old_fork, path in new_fork
        if in_ours and in_old and in_new:
            ob, nb = git.show(f1, path), git.show(f2, path)
            if ob != nb:
                merged, n = merge3(ctx.overlay.read(path), ob or b"", nb or b"", ("overlay", "fork-old", "fork-new"))
                write(path, merged)
                if n:
                    conflicts.append(f"{path}: {n} conflict(s)")
        elif in_ours and in_old and not in_new:
            if ctx.overlay.read(path) == git.show(f1, path):
                safe_dest(out.root, path).unlink()
                changed.append(path)
            else:
                conflicts.append(f"{path}: the fork deleted it but the overlay changes it; kept, decide by hand")
        elif in_ours and not in_old and in_new:
            nb = git.show(f2, path) or b""
            if nb != ctx.overlay.read(path):
                merged, n = merge3(ctx.overlay.read(path), b"", nb, ("overlay", "none", "fork-new"))
                write(path, merged)
                conflicts.append(f"{path}: the fork now has a file the overlay carries locally; merged with conflict markers ({n})")
        elif not in_ours and in_old:
            if in_new and git.show(f1, path) != git.show(f2, path):
                notes.append(f"{path}: dropped locally, but the fork changed it; check the drop still holds")
        elif not in_ours and not in_old and in_new:
            write(path, git.show(f2, path) or b"")
            notes.append(f"{path}: new in the fork, added")

    # Fork hooks outside the overlay.
    core = set(ctx.overlay.core_files())
    def hooks(b: str, f: str) -> set[str]:
        return {
            p
            for p in git.changed_paths(b, f)
            if (p.startswith("mlx/") or p == "CMakeLists.txt") and not p.startswith(BACKEND_PREFIX) and p not in core
        }
    for path in sorted(hooks(b2, f2) - hooks(b1, f1)):
        notes.append(f"{path}: the new fork commit changes this MLX file, which the overlay does not carry; decide whether it becomes a core overlay")

    upstream_path = out.root / "UPSTREAM"
    new_up = update_upstream(upstream_path.read_text(), {"commit": f2, "fork_upstream_merge_base": b2})
    if new_up != upstream_path.read_text():
        upstream_path.write_text(new_up)
        changed.append("UPSTREAM")

    for n in notes:
        print(f"[rocm-overlay] sync note: {n}")
    for c in conflicts:
        print(f"[rocm-overlay] sync CONFLICT: {c}", file=sys.stderr)

    if args.check:
        try:
            diffs = compare_trees(ctx.overlay.root, out.root)
        finally:
            shutil.rmtree(out.root.parent, ignore_errors=True)
        if diffs:
            for d in diffs:
                print(f"[rocm-overlay] sync --check: {d}", file=sys.stderr)
            return 1
        print(f"[rocm-overlay] sync --check OK: syncing to {f2[:8]} reproduces the committed overlay byte for byte")
        return 0

    print(f"[rocm-overlay] synced {out.root} to fork {f2[:8]} (merge base {b2[:8]}): {len(changed)} files changed, {len(conflicts)} conflicts")
    if changed and f2 != f1:
        print("[rocm-overlay] next: resolve conflicts, build with --features rocm (check_api_drift.sh), run `drift`, record the sync in LOCAL_FIXES.md")
    return 1 if conflicts else 0


def compare_trees(a: Path, b: Path) -> list[str]:
    diffs = []
    fa = {p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file()}
    for rel in sorted(fa - fb):
        diffs.append(f"{rel}: missing after sync")
    for rel in sorted(fb - fa):
        diffs.append(f"{rel}: added by sync")
    for rel in sorted(fa & fb):
        if (a / rel).read_bytes() != (b / rel).read_bytes():
            diffs.append(f"{rel}: content differs after sync")
    return diffs


# --------------------------------------------------------------------------
# retarget


def cmd_retarget(args: argparse.Namespace) -> int:
    ctx = make_context(args, need_git=True)
    git = ctx.git
    assert git is not None
    up = ctx.overlay.upstream()
    old_pin = up["retargeted_to_mlx_pin"]
    new_pin = args.to or pinned_commit()
    if not SHA_RE.match(new_pin):
        raise ToolError(f"--to must be a full commit SHA, got {new_pin!r}")
    if new_pin == old_pin:
        print(f"[rocm-overlay] already retargeted to {new_pin[:8]}")
        return 0
    git.ensure_commit(old_pin, "upstream")
    git.ensure_commit(new_pin, "upstream")
    if not git.is_ancestor(old_pin, new_pin):
        print(f"[rocm-overlay] warning: {old_pin[:8]} is not an ancestor of {new_pin[:8]}; merging anyway", file=sys.stderr)
    conflicts = []
    for path in ctx.overlay.core_files():
        ob, nb = git.show(old_pin, path), git.show(new_pin, path)
        if nb is None:
            conflicts.append(f"{path}: not in MLX at {new_pin[:8]} (moved or removed upstream); resolve by hand")
            continue
        merged, n = merge3(ctx.overlay.read(path), ob or b"", nb, ("overlay", "old-pin", "new-pin"))
        safe_dest(ctx.overlay.root, path).write_bytes(merged)
        if n:
            conflicts.append(f"{path}: {n} conflict(s)")
        elif ob != nb:
            print(f"  merged   {path}")
    upstream_path = ctx.overlay.root / "UPSTREAM"
    upstream_path.write_text(update_upstream(upstream_path.read_text(), {"retargeted_to_mlx_pin": new_pin}))
    for c in conflicts:
        print(f"[rocm-overlay] retarget CONFLICT: {c}", file=sys.stderr)
    print(
        f"[rocm-overlay] retargeted the core files from {old_pin[:8]} to {new_pin[:8]}. Next: resolve conflicts, "
        "run check_api_drift.sh (or build with --features rocm), fix the backend, run `drift`, and record every fix in LOCAL_FIXES.md"
    )
    return 1 if conflicts else 0


# --------------------------------------------------------------------------
# export-tree and api-report (check_api_drift.sh)


def cmd_export_tree(args: argparse.Namespace) -> int:
    ctx = make_context(args, need_git=True)
    git = ctx.git
    assert git is not None
    commit = args.mlx_commit or pinned_commit()
    git.ensure_commit(commit, "upstream")
    dest = Path(args.dest).resolve()
    if dest.exists():
        raise ToolError(f"{dest} already exists")
    dest.mkdir(parents=True)
    archive = git.run("archive", "--format=tar", commit).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    core = ctx.overlay.core_files()
    for rel in core:
        target = safe_dest(dest, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(ctx.overlay.read(rel))
    if args.rocm_from == "overlay":
        backend_src = "overlay"
        for rel in ctx.overlay.backend_files():
            target = safe_dest(dest, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(ctx.overlay.read(rel))
    elif args.rocm_from.startswith("fork:"):
        fork = args.rocm_from[len("fork:"):]
        git.ensure_commit(fork, "fork")
        backend_src = f"fork {fork[:8]}"
        for rel in git.ls_tree(fork, BACKEND_PREFIX):
            target = safe_dest(dest, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git.show(fork, rel) or b"")
    else:
        raise ToolError("--rocm-from must be `overlay` or `fork:<sha>`")
    print(f"[rocm-overlay] exported MLX {commit[:8]} + {len(core)} core overlay files + backend from {backend_src} to {dest}")
    return 0


NM_RE = re.compile(r"^\s*(?:[0-9a-fA-F]+\s+)?[A-Za-z?]\s+(\S.*)$")
ERROR_RE = re.compile(r"^(?P<loc>[^\s:][^:]*:\d+(?::\d+)?): (?:fatal )?error: (?P<msg>.*)$")


def cmd_api_report(args: argparse.Namespace) -> int:
    log = Path(args.build_log).read_text(errors="replace")
    src = args.source_dir.rstrip("/") + "/" if args.source_dir else ""
    errors = []
    seen = set()
    for line in log.splitlines():
        m = ERROR_RE.match(line.strip())
        if not m:
            continue
        loc = m.group("loc")
        if src and loc.startswith(src):
            loc = loc[len(src):]
        key = (loc, m.group("msg"))
        if key not in seen:
            seen.add(key)
            errors.append(f"{loc}: {m.group('msg')}")
    defined: set[str] = set()
    undefined: set[str] = set()
    for path, bucket in ((args.defined, defined), (args.undefined, undefined)):
        if not path:
            continue
        for line in Path(path).read_text(errors="replace").splitlines():
            # `nm -C` lines: "<addr> <type> <name>" or "<spaces> U <name>";
            # per-file headers ("foo.o:") and blank lines do not match.
            m = NM_RE.match(line)
            if m:
                bucket.add(m.group(1))
    missing = sorted(n for n in undefined - defined if "mlx::core::" in n)
    out = [f"compile errors: {len(errors)}"]
    out += [f"  {e}" for e in errors]
    out.append(f"undefined mlx::core symbols: {len(missing)}")
    out += [f"  {n}" for n in missing]
    if errors and missing:
        out.append("  (symbols defined in a file that failed to compile show up here too; fix compile errors first)")
    report_text = "\n".join(out) + "\n"
    sys.stdout.write(report_text)
    if args.output:
        Path(args.output).write_text(report_text)
    return 1 if errors or missing else 0


# --------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overlay-dir", default=str(DEFAULT_OVERLAY), help="the ROCm overlay (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def git_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--git-dir", help="git cache holding fork and MLX objects (default: $MLXCELVERSE_GIT_DIR or ~/.cache/mlxcel/mlxcelverse/mlx.git)")
        p.add_argument("--no-fetch", action="store_true", help="never touch the network; fail if an object is missing")
        p.add_argument("--fork-url", default=DEFAULT_FORK_URL)
        p.add_argument("--mlx-url", default=DEFAULT_MLX_URL)

    p = sub.add_parser("records", help="offline consistency check of the overlay records")
    p.add_argument("--pin", help="MLX pin to compare against (default: scripts/ci/mlx_pinned_commit.sh)")
    p.set_defaults(func=cmd_records)

    p = sub.add_parser("drift", help="compare the overlay with pin + fork delta; check the residual record")
    git_options(p)
    p.add_argument("--write", action="store_true", help=f"rewrite {RESIDUAL_FILE} (keeps notes)")
    p.add_argument("--show", action="store_true", help="print the whole residual")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("sync", help="move the overlay to a new fork commit")
    git_options(p)
    p.add_argument("--fork-commit", help="fork commit SHA or branch (default: the UPSTREAM branch head)")
    p.add_argument("--fork-merge-base", help="the new fork commit's merge base with upstream MLX (default: computed)")
    p.add_argument("--allow-fork-ahead", action="store_true", help="merge even if the fork has merged MLX newer than the pin")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--out", help="write the synced overlay to this new directory instead of in place")
    g.add_argument("--check", action="store_true", help="sync into a scratch copy and require it to equal the committed overlay")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("retarget", help="3-way merge the core files onto a new MLX pin")
    git_options(p)
    p.add_argument("--to", help="new MLX pin (default: the build's pin from mlx_pinned_commit.sh)")
    p.set_defaults(func=cmd_retarget)

    p = sub.add_parser("export-tree", help="write MLX at a commit with the ROCm files over it")
    git_options(p)
    p.add_argument("--mlx-commit", help="MLX commit (default: the build's pin)")
    p.add_argument("--rocm-from", default="overlay", help="`overlay` (default) or `fork:<sha>` for the backend directory")
    p.add_argument("--dest", required=True)
    p.set_defaults(func=cmd_export_tree)

    p = sub.add_parser("api-report", help="summarize a build log and nm output")
    p.add_argument("--build-log", required=True)
    p.add_argument("--defined", help="`nm -C --defined-only` output")
    p.add_argument("--undefined", help="`nm -C --undefined-only` output")
    p.add_argument("--source-dir", help="prefix to strip from error locations")
    p.add_argument("--output", help="also write the report here")
    p.set_defaults(func=cmd_api_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ToolError as e:
        print(f"[rocm-overlay] error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
