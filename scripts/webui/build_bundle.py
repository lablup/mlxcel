#!/usr/bin/env python3
"""Build and verify the checked-in mlxcel WebUI asset bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEBUI = ROOT / "webui"
ASSETS = ROOT / "src" / "webui" / "assets"
MANIFEST_NAME = "mlxcel-webui-manifest.json"
MAX_INITIAL_JS_GZIP = 200 * 1024
MAX_TOTAL_JS_GZIP = 700 * 1024
MAX_EMBEDDED_BYTES = 5 * 1024 * 1024
LICENSE_PATH = "third-party-licenses.txt"
REQUIRED_NODE_VERSION = "v26.5.1"
REQUIRED_PNPM_VERSION = "11.18.0"
SOURCE_FILES = [
    ROOT / "scripts" / "webui" / "build_bundle.py",
    WEBUI / "package.json",
    WEBUI / "pnpm-lock.yaml",
    WEBUI / ".node-version",
    WEBUI / "tsconfig.json",
    WEBUI / "vite.config.ts",
    WEBUI / "vitest.config.ts",
    WEBUI / "index.html",
]
SOURCE_DIRS = [WEBUI / "src", WEBUI / "public"]


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_sources() -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    paths.extend(path for path in SOURCE_FILES if path.exists())
    for directory in SOURCE_DIRS:
        paths.extend(iter_files(directory))
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT).as_posix().encode()
        data = path.read_bytes()
        digest.update(rel + b"\0" + hashlib.sha256(data).digest() + b"\0")
    return digest.hexdigest()


def gzip_len(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def require_toolchain() -> tuple[str, str]:
    node_version = run(["node", "--version"], cwd=WEBUI).strip()
    pnpm_version = run(["pnpm", "--version"], cwd=WEBUI).strip()
    if node_version != REQUIRED_NODE_VERSION:
        raise SystemExit(f"WebUI build requires node {REQUIRED_NODE_VERSION}, got {node_version}")
    if pnpm_version != REQUIRED_PNPM_VERSION:
        raise SystemExit(f"WebUI build requires pnpm {REQUIRED_PNPM_VERSION}, got {pnpm_version}")
    return node_version, pnpm_version


def package_versions() -> dict[str, str]:
    package = json.loads((WEBUI / "package.json").read_text())
    versions: dict[str, str] = {}
    for section in ("dependencies", "devDependencies"):
        for name, version in package.get(section, {}).items():
            versions[name] = version
    return dict(sorted(versions.items()))


def initial_javascript_paths(out_dir: Path) -> set[str]:
    html = (out_dir / "index.html").read_text()
    paths = set()
    for match in re.finditer(r"(?:src|href)=[\"']\./([^\"']+\.js)[\"']", html):
        paths.add(match.group(1))
    if not paths:
        raise SystemExit("index.html does not reference an initial JavaScript module")
    return paths


def manifest_for(out_dir: Path, manifest_bytes: int = 0) -> dict[str, object]:
    node_version, pnpm_version = require_toolchain()
    files = []
    embedded_bytes = manifest_bytes
    initial_js_gzip = 0
    total_js_gzip = 0
    initial_js_paths = initial_javascript_paths(out_dir)
    for path in iter_files(out_dir):
        if path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(out_dir).as_posix()
        data = path.read_bytes()
        size = len(data)
        gz_size = gzip_len(data)
        embedded_bytes += size
        if rel.endswith(".js"):
            total_js_gzip += gz_size
            if rel in initial_js_paths:
                initial_js_gzip += gz_size
        files.append(
            {
                "path": rel,
                "sha256": sha256_bytes(data),
                "bytes": size,
                "gzip_bytes": gz_size,
            }
        )
    files.sort(key=lambda item: item["path"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "package_manager": "pnpm@11.18.0",
        "node_version": node_version,
        "pnpm_version": pnpm_version,
        "source_digest_sha256": digest_sources(),
        "packages": package_versions(),
        "budgets": {
            "initial_js_gzip_bytes": initial_js_gzip,
            "initial_js_gzip_limit_bytes": MAX_INITIAL_JS_GZIP,
            "total_js_gzip_bytes": total_js_gzip,
            "total_js_gzip_limit_bytes": MAX_TOTAL_JS_GZIP,
            "embedded_asset_bytes": embedded_bytes,
            "embedded_asset_limit_bytes": MAX_EMBEDDED_BYTES,
        },
        "files": files,
    }
    return manifest


def write_manifest(out_dir: Path) -> None:
    manifest_bytes = 0
    previous_text = ""
    for _ in range(8):
        manifest = manifest_for(out_dir, manifest_bytes=manifest_bytes)
        budgets = manifest["budgets"]
        assert isinstance(budgets, dict)
        if budgets["initial_js_gzip_bytes"] > MAX_INITIAL_JS_GZIP:
            raise SystemExit("initial JavaScript gzip budget exceeded")
        if budgets["total_js_gzip_bytes"] > MAX_TOTAL_JS_GZIP:
            raise SystemExit("total JavaScript gzip budget exceeded")
        if budgets["embedded_asset_bytes"] > MAX_EMBEDDED_BYTES:
            raise SystemExit("embedded WebUI asset budget exceeded")
        text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if text == previous_text:
            (out_dir / MANIFEST_NAME).write_text(text)
            return
        previous_text = text
        manifest_bytes = len(text.encode())
    raise SystemExit("WebUI manifest did not reach a deterministic size fixed point")


def build(out_dir: Path) -> None:
    require_toolchain()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    run(["pnpm", "--dir", str(WEBUI), "run", "build"], env={"MLXCEL_WEBUI_OUT_DIR": str(out_dir)})
    write_manifest(out_dir)
    assert_bundle(out_dir)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix().encode()
        digest.update(rel + b"\0" + hashlib.sha256(path.read_bytes()).digest() + b"\0")
    return digest.hexdigest()


def assert_bundle(root: Path) -> None:
    if not root.exists() or not any(root.iterdir()):
        raise SystemExit(f"WebUI asset directory is missing or empty: {root}")
    for required in ("index.html", MANIFEST_NAME, LICENSE_PATH):
        if not (root / required).is_file():
            raise SystemExit(f"required WebUI asset is missing: {required}")
    js_files = sorted(root.glob("assets/*.js"))
    if not js_files:
        raise SystemExit("WebUI bundle contains no JavaScript asset")
    assert_index_html(root)
    assert_license_notice(root)
    assert_manifest_matches_files(root)
    assert_no_active_external_or_debug_artifacts(root)


def assert_index_html(root: Path) -> None:
    html = (root / "index.html").read_text()
    if "http://" in html or "https://" in html or "//" in html:
        raise SystemExit("index.html must not reference external origins")
    if 'id="root"' not in html or 'type="module"' not in html or './assets/' not in html:
        raise SystemExit("index.html must contain the WebUI root and relative module assets")
    for token in html.replace('>', '>\n').splitlines():
        for attr in ('src="./', 'href="./'):
            if attr in token:
                rel = token.split(attr, 1)[1].split('"', 1)[0]
                if not (root / rel).is_file():
                    raise SystemExit(f"index.html references missing asset: {rel}")


def assert_license_notice(root: Path) -> None:
    notice = (root / LICENSE_PATH).read_text()
    for required in ("react 19.3.0", "react-dom 19.3.0", "scheduler 0.28.0", "MIT License", "Copyright (c) Meta Platforms, Inc. and affiliates."):
        if required not in notice:
            raise SystemExit(f"WebUI license notice is missing: {required}")


def assert_manifest_matches_files(root: Path) -> None:
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    manifest_files = {item["path"]: item for item in manifest.get("files", [])}
    actual = {path.relative_to(root).as_posix(): path for path in iter_files(root) if path.name != MANIFEST_NAME}
    if set(manifest_files) != set(actual):
        missing = sorted(set(manifest_files) ^ set(actual))
        raise SystemExit(f"manifest file set does not match generated assets: {missing}")
    for rel, path in actual.items():
        data = path.read_bytes()
        entry = manifest_files[rel]
        if entry.get("sha256") != sha256_bytes(data) or entry.get("bytes") != len(data) or entry.get("gzip_bytes") != gzip_len(data):
            raise SystemExit(f"manifest digest/size is stale for generated asset: {rel}")
    if manifest.get("source_digest_sha256") != digest_sources():
        raise SystemExit("manifest source digest is stale; rebuild the WebUI bundle")
    budgets = manifest.get("budgets", {})
    embedded_bytes = sum(path.stat().st_size for path in actual.values()) + (root / MANIFEST_NAME).stat().st_size
    if budgets.get("embedded_asset_bytes") != embedded_bytes:
        raise SystemExit("manifest embedded asset budget is stale")


def assert_no_active_external_or_debug_artifacts(root: Path) -> None:
    repo_marker = ROOT.as_posix()
    active_external_markers = (
        "fetch(\"http://",
        "fetch('http://",
        "import(\"http://",
        "import('http://",
        "importScripts(\"http://",
        "importScripts('http://",
        "new Worker(\"http://",
        "new Worker('http://",
        "EventSource(\"http://",
        "EventSource('http://",
        "WebSocket(\"ws://",
        "WebSocket('ws://",
        "url(http://",
        "url(https://",
    )
    for path in iter_files(root):
        text = path.read_text(errors="ignore")
        if "sourceMappingURL" in text:
            raise SystemExit(f"source map reference leaked into WebUI asset: {path.relative_to(root).as_posix()}")
        if repo_marker in text or "node_modules" in text:
            raise SystemExit(f"local build path leaked into WebUI asset: {path.relative_to(root).as_posix()}")
        lowered = text.lower()
        for marker in active_external_markers:
            if marker.lower() in lowered:
                raise SystemExit(f"active external-origin reference in WebUI asset: {path.relative_to(root).as_posix()}")


def compare_trees(left: Path, right: Path, label: str) -> None:
    left_files = {path.relative_to(left).as_posix(): path for path in iter_files(left)}
    right_files = {path.relative_to(right).as_posix(): path for path in iter_files(right)}
    if set(left_files) != set(right_files):
        missing = sorted(set(left_files) ^ set(right_files))
        raise SystemExit(f"{label} differs in file set: {missing}")
    changed = [rel for rel in sorted(left_files) if left_files[rel].read_bytes() != right_files[rel].read_bytes()]
    if changed:
        raise SystemExit(f"{label} differs in generated bytes: {changed}")


def verify() -> None:
    assert_bundle(ASSETS)
    with tempfile.TemporaryDirectory(prefix="mlxcel-webui-bundle-") as first_raw, tempfile.TemporaryDirectory(prefix="mlxcel-webui-bundle-") as second_raw:
        first = Path(first_raw) / "assets"
        second = Path(second_raw) / "assets"
        build(first)
        build(second)
        compare_trees(first, second, "two clean WebUI builds")
        compare_trees(first, ASSETS, "checked-in WebUI bundle")
        print(f"WebUI bundle verified: digest={tree_digest(ASSETS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ASSETS)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        build(args.out_dir.resolve())


if __name__ == "__main__":
    main()
