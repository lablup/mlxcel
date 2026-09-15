#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Root-owned real Activity performance acceptance harness.

Runs a real installed server/checkpoint through the production WebUI API and
the existing headed activity-performance.mjs full mode. Unsupported hosts fail;
there is no synthetic visibility, auto-install, or exit-code-only green path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MAX_BODY = 16 * 1024 * 1024
MODEL_ID_RE = re.compile(r"^mdl_[A-Za-z0-9_-]{43}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+")
REQUIRED_MODES = {"one-visible", "two-visible", "hidden"}
VISIBLE_ONLY_MODES = {"one-visible", "two-visible"}
EXPECTED_SAMPLE_COUNTS = {"warmup": 1, "off": 15, "one-visible": 5, "two-visible": 5, "hidden": 5}
EXPECTED_PREFLIGHT_COUNTS = {"one-visible": 6, "two-visible": 12, "hidden": 6}
# A visible-only run drops the hidden mode: its five paired runs and the six preflight records
# that belong to it, and the five "off" baselines they were paired against.
VISIBLE_ONLY_SAMPLE_COUNTS = {"warmup": 1, "off": 10, "one-visible": 5, "two-visible": 5}
VISIBLE_ONLY_PREFLIGHT_COUNTS = {"one-visible": 6, "two-visible": 12}

@dataclass
class OwnedProcess:
    label: str
    proc: subprocess.Popen[bytes]
    log_path: Path | None = None

@dataclass
class Harness:
    args: argparse.Namespace
    work: Path
    model_for_server: Path
    key_file: Path
    base_root: str
    base_prefixed: str
    env: dict[str, str]
    secrets: list[str]
    processes: list[OwnedProcess] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def flush(self) -> None:
        self.args.evidence.write_text(json.dumps(redact(self.evidence, self.secrets), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def record(self, section: str, value: Any) -> None:
        self.evidence.setdefault(section, []).append(redact(value, self.secrets))
        self.flush()

def redact(value: Any, secrets_to_hide: Iterable[str]) -> Any:
    if isinstance(value, str):
        text = SECRET_RE.sub(r"\1<redacted>", value)
        for secret in secrets_to_hide:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text
    if isinstance(value, list):
        return [redact(item, secrets_to_hide) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, secrets_to_hide) for key, item in value.items()}
    return value

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_private(path: Path, body: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write(body)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise AssertionError(f"{path} mode is {mode:o}, want 600")

def open_private_append(path: Path) -> Any:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "ab", buffering=0)

def unique_work_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mlxcel-activity-performance-", dir=str(parent)))
    os.chmod(work, 0o700)
    return work

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def playwright_browsers_path() -> str | None:
    existing = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if existing:
        return existing
    candidates = [Path.home() / "Library/Caches/ms-playwright", Path.home() / ".cache/ms-playwright"]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None

def clean_env(home: Path, store: Path, display: str | None) -> dict[str, str]:
    allowed = {"PATH", "SystemRoot", "WINDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "TMPDIR", "TMP", "TEMP"}
    env = {key: value for key, value in os.environ.items() if key in allowed and not key.lower().endswith("proxy")}
    env.update({"HOME": str(home), "MLXCEL_MODELS_DIR": str(store), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "*", "no_proxy": "*"})
    if display:
        env["DISPLAY"] = display
    browsers = playwright_browsers_path()
    if browsers:
        env["PLAYWRIGHT_BROWSERS_PATH"] = browsers
    return env

def request(url: str, *, key: str | None = None, method: str = "GET", data: bytes | None = None, timeout: float = 15.0) -> tuple[int, dict[str, str], bytes]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise AssertionError(f"response exceeded {MAX_BODY} bytes")
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, body
    except urllib.error.HTTPError as exc:
        body = exc.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise AssertionError(f"error response exceeded {MAX_BODY} bytes")
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, body

def json_request(url: str, *, key: str | None = None, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    status, _, body = request(url, key=key, method=method, data=data, timeout=timeout)
    if not (200 <= status < 300):
        raise AssertionError(f"{method} {url} failed with {status}: {body[:512].decode(errors='replace')}")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise AssertionError(f"{method} {url} returned non-object JSON")
    return parsed

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def summarize_checkpoint(path: Path, revision: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")
    config = path / "config.json"
    if not config.is_file():
        raise AssertionError(f"checkpoint is missing config.json: {path}")
    files = [item for item in path.rglob("*") if item.is_file()]
    named_hashes: dict[str, str] = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        candidate = path / name
        if candidate.is_file():
            named_hashes[name] = sha256(candidate)
    safetensors = sorted(path.rglob("*.safetensors"))
    if not safetensors:
        raise AssertionError(f"checkpoint has no safetensors weights: {path}")
    weight_hashes = []
    for item in safetensors:
        size = item.stat().st_size
        if size <= 0:
            raise AssertionError(f"checkpoint has empty safetensors weight: {item}")
        weight_hashes.append({"relative_path": str(item.relative_to(path)), "size": size, "sha256": sha256(item)})
    return {
        "path": str(path.resolve()),
        "display_name": path.name,
        "revision": revision,
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "metadata_sha256": named_hashes,
        "safetensors_count": len(safetensors),
        "safetensors_total_bytes": sum(item.stat().st_size for item in safetensors),
        "safetensors_sha256": weight_hashes,
    }

def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise AssertionError(f"{label} must be finite")
    if positive and number <= 0:
        raise AssertionError(f"{label} must be positive")
    return number

def assert_no_symlinks_or_hardlinks(path: Path) -> None:
    if path.is_symlink():
        raise AssertionError(f"model view must not contain symlinks: {path}")
    for item in path.rglob("*"):
        if item.is_symlink():
            raise AssertionError(f"model view must not contain symlinks: {item}")
        if item.is_file() and item.stat().st_nlink != 1:
            raise AssertionError(f"model view must not contain hardlinked files: {item}")

def materialize_model_view(source: Path, work: Path) -> tuple[Path, str]:
    models_root = work / "models"
    models_root.mkdir(mode=0o700, exist_ok=False)
    dest = models_root / source.name
    if sys.platform == "darwin":
        commands = [(["/bin/cp", "-cRL", str(source), str(dest)], "mac-cp-clonefile-deref"), (["/bin/cp", "-RL", str(source), str(dest)], "mac-cp-recursive-deref")]
    else:
        commands = [(["cp", "-RL", "--reflink=auto", str(source), str(dest)], "linux-cp-reflink-auto-deref")]
    errors = []
    for command, copy_kind in commands:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if completed.returncode == 0:
            assert_no_symlinks_or_hardlinks(dest)
            return dest, copy_kind
        errors.append(f"{' '.join(command[:3])}: {completed.stderr[-1000:]}")
        if dest.exists():
            shutil.rmtree(dest)
    raise AssertionError(f"failed to create isolated model view: {'; '.join(errors)}")

def validate_activity_output(path: Path, perf_mode: str = "full") -> dict[str, Any]:
    # A visible-only run is a declared deferral of the hidden acceptance, not a weaker pass: every
    # numeric budget below still applies, and the expected mode, sample and preflight counts move
    # to the visible-only set so a run that silently dropped a mode is still caught.
    hidden_deferred = perf_mode != "full"
    expected_status = "incomplete" if hidden_deferred else "within-target"
    expected_hidden_native = "not-run" if hidden_deferred else "measured"
    expected_modes = VISIBLE_ONLY_MODES if hidden_deferred else REQUIRED_MODES
    expected_preflight = VISIBLE_ONLY_PREFLIGHT_COUNTS if hidden_deferred else EXPECTED_PREFLIGHT_COUNTS
    expected_samples = VISIBLE_ONLY_SAMPLE_COUNTS if hidden_deferred else EXPECTED_SAMPLE_COUNTS
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("activity performance output must be a JSON object")
    if data.get("status") != expected_status:
        raise AssertionError(f"activity performance status is {data.get('status')!r}, want {expected_status!r}")
    if data.get("hidden_native") != expected_hidden_native:
        raise AssertionError(f"hidden_native is {data.get('hidden_native')!r}, want {expected_hidden_native!r}")
    summaries = data.get("summaries")
    if not isinstance(summaries, list) or not summaries:
        raise AssertionError("activity performance output must include summaries")
    bad = [item for item in summaries if not isinstance(item, dict) or item.get("status") != "within-target"]
    if bad:
        raise AssertionError(f"activity performance has non-target summaries: {bad}")
    seen_summary_modes: set[str] = set()
    for item in summaries:
        mode = item.get("mode")
        if mode not in expected_modes:
            raise AssertionError(f"unexpected summary mode: {mode!r}")
        if mode in seen_summary_modes:
            raise AssertionError(f"duplicate summary mode: {mode}")
        seen_summary_modes.add(mode)
        if item.get("paired_runs") != 5:
            raise AssertionError(f"{mode} summary must have paired_runs=5")
        degradation = finite_number(item.get("median_decode_degradation_percent"), f"{mode} degradation")
        cv = finite_number(item.get("baseline_cv_percent"), f"{mode} baseline CV")
        if cv < 0:
            raise AssertionError(f"{mode} baseline CV must not be negative")
        if degradation > 2 or cv > 5:
            raise AssertionError(f"{mode} summary exceeded targets despite within-target status")
        pair_range = item.get("paired_range_percent")
        if not isinstance(pair_range, list) or len(pair_range) != 2:
            raise AssertionError(f"{mode} paired_range_percent must have two entries")
        range_min = finite_number(pair_range[0], f"{mode} paired range min")
        range_max = finite_number(pair_range[1], f"{mode} paired range max")
        if range_min > range_max:
            raise AssertionError(f"{mode} paired_range_percent must be ordered")
    missing = expected_modes - seen_summary_modes
    if missing:
        raise AssertionError(f"activity performance missing mode summaries: {sorted(missing)}")
    preflight = data.get("preflight")
    if not isinstance(preflight, list) or not preflight:
        raise AssertionError("activity performance output must include preflight geometry/visibility records")
    preflight_counts: dict[str, int] = {}
    for item in preflight:
        if not isinstance(item, dict) or item.get("mode") not in expected_modes:
            raise AssertionError("preflight records must identify a required mode")
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            raise AssertionError("preflight records must include geometry")
        for field in ("innerWidth", "innerHeight", "visualWidth", "visualHeight"):
            finite_number(geometry.get(field), f"preflight {field}", positive=True)
        preflight_counts[item["mode"]] = preflight_counts.get(item["mode"], 0) + 1
    if preflight_counts != expected_preflight:
        raise AssertionError(f"unexpected preflight counts: {preflight_counts}")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise AssertionError("activity performance output must include samples")
    sample_counts: dict[str, int] = {}
    for item in samples:
        if not isinstance(item, dict) or not isinstance(item.get("mode"), str):
            raise AssertionError("sample records must include mode")
        sample_counts[item["mode"]] = sample_counts.get(item["mode"], 0) + 1
        finite_number(item.get("tokens_per_second"), f"{item['mode']} tokens_per_second", positive=True)
        finite_number(item.get("predicted_tokens"), f"{item['mode']} predicted_tokens", positive=True)
        finite_number(item.get("predicted_ms"), f"{item['mode']} predicted_ms", positive=True)
    if sample_counts != expected_samples:
        raise AssertionError(f"unexpected sample counts: {sample_counts}")
    return data

def parse_features(raw: str) -> list[str]:
    features = [item.strip() for item in raw.split(",") if item.strip()]
    if not features:
        raise argparse.ArgumentTypeError("features must contain at least one item")
    return features

def check_require_models() -> None:
    if os.environ.get("MLXCEL_REQUIRE_MODELS") != "1":
        raise SystemExit("MLXCEL_REQUIRE_MODELS=1 is required; missing hardware/checkpoints are blockers, not skips")

def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AssertionError(f"required tool not found in PATH: {name}")
    return path

def start_virtual_display(work: Path) -> tuple[str, list[OwnedProcess]]:
    xvfb = require_tool("Xvfb")
    openbox = require_tool("openbox")
    display_num = 90 + (os.getpid() % 100)
    display = f":{display_num}"
    xvfb_log = work / "xvfb.log"
    openbox_log = work / "openbox.log"
    processes: list[OwnedProcess] = []
    try:
        xvfb_file = open_private_append(xvfb_log)
        try:
            xvfb_proc = subprocess.Popen([xvfb, display, "-screen", "0", "1600x1100x24", "-nolisten", "tcp"], stdout=xvfb_file, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            xvfb_file.close()
            raise
        owned_xvfb = OwnedProcess("xvfb", xvfb_proc, xvfb_log)
        setattr(owned_xvfb, "_log_file", xvfb_file)
        processes.append(owned_xvfb)
        time.sleep(0.5)
        if xvfb_proc.poll() is not None:
            raise RuntimeError(f"Xvfb exited early with {xvfb_proc.returncode}; log={xvfb_log}")
        openbox_file = open_private_append(openbox_log)
        try:
            openbox_proc = subprocess.Popen([openbox], env={**os.environ, "DISPLAY": display}, stdout=openbox_file, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            openbox_file.close()
            raise
        owned_openbox = OwnedProcess("openbox", openbox_proc, openbox_log)
        setattr(owned_openbox, "_log_file", openbox_file)
        processes.append(owned_openbox)
        time.sleep(0.5)
        if openbox_proc.poll() is not None:
            raise RuntimeError(f"openbox exited early with {openbox_proc.returncode}; log={openbox_log}")
        return display, processes
    except Exception:
        terminate_owned(processes)
        raise

def terminate_owned(processes: list[OwnedProcess]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for owned in reversed(processes):
        proc = owned.proc
        forced = False
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                forced = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
        close = getattr(owned, "_log_file", None)
        if close:
            close.close()
        results.append({"label": owned.label, "exit_code": proc.returncode, "forced": forced, "process_group_empty": process_group_empty(proc.pid, 0.5), "log": str(owned.log_path) if owned.log_path else None})
    return results

def process_group_empty(pgid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)
    return False

def stop_process_group(proc: subprocess.Popen[bytes], *, first: int = signal.SIGTERM) -> None:
    for sig, wait in ((first, 5.0), (signal.SIGKILL, 5.0)):
        if process_group_empty(proc.pid, 0.1):
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        if proc.poll() is None:
            try:
                proc.wait(timeout=min(wait, 1.0))
            except subprocess.TimeoutExpired:
                pass
        deadline = time.time() + wait
        while time.time() < deadline:
            if process_group_empty(proc.pid, 0.1):
                return
    if not process_group_empty(proc.pid, 0.1):
        raise RuntimeError(f"process group {proc.pid} did not terminate")

def start_server(h: Harness) -> OwnedProcess:
    model_parent = h.model_for_server.resolve().parent
    log_path = h.work / "server.log"
    log_file = open_private_append(log_path)
    command = [
        str(h.args.server_bin.resolve()), "--host", "127.0.0.1", "--port", str(h.args.port),
        "--models-dir", str(model_parent), "--model-store-root", str((h.work / "store").resolve()),
        "--api-key-file", str(h.key_file), "--api-prefix", "/lab", "--no-models-autoload",
        "--settings", "--props", "--metrics", "--webui",
    ]
    try:
        proc = subprocess.Popen(command, cwd=h.work, env=h.env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception:
        log_file.close()
        raise
    owned = OwnedProcess("mlxcel-server", proc, log_path)
    setattr(owned, "_log_file", log_file)
    h.processes.append(owned)
    return owned

def wait_health(base: str, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + 90
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}: {last}")
        try:
            status, _, body = request(f"{base}/health", timeout=5)
            last = body[:512].decode(errors="replace")
            if status == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"server did not become ready: {last}")

def catalog_pages(base: str, key: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(20):
        suffix = "?limit=200" if cursor is None else f"?cursor={urllib.parse.quote(cursor)}"
        page = json_request(f"{base}/ui-api/v1/catalog{suffix}", key=key)
        pages.append(page)
        pagination = page.get("pagination")
        if not isinstance(pagination, dict):
            raise AssertionError("catalog page missing pagination")
        next_cursor = pagination.get("next_cursor")
        if next_cursor is None:
            return pages
        if not isinstance(next_cursor, str):
            raise AssertionError("catalog next_cursor must be string or null")
        cursor = next_cursor
    raise AssertionError("catalog pagination exceeded bounded 20-page limit")

def all_catalog_items(base: str, key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page in catalog_pages(base, key):
        page_items = page.get("items")
        if not isinstance(page_items, list):
            raise AssertionError("catalog items must be a list")
        items.extend(item for item in page_items if isinstance(item, dict))
    return items

def select_entry(items: list[dict[str, Any]], model_path: Path, requested_id: str | None) -> dict[str, Any]:
    if requested_id:
        if not MODEL_ID_RE.fullmatch(requested_id):
            raise AssertionError("--model-id must be an opaque WebUI model id")
        matches = [item for item in items if item.get("identity", {}).get("id") == requested_id]
    else:
        basename = model_path.name
        matches = [item for item in items if item.get("identity", {}).get("display_name") == basename or item.get("identity", {}).get("inference_id") == basename]
    if len(matches) != 1:
        names = [item.get("identity", {}).get("display_name") for item in items[:20]]
        raise AssertionError(f"could not uniquely identify checkpoint {model_path}; matches={len(matches)} first_catalog_names={names}; pass --model-id")
    return matches[0]


def entry_identity(entry: dict[str, Any]) -> dict[str, Any]:
    identity = entry.get("identity")
    if not isinstance(identity, dict):
        raise AssertionError("catalog entry missing identity")
    model_id = identity.get("id")
    inference_id = identity.get("inference_id")
    revision = identity.get("revision")
    if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
        raise AssertionError("catalog entry has invalid model id")
    if not isinstance(inference_id, str) or not inference_id:
        raise AssertionError("catalog entry has invalid inference id")
    if not isinstance(revision, int):
        raise AssertionError("catalog entry has invalid revision")
    return {"model_id": model_id, "inference_id": inference_id, "revision": revision, "display_name": identity.get("display_name")}


def load_model(base: str, key: str, model_id: str, revision: int) -> dict[str, Any]:
    accepted = json_request(f"{base}/ui-api/v1/model-actions", key=key, method="POST", payload={"model_id": model_id, "action": "load", "expected_revision": revision, "idempotency_key": f"activity-load-{secrets.token_hex(8)}"})
    return accepted


def unload_model(base: str, key: str, model_id: str, revision: int) -> dict[str, Any]:
    return json_request(f"{base}/ui-api/v1/model-actions", key=key, method="POST", payload={"model_id": model_id, "action": "unload", "expected_revision": revision, "idempotency_key": f"activity-unload-{secrets.token_hex(8)}"})


def wait_entry(base: str, key: str, model_id: str, predicate: Any, label: str, timeout: float = 240.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        for item in all_catalog_items(base, key):
            if item.get("identity", {}).get("id") == model_id:
                last = item
                if predicate(item):
                    return item
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {label}; last={last}")


def lifecycle_state(entry: dict[str, Any]) -> dict[str, Any]:
    lifecycle = entry.get("lifecycle")
    if not isinstance(lifecycle, dict):
        raise AssertionError("catalog entry missing lifecycle")
    return lifecycle


def runtime_snapshot(base: str, key: str, model_id: str) -> dict[str, Any]:
    return json_request(f"{base}/ui-api/v1/runtime?model_id={urllib.parse.quote(model_id)}&autoload=false", key=key)


def run_activity_script(h: Harness, model_id: str, inference_id: str) -> tuple[dict[str, Any], Path, Path]:
    output = h.work / "activity-performance.json"
    log_path = h.work / "activity-performance.log"
    env = {**h.env, "WEBUI_PERF_BASE": h.base_prefixed + "/", "WEBUI_PERF_KEY_FILE": str(h.key_file), "WEBUI_PERF_INFERENCE_MODEL": inference_id, "WEBUI_PERF_MODEL_ID": model_id, "WEBUI_PERF_OUTPUT": str(output), "WEBUI_PERF_MODE": h.args.perf_mode}
    if h.args.prompt_file:
        env["WEBUI_PERF_PROMPT_FILE"] = str(h.args.prompt_file.resolve())
    command = [h.args.node_bin, "webui/scripts/activity-performance.mjs"]
    with open_private_append(log_path) as log_file:
        proc = subprocess.Popen(command, cwd=h.args.repo_root, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=h.args.activity_timeout)
        except subprocess.TimeoutExpired as exc:
            stop_process_group(proc)
            raise AssertionError(f"activity-performance.mjs timed out after {h.args.activity_timeout}s; log={log_path}") from exc
        if returncode != 0:
            stop_process_group(proc)
            raise AssertionError(f"activity-performance.mjs exited {returncode}; log={log_path}")
        if not process_group_empty(proc.pid, 5.0):
            stop_process_group(proc)
            raise AssertionError(f"activity-performance.mjs left child processes in group {proc.pid}; log={log_path}")
    data = validate_activity_output(output, h.args.perf_mode)
    return data, output, log_path


def stop_server_strict(owned: OwnedProcess) -> dict[str, Any]:
    proc = owned.proc
    forced = False
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            forced = True
            stop_process_group(proc, first=signal.SIGTERM)
    close = getattr(owned, "_log_file", None)
    if close:
        close.close()
    if proc.returncode != 0:
        raise RuntimeError(f"server exited with {proc.returncode}; strict SIGINT exit 0 required; log={owned.log_path}")
    if not process_group_empty(proc.pid, 1.0):
        raise RuntimeError(f"server left owned process group {proc.pid} after SIGINT 0; log={owned.log_path}")
    return {"exit_code": proc.returncode, "forced": forced, "process_group_empty": True, "log": str(owned.log_path)}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Read-only local checkpoint directory to expose through --models-dir <parent>")
    parser.add_argument("--model-id", help="Optional opaque WebUI model id when basename lookup is ambiguous")
    parser.add_argument("--perf-mode", default="full", choices=["full", "visible-only-headed", "visible-only-headless"], help="full runs the native hidden acceptance; the visible-only modes measure observation overhead and record hidden_native as not-run")
    parser.add_argument("--checkpoint-revision", required=True, help="Pinned checkpoint revision/commit/hash recorded in evidence")
    parser.add_argument("--source-sha", required=True, help="40-character source commit used to build the server artifact")
    parser.add_argument("--features", type=parse_features, required=True, help="Comma-separated build features, e.g. cuda,webui")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Parent directory for a new private 0700 run directory; never reused directly")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--node-bin", default=shutil.which("node") or "node")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--activity-timeout", type=float, default=1800.0)
    parser.add_argument("--virtual-display", action="store_true", help="Start owned Xvfb + openbox instead of using DISPLAY")
    args = parser.parse_args(argv)
    if not HEX40_RE.fullmatch(args.source_sha):
        parser.error("--source-sha must be a 40-character lowercase git SHA")
    if not args.server_bin.is_file() or not os.access(args.server_bin, os.X_OK):
        parser.error("--server-bin must be an executable file")
    if not args.model.is_dir():
        parser.error("--model must be a checkpoint directory")
    if args.port == 0:
        args.port = free_port()
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    check_require_models()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    work = unique_work_dir(args.work_dir or args.evidence.resolve().parent)
    (work / "home").mkdir(mode=0o700)
    (work / "store").mkdir(mode=0o700)
    key = f"mlxcel-activity-{secrets.token_urlsafe(32)}"
    key_file = work / "api-key.txt"
    h: Harness | None = None
    server: OwnedProcess | None = None
    virtual_processes: list[OwnedProcess] = []
    try:
        write_private(key_file, key + "\n")
        if args.virtual_display:
            display, virtual_processes = start_virtual_display(work)
        else:
            display = os.environ.get("DISPLAY")
            if not display and sys.platform != "darwin":
                raise AssertionError("DISPLAY is required on non-macOS hosts unless --virtual-display is used; native hidden cannot be measured headlessly")
        base_root = f"http://127.0.0.1:{args.port}"
        base_prefixed = f"{base_root}/lab"
        h = Harness(args=args, work=work, model_for_server=args.model, key_file=key_file, base_root=base_root, base_prefixed=base_prefixed, env=clean_env(work / "home", work / "store", display), secrets=[key])
        h.processes.extend(virtual_processes)
        h.evidence = {"status": "in-progress", "source_commit": args.source_sha, "server_bin_sha256": sha256(args.server_bin), "features": args.features, "work_dir": str(work), "display": {"value": display, "virtual": args.virtual_display, "minimum_geometry": "1600x1100x24" if args.virtual_display else None}, "started_at": now()}
        h.flush()
        h.evidence["checkpoint"] = summarize_checkpoint(args.model, args.checkpoint_revision)
        model_view, copy_kind = materialize_model_view(args.model, work)
        h.model_for_server = model_view
        h.evidence["model_view"] = {"path": str(model_view), "copy_kind": copy_kind}
        h.flush()
        server = start_server(h)
        wait_health(h.base_root, server.proc)
        h.record("server", {"event": "ready", "base": h.base_root, "log": str(server.log_path)})
        entry = select_entry(all_catalog_items(h.base_prefixed, key), args.model, args.model_id)
        identity = entry_identity(entry)
        h.record("runtime_observations", {"stage": "catalog_initial", "entry": entry})
        accepted = load_model(h.base_prefixed, key, identity["model_id"], identity["revision"])
        h.record("runtime_observations", {"stage": "load_accepted", "accepted": accepted})
        loaded = wait_entry(h.base_prefixed, key, identity["model_id"], lambda item: lifecycle_state(item).get("state") == "ready", "model ready after load")
        runtime_loaded = runtime_snapshot(h.base_prefixed, key, identity["model_id"])
        h.record("runtime_observations", {"stage": "loaded", "entry": loaded, "runtime": runtime_loaded})
        activity, activity_output, activity_log = run_activity_script(h, identity["model_id"], identity["inference_id"])
        h.record("activity_performance", {"status": "passed", "output": str(activity_output), "output_sha256": sha256(activity_output), "log": str(activity_log), "summary": {"status": activity.get("status"), "hidden_native": activity.get("hidden_native"), "summaries": activity.get("summaries")}})
        latest = wait_entry(h.base_prefixed, key, identity["model_id"], lambda item: isinstance(item.get("identity"), dict), "latest revision before unload", timeout=30)
        latest_identity = entry_identity(latest)
        unload = unload_model(h.base_prefixed, key, identity["model_id"], latest_identity["revision"])
        h.record("runtime_observations", {"stage": "unload_accepted", "accepted": unload})
        unloaded = wait_entry(h.base_prefixed, key, identity["model_id"], lambda item: lifecycle_state(item).get("state") == "unloaded" and lifecycle_state(item).get("worker_exit_observed") is True, "worker exit observed after unload")
        h.record("runtime_observations", {"stage": "unloaded_worker_exit", "entry": unloaded})
        shutdown = stop_server_strict(server)
        h.record("cleanup", {"stage": "server_sigint", **shutdown})
        h.evidence.update({"status": "passed", "completed_at": now(), "server_shutdown": shutdown})
        h.flush()
        return h.evidence
    except Exception as exc:
        if h is not None:
            h.evidence.update({"status": "failed", "failed_at": now(), "error": str(redact(str(exc), [key]))})
            h.flush()
        raise
    finally:
        if server is not None:
            terminate_owned([server])
        remaining = [proc for proc in (h.processes if h is not None else virtual_processes) if proc is not server]
        cleanup = terminate_owned(remaining)
        if cleanup and h is not None:
            h.evidence.setdefault("cleanup", []).append(cleanup)
            h.flush()
        key_file.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"activity performance verification failed: {redact(str(exc), [])}", file=sys.stderr)
        return 1
    print(f"validated Activity native-hidden performance evidence: {args.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
