#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Installed WebUI artifact gate for issue #1848.

The gate launches relocated Rust artifacts, never Vite preview, and verifies the
model-free bundled WebUI surface for both `mlxcel-server` and `mlxcel serve`:
authentication, prefixing, public/private route split, CSP/cache headers, UI-off
regression, feature-off artifacts, generated-key startup, optional TLS, and
best-effort OS network-denial evidence. It intentionally does not load a model;
actual checkpoint, browser, Safari/VoiceOver, and GPU gates remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pty
import re
import secrets
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
MAX_BODY = 8 * 1024 * 1024
SECRET_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def sha256(path: Path) -> str:
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def redact(value: Any, secrets_to_hide: list[str]) -> Any:
    if isinstance(value, str):
        text = SECRET_RE.sub(r"\1<redacted>", value)
        for secret in secrets_to_hide:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text
    if isinstance(value, list):
        return [redact(v, secrets_to_hide) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, secrets_to_hide) for k, v in value.items()}
    return value


def write_private(path: Path, body: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fp:
        fp.write(body)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise AssertionError(f"{path} mode is {mode:o}, want 600")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def clean_env(home: Path, store: Path) -> dict[str, str]:
    allowed = {"PATH", "SystemRoot", "WINDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "TMPDIR", "TMP", "TEMP"}
    env = {k: v for k, v in os.environ.items() if k in allowed and not k.lower().endswith("proxy")}
    env.update({"HOME": str(home), "MLXCEL_MODELS_DIR": str(store), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "*", "no_proxy": "*"})
    return env


def request(url: str, *, key: str | None = None, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None, https: bool = False) -> tuple[int, dict[str, str], bytes]:
    all_headers = dict(headers or {})
    if key is not None:
        all_headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, method=method, headers=all_headers)
    opener = NO_PROXY_OPENER
    if https:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    try:
        with opener.open(req, timeout=8) as resp:
            body = resp.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise AssertionError("response exceeded bounded buffer")
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, body
    except urllib.error.HTTPError as exc:
        body = exc.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise AssertionError("error response exceeded bounded buffer")
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, body


def open_headers(url: str, *, key: str | None = None, https: bool = False) -> tuple[int, dict[str, str]]:
    headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
    req = urllib.request.Request(url, headers=headers)
    opener = NO_PROXY_OPENER
    if https:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    try:
        resp = opener.open(req, timeout=8)
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}
    with resp:
        return resp.status, {k.lower(): v for k, v in resp.headers.items()}


@dataclass
class Artifact:
    name: str
    source: Path
    relocated: Path
    command: list[str]
    sha256: str


@dataclass
class Harness:
    args: argparse.Namespace
    root: Path
    evidence: dict[str, Any]
    secrets: list[str] = field(default_factory=list)

    def add(self, section: str, value: Any) -> None:
        self.evidence.setdefault(section, []).append(redact(value, self.secrets))
        self.flush()

    def flush(self) -> None:
        Path(self.args.evidence).write_text(json.dumps(redact(self.evidence, self.secrets), indent=2, sort_keys=True) + "\n")


def relocate(src: Path, install: Path, name: str, subcommand: list[str]) -> Artifact:
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    install.mkdir(parents=True, exist_ok=True)
    dest = install / src.name
    shutil.copy2(src, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR)
    return Artifact(name, src, dest, [str(dest), *subcommand], sha256(dest))


def launch(command: list[str], work: Path, key_file: Path, port: int, models: Path, store: Path, webui: bool, *, extra: list[str] | None = None, https: bool = False) -> tuple[subprocess.Popen[bytes], Any, Path, str]:
    args = command + ["--host", "127.0.0.1", "--port", str(port), "--models-dir", str(models), "--model-store-root", str(store), "--api-key-file", str(key_file), "--api-prefix", "/lab", "--no-models-autoload", "--settings", "--props", "--metrics", "--webui" if webui else "--no-webui"]
    if extra:
        args.extend(extra)
    log_path = work / ("server-webui-on.log" if webui else "server-webui-off.log")
    log_file = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(args, cwd=work, env=clean_env(work / "home", store), stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file, log_path, f"{'https' if https else 'http'}://127.0.0.1:{port}"


def wait_ready(base: str, proc: subprocess.Popen[bytes], *, https: bool = False) -> None:
    deadline = time.time() + 45
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}: {last}")
        try:
            status, _, body = request(f"{base}/health", https=https)
            last = body[:512].decode(errors="replace")
            if status == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.15)
    raise RuntimeError(f"server did not become ready: {last}")


def stop_proc(proc: subprocess.Popen[bytes], log_file: Any, log_path: Path) -> dict[str, Any]:
    forced = False
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced = True
                proc.kill()
                proc.wait(timeout=5)
        if proc.returncode not in (0, -signal.SIGINT):
            raise RuntimeError(f"server exited with {proc.returncode}; log: {log_path}")
        return {"exit_code": proc.returncode, "forced": forced, "log": str(log_path)}
    finally:
        close = getattr(log_file, "close", None)
        if close:
            close()


def assert_html_and_assets(base: str, key: str, *, https: bool = False) -> dict[str, Any]:
    shell = f"{base}/lab/webui/"
    status, headers, html = request(shell, https=https)
    assert status == 200 and b"<!" in html[:64], (status, html[:200])
    csp = headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp and "unsafe-eval" not in csp and "unsafe-inline" not in csp, csp
    etag = headers.get("etag")
    assert etag, "HTML must carry an ETag"
    head_status, head_headers, head_body = request(shell, method="HEAD", https=https)
    assert head_status == 200 and head_body == b"" and head_headers.get("etag"), head_headers
    assert request(shell, headers={"If-None-Match": etag}, https=https)[0] == 304
    redirect_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with redirect_opener.open(urllib.request.Request(shell[:-1]), timeout=5) as resp:
            redirect_status, redirect_location = resp.status, resp.headers.get("Location")
    except urllib.error.HTTPError as exc:
        redirect_status, redirect_location = exc.code, exc.headers.get("Location")
    assert redirect_status in (301, 307, 308) and redirect_location == "/lab/webui/", (redirect_status, redirect_location)
    assets = re.findall(rb'(?:src|href)="([^"#]+)"', html)
    assert assets, "HTML shell must reference bundled assets"
    checked: list[str] = []
    for raw in assets[:8]:
        name = raw.decode()
        assert not name.startswith(("http:", "https:", "//", "/")), name
        url = shell + name.removeprefix("./")
        status, asset_headers, asset = request(url, https=https)
        assert status == 200 and asset, name
        assert request(url, method="HEAD", https=https)[2] == b""
        if asset_headers.get("etag"):
            assert request(url, headers={"If-None-Match": asset_headers["etag"]}, https=https)[0] == 304
        checked.append(name)
    assert request(shell + "missing-asset-does-not-exist.js", https=https)[0] == 404
    assert key.encode() not in html
    return {"csp": csp, "html_etag": bool(etag), "assets_checked": checked, "missing_asset": 404}


def assert_private_api(base: str, key: str, *, https: bool = False) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name in ("bootstrap", "catalog", "operations"):
        path = f"{base}/lab/ui-api/v1/{name}"
        assert request(path, https=https)[0] == 401, name
        status, _, body = request(path, key=key, https=https)
        assert status == 200, (name, status, body[:200])
        value = json.loads(body)
        assert key.encode() not in body
        observed[name] = value
    assert observed["bootstrap"]["server"]["api_base"] == "/lab", observed["bootstrap"]["server"]
    assert observed["catalog"]["items"] == []
    unknown = f"{base}/lab/ui-api/v1/runtime?model_id=mdl_{'a' * 43}"
    assert request(unknown, key=key, https=https)[0] == 404
    assert request(f"{base}/lab/ui-api/v1/events", https=https)[0] == 401
    status, headers = open_headers(f"{base}/lab/ui-api/v1/events", key=key, https=https)
    assert status == 200 and "text/event-stream" in headers.get("content-type", "")
    attack = {"Host": "foreign.invalid", "Origin": "http://foreign.invalid", "Sec-Fetch-Site": "cross-site", "Content-Type": "application/json"}
    assert request(f"{base}/lab/ui-api/v1/model-actions", key=key, method="POST", data=b"{}", headers=attack, https=https)[0] == 403
    assert request(f"{base}/lab/v1/chat/completions", key=key, method="POST", data=b"{}", headers=attack, https=https)[0] == 403
    assert request(f"{base}/lab/tools", key=key, method="POST", data=b"{}", headers={"Content-Type": "application/json"}, https=https)[0] in (403, 404, 405)
    assert request(f"{base}/lab/cors-proxy/http://169.254.169.254/latest/meta-data/", key=key, https=https)[0] in (403, 404)
    return {"mode": observed["bootstrap"]["server"].get("mode"), "catalog_items": 0, "unknown_runtime": 404, "events_auth": True, "hostile_origin": 403, "legacy_attack_routes_denied": True}


def run_on_off(h: Harness, artifact: Artifact, *, tls_extra: list[str] | None = None, https: bool = False, label_suffix: str = "") -> None:
    work = h.root / f"{artifact.name}{label_suffix}"
    work.mkdir(parents=True)
    for d in ("home", "empty-models", "store"):
        (work / d).mkdir()
    key = secrets.token_urlsafe(32); h.secrets.append(key)
    key_file = work / "api-key.txt"
    write_private(key_file, key + "\n")
    proc, log_file, log_path, base = launch(artifact.command, work, key_file, free_port(), work / "empty-models", work / "store", True, extra=["--sse-ping-interval", "5", *(tls_extra or [])], https=https)
    try:
        wait_ready(base, proc, https=https)
        result = {"label": artifact.name + label_suffix, "command": [artifact.relocated.name, *artifact.command[1:], "<flags>", "--api-key-file", str(key_file)], "public_shell": assert_html_and_assets(base, key, https=https), "private_api": assert_private_api(base, key, https=https)}
        assert not (work / "home/.cache/mlxcel/models").exists(), "model-free startup created cache"
    finally:
        shutdown = stop_proc(proc, log_file, log_path)
        key_file.unlink(missing_ok=True)
    assert key not in log_path.read_text(errors="replace"), "credential leaked to server log"
    result["shutdown"] = shutdown
    result["no_model_autoload"] = True
    h.add("webui_on", result)

    write_private(key_file, key + "\n")
    proc, log_file, log_path, base = launch(artifact.command, work, key_file, free_port(), work / "empty-models", work / "store", False)
    try:
        wait_ready(base, proc)
        assert request(f"{base}/health")[0] == 200
        assert request(f"{base}/lab/webui/")[0] in (401, 404)
        assert request(f"{base}/lab/ui-api/v1/bootstrap", key=key)[0] in (400, 404)
    finally:
        shutdown = stop_proc(proc, log_file, log_path)
        key_file.unlink(missing_ok=True)
    h.add("webui_off", {"label": artifact.name + label_suffix, "health": 200, "ui_routes_absent": True, "shutdown": shutdown})


def run_feature_off(h: Harness, artifact: Artifact) -> None:
    work = h.root / f"{artifact.name}-feature-off"
    work.mkdir(parents=True)
    for d in ("home", "empty-models", "store"):
        (work / d).mkdir()
    key_file = work / "api-key.txt"
    key = secrets.token_urlsafe(32); h.secrets.append(key)
    write_private(key_file, key + "\n")
    proc, log_file, log_path, base = launch(artifact.command, work, key_file, free_port(), work / "empty-models", work / "store", False)
    try:
        wait_ready(base, proc)
        assert request(f"{base}/health")[0] == 200
        assert request(f"{base}/lab/webui/")[0] in (401, 404)
    finally:
        shutdown = stop_proc(proc, log_file, log_path)
    fail = subprocess.run(artifact.command + ["--webui", "--host", "127.0.0.1", "--port", str(free_port()), "--api-key-file", str(key_file)], cwd=work, env=clean_env(work / "home", work / "store"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    text = fail.stdout.decode(errors="replace")
    assert fail.returncode != 0 and "webui" in text.lower(), text
    key_file.unlink(missing_ok=True)
    h.add("feature_off", {"label": artifact.name, "no_webui_health": 200, "webui_flag_rejected": True, "shutdown": shutdown})


def generate_tls(work: Path) -> list[str] | None:
    openssl = shutil.which("openssl")
    if not openssl:
        return None
    work.mkdir(exist_ok=True)
    cert, key = work / "cert.pem", work / "key.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1", "-keyout", str(key), "-out", str(cert), "-days", "1", "-addext", "subjectAltName=IP:127.0.0.1"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    return ["--ssl-cert-file", str(cert), "--ssl-key-file", str(key)]


def run_generated_key(h: Harness, artifact: Artifact) -> None:
    if platform.system() == "Windows":
        h.add("generated_key", {"label": artifact.name, "status": "unsupported", "reason": "pty unavailable on Windows"}); return
    work = h.root / f"{artifact.name}-generated-key"
    work.mkdir(parents=True); (work / "home").mkdir(); (work / "models").mkdir(); (work / "store").mkdir()
    env = clean_env(work / "home", work / "store"); env["RUST_LOG"] = "info"
    port = free_port()
    cmd = artifact.command + ["--webui", "--host", "127.0.0.1", "--port", str(port), "--api-prefix", "/gen", "--models-dir", str(work / "models"), "--model-store-root", str(work / "store"), "--no-models-autoload"]
    headless = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, timeout=20)
    assert headless.returncode != 0 and b"session key" not in headless.stdout.lower(), headless.stdout[:2000]
    pid, fd = pty.fork(); transcript = bytearray(); key: str | None = None; status = 0
    if pid == 0:
        os.chdir(work); os.execve(artifact.command[0], cmd, env)
    try:
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                transcript.extend(os.read(fd, 65536))
            except OSError:
                pass
            match = re.search(rb"session key \(shown once\): ([^\s]+)", transcript)
            if match:
                key = match.group(1).decode(); h.secrets.append(key); break
            time.sleep(0.05)
        assert key, transcript.decode(errors="replace")[-2000:]
        assert request(f"http://127.0.0.1:{port}/gen/ui-api/v1/bootstrap", key=key)[0] == 200
        assert request(f"http://127.0.0.1:{port}/gen/ui-api/v1/bootstrap")[0] == 401
    finally:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                waited, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                waited = pid
            if waited:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                _, status = os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            raise AssertionError("generated-key server did not exit after SIGINT and had to be killed")
        os.close(fd)
    h.add("generated_key", {"label": artifact.name, "headless_without_key_rejected": True, "tty_key_authenticated": True, "key_in_argv_or_env": False, "exit_status": status})


def network_denial(h: Harness) -> None:
    if os.environ.get("MLXCEL_WEBUI_NETNS_ACTIVE") == "1":
        try:
            with NO_PROXY_OPENER.open("http://93.184.216.34/", timeout=2) as response:
                response.read(1)
        except (OSError, TimeoutError, urllib.error.URLError):
            h.add("network_denial", {"status": "enforced", "mechanism": "unshare-net", "negative_control": "external IPv4 fetch denied", "loopback_server_tests": "passed above"})
            return
        raise AssertionError("external network reachable inside requested network-denial namespace")
    value = {"status": "not_enforced_in_this_process", "mechanism": "none", "note": "run under `unshare -Urn` or macOS sandbox-exec for hard no-network evidence; offline env/proxy stripping is still applied"}
    if os.environ.get("WEBUI_NETWORK_SANDBOX") == "require":
        raise AssertionError(value["note"])
    h.add("network_denial", value)


def maybe_reexec_netns(argv: list[str]) -> None:
    mode = os.environ.get("WEBUI_NETWORK_SANDBOX", "auto")
    if mode == "off" or os.environ.get("MLXCEL_WEBUI_NETNS_ACTIVE") == "1" or platform.system() != "Linux":
        return
    unshare = shutil.which("unshare")
    if not unshare:
        return
    probe = subprocess.run([unshare, "-Urn", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        env = os.environ.copy(); env["MLXCEL_WEBUI_NETNS_ACTIVE"] = "1"
        raise SystemExit(subprocess.call([unshare, "-Urn", sys.executable, *argv], env=env))
    if mode == "require":
        raise SystemExit("WEBUI_NETWORK_SANDBOX=require but `unshare -Urn` is unavailable")


def main() -> int:
    maybe_reexec_netns(sys.argv)
    p = argparse.ArgumentParser()
    p.add_argument("--server-bin", default=os.environ.get("WEBUI_SERVER_BIN"))
    p.add_argument("--cli-bin", default=os.environ.get("WEBUI_CLI_BIN"))
    p.add_argument("--feature-off-server-bin", default=os.environ.get("WEBUI_FEATURE_OFF_SERVER_BIN"))
    p.add_argument("--feature-off-cli-bin", default=os.environ.get("WEBUI_FEATURE_OFF_CLI_BIN"))
    p.add_argument("--evidence", default=os.environ.get("WEBUI_INSTALLED_EVIDENCE", "webui-installed-evidence.json"))
    args = p.parse_args()
    if not args.server_bin or not args.cli_bin:
        p.error("--server-bin and --cli-bin are required; issue #1848 must cover both mlxcel-server and `mlxcel serve`")
    evidence: dict[str, Any] = {"result": "fail", "scope": "installed relocated Rust artifacts; model-free bundled WebUI/security/compatibility", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": platform.platform(), "feature_set": "webui-on artifacts plus optional feature-off artifacts", "limits": ["no real checkpoint inference", "no browser/Safari/VoiceOver/manual hardware validation"]}
    evidence_path = Path(args.evidence).resolve()
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root = evidence_path.parent / f"{evidence_path.stem}-artifacts-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    h = Harness(args, root, evidence)
    h.evidence["artifact_dir"] = str(root)
    try:
        artifacts = [relocate(Path(args.server_bin), root / "installed", "mlxcel-server", []), relocate(Path(args.cli_bin), root / "installed", "mlxcel-serve", ["serve"])]
        h.evidence["artifacts"] = [{"label": a.name, "source": str(a.source), "relocated": str(a.relocated), "sha256": a.sha256, "command_shape": [a.relocated.name, *a.command[1:]]} for a in artifacts]
        source_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        h.evidence["source_head"] = source_head.stdout.strip() if source_head.returncode == 0 else "unknown"
        h.flush()
        for artifact in artifacts:
            run_on_off(h, artifact)
        tls_extra = generate_tls(root / "tls")
        if tls_extra:
            run_on_off(h, artifacts[0], tls_extra=tls_extra, https=True, label_suffix="-tls")
        else:
            if os.environ.get("WEBUI_REQUIRE_TLS") == "1":
                raise AssertionError("WEBUI_REQUIRE_TLS=1 but openssl is unavailable for self-signed certificate generation")
            h.add("tls", {"status": "not_run", "reason": "openssl unavailable for self-signed certificate generation"})
        run_generated_key(h, artifacts[0])
        if args.feature_off_server_bin and args.feature_off_cli_bin:
            for artifact in [relocate(Path(args.feature_off_server_bin), root / "installed-feature-off", "mlxcel-server", []), relocate(Path(args.feature_off_cli_bin), root / "installed-feature-off", "mlxcel-serve", ["serve"] )]:
                run_feature_off(h, artifact)
        else:
            if os.environ.get("WEBUI_REQUIRE_FEATURE_OFF", "1") == "1":
                raise AssertionError("feature-off artifact paths are required; set WEBUI_REQUIRE_FEATURE_OFF=0 only for local helper development")
            h.add("feature_off", {"status": "not_run", "reason": "feature-off artifact paths not provided"})
        network_denial(h)
        h.evidence["result"] = "pass"; h.evidence["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); h.flush()
        print(json.dumps(redact(h.evidence, h.secrets), indent=2, sort_keys=True)); return 0
    except BaseException as exc:
        h.evidence["result"] = "fail"; h.evidence["error"] = {"type": type(exc).__name__, "message": redact(str(exc), h.secrets)}; h.evidence["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); h.flush(); raise


if __name__ == "__main__":
    raise SystemExit(main())
