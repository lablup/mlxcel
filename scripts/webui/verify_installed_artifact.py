#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import argparse, hashlib, json, os, platform, re, secrets, shutil, signal, socket, ssl, stat, subprocess, sys, time, urllib.error, urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
MAX_BODY = 8 * 1024 * 1024
SECRET_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+")
GENERATED_KEY_RE = re.compile(rb"session key \(shown once\): ([^\s]+)")
DISABLED_FEATURE_MESSAGE, ROUTER_MISSING_MODEL_MESSAGE, PROBE_MODEL = "this feature is disabled", "model name is missing from the request", "mlxcel-installed-empty-router-probe"
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None  # type: ignore[no-untyped-def]  # noqa: E701
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
def generated_key_marker_present(output: bytes) -> bool: return GENERATED_KEY_RE.search(output) is not None  # noqa: E701
def generated_key_helper_failure(status: int, stdout: bytes, stderr: bytes) -> AssertionError:
    return AssertionError(f"generated-key helper failed: status={status} stdout_bytes={len(stdout)} stderr_bytes={len(stderr)}")
def run_generated_key_helper_process(helper: Path, config: Path, work: Path, env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.Popen([sys.executable, str(helper), str(config)], cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate(timeout=5)
        return subprocess.CompletedProcess(proc.args, -signal.SIGKILL, stdout, stderr)
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
def generated_key_error(stage: str, status: int | None, transcript: bytes) -> AssertionError:
    lines = transcript.count(b"\n") + (1 if transcript else 0)
    marker = generated_key_marker_present(transcript)
    return AssertionError(f"generated-key {stage}: status={status} bytes={len(transcript)} lines={lines} credential_marker_present={marker}")
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
def opener_for(*, https: bool = False, redirects: bool = True) -> urllib.request.OpenerDirector:
    handlers: list[Any] = [urllib.request.ProxyHandler({})]
    if https:
        handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    if not redirects:
        handlers.append(NoRedirect())
    return urllib.request.build_opener(*handlers)
def request(url: str, *, key: str | None = None, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None, https: bool = False) -> tuple[int, dict[str, str], bytes]:
    all_headers = dict(headers or {})
    if key is not None:
        all_headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, method=method, headers=all_headers)
    try:
        with opener_for(https=https).open(req, timeout=8) as resp:
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
    try:
        resp = opener_for(https=https).open(req, timeout=8)
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}
    with resp:
        return resp.status, {k.lower(): v for k, v in resp.headers.items()}
def error_summary(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body[:MAX_BODY])
    except Exception: return {"body_class": "non_json", "bytes": len(body)}  # noqa: BLE001,E701
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return {"body_class": "json", "bytes": len(body)}
    out = {"body_class": "json_error", "bytes": len(body)}
    for src, dst in (("type", "error_type"), ("code", "error_code"), ("message", "error_message")):
        if src in err: out[dst] = str(err[src])[:240]
    return out
def assert_compat_surface(label: str, status: int, body: bytes) -> dict[str, Any]:
    summary = error_summary(body)
    if status == 403 and summary.get("error_message") == DISABLED_FEATURE_MESSAGE and summary.get("error_type") == "feature_disabled":
        return {"label": label, "status": status, "mode": "disabled_feature_stub", **summary}
    missing_model_code = summary.get("error_type") == "invalid_request_error" or summary.get("error_code") == "invalid_request_error"
    if status == 400 and summary.get("error_message") == ROUTER_MISSING_MODEL_MESSAGE and missing_model_code:
        return {"label": label, "status": status, "mode": "router_dispatch_missing_model", **summary}
    if status == 400 and summary.get("error_message") == f"model '{PROBE_MODEL}' not found" and missing_model_code:
        return {"label": label, "status": status, "mode": "router_dispatch_scope_not_forwarded", **summary}
    raise AssertionError(f"{label} exposed unexpected compatibility-surface response: status={status} summary={summary}")
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
        if proc.returncode != 0:
            raise RuntimeError(f"server exited with {proc.returncode}; log: {log_path}")
        return {"exit_code": proc.returncode, "forced": forced, "log": str(log_path)}
    finally:
        close = getattr(log_file, "close", None)
        if close:
            close()
def stop_proc_with_key_cleanup(proc: subprocess.Popen[bytes], log_file: Any, log_path: Path, key_file: Path, stopper: Any = stop_proc) -> dict[str, Any]:
    try:
        return stopper(proc, log_file, log_path)
    finally:
        key_file.unlink(missing_ok=True)
def run_feature_off_probe(command: list[str], work: Path, key_file: Path, env: dict[str, str], runner: Any = subprocess.run) -> None:
    fail = runner(command + ["--webui", "--host", "127.0.0.1", "--port", str(free_port()), "--api-key-file", str(key_file)], cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    text = fail.stdout.decode(errors="replace")
    assert fail.returncode != 0 and "webui" in text.lower(), text
def run_feature_off_probe_with_key_cleanup(command: list[str], work: Path, key_file: Path, env: dict[str, str], runner: Any = subprocess.run) -> None:
    try:
        run_feature_off_probe(command, work, key_file, env, runner)
    finally:
        key_file.unlink(missing_ok=True)
def cleanup_tls_material(tls_extra: list[str] | None) -> None:
    if not tls_extra:
        return
    for flag in ("--ssl-key-file", "--ssl-cert-file"):
        if flag in tls_extra:
            Path(tls_extra[tls_extra.index(flag) + 1]).unlink(missing_ok=True)
def assert_wait_status_zero(status: int, label: str) -> None:
    if os.WIFEXITED(status):
        exit_value = os.WEXITSTATUS(status)
        if exit_value == 0:
            return
        raise AssertionError(f"{label} exited {exit_value}")
    if os.WIFSIGNALED(status):
        raise AssertionError(f"{label} died from signal {os.WTERMSIG(status)}")
    raise AssertionError(f"{label} ended with unexpected wait status {status}")
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
    try:
        with opener_for(https=https, redirects=False).open(urllib.request.Request(shell[:-1]), timeout=5) as resp:
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
    surfaces = []
    for label, path, method, data in (("tools_post", "/lab/tools", "POST", b"{}"), ("tools_probe_model_post", "/lab/tools", "POST", json.dumps({"model": PROBE_MODEL}).encode()), ("tools_unprefixed_post", "/tools", "POST", b"{}"), ("tools_unprefixed_probe_model_post", "/tools", "POST", json.dumps({"model": PROBE_MODEL}).encode()), ("cors_proxy_get", "/lab/cors-proxy", "GET", None), ("cors_proxy_unprefixed_get", "/cors-proxy", "GET", None)):
        status, _, body = request(f"{base}{path}", key=key, method=method, data=data, headers={"Content-Type": "application/json"} if data else None, https=https)
        surfaces.append(assert_compat_surface(label, status, body))
    return {"mode": observed["bootstrap"]["server"].get("mode"), "catalog_items": 0, "unknown_runtime": 404, "events_auth": True, "hostile_origin": 403, "disabled_feature_stubs": [s for s in surfaces if s["mode"] == "disabled_feature_stub"], "router_dispatch_rejections": [s for s in surfaces if s["mode"] != "disabled_feature_stub"], "compatibility_surface_scope": "router-mode empty-artifact probes prove non-successful dispatch only; hostile-origin checks are the installed security proof, while true feature-disabled 403 proof belongs to the ordinary/loaded-router harness"}
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
        result = {"label": artifact.name + label_suffix, "command": [artifact.relocated.name, *artifact.command[1:], "<flags>", "--api-key-file", "<private-key-file>"], "public_shell": assert_html_and_assets(base, key, https=https), "private_api": assert_private_api(base, key, https=https)}
        assert not (work / "home/.cache/mlxcel/models").exists(), "model-free startup created cache"
    finally:
        shutdown = stop_proc_with_key_cleanup(proc, log_file, log_path, key_file)
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
        shutdown = stop_proc_with_key_cleanup(proc, log_file, log_path, key_file)
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
        shutdown = stop_proc_with_key_cleanup(proc, log_file, log_path, key_file)
    write_private(key_file, key + "\n")
    run_feature_off_probe_with_key_cleanup(artifact.command, work, key_file, clean_env(work / "home", work / "store"))
    h.add("feature_off", {"label": artifact.name, "no_webui_health": 200, "webui_flag_rejected": True, "shutdown": shutdown})
def run_tls_matrix(h: Harness, artifacts: list[Artifact]) -> None:
    tls_extra = generate_tls(h.root / "tls")
    if tls_extra:
        try:
            for artifact in artifacts:
                run_on_off(h, artifact, tls_extra=tls_extra, https=True, label_suffix="-tls")
        finally:
            cleanup_tls_material(tls_extra)
    else:
        if os.environ.get("WEBUI_REQUIRE_TLS") == "1":
            raise AssertionError("WEBUI_REQUIRE_TLS=1 but openssl is unavailable for self-signed certificate generation")
        h.add("tls", {"status": "not_run", "reason": "openssl unavailable for self-signed certificate generation"})
def generate_tls(work: Path) -> list[str] | None:
    openssl = shutil.which("openssl")
    if not openssl:
        return None
    work.mkdir(exist_ok=True)
    cert, key = work / "cert.pem", work / "key.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1", "-keyout", str(key), "-out", str(cert), "-days", "1", "-addext", "subjectAltName=IP:127.0.0.1"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    key.chmod(0o600)
    if stat.S_IMODE(key.stat().st_mode) != 0o600:
        raise AssertionError(f"TLS key {key} is not 0600")
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
    if headless.returncode == 0:
        raise AssertionError("headless keyless WebUI unexpectedly succeeded")
    if generated_key_marker_present(headless.stdout):
        raise AssertionError("headless keyless WebUI emitted a generated credential marker")
    config = work / "generated-key-helper.json"
    write_private(config, json.dumps({"cmd": cmd, "work": str(work), "port": port}) + "\n")
    helper = Path(__file__).with_name("verify_generated_key_helper.py")
    try:
        result = run_generated_key_helper_process(helper, config, work, env)
    finally:
        config.unlink(missing_ok=True)
    if result.returncode != 0:
        raise generated_key_helper_failure(result.returncode, result.stdout, result.stderr)
    helper_result = json.loads(result.stdout or b"{}")
    helper_result.update({"label": artifact.name, "headless_without_key_rejected": True})
    h.add("generated_key", helper_result)
def network_interfaces() -> list[dict[str, Any]]:
    interfaces: list[dict[str, Any]] = []
    root = Path("/sys/class/net")
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            try:
                interfaces.append({"name": entry.name, "operstate": (entry / "operstate").read_text().strip(), "flags": (entry / "flags").read_text().strip()})
            except OSError:
                interfaces.append({"name": entry.name, "operstate": "unknown", "flags": "unknown"})
    return interfaces
def default_routes() -> list[str]:
    route = Path("/proc/net/route")
    if not route.is_file():
        return []
    rows = []
    for line in route.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "00000000":
            rows.append(line)
    return rows
def assert_network_namespace_isolated() -> dict[str, Any]:
    interfaces = network_interfaces()
    non_loopback = [iface for iface in interfaces if iface.get("name") != "lo"]
    routes = default_routes()
    if non_loopback or routes:
        raise AssertionError(f"network namespace still exposes non-loopback interfaces/routes: interfaces={non_loopback} routes={routes}")
    lo = next((iface for iface in interfaces if iface.get("name") == "lo"), None)
    if not lo or lo.get("operstate") not in ("unknown", "up"):
        raise AssertionError(f"loopback is not available/up inside network namespace: {lo}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.connect(("198.51.100.1", 80))
        except OSError as exc:
            return {"status": "enforced", "mechanism": "unshare-net", "interfaces": interfaces, "default_routes": routes, "negative_control": f"TEST-NET external TCP connect denied: {type(exc).__name__}", "loopback_server_tests": "passed above"}
    raise AssertionError("external TEST-NET TCP connect unexpectedly succeeded inside network namespace")
def network_denial(h: Harness) -> None:
    if os.environ.get("MLXCEL_WEBUI_NETNS_ACTIVE") == "1":
        h.add("network_denial", assert_network_namespace_isolated())
        return
    value = {"status": "not_enforced_in_this_process", "mechanism": "none", "note": "run under WEBUI_NETWORK_SANDBOX=require on Linux so the script re-execs with unshare -Urn and loopback brought up; offline env/proxy stripping is still applied"}
    if os.environ.get("WEBUI_NETWORK_SANDBOX") == "require":
        raise AssertionError(value["note"])
    h.add("network_denial", value)
def maybe_reexec_netns(argv: list[str]) -> None:
    mode = os.environ.get("WEBUI_NETWORK_SANDBOX", "auto")
    if mode == "off" or os.environ.get("MLXCEL_WEBUI_NETNS_ACTIVE") == "1" or platform.system() != "Linux":
        return
    unshare = shutil.which("unshare")
    ip = shutil.which("ip")
    if not unshare or not ip:
        if mode == "require":
            raise SystemExit("WEBUI_NETWORK_SANDBOX=require needs both `unshare` and `ip` to create a loopback-only namespace")
        return
    probe = subprocess.run([unshare, "-Urn", ip, "link", "set", "lo", "up"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        env = os.environ.copy(); env["MLXCEL_WEBUI_NETNS_ACTIVE"] = "1"
        script = '"$1" link set lo up && shift && exec "$@"'
        raise SystemExit(subprocess.call([unshare, "-Urn", "/bin/sh", "-c", script, "sh", ip, sys.executable, *argv], env=env))
    if mode == "require":
        raise SystemExit("WEBUI_NETWORK_SANDBOX=require but `unshare -Urn ip link set lo up` failed")
def main() -> int:
    maybe_reexec_netns(sys.argv)
    p = argparse.ArgumentParser()
    p.add_argument("--server-bin", default=os.environ.get("WEBUI_SERVER_BIN"))
    p.add_argument("--cli-bin", default=os.environ.get("WEBUI_CLI_BIN"))
    p.add_argument("--feature-off-server-bin", default=os.environ.get("WEBUI_FEATURE_OFF_SERVER_BIN"))
    p.add_argument("--feature-off-cli-bin", default=os.environ.get("WEBUI_FEATURE_OFF_CLI_BIN"))
    p.add_argument("--build-source-head", default=os.environ.get("WEBUI_BUILD_SOURCE_HEAD"), help="Git commit used to build the WebUI-enabled artifacts, if known")
    p.add_argument("--feature-off-build-source-head", default=os.environ.get("WEBUI_FEATURE_OFF_BUILD_SOURCE_HEAD"), help="Git commit used to build the feature-off artifacts, if known")
    p.add_argument("--evidence", default=os.environ.get("WEBUI_INSTALLED_EVIDENCE", "webui-installed-evidence.json"))
    args = p.parse_args()
    if not args.server_bin or not args.cli_bin:
        p.error("--server-bin and --cli-bin are required; issue #1848 must cover both mlxcel-server and `mlxcel serve`")
    evidence: dict[str, Any] = {"result": "fail", "scope": "installed relocated Rust artifacts; model-free bundled WebUI/security/compatibility", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": platform.platform(), "feature_set": "webui-on artifacts plus caller-provided required feature-off artifacts", "limits": ["no real checkpoint inference", "no browser/Safari/VoiceOver/manual hardware validation"]}
    evidence_path = Path(args.evidence).resolve()
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root = evidence_path.parent / f"{evidence_path.stem}-artifacts-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    h = Harness(args, root, evidence)
    h.evidence["artifact_dir"] = str(root)
    h.evidence["log_dir"] = str(root)
    try:
        artifacts = [relocate(Path(args.server_bin), root / "installed", "mlxcel-server", []), relocate(Path(args.cli_bin), root / "installed", "mlxcel-serve", ["serve"])]
        h.evidence["artifacts"] = [{"label": a.name, "source": str(a.source), "relocated": str(a.relocated), "sha256": a.sha256, "command_shape": [a.relocated.name, *a.command[1:]], "build_source_head": args.build_source_head or "unknown"} for a in artifacts]
        script_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        h.evidence["script_worktree_head"] = script_head.stdout.strip() if script_head.returncode == 0 else "unknown"
        h.evidence["build_provenance"] = {"webui_artifacts_source_head": args.build_source_head or "unknown", "feature_off_artifacts_source_head": args.feature_off_build_source_head or "unknown", "note": "script_worktree_head identifies the verifier source; build_source_head identifies the already-built binaries when the caller provides it"}
        h.flush()
        for artifact in artifacts:
            run_on_off(h, artifact)
        run_tls_matrix(h, artifacts)
        for artifact in artifacts:
            run_generated_key(h, artifact)
        if args.feature_off_server_bin and args.feature_off_cli_bin:
            feature_off_artifacts = [relocate(Path(args.feature_off_server_bin), root / "installed-feature-off", "mlxcel-server", []), relocate(Path(args.feature_off_cli_bin), root / "installed-feature-off", "mlxcel-serve", ["serve"] )]
            h.evidence["feature_off_artifacts"] = [{"label": a.name, "source": str(a.source), "relocated": str(a.relocated), "sha256": a.sha256, "command_shape": [a.relocated.name, *a.command[1:]], "build_source_head": args.feature_off_build_source_head or "unknown"} for a in feature_off_artifacts]
            h.flush()
            for artifact in feature_off_artifacts:
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
