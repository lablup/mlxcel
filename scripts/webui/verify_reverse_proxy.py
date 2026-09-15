#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import argparse, http.server, json, os, re, secrets, signal, ssl, subprocess, threading, time, urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))
import verify_installed_artifact as inst  # noqa: E402
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
QUERY_CREDENTIAL_NAMES = {"api_key", "key", "token", "access_token", "auth"}

@dataclass
class ProxyState:
    backend_base: str
    backend_host: str
    backend_origin: str
    public_host: str
    public_origin: str
    forwarded: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    sse_cancelled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

def header_values(headers: Any, name: str) -> list[str]:
    if hasattr(headers, "get_all"):
        return [str(v) for v in (headers.get_all(name) or [])]
    return [str(v) for k, v in dict(headers).items() if k.lower() == name.lower()]

def first_header(headers: Any, name: str) -> str | None:
    values = header_values(headers, name)
    return values[0] if len(values) == 1 else None

def query_carries_credential(path: str) -> bool:
    query = urllib.parse.urlsplit(path).query
    return any(name.lower() in QUERY_CREDENTIAL_NAMES for name, _ in urllib.parse.parse_qsl(query, keep_blank_values=True))

def is_public_request(method: str, path: str) -> bool:
    return method in {"GET", "HEAD"} and (path == "/lab/webui" or path.startswith("/lab/webui/"))

def edge_rejection(method: str, path: str, headers: Any, public_host: str, public_origin: str) -> str | None:
    hosts, origins = header_values(headers, "Host"), header_values(headers, "Origin")
    if len(hosts) != 1 or hosts[0] != public_host: return "invalid_host"  # noqa: E701
    if query_carries_credential(path): return "query_credentials"  # noqa: E701
    public = is_public_request(method, urllib.parse.urlsplit(path).path)
    mutation = method not in {"GET", "HEAD", "OPTIONS"}
    if len(origins) > 1 or (origins and origins[0] != public_origin) or (mutation and len(origins) != 1): return "invalid_origin"  # noqa: E701
    site = first_header(headers, "Sec-Fetch-Site")
    if public and site not in {None, "none", "same-origin"}: return "invalid_fetch_metadata"  # noqa: E701
    if not public and site not in {None, "same-origin"}: return "invalid_fetch_metadata"  # noqa: E701
    if mutation and site != "same-origin": return "invalid_fetch_metadata"  # noqa: E701
    return None

def make_upstream_headers(headers: Any, backend_host: str, backend_origin: str) -> dict[str, str]:
    outgoing: dict[str, str] = {}
    for key, value in dict(headers).items():
        lower = key.lower()
        if lower in HOP_BY_HOP or lower in {"host", "origin", "content-length"}:
            continue
        outgoing[key] = str(value)
    outgoing["Host"] = backend_host
    if header_values(headers, "Origin"):
        outgoing["Origin"] = backend_origin
    return outgoing

def response_headers(headers: Any, body_len: int | None) -> list[tuple[str, str]]:
    out = [(k, v) for k, v in headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "content-length"]
    if body_len is not None:
        out.append(("Content-Length", str(body_len)))
    return out

def forward_record(method: str, url: str, headers: dict[str, str], status: int | None = None) -> dict[str, Any]:
    return {"method": method, "path": urllib.parse.urlsplit(url).path, "status": status, "authorization_header": bool(headers.get("Authorization")), "url_has_credential_query": query_carries_credential(url), "host": headers.get("Host"), "origin": headers.get("Origin"), "fetch_site": headers.get("Sec-Fetch-Site")}

class ReverseProxyHandler(http.server.BaseHTTPRequestHandler):
    server: Any
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt: str, *args: Any) -> None: return None  # noqa: E701
    def do_GET(self) -> None: self._proxy()  # noqa: N802,E701
    def do_HEAD(self) -> None: self._proxy()  # noqa: N802,E701
    def do_POST(self) -> None: self._proxy()  # noqa: N802,E701
    def _reject(self, code: int, reason: str) -> None:
        body = json.dumps({"error": {"type": reason, "message": "reverse proxy edge rejected request"}}).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers()
        if self.command != "HEAD": self.wfile.write(body)
    def _proxy(self) -> None:
        state: ProxyState = self.server.state
        reason = edge_rejection(self.command, self.path, self.headers, state.public_host, state.public_origin)
        if reason:
            with state.lock: state.blocked.append({"reason": reason, "path": urllib.parse.urlsplit(self.path).path})
            self._reject(400 if reason == "query_credentials" else 403, reason); return
        length = int(first_header(self.headers, "Content-Length") or "0")
        if length > inst.MAX_BODY:
            self._reject(413, "payload_too_large"); return
        data = self.rfile.read(length) if length else None
        url = state.backend_base + self.path
        upstream_headers = make_upstream_headers(self.headers, state.backend_host, state.backend_origin)
        with state.lock: state.forwarded.append(forward_record(self.command, url, upstream_headers))
        req = urllib.request.Request(url, data=data, method=self.command, headers=upstream_headers)
        try:
            resp = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=12)
            status, headers = resp.status, {k: v for k, v in resp.headers.items()}
            body = b"" if self.command == "HEAD" else self._read_response(resp, headers)
        except urllib.error.HTTPError as exc:
            status, headers = exc.code, {k: v for k, v in exc.headers.items()}
            body = b"" if self.command == "HEAD" else self._read_response(exc, headers)
        with state.lock: state.forwarded[-1]["status"] = status
        self.send_response(status)
        is_sse = headers.get("content-type", "").startswith("text/event-stream")
        for key, value in response_headers(headers, None if is_sse else len(body)):
            self.send_header(key, value)
        self.end_headers()
        if is_sse:
            with state.lock: state.sse_cancelled = True
            return
        if self.command != "HEAD": self.wfile.write(body)
    def _read_response(self, resp: Any, headers: dict[str, str]) -> bytes:
        if headers.get("content-type", "").startswith("text/event-stream"):
            resp.close(); return b""
        body = resp.read(inst.MAX_BODY + 1)
        if len(body) > inst.MAX_BODY:
            raise RuntimeError("reverse proxy response exceeded bounded buffer")
        return body

@dataclass
class Harness:
    args: argparse.Namespace
    root: Path
    evidence: dict[str, Any]
    secrets: list[str] = field(default_factory=list)
    evidence_created: bool = False
    def flush(self) -> None:
        path = Path(self.args.evidence)
        if path.is_symlink():
            raise AssertionError(f"refusing to write evidence through symlink: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (0 if self.evidence_created else os.O_EXCL)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "w") as fp:
            fp.write(json.dumps(inst.redact(self.evidence, self.secrets), indent=2, sort_keys=True) + "\n")
        os.chmod(path, 0o600); self.evidence_created = True
    def add(self, section: str, value: Any) -> None:
        self.evidence.setdefault(section, []).append(inst.redact(value, self.secrets)); self.flush()

def start_tls_proxy(work: Path, backend_base: str) -> tuple[http.server.ThreadingHTTPServer, threading.Thread, str, list[str]]:
    extra = inst.generate_tls(work / "tls")
    if not extra:
        raise AssertionError("openssl is required for reverse proxy TLS edge verification")
    cert, key = extra[extra.index("--ssl-cert-file") + 1], extra[extra.index("--ssl-key-file") + 1]
    parsed = urllib.parse.urlsplit(backend_base)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ReverseProxyHandler)
    public_host = f"127.0.0.1:{server.server_address[1]}"
    server.state = ProxyState(backend_base, parsed.netloc, f"{parsed.scheme}://{parsed.netloc}", public_host, f"https://{public_host}")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(cert, key)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, name=f"reverse-proxy-{public_host}", daemon=True); thread.start()
    return server, thread, f"https://{public_host}", extra

def stop_tls_proxy(server: http.server.ThreadingHTTPServer | None, thread: threading.Thread | None) -> dict[str, Any]:
    if not server or not thread: return {"stopped": False}  # noqa: E701
    server.shutdown(); server.server_close(); thread.join(timeout=5)
    if thread.is_alive(): raise RuntimeError("reverse proxy thread did not stop")
    return {"stopped": True, "thread_alive": False}

def edge_request(url: str, key: str | None, origin: str | None = None, *, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    all_headers = {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors", **(headers or {})}
    if origin is not None:
        all_headers["Origin"] = origin
    return inst.request(url, key=key, method=method, data=data, headers=all_headers, https=True)

def edge_open_headers(url: str, key: str, origin: str | None = None) -> tuple[int, dict[str, str]]:
    headers = {"Authorization": f"Bearer {key}", "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}
    if origin is not None:
        headers["Origin"] = origin
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = inst.opener_for(https=True).open(req, timeout=8)
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}
    with resp:
        return resp.status, {k.lower(): v for k, v in resp.headers.items()}

def hostile_probe(edge_base: str, state: ProxyState, key: str, *, host: str | None = None, origin: str | None = None) -> dict[str, Any]:
    before = len(state.forwarded)
    headers = {"Origin": origin or state.public_origin, "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}
    if host: headers["Host"] = host
    status, _, _ = inst.request(edge_base + "/lab/ui-api/v1/bootstrap", key=key, headers=headers, https=True)
    return {"status": status, "forwarded": len(state.forwarded) != before}

def missing_mutation_origin_probe(edge_base: str, state: ProxyState, key: str) -> dict[str, Any]:
    before = len(state.forwarded)
    headers = {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors", "Content-Type": "application/json"}
    status, _, _ = inst.request(edge_base + "/lab/ui-api/v1/catalog/refresh", key=key, method="POST", data=b"", headers=headers, https=True)
    return {"status": status, "forwarded": len(state.forwarded) != before}

def launch_backend(command: list[str], work: Path, key_file: Path, port: int, models: Path, store: Path) -> tuple[subprocess.Popen[bytes], Any, Path, str]:
    args = command + ["--host", "127.0.0.1", "--port", str(port), "--models-dir", str(models), "--model-store-root", str(store), "--api-key-file", str(key_file), "--api-prefix", "/lab", "--no-models-autoload", "--settings", "--props", "--metrics", "--webui", "--sse-ping-interval", "5"]
    log_path = work / "backend.log"; log_file = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(args, cwd=work, env=inst.clean_env(work / "home", store), stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log_file, log_path, f"http://127.0.0.1:{port}"

def stop_backend_group(proc: subprocess.Popen[bytes], log_file: Any, log_path: Path, key_file: Path) -> dict[str, Any]:
    forced = False
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced = True; os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"backend exited with {proc.returncode}; log: {log_path}")
        return {"exit_code": proc.returncode, "forced": forced, "log": str(log_path)}
    finally:
        log_file.close(); key_file.unlink(missing_ok=True)

def verify_artifact(h: Harness, artifact: inst.Artifact) -> None:
    work = h.root / artifact.name; work.mkdir(mode=0o700)
    for name in ("home", "models", "store"): (work / name).mkdir(mode=0o700)
    key = secrets.token_urlsafe(32); h.secrets.append(key); key_file = work / "api-key.txt"; inst.write_private(key_file, key + "\n")
    proc = log_file = log_path = None; proxy = thread = None; tls_extra = None
    try:
        proc, log_file, log_path, backend = launch_backend(artifact.command, work, key_file, inst.free_port(), work / "models", work / "store")
        inst.wait_ready(backend, proc)
        proxy, thread, edge, tls_extra = start_tls_proxy(work, backend); state: ProxyState = proxy.state
        status, headers, html = inst.request(edge + "/lab/webui/", https=True)
        assert status == 200 and b"<!" in html[:64] and headers.get("content-security-policy"), status
        assets = re.findall(rb'(?:src|href)="([^"#]+)"', html)
        assert assets, "public shell had no asset references"
        asset_url = edge + "/lab/webui/" + assets[0].decode().removeprefix("./")
        assert inst.request(asset_url, https=True)[0] == 200
        assert inst.request(asset_url, method="HEAD", https=True)[2] == b""
        status, _, body = edge_request(edge + "/lab/ui-api/v1/bootstrap", key)
        assert status == 200 and key.encode() not in body, status
        assert json.loads(body)["server"]["api_base"] == "/lab"
        status, _, body = edge_request(edge + "/lab/ui-api/v1/catalog/refresh", key, state.public_origin, method="POST", data=b"", headers={"Content-Type": "application/json"})
        if status != 202 or not json.loads(body).get("operation_id"):
            raise AssertionError(f"catalog refresh returned status={status}")
        sse_status, sse_headers = edge_open_headers(edge + "/lab/ui-api/v1/events", key)
        assert sse_status == 200 and "text/event-stream" in sse_headers.get("content-type", ""), sse_headers
        hostile_origin = hostile_probe(edge, state, key, origin="https://foreign.invalid")
        hostile_host = hostile_probe(edge, state, key, host="foreign.invalid")
        missing_mutation_origin = missing_mutation_origin_probe(edge, state, key)
    finally:
        proxy_shutdown = stop_tls_proxy(proxy, thread)
        if tls_extra: inst.cleanup_tls_material(tls_extra)
        backend_shutdown = stop_backend_group(proc, log_file, log_path, key_file) if proc and log_file and log_path else {"exit_code": None}
    if log_path and key in Path(log_path).read_text(errors="replace"):
        raise AssertionError("credential leaked to backend log")
    with state.lock:
        forwards, blocked, sse_cancelled = list(state.forwarded), list(state.blocked), state.sse_cancelled
    assert hostile_origin == {"status": 403, "forwarded": False} and hostile_host == {"status": 403, "forwarded": False} and missing_mutation_origin == {"status": 403, "forwarded": False}
    assert any(r["authorization_header"] and not r["url_has_credential_query"] for r in forwards), forwards
    h.add("reverse_proxy", {"label": artifact.name, "tls_edge": True, "public_shell_without_origin": 200, "public_asset_without_origin": 200, "private_bootstrap_without_origin": 200, "catalog_refresh": 202, "sse_headers_cancelled": sse_cancelled, "hostile_origin": hostile_origin, "hostile_host": hostile_host, "missing_mutation_origin": missing_mutation_origin, "forwarded": forwards, "blocked": blocked, "proxy_shutdown": proxy_shutdown, "backend_shutdown": backend_shutdown, "scope": "trusted reverse-proxy edge policy only; direct backend security remains covered by product middleware and installed-artifact checks"})

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server-bin", default=os.environ.get("WEBUI_SERVER_BIN")); p.add_argument("--cli-bin", default=os.environ.get("WEBUI_CLI_BIN"))
    p.add_argument("--build-source-head", default=os.environ.get("WEBUI_BUILD_SOURCE_HEAD")); p.add_argument("--evidence", default=os.environ.get("WEBUI_REVERSE_PROXY_EVIDENCE", "webui-reverse-proxy-evidence.json"))
    args = p.parse_args()
    if not args.server_bin or not args.cli_bin: p.error("--server-bin and --cli-bin are required")
    evidence = {"result": "fail", "scope": "loopback-only TLS reverse proxy test for installed WebUI artifacts", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "limits": ["not a production proxy implementation or recommendation", "no real checkpoint inference", "no browser automation"]}
    evidence_path = Path(args.evidence).resolve(); root = evidence_path.parent / f"{evidence_path.stem}-artifacts-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"; root.mkdir(mode=0o700, parents=True); os.chmod(root, 0o700)
    h = Harness(args, root, evidence); h.evidence["artifact_dir"] = str(root); h.flush()
    try:
        artifacts = [inst.relocate(Path(args.server_bin), root / "installed", "mlxcel-server", []), inst.relocate(Path(args.cli_bin), root / "installed", "mlxcel-serve", ["serve"])]
        h.evidence["artifacts"] = [{"label": a.name, "source": str(a.source), "relocated": str(a.relocated), "sha256": a.sha256, "command_shape": [a.relocated.name, *a.command[1:]], "build_source_head": args.build_source_head or "unknown"} for a in artifacts]; h.flush()
        for artifact in artifacts: verify_artifact(h, artifact)
        h.evidence["result"] = "pass"; h.evidence["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); h.flush(); print(json.dumps(inst.redact(h.evidence, h.secrets), indent=2, sort_keys=True)); return 0
    except BaseException as exc:
        h.evidence["result"] = "fail"; h.evidence["error"] = {"type": type(exc).__name__, "message": inst.redact(str(exc), h.secrets)}; h.evidence["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); h.flush(); print(f"reverse proxy verification failed: {type(exc).__name__}: {inst.redact(str(exc), h.secrets)}", file=sys.stderr); return 1
if __name__ == "__main__":
    raise SystemExit(main())
