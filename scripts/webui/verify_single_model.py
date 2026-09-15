#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Explicit real-checkpoint single-model gate; never starts inference on import.

Run with MLXCEL_REQUIRE_MODELS=1 and the pinned WebUI contract Python environment.
Four fresh processes run serially: server/CLI serve, each with WebUI on and off.
Successful capture still requires semantic review of the actual model replies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MAX_BODY = 1024 * 1024
MAX_LOG = 16 * 1024 * 1024


class GateError(Exception):
    """Messages are fixed stage codes, never raw server responses or secrets."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise GateError(code)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_provenance(model: Path) -> dict[str, Any]:
    require(model.is_dir(), "checkpoint_directory_required")
    config = model / "config.json"
    require(config.is_file(), "checkpoint_config_required")
    require(config.stat().st_size <= MAX_BODY, "checkpoint_config_too_large")
    require(
        isinstance(json.loads(config.read_text()), dict),
        "checkpoint_config_object_required",
    )
    weights = sorted(model.glob("*.safetensors"))
    require(
        bool(weights) and all(p.is_file() and p.stat().st_size > 0 for p in weights),
        "nonempty_checkpoint_weights_required",
    )
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    revision = None
    if metadata.is_file():
        with metadata.open("r") as stream:
            candidate = stream.readline(128).strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
            revision = candidate
    return {
        "name": model.name,
        "config_sha256": sha256(config),
        "weights_sha256": {p.name: sha256(p) for p in weights},
        "cached_config_revision": revision,
        "revision_source": "local_download_metadata" if revision else "unknown",
    }


def private_write(path: Path, body: bytes) -> None:
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        stream.write(body)


def clean_env(home: Path, store: Path) -> dict[str, str]:
    env = {
        key: os.environ[key]
        for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
        if key in os.environ
    }
    env.update(
        HOME=str(home),
        XDG_CACHE_HOME=str(home / "cache"),
        HF_HOME=str(home / "hf"),
        MLXCEL_MODELS_DIR=str(store),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_PROXY="*",
        no_proxy="*",
        RUST_LOG="warn",
    )
    return env


def validate_prefix(prefix: str) -> str:
    require(
        bool(re.fullmatch(r"(?:/[A-Za-z0-9_-]+)+", prefix)),
        "nonempty_unambiguous_api_prefix_required",
    )
    return prefix


def command(
    binary: Path,
    cli: bool,
    ui: bool,
    model: Path,
    store: Path,
    keyfile: Path,
    port: int,
    prefix: str,
) -> list[str]:
    result = [str(binary)] + (["serve"] if cli else [])
    result += [
        "-m",
        str(model),
        "--model-store-root",
        str(store),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-prefix",
        validate_prefix(prefix),
        "--api-key-file",
        str(keyfile),
        "--ctx-size",
        "2048",
        "--parallel",
        "1",
    ]
    return result + (["--webui"] if ui else ["--no-webui"])


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """POSIX wall-clock bound includes trickling HTTP headers and body reads."""

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError("request_deadline")

    handler = signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        if previous[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous[0] - (time.monotonic() - started)),
                previous[1],
            )


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def request(
    origin: str,
    path: str,
    key: str | None,
    timeout: float,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    headers = {"Authorization": "Bearer " + key} if key else {}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(
        origin + path,
        data=data,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with deadline(timeout):
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read(MAX_BODY + 1)
            require(len(raw) <= MAX_BODY, "response_body_limit")
            require(not key or key.encode() not in raw, "credential_in_response")
            return (
                response.status,
                {k.lower(): v for k, v in response.headers.items()},
                raw,
            )


def completion_text(payload: Any) -> str:
    require(isinstance(payload, dict), "completion_object_required")
    choices = payload.get("choices")
    require(
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict),
        "completion_choice_required",
    )
    choice = choices[0]
    require(
        choice.get("finish_reason") in ("stop", "length"), "completion_finish_required"
    )
    message = choice.get("message")
    require(isinstance(message, dict), "completion_message_required")
    content = message.get("content")
    require(
        isinstance(content, str) and bool(content.strip()) and len(content) <= 65536,
        "nonempty_bounded_completion_required",
    )
    return content


def shutdown(process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    alive = process.poll() is None
    forced = False
    error = None
    if alive:
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as failure:
            error = type(failure).__name__
            forced = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                error = "kill_reap_timeout"
    return {
        "alive_before_shutdown": alive,
        "forced": forced,
        "exit_code": process.returncode,
        "error": error,
        "passed": alive and not forced and process.returncode == 0,
    }


def sanitize_log(path: Path, key: str) -> bool:
    with path.open("rb") as stream:
        raw = stream.read(MAX_LOG + 1)
    leaked = key.encode() in raw
    if len(raw) > MAX_LOG:
        safe = b"Log exceeded the bounded retained log size; raw contents omitted.\n"
    else:
        safe = re.sub(
            rb"(?i)Bearer\s+[^\s]+",
            b"Bearer <redacted>",
            raw.replace(key.encode(), b"<redacted>"),
        )
    path.write_bytes(safe)
    return leaked or len(raw) > MAX_LOG


class Contract:
    def __init__(self, source: Path):
        sys.path.insert(0, str(source / "scripts" / "ci"))
        from webui_contract_checks import load_contract, validator_for

        self.contract = load_contract()
        self.validator_for = validator_for

    def check(self, schema: str, value: Any) -> Any:
        self.validator_for(self.contract, schema).validate(value)
        return value


def exercise(
    origin: str, key: str, ui: bool, args: argparse.Namespace, contract: Contract
) -> dict[str, Any]:
    prefix = args.api_prefix

    def get(path: str, schema: str) -> Any:
        status, _, raw = request(origin, prefix + path, None, args.request_timeout)
        require(status == 401, "unauthenticated_control_not_refused")
        contract.check("ErrorEnvelope", json.loads(raw))
        status, _, raw = request(origin, prefix + path, key, args.request_timeout)
        require(status == 200, "canonical_get_status")
        return contract.check(schema, json.loads(raw))

    status, _, raw = request(origin, prefix + "/v1/models", key, args.request_timeout)
    require(status == 200, "models_status")
    models = json.loads(raw).get("data")
    require(
        isinstance(models, list) and len(models) == 1, "single_inference_model_required"
    )
    inference_id = models[0].get("id")
    require(
        isinstance(inference_id, str) and bool(inference_id),
        "inference_identity_required",
    )
    result: dict[str, Any] = {"inference_id": inference_id, "ui_enabled": ui}
    for path in ("/webui/", "/ui-api/v1/bootstrap", "/v1/models"):
        require(
            request(origin, path, key, args.request_timeout)[0] == 404,
            "unprefixed_route_exposed",
        )
    if ui:
        status, headers, raw = request(
            origin, prefix + "/webui/", None, args.request_timeout
        )
        require(
            status == 200 and "text/html" in headers.get("content-type", ""),
            "bundled_ui_required",
        )
        csp = headers.get("content-security-policy", "")
        require(
            "script-src 'self'" in csp
            and "'unsafe-inline'" not in csp
            and "'unsafe-eval'" not in csp,
            "served_csp_required",
        )
        bootstrap = get("/ui-api/v1/bootstrap", "BootstrapResponse")
        require(
            bootstrap["server"]["mode"] == "single_model", "single_model_mode_required"
        )
        require(bootstrap["server"]["api_base"] == prefix, "bootstrap_prefix_mismatch")
        for action in ("load", "unload", "download", "cache_delete"):
            state = bootstrap["actions"][action]
            require(
                state["state"] in ("disabled", "read_only") and bool(state["reason"]),
                "single_model_actions_not_readonly",
            )
        catalog = get("/ui-api/v1/catalog", "CatalogListResponse")
        require(
            len(catalog["items"]) == 1 and catalog["pagination"]["next_cursor"] is None,
            "single_catalog_required",
        )
        entry = catalog["items"][0]
        identity = entry["identity"]
        require(
            identity["source"] == "single_model"
            and entry["lifecycle"]["state"] == "ready",
            "single_ready_entry_required",
        )
        require(
            identity["inference_id"] == inference_id,
            "inference_catalog_identity_mismatch",
        )
        query = urllib.parse.urlencode(
            {"model_id": identity["id"], "autoload": "false"}
        )
        runtime = get("/ui-api/v1/runtime?" + query, "RuntimeSnapshot")
        require(
            runtime["model_id"] == identity["id"]
            and runtime["server_instance_id"]
            == bootstrap["server"]["server_instance_id"],
            "runtime_identity_mismatch",
        )
        common = {
            "model_id": identity["id"],
            "expected_revision": identity["revision"],
            "idempotency_key": "single-readonly-check",
        }
        mutations = [
            ("model-actions", "ModelActionRequest", {**common, "action": action})
            for action in ("load", "unload")
        ]
        mutations += [
            ("model-removals", "RemovalRequest", common),
            (
                "downloads",
                "DownloadRequest",
                {
                    "repo_id": "mlx-community/Qwen3-4B-4bit",
                    "idempotency_key": "single-readonly-download",
                },
            ),
        ]
        refusals = []
        for route, schema, body in mutations:
            contract.check(schema, body)
            status, _, raw = request(
                origin, prefix + "/ui-api/v1/" + route, key, args.request_timeout, body
            )
            require(status == 422, "readonly_mutation_status")
            error = contract.check("ErrorEnvelope", json.loads(raw))
            require(error["error"]["code"] == "unsupported", "readonly_mutation_reason")
            refusals.append(
                {"route": route, "status": status, "code": error["error"]["code"]}
            )
        after = get("/ui-api/v1/catalog", "CatalogListResponse")
        require(
            len(after["items"]) == 1
            and after["items"][0]["identity"] == identity
            and after["items"][0]["lifecycle"]["state"] == "ready",
            "readonly_mutation_changed_catalog",
        )
        result.update(
            bootstrap=bootstrap,
            catalog=catalog,
            runtime=runtime,
            readonly_refusals=refusals,
            csp=csp,
        )
    else:
        for route in ("/webui/", "/ui-api/v1/bootstrap", "/ui-api/v1/catalog"):
            require(
                request(origin, prefix + route, key, args.request_timeout)[0] == 404,
                "ui_off_route_exposed",
            )
        result["ui_routes_absent"] = True
    body = {
        "model": inference_id,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
    }
    path = prefix + "/v1/chat/completions?autoload=false"
    require(
        request(origin, path, None, args.request_timeout, body)[0] == 401,
        "unauthenticated_inference_not_refused",
    )
    status, _, raw = request(origin, path, key, args.request_timeout, body)
    require(status == 200, "actual_completion_status")
    result["actual_reply"] = completion_text(json.loads(raw))
    result["semantic_review"] = (
        "REQUIRED; nonempty actual model output is not semantic acceptance"
    )
    return result


def run_arm(
    binary: Path,
    cli: bool,
    ui: bool,
    model: Path,
    args: argparse.Namespace,
    root: Path,
    contract: Contract,
) -> dict[str, Any]:
    directory = root / (("cli" if cli else "server") + ("-ui-on" if ui else "-ui-off"))
    directory.mkdir(mode=0o700)
    home, store = directory / "home", directory / "store"
    home.mkdir(mode=0o700)
    store.mkdir(mode=0o700)
    key = secrets.token_urlsafe(32)
    keyfile, logfile = directory / "private-key", directory / "server.log"
    result: dict[str, Any] = {
        "entrypoint": "cli serve" if cli else "server",
        "ui_enabled": ui,
        "status": "FAIL",
        "log": str(logfile),
    }
    process = None
    try:
        private_write(keyfile, (key + "\n").encode())
        private_write(logfile, b"")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        origin = f"http://127.0.0.1:{port}"
        started = time.monotonic()
        with logfile.open("ab") as log:
            process = subprocess.Popen(
                command(binary, cli, ui, model, store, keyfile, port, args.api_prefix),
                cwd=directory,
                env=clean_env(home, store),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        until = started + args.startup_timeout
        while True:
            require(process.poll() is None, "server_exited_before_ready")
            remaining = until - time.monotonic()
            require(remaining > 0, "startup_deadline")
            try:
                if (
                    request(
                        origin, args.api_prefix + "/v1/models", key, min(2, remaining)
                    )[0]
                    == 200
                ):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(min(0.1, max(0, until - time.monotonic())))
        result["startup"] = {
            "seconds_to_models_ready": time.monotonic() - started,
            "scope": "Includes real checkpoint loading; not isolated WebUI overhead",
            "rss_bytes": None,
            "gpu_memory_bytes": None,
        }
        result.update(exercise(origin, key, ui, args, contract))
        result["status"] = "CAPTURED_REQUIRES_SEMANTIC_REVIEW"
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - fail closed without leaking server/schema error payloads
        result["error"] = (
            str(error) if isinstance(error, GateError) else type(error).__name__
        )
    finally:
        try:
            if process is not None:
                result["shutdown"] = shutdown(process, args.shutdown_timeout)
                if not result["shutdown"]["passed"]:
                    result["status"] = "FAIL"
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - fail closed without leaking server/schema error payloads
            result.update(status="FAIL", shutdown_error=type(error).__name__)
        finally:
            keyfile.unlink(missing_ok=True)
            if logfile.exists() and sanitize_log(logfile, key):
                result.update(status="FAIL", log_redacted_or_oversized=True)
    encoded = json.dumps(result, indent=2).encode()
    if key.encode() in encoded:
        result = {"status": "FAIL", "error": "credential_in_evidence"}
        encoded = json.dumps(result).encode()
    private_write(directory / "result.json", encoded)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--cli-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--build-sha",
        required=True,
        help="Caller-declared source SHA used to build both binaries",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Caller-declared comma-separated features, including webui",
    )
    parser.add_argument("--api-prefix", default="/single-check")
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--startup-resource-report",
        type=Path,
        help="Optional separate measure_startup.py report to reference by hash",
    )
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {"status": "FAIL", "arms": []}
    root = None
    try:
        require(
            os.environ.get("MLXCEL_REQUIRE_MODELS") == "1",
            "MLXCEL_REQUIRE_MODELS_1_required",
        )
        require(hasattr(signal, "setitimer"), "posix_request_deadline_required")
        require(
            bool(re.fullmatch(r"[0-9a-fA-F]{40}", args.build_sha)),
            "full_build_sha_required",
        )
        features = args.features.split(",")
        require(
            "webui" in features
            and all(re.fullmatch(r"[a-zA-Z0-9_-]+", value) for value in features),
            "declared_features_including_webui_required",
        )
        validate_prefix(args.api_prefix)
        for value, limit in (
            (args.startup_timeout, 600),
            (args.request_timeout, 300),
            (args.shutdown_timeout, 180),
        ):
            require(0 < value <= limit, "finite_bounded_timeout_required")
        model = args.model.resolve(strict=True)
        provenance = checkpoint_provenance(model)
        binaries = [
            args.server_bin.resolve(strict=True),
            args.cli_bin.resolve(strict=True),
        ]
        require(
            all(p.is_file() and os.access(p, os.X_OK) for p in binaries),
            "executable_binaries_required",
        )
        if args.output_dir:
            candidate_root = args.output_dir.resolve()
            candidate_root.mkdir(mode=0o700, parents=True, exist_ok=False)
            root = candidate_root
        else:
            root = Path(tempfile.mkdtemp(prefix="mlxcel-single-"))
        report.update(
            caller_build_sha=args.build_sha,
            caller_features=features,
            binary_sha256={
                label: sha256(p) for label, p in zip(("server", "cli"), binaries)
            },
            checkpoint=provenance,
            platform={
                "system": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "mac_ver": platform.mac_ver()[0],
            },
            api_prefix=args.api_prefix,
            startup_resources="Use measure_startup.py for isolated UI-on/off startup RSS/resource comparison; readiness here includes model loading.",
        )
        if args.startup_resource_report:
            report["startup_resource_report"] = {
                "path": str(args.startup_resource_report.resolve(strict=True)),
                "sha256": sha256(args.startup_resource_report),
            }
        contract = Contract(args.source_root.resolve())
        for binary, cli in zip(binaries, (False, True)):
            for ui in (True, False):
                arm = run_arm(binary, cli, ui, model, args, root, contract)
                report["arms"].append(arm)
                require(
                    arm["status"] == "CAPTURED_REQUIRES_SEMANTIC_REVIEW",
                    "single_model_arm_failed",
                )
        require(
            checkpoint_provenance(model) == provenance,
            "checkpoint_changed_during_verification",
        )
        report.update(
            status="CAPTURED_REQUIRES_SEMANTIC_REVIEW", checkpoint_unchanged=True
        )
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - fail closed without leaking server/schema error payloads
        report["error"] = (
            str(error) if isinstance(error, GateError) else type(error).__name__
        )
    if root is not None:
        private_write(root / "result.json", json.dumps(report, indent=2).encode())
    print(
        json.dumps(
            {
                "status": report["status"],
                "result": str(root / "result.json") if root else None,
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["status"] == "CAPTURED_REQUIRES_SEMANTIC_REVIEW" else 1


if __name__ == "__main__":
    sys.exit(main())
