#!/usr/bin/env python3
# Copyright 2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate the WebUI OpenAPI contract, generated DTOs, and shared fixtures."""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry
    from referencing.jsonschema import DRAFT202012
except ImportError as e:  # pragma: no cover - exercised by developer machines without the venv.
    print(
        "error: jsonschema is required. Install the pinned isolated verifier with: "
        "python3 -m venv /tmp/mlxcel-webui-contract && "
        "/tmp/mlxcel-webui-contract/bin/python -m pip install -r scripts/ci/webui_contract_requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2) from e

ROOT = Path(__file__).resolve().parents[2]
OPENAPI = ROOT / "docs/webui/api.yaml"
DTO = ROOT / "docs/webui/generated/ui-api.d.ts"
FIXTURES = ROOT / "tests/fixtures/webui"

IDENT = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
RFC3339_DATE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
FORMAT_CHECKER = FormatChecker()
SEEDED_SENSITIVE_MARKERS = (
    "/Users/mlxcel-seeded-secret/",
    "/Volumes/mlxcel-seeded-secret/",
    "/private/mlxcel-seeded-secret/",
    "hf_MLXCELE2ESECRET",
    "Bearer mlxcel-seeded-secret-token",
)
ALLOWED_SCHEMA_KEYS = {
    "$ref",
    "additionalProperties",
    "anyOf",
    "const",
    "description",
    "enum",
    "format",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "oneOf",
    "pattern",
    "properties",
    "required",
    "type",
}


class ContractError(Exception):
    pass


@FORMAT_CHECKER.checks("date-time", raises=(ValueError,))
def is_rfc3339_date_time(value: object) -> bool:
    if not isinstance(value, str) or not RFC3339_DATE_TIME.fullmatch(value):
        return False
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.tzinfo is not None


def load_contract() -> dict[str, Any]:
    try:
        with OPENAPI.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ContractError(f"{OPENAPI} must remain JSON-compatible YAML: {e}") from e
    if data.get("openapi") != "3.1.0":
        raise ContractError("WebUI contract must stay on OpenAPI 3.1.0 so JSON Schema validation is unambiguous")
    if not isinstance(data.get("components", {}).get("schemas"), dict):
        raise ContractError("OpenAPI components.schemas is required")
    lint_schema_keywords(data["components"]["schemas"])
    require_active_format_checkers(data)
    for name, schema in data["components"]["schemas"].items():
        Draft202012Validator.check_schema(schema)
        if isinstance(schema, dict) and schema.get("type") == "object" and schema.get("additionalProperties") is not False:
            raise ContractError(f"schema {name} must be strict: set additionalProperties to false or a typed schema")
    return data


def iter_schema_formats(node: Any, path: str = "#") -> list[tuple[str, str]]:
    formats: list[tuple[str, str]] = []
    if isinstance(node, dict):
        fmt = node.get("format")
        if isinstance(fmt, str):
            formats.append((path, fmt))
        for key, value in node.items():
            formats.extend(iter_schema_formats(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            formats.extend(iter_schema_formats(value, f"{path}/{i}"))
    return formats


def require_active_format_checkers(contract: dict[str, Any]) -> None:
    missing = sorted({fmt for _, fmt in iter_schema_formats(contract) if fmt not in FORMAT_CHECKER.checkers})
    if missing:
        raise ContractError(f"schema uses format(s) without active checker: {', '.join(missing)}")


def lint_schema_keywords(node: Any, path: str = "#/components/schemas") -> None:
    if isinstance(node, dict):
        if path.endswith("/properties"):
            for key, value in node.items():
                lint_schema_keywords(value, f"{path}/{key}")
            return
        schema_like = any(k in node for k in ("type", "$ref", "oneOf", "anyOf", "properties", "items", "enum", "const"))
        if schema_like:
            unknown = sorted(k for k in node if k not in ALLOWED_SCHEMA_KEYS)
            if unknown:
                raise ContractError(f"unknown schema keyword(s) at {path}: {', '.join(unknown)}")
        for key, value in node.items():
            lint_schema_keywords(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            lint_schema_keywords(value, f"{path}/{i}")


def schema_name_from_ref(ref: str) -> str:
    prefix = "#/components/schemas/"
    if not ref.startswith(prefix):
        raise ContractError(f"unsupported local ref {ref!r}")
    return ref[len(prefix) :]


def resolve_ref(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        return components[schema_name_from_ref(schema["$ref"])]
    return schema


def ts_prop(name: str) -> str:
    return name if IDENT.match(name) else json.dumps(name)


def ts_type(schema: dict[str, Any], components: dict[str, Any]) -> str:
    if "$ref" in schema:
        return schema_name_from_ref(schema["$ref"])
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return " | ".join(json.dumps(v) for v in schema["enum"])
    if "oneOf" in schema:
        return " | ".join(ts_type(s, components) for s in schema["oneOf"])
    if "anyOf" in schema:
        return " | ".join(ts_type(s, components) for s in schema["anyOf"])
    typ = schema.get("type")
    if isinstance(typ, list):
        return " | ".join("null" if t == "null" else ts_type({**schema, "type": t}, components) for t in typ)
    if typ == "null":
        return "null"
    if typ == "string":
        return "string"
    if typ in ("integer", "number"):
        return "number"
    if typ == "boolean":
        return "boolean"
    if typ == "array":
        return f"ReadonlyArray<{ts_type(schema.get('items', {}), components)}>"
    if typ == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not props:
            addl = schema.get("additionalProperties", False)
            if addl is False:
                return "Record<string, never>"
            if isinstance(addl, dict):
                return f"Record<string, {ts_type(addl, components)}>"
            raise ContractError("object schemas must be strict or typed maps")
        parts = []
        for key, value in props.items():
            opt = "" if key in required else "?"
            parts.append(f"readonly {ts_prop(key)}{opt}: {ts_type(value, components)};")
        if isinstance(schema.get("additionalProperties"), dict):
            parts.append(f"readonly [key: string]: {ts_type(schema['additionalProperties'], components)};")
        return "{ " + " ".join(parts) + " }"
    return "unknown"


def generate_dto(contract: dict[str, Any]) -> str:
    components = contract["components"]["schemas"]
    lines = [
        "// Generated by scripts/ci/check_webui_contract.py --fix from docs/webui/api.yaml.",
        "// Do not edit by hand; update the schema and fixtures together.",
        "",
    ]
    for name, schema in components.items():
        if schema.get("type") == "object" and "properties" in schema and "oneOf" not in schema and "anyOf" not in schema:
            lines.append(f"export interface {name} {{")
            required = set(schema.get("required", []))
            for key, value in schema.get("properties", {}).items():
                opt = "" if key in required else "?"
                lines.append(f"  readonly {ts_prop(key)}{opt}: {ts_type(value, components)};")
            if isinstance(schema.get("additionalProperties"), dict):
                lines.append(f"  readonly [key: string]: {ts_type(schema['additionalProperties'], components)};")
            lines.append("}")
            lines.append("")
        else:
            lines.append(f"export type {name} = {ts_type(schema, components)};")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def iter_fixture_files() -> list[Path]:
    return sorted(p for p in FIXTURES.rglob("*.json") if p.is_file())


def fixture_schema_name(path: Path, value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("$schemaName"), str):
        return value["$schemaName"]
    name = path.name
    if name == "requirement-map.json":
        return "RequirementMap"
    if name == "strings.json":
        return "StringCatalog"
    if name == "identity-vectors.json":
        return "IdentityVectors"
    if name.startswith("bootstrap"):
        return "BootstrapResponse"
    if name.startswith("catalog"):
        return "CatalogListResponse"
    if name.startswith("operation"):
        return "Operation"
    if name.startswith("runtime"):
        return "RuntimeSnapshot"
    if name.startswith("error"):
        return "ErrorEnvelope"
    if "event" in name:
        return "UiEvent"
    return "WebUiContractFixture"


def validator_for(contract: dict[str, Any], schema_name: str) -> Draft202012Validator:
    base_uri = "urn:mlxcel:webui:contract"
    registry = Registry().with_resource(base_uri, DRAFT202012.create_resource(contract))
    return Draft202012Validator({"$ref": f"{base_uri}#/components/schemas/{schema_name}"}, registry=registry, format_checker=FORMAT_CHECKER)


def seeded_sensitive_markers(value: Any) -> list[str]:
    rendered = json.dumps(value, ensure_ascii=False)
    return [marker for marker in SEEDED_SENSITIVE_MARKERS if marker in rendered]


def check_no_seeded_sensitive_markers() -> None:
    failures = []
    for path in iter_fixture_files():
        value = json.loads(path.read_text(encoding="utf-8"))
        markers = seeded_sensitive_markers(value)
        if markers:
            failures.append(f"{path}: contains seeded sensitive marker(s): {', '.join(markers)}")
    if failures:
        raise ContractError("fixture redaction check failed:\n" + "\n".join(failures))


def validate_fixtures(contract: dict[str, Any]) -> None:
    files = iter_fixture_files()
    if not files:
        raise ContractError("no WebUI fixtures found")
    failures: list[str] = []
    scenario_names: set[str] = set()
    validators: dict[str, Draft202012Validator] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        schema_name = fixture_schema_name(path, value)
        if isinstance(value, dict) and "$schemaName" in value:
            value = {k: v for k, v in value.items() if k != "$schemaName"}
        if schema_name not in contract["components"]["schemas"]:
            failures.append(f"{path}: unknown schema {schema_name}")
            continue
        validator = validators.setdefault(schema_name, validator_for(contract, schema_name))
        for error in sorted(validator.iter_errors(value), key=lambda e: list(e.path)):
            location = "$" + "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in error.path)
            failures.append(f"{path}:{location}: {error.message}")
        if schema_name == "WebUiContractFixture":
            scenario_names.add(value.get("scenario", ""))
    required_scenarios = {
        "duplicate_load",
        "load_unload_race",
        "busy_eviction",
        "stale_revision",
        "failed_load",
        "download_cancel",
        "deletion_refusal",
        "sse_gap",
        "server_restart",
        "unknown_null_partial_error",
    }
    missing = sorted(required_scenarios - scenario_names)
    if missing:
        failures.append(f"missing required table fixtures: {', '.join(missing)}")
    if failures:
        raise ContractError("fixture validation failed:\n" + "\n".join(failures))


def check_requirement_map() -> None:
    path = FIXTURES / "requirement-map.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    failures = []
    for req in data.get("requirements", []):
        if not req.get("child") or not req.get("contract_artifact") or not req.get("test_fixture"):
            failures.append(f"requirement {req.get('id')} must map to a child, artifact, and fixture")
            continue
        if not (ROOT / req["contract_artifact"]).exists():
            failures.append(f"requirement {req.get('id')} artifact is missing: {req['contract_artifact']}")
        if not (ROOT / req["test_fixture"]).exists():
            failures.append(f"requirement {req.get('id')} fixture is missing: {req['test_fixture']}")
    if failures:
        raise ContractError("requirement map is incomplete:\n" + "\n".join(failures))



def canonical_identity_id(vector: dict[str, Any]) -> tuple[str, str]:
    source_key_hash = hashlib.sha256(vector["redacted_source_key"].encode("utf-8")).hexdigest()
    canonical = {
        "entry_key": vector["entry_key"],
        "namespace_hash": source_key_hash,
        "source": vector["source"],
        "source_rank": vector["source_rank"],
        "version": 1,
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    stable_id = "mdl_" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return stable_id, source_key_hash


def check_identity_vectors() -> None:
    path = FIXTURES / "identity-vectors.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    failures = []
    for index, vector in enumerate(data.get("identity_vectors", [])):
        expected_id, source_key_hash = canonical_identity_id(vector)
        if vector.get("expected_id") != expected_id:
            failures.append(f"vector {index} expected_id mismatch: {vector.get('expected_id')} != {expected_id}")
        if vector.get("source_key_hash") != source_key_hash:
            failures.append(f"vector {index} source_key_hash mismatch: {vector.get('source_key_hash')} != {source_key_hash}")
        if vector.get("canonical_identity") != {
            "entry_key": vector["entry_key"],
            "namespace_hash": source_key_hash,
            "source": vector["source"],
            "source_rank": vector["source_rank"],
            "version": 1,
        }:
            failures.append(f"vector {index} canonical_identity is not the exact sorted-input object")
    if failures:
        raise ContractError("identity vector validation failed:\n" + "\n".join(failures))

def check_identity_collision_fixture() -> None:
    path = FIXTURES / "scenarios" / "identity-collision.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"]
    ids = [item["identity"]["id"] for item in items]
    source_hashes = [item["identity"]["source_key_hash"] for item in items]
    inference_ids = {item["identity"]["inference_id"] for item in items}
    if len(items) < 2 or len(set(ids)) != len(ids):
        raise ContractError("identity collision fixture must prove colliding sources keep distinct stable IDs")
    if len(inference_ids) != 1 or len(set(source_hashes)) != len(source_hashes):
        raise ContractError("identity collision fixture must share inference_id while varying redacted source hashes")
    forbidden_path_markers = ("/Users/", "/Volumes/", "\\", "../")
    rendered = json.dumps(data)
    if any(marker in rendered for marker in forbidden_path_markers):
        raise ContractError("identity collision fixture must not expose filesystem paths")
    vectors = json.loads((FIXTURES / "identity-vectors.json").read_text(encoding="utf-8"))["identity_vectors"]
    expected_ids = {vector["expected_id"] for vector in vectors}
    if not set(ids).issubset(expected_ids):
        raise ContractError("identity collision fixture IDs must come from canonical identity vectors")


def check_transition_fixtures() -> None:
    allowed = {
        ("unloaded", "load", "loading"),
        ("loading", "provider_ready", "ready"),
        ("loading", "loader_error", "failed"),
        ("loading", "unload", "draining"),
        ("ready", "unload", "draining"),
        ("ready", "explicit_eviction", "draining"),
        ("draining", "requests_complete", "unloading"),
        ("unloading", "worker_exit_observed", "unloaded"),
        ("failed", "retry_after_worker_absent", "loading"),
        ("unloaded", "download", "unloaded"),
        ("unloaded", "cancel_download", "unloaded"),
        ("unloaded", "downloader_stopped", "unloaded"),
        ("ready", "patch_settings_partial", "ready"),
        ("ready", "download_progress_unknown_total", "ready"),
    }
    failures = []
    expected_keys = {
        "duplicate_load": {"single_operation"},
        "load_unload_race": {"worker_exit_required", "continuous"},
        "busy_eviction": {"surprise_eviction"},
        "stale_revision": {"refresh_required"},
        "failed_load": {"resource_owner_absent_precondition"},
        "download_cancel": {"final_state", "indeterminate_not_percent"},
        "deletion_refusal": {"requires_cache_and_not_busy"},
        "sse_gap": {"resnapshot"},
        "server_restart": {"do_not_repost_actions"},
        "unknown_null_partial_error": {"partial_errors", "progress"},
    }
    given_keys = {
        "duplicate_load": {"model_id", "revision"},
        "load_unload_race": {"model_id", "revision"},
        "busy_eviction": {"model_id", "eviction_target_id"},
        "stale_revision": {"model_id", "client_revision", "server_revision"},
        "failed_load": {"model_id"},
        "download_cancel": {"operation_id", "progress"},
        "deletion_refusal": {"model_id", "source"},
        "sse_gap": {"server_instance_id", "snapshot_sequence", "last_event_id"},
        "server_restart": {"old_server_instance_id", "new_server_instance_id"},
        "unknown_null_partial_error": {"settings_patch", "download_total"},
    }
    for path in sorted((FIXTURES / "scenarios").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("$schemaName"):
            continue
        scenario = data.get("scenario", "")
        if set(data.get("given", {})) != given_keys.get(scenario, set()):
            failures.append(f"{path}: scenario given keys are not the executable contract set for {scenario}")
        if set(data.get("expected", {})) != expected_keys.get(scenario, set()):
            failures.append(f"{path}: scenario expected keys are not the executable contract set for {scenario}")
        steps = data.get("steps", [])
        if data.get("expected", {}).get("continuous"):
            for prev, nxt in zip(steps, steps[1:]):
                if prev["to"] != nxt["from"]:
                    failures.append(f"{path}: non-contiguous transition {prev['action']} -> {nxt['action']}")
        for step in steps:
            edge = (step["from"], step["action"], step["to"])
            if step["allowed"] and edge not in allowed:
                failures.append(f"{path}: allowed edge is not in the contract table: {edge}")
            if not step["allowed"] and step.get("error") is None:
                failures.append(f"{path}: rejected edge must carry a structured error: {edge}")
        if scenario == "sse_gap" and not str(data["given"]["last_event_id"]).startswith("evt_"):
            failures.append(f"{path}: Last-Event-ID fixture must use the same opaque event id format as UiEvent.event_id")
        if scenario == "unknown_null_partial_error":
            progress = data["expected"]["progress"]
            if progress.get("total_bytes") is not None or progress.get("indeterminate") is not True:
                failures.append(f"{path}: unknown total progress must stay indeterminate with total_bytes null")
    if failures:
        raise ContractError("transition fixture validation failed:\n" + "\n".join(failures))


def self_test() -> None:
    contract = load_contract()
    bootstrap = json.loads((FIXTURES / "examples/bootstrap.model-free.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(bootstrap)
    bad["server"]["auth_required"] = 1
    errors = list(validator_for(contract, "BootstrapResponse").iter_errors(bad))
    if not errors:
        raise ContractError("negative self-test failed: boolean schema accepted integer 1")
    bad2 = copy.deepcopy(bootstrap)
    bad2["unexpected"] = True
    errors = list(validator_for(contract, "BootstrapResponse").iter_errors(bad2))
    if not errors:
        raise ContractError("negative self-test failed: strict object accepted an unknown property")
    bad_api_base = copy.deepcopy(bootstrap)
    bad_api_base["server"]["api_base"] = "//evil.invalid/ui"
    errors = list(validator_for(contract, "BootstrapResponse").iter_errors(bad_api_base))
    if not errors:
        raise ContractError("negative self-test failed: protocol-relative api_base was accepted")
    bad_timestamp = copy.deepcopy(json.loads((FIXTURES / "examples" / "event.1.json").read_text(encoding="utf-8")))
    bad_timestamp["emitted_at"] = "not a timestamp"
    errors = list(validator_for(contract, "UiEvent").iter_errors(bad_timestamp))
    if not errors:
        raise ContractError("negative self-test failed: malformed RFC3339 date-time was accepted")
    bad_catalog = copy.deepcopy(json.loads((FIXTURES / "examples" / "catalog.page.json").read_text(encoding="utf-8")))
    bad_catalog["items"] = bad_catalog["items"] * 201
    bad_catalog["pagination"]["next_cursor"] = "x" * 513
    errors = list(validator_for(contract, "CatalogListResponse").iter_errors(bad_catalog))
    if not errors:
        raise ContractError("negative self-test failed: over-limit catalog page/cursor was accepted")
    bad_download = {
        "$schemaName": "DownloadRequest",
        "repo_id": "../..",
        "revision": "../../weights\nwith-control",
        "idempotency_key": "download\nsplice",
    }
    errors = list(validator_for(contract, "DownloadRequest").iter_errors({k: v for k, v in bad_download.items() if k != "$schemaName"}))
    if not errors:
        raise ContractError("negative self-test failed: unsafe repo/revision/idempotency input was accepted")
    bad_action = copy.deepcopy(json.loads((FIXTURES / "examples" / "request.model-action.load.json").read_text(encoding="utf-8")))
    bad_action.pop("$schemaName", None)
    bad_action["load_profile"] = {
        "ctx_size": 10**18,
        "n_parallel": 1024,
        "kv_cache_mode": "q8_0",
        "sampling_preset": "balanced",
    }
    errors = list(validator_for(contract, "ModelActionRequest").iter_errors(bad_action))
    if not errors:
        raise ContractError("negative self-test failed: unsupported or unbounded load profile was accepted")
    leaked_error = copy.deepcopy(json.loads((FIXTURES / "examples" / "error.stale-revision.json").read_text(encoding="utf-8")))
    leaked_error["error"]["message"] = "failed under /Users/mlxcel-seeded-secret/.cache with token hf_MLXCELE2ESECRET"
    if not seeded_sensitive_markers(leaked_error):
        raise ContractError("negative self-test setup failed: seeded sensitive marker was not detected")
    harmless_auth_word = {"error": {"message": "Authorization required"}}
    if seeded_sensitive_markers(harmless_auth_word):
        raise ContractError("negative self-test failed: harmless Authorization wording was treated as a seeded secret")
    vectors = json.loads((FIXTURES / "identity-vectors.json").read_text(encoding="utf-8"))
    bad_vector = copy.deepcopy(vectors["identity_vectors"][0])
    bad_vector["expected_id"] = "mdl_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    expected_id, _ = canonical_identity_id(bad_vector)
    if bad_vector["expected_id"] == expected_id:
        raise ContractError("negative self-test setup failed: bad identity vector still matches")
    scenario = json.loads((FIXTURES / "scenarios" / "load-unload-race.json").read_text(encoding="utf-8"))
    scenario["steps"][1]["to"] = "failed"
    try:
        for prev, nxt in zip(scenario["steps"], scenario["steps"][1:]):
            if prev["to"] != nxt["from"]:
                raise ContractError("expected transition continuity failure")
    except ContractError:
        pass
    else:
        raise ContractError("negative self-test failed: mutated transition continuity was accepted")
    mutated = copy.deepcopy(contract)
    mutated["components"]["schemas"]["BootstrapResponse"]["properties"]["schema_version"]["unknownKeyword"] = True
    try:
        lint_schema_keywords(mutated["components"]["schemas"])
    except ContractError:
        pass
    else:
        raise ContractError("negative self-test failed: unknown schema keyword was not rejected")
    event = json.loads((FIXTURES / "examples" / "event.1.json").read_text(encoding="utf-8"))
    missing_event_id = copy.deepcopy(event)
    missing_event_id.pop("event_id", None)
    errors = list(validator_for(contract, "UiEvent").iter_errors(missing_event_id))
    if not errors:
        raise ContractError("negative self-test failed: event without an opaque event_id was accepted")
    operation = json.loads((FIXTURES / "examples" / "operation.running.json").read_text(encoding="utf-8"))
    permissive_target = copy.deepcopy(operation)
    permissive_target["target"]["filesystem_path"] = "/Volumes/private/model"
    errors = list(validator_for(contract, "Operation").iter_errors(permissive_target))
    if not errors:
        raise ContractError("negative self-test failed: operation target accepted a permissive metadata bag")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="rewrite generated TypeScript DTOs")
    parser.add_argument("--self-test", action="store_true", help="run validator negative tests")
    args = parser.parse_args()
    contract = load_contract()
    generated = generate_dto(contract)
    if args.fix:
        DTO.parent.mkdir(parents=True, exist_ok=True)
        DTO.write_text(generated, encoding="utf-8")
    elif not DTO.exists() or DTO.read_text(encoding="utf-8") != generated:
        print("docs/webui/generated/ui-api.d.ts is stale; run python3 scripts/ci/check_webui_contract.py --fix", file=sys.stderr)
        return 1
    validate_fixtures(contract)
    check_no_seeded_sensitive_markers()
    check_requirement_map()
    check_identity_vectors()
    check_identity_collision_fixture()
    check_transition_fixtures()
    if args.self_test:
        self_test()
    print(f"validated {len(iter_fixture_files())} WebUI contract fixtures, DTO drift, and schema strictness")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
