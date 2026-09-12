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
"""Negative regression cases for the shared WebUI contract verifier."""
import copy
import json
from typing import Any

from webui_contract_checks import (
    ContractError,
    FIXTURES,
    canonical_identity_id,
    lint_schema_keywords,
    load_contract,
    require_active_format_checkers,
    seeded_sensitive_markers,
    validator_for,
)


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

    def rejects(
        schema_name: str, original: dict[str, Any], path: list[str], invalid_values: list[Any]
    ) -> None:
        validator = validator_for(contract, schema_name)
        baseline = {k: v for k, v in original.items() if k != "$schemaName"}
        if list(validator.iter_errors(baseline)):
            raise ContractError(f"negative self-test setup failed: invalid {schema_name} baseline")
        for invalid in invalid_values:
            candidate = copy.deepcopy(baseline)
            target = candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = invalid
            if not list(validator.iter_errors(candidate)):
                raise ContractError(f"negative self-test failed: {schema_name}.{'.'.join(path)} accepted {invalid!r}")

    def example(name: str) -> dict[str, Any]:
        return json.loads((FIXTURES / "examples" / name).read_text(encoding="utf-8"))

    rejects("UiEvent", example("event.1.json"), ["emitted_at"], [
        "not a timestamp", "2026-09-31T12:00:00Z", "2026-09-12T12:00:00",
        "2026-09-12T12:00:00+00:60", "2026-09-12T12:00:00+24:00",
    ])
    catalog = example("catalog.page.json")
    rejects("CatalogListResponse", catalog, ["items"], [catalog["items"] * 201])
    rejects("CatalogListResponse", catalog, ["pagination", "next_cursor"], ["x" * 513])
    rejects("BootstrapResponse", bootstrap, ["server", "api_base"], ["\n", "/ui\n", "//evil.invalid"])
    download = example("request.download.json")
    rejects("DownloadRequest", download, ["repo_id"], ["../..", "owner/..", "https://host/model", "o/n\n"])
    rejects("DownloadRequest", download, ["revision"], ["../../weights", "main\n", "x" * 129])
    rejects("DownloadRequest", download, ["idempotency_key"], ["key\nsplice", "validkey\n", "x" * 129])
    action = example("request.model-action.load.json")
    rejects("ModelActionRequest", action, ["load_profile", "ctx_size"], [0, 262145])
    rejects("ModelActionRequest", action, ["load_profile", "n_parallel"], [0, 33])
    rejects("ModelActionRequest", action, ["load_profile", "kv_cache_mode"], ["q8_0"])
    rejects("ModelActionRequest", action, ["load_profile", "sampling_preset"], ["balanced"])
    missing_format = copy.deepcopy(contract)
    missing_format["components"]["schemas"]["BootstrapResponse"]["properties"]["schema_version"]["format"] = "unsupported-webui-format"
    try:
        require_active_format_checkers(missing_format)
    except ContractError:
        pass
    else:
        raise ContractError("negative self-test failed: unavailable format checker was accepted")
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
