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
import argparse
import sys

from webui_contract_checks import (
    ContractError,
    DTO,
    check_identity_collision_fixture,
    check_identity_vectors,
    check_no_seeded_sensitive_markers,
    check_requirement_map,
    check_transition_fixtures,
    generate_dto,
    iter_fixture_files,
    load_contract,
    validate_fixtures,
)
from webui_contract_self_tests import self_test


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
