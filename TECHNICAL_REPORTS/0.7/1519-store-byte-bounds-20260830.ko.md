# 기술 보고서: 저장소 바이트 상한

## 요약

PR #1519는 Responses API의 인메모리 응답 저장소와 대화 transcript 저장소가 엔트리 수와 TTL뿐 아니라 근사 retained byte 기준으로도 제한되도록 하여 이슈 #1248을 해결한다.

이 변경은 saturating JSON byte 계수, 엔트리별 크기 스냅샷, running total, 두 서버 바이너리의 byte budget 설정을 추가한다. 또한 기존의 map 전체 LRU victim 선형 스캔을 `BTreeSet` LRU 인덱스로 교체하여 live entry 하나가 bounded metadata node 하나만 보유하고 eviction마다 가장 오래된 엔트리를 O(log n)에 제거한다.

## 문제

기존 저장소는 보존 엔트리 수만 제한했다. 응답 엔트리는 전체 요청 입력과 응답 객체를 보관할 수 있고 대화 엔트리는 계속 커지는 transcript를 보관할 수 있으므로, 적은 수의 큰 multimodal 요청만으로도 엔트리 수 제한이 암시하는 것보다 훨씬 큰 메모리를 고정할 수 있었다.

응답 저장소는 eviction마다 전체 map을 선형 스캔하여 LRU victim을 골랐다. Byte pressure 상황에서는 한 번의 insert가 여러 엔트리를 evict할 수 있으므로, 이미 메모리 압력이 높은 상황에서 scan cost가 반복될 수 있었다.

## 구현

- `ResponsesStoreConfig::max_bytes`와 `ConversationStoreConfig::max_bytes`를 추가하고 기본값을 각각 256 MiB와 64 MiB로 설정했다.
- Writer를 통해 serialized JSON byte를 세고 serialization 실패 시 `usize::MAX`를 반환하는 `store_budget::serialized_json_len_saturating`을 추가하여 accounting이 fail closed되도록 했다.
- 두 저장소에 live entry별 `LruKey` 하나와 `BTreeSet` 인덱스를 추가하고, delete, refresh, replacement, TTL sweep, eviction에서 함께 제거하여 stale LRU metadata가 map과 별도로 증가하지 않도록 했다.
- `max_entries`와 `max_bytes`가 모두 만족될 때까지 eviction하도록 했으며, oversized single entry는 삽입 직후 자기 자신을 evict한다.
- `mlxcel serve`와 `mlxcel-server`에 `--responses-store-max-bytes` / `MLXCEL_RESPONSES_STORE_MAX_BYTES`, `--conversation-store-max-bytes` / `MLXCEL_CONVERSATION_STORE_MAX_BYTES`를 추가했다.
- `docs/environment-variables.md`와 `docs/responses-api.md`에 새 byte-budget knob를 문서화했다.

## 정합성

저장소는 삽입 또는 transcript replacement 시점에 크기를 계산하고 saturating add/subtract로 running total을 유지한다. Replacement는 기존 엔트리를 먼저 제거한 뒤 갱신된 엔트리를 삽입하므로 total과 LRU metadata가 같은 id를 중복 계산하지 않는다.

읽기는 접근 순서를 갱신하되 miss에서는 LRU sequence를 증가시키지 않는다. TTL sweep은 map entry와 index key를 함께 제거한다. Byte budget이 0이면 entry count가 0이 아닌 경우 route/store surface는 유지되지만 저장 엔트리는 즉시 자기 자신을 evict하고 조회는 일반 miss로 끝난다.

## 보안 및 리소스 경계

Byte limit은 stored request item 안의 base64 image data를 포함한 well-formed large input이 TTL window 동안 과도한 메모리를 고정하는 blast radius를 줄인다. Serialization 오류와 extreme value는 maximum-size entry로 취급되어 under-accounting 대신 eviction을 유도한다.

운영자가 극단적인 `max_entries` 값을 설정해도 초기 `HashMap` capacity는 제한되어 capacity knob 자체로 큰 upfront allocation이 발생하지 않는다. TTL sweep은 live entry에 대해 O(n)으로 남아 있지만, LRU victim selection은 byte pressure 상황에서 반복되는 O(n) scan이 아니다.

## 검증

- `cargo test --lib responses_store -- --nocapture` 통과: 16개 테스트, byte-only eviction, count와 byte 동시 pressure, oversized single entry, exact boundary, replacement, access-order refresh, TTL, zero/extreme budgets, running-total consistency 포함.
- `cargo test --lib conversation_store -- --nocapture` 통과: 13개 테스트, transcript에 대한 byte, count, oversized, exact-boundary, update, LRU, TTL, zero/extreme, total-consistency 사례 포함.
- `cargo test --lib store_byte_budgets_round_trip_through_into_startup_config -- --nocapture` 통과.
- `cargo test --bin mlxcel serve_store_byte_budget -- --nocapture` 통과.
- `cargo test --bin mlxcel-server store_byte_budget -- --nocapture` 통과.
- `cargo test --bin mlxcel-server settings_cli_mlxcel_server -- --nocapture` 통과, runtime-settings PR #1516을 보존하면서 chat-template-cache head인 PR #1518까지 포함한 main 위로 rebase한 뒤 확인.
- `cargo test --bin mlxcel settings_cli_mlxcel_serve -- --nocapture` 통과, runtime-settings PR #1516을 보존하면서 chat-template-cache head인 PR #1518까지 포함한 main 위로 rebase한 뒤 확인.
- `cargo test --bin mlxcel settings_cli_build_startup_input_defaults_off_and_propagates_enablement -- --nocapture` 통과, runtime-settings PR #1516을 보존하면서 chat-template-cache head인 PR #1518까지 포함한 main 위로 rebase한 뒤 확인.
- `cargo test --lib settings_cli -- --nocapture` 통과, runtime-settings PR #1516을 보존하면서 chat-template-cache head인 PR #1518까지 포함한 main 위로 rebase한 뒤 확인.
- `cargo test --test llama_compat_manifest manifest_option_claims_hold_on_both_server_binaries -- --nocapture` 통과.
- `python3 scripts/ci/check_llama_compat_manifest.py` 통과.
- `cargo fmt --all --check` 통과.
- `cargo clippy --lib --bin mlxcel --bin mlxcel-server -- -D warnings` 통과.
- `git diff --check` 통과.
- Static scan에서 touched store 파일에 conflict marker나 기존 `min_by`, `min_by_key`, `evict_to_capacity` victim-scan code가 남아 있지 않음을 확인했다.

## 생략한 검증

Wave-runner watchdog guard에 따라 broad cargo workspace tests, serial all-tests, workspace clippy, cold release build는 실행하지 않았다. 이 변경은 인메모리 server-store accounting과 CLI/config wiring에 한정되므로 real checkpoint, ffmpeg, hardware qualification은 실행하지 않았다.

## 위험 사항

Byte accounting은 의도적으로 approximate이다. 문자열 JSON escaping과 response/output structure의 serialized JSON은 계수하지만, Rust allocation overhead와 container internal overhead는 budget에 포함하지 않는다.

대화 update는 기존 transcript accounting을 제거한 뒤 append된 새 transcript로 교체한다. 갱신된 transcript가 byte limit을 초과하면 자기 자신을 evict하며, 이는 over-budget retained state에 대한 의도된 fail-closed 동작이다.
