# PR #2324: WebUI 릴리스 문서 추가 및 --help에 --webui 노출

**날짜**: 2026-10-16
**상태**: 머지 전 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료, 리버스 프록시와 통합 매트릭스 일부는 미검증 ("미검증 및 후속" 참고)
**위험도**: 낮음 (help 출력, 문서, NOTICE, 테스트 강화뿐이며 런타임 서빙 경로는 바뀌지 않음)

## 요약

에픽 #1910의 마지막 문서 하위 이슈인 #1900은 내장 WebUI를 찾을 수 있게 하고 문서화한다.

1. `--webui`와 `--no-webui`가 `mlxcel-server --help`와 `mlxcel serve --help`의 "WebUI Options" 제목 아래에 표시된다. `--ui`와 `--no-ui`는 숨겨진 alias로 남는다. 나머지 b10621 UI, tools, MCP, proxy, agent 인자는 계속 숨겨져 있고 기동 시 진단과 함께 실패한다.
2. 새 문서: `docs/webui/README.md`, `user-guide.md`와 `.ko.md`, `contributing.md`와 `.ko.md`. `README.md`, `docs/README.md`, `docs/server-features.md`, `docs/llama-server-compat.md`에서 링크한다.
3. `NOTICE`에 `@lablup/ui-common` 섹션을 추가한다. 컴파일된 코드가 번들에 들어 있는데 라이선스 문구는 `third-party-licenses.txt`에만 있었기 때문이다.

## 문제 정의

WebUI는 `--help`에 나오지 않는 플래그 뒤에 있었다. 사용자는 b10621 표기인 `--ui`를 알아야 했고, llama-compat manifest도 `--ui`를 hidden으로 기록했다. 문서는 흩어진 메모뿐이었고 그중 하나는 mlxcel이 UI를 제공하지 않는다고 적고 있었다. 또 manifest의 `hidden` 주장은 이번에 중요한 방향으로는 검증되지 않았다. "visible"이라고 적힌 항목이 이후 clap 변경으로 조용히 다시 숨겨질 수 있었다.

## 기술적 선택과 그 이유

- `--webui`를 기본 long 이름으로, `--ui`를 alias로 둔다(반대가 아니라). clap은 기본 이름을 보여주며, `--webui`가 mlxcel 자체 제품 표기다.
- 이 쌍만 노출한다. b10621 그룹의 나머지는 기동 시 실패하는 호환 표면이므로, `--help`에 올리면 동작할 수 없는 인자를 쓰도록 유도하게 된다.
- 제목은 `next_help_heading`이 아니라 인자별(`WEBUI_HELP_HEADING`)로 지정한다. `UiCompatArgs` 뒤에 flatten되는 플래그 그룹으로 번지지 않게 하기 위해서다.
- `tests/llama_compat_manifest.rs`가 `hidden` 주장을 양방향으로 검사한다. 기존 테스트는 hidden이라 주장했는데 보이는 항목만 잡았다. 새 분기는 visible이라 주장했는데 숨겨진 항목을 잡으며, 이 PR이 겪을 수 있는 회귀가 바로 그것이다.
- 단위 테스트에 더해 바이너리 수준 테스트(`tests/webui_cli_help.rs`)로 help 문구를 고정한다. 섹션, 숨겨진 나머지, 두 진입점을 빌드된 바이너리로 확인한다.
- 문서의 오류 메시지는 바꿔 쓰지 않고 바이너리에서 그대로 캡처한다. 문제 해결 절을 실제 출력과 대조할 수 있다.
- 통합 매트릭스의 "Bundle budgets and deterministic assets" 행은 CI `WebUI bundle` 잡 결과와 digest로 기록해, 가이드가 주장이 아닌 증거를 링크하게 한다.

## 변경 요약

- `src/cli/ui_compat_args.rs`: `--webui`/`--no-webui`를 `WEBUI_HELP_HEADING` 아래에 표시, `--ui`/`--no-ui`는 숨겨진 alias, 문서 주석 갱신.
- `src/main.rs`, `src/bin/mlx_server.rs`: 각 `after_help`에 짧은 "WebUI" 안내 추가.
- `compat/llama-server/b10621/ui-tools-mcp-gcp.json`: `--ui`를 `hidden: false`로 기록, `--ui-config` 노트에서 "UI 미제공" 문구 제거.
- 테스트: `src/cli/ui_compat_args_tests.rs`, `tests/webui_cli_help.rs`(신규), `tests/llama_compat_manifest.rs`.
- 문서: `docs/webui/`(신규 5개 파일과 README), `README.md`, `docs/README.md`, `docs/server-features.md`, `docs/llama-server-compat.md`, `docs/webui-integration-matrix.md`, `NOTICE` 수정.

## 동작 변경

help 출력에 섹션과 `after_help` 문단이 추가된다. 플래그의 추가나 제거는 없고, `--ui`, `--no-ui`와 나머지 b10621 인자의 파싱도 그대로다. 숨겨진 그룹을 `--help`에서 grep하는 스크립트는 여전히 아무것도 찾지 못한다.

## 검증

실제 체크포인트 `~/models/mlx/qwen3-0.6b-4bit`, `mlxcel-server --webui`와 `mlxcel serve --webui` 모두, `test-fast` 프로파일과 `--features cuda`로 빌드.

- 두 `--help` 모두 `--webui`와 `--no-webui`를 표시하고, `--ui-config`, `--tools`, `--mcp-servers`, `--ui-mcp-proxy`, `--agent`를 grep하면 아무것도 나오지 않는다.
- 카탈로그 192개 항목, 시작 시 로드된 모델 없음. 로드되지 않은 모델로의 chat은 400. `POST /ui-api/v1/model-actions`로 로드하면 202, 2초 만에 ready. 스트리밍 chat은 `stop`과 `[DONE]`으로 종료. unload는 202.
- 문서의 모든 오류 메시지를 바이너리에서 그대로 캡처했다. UI 미컴파일 메시지는 `--no-default-features --features cuda,surgery` 빌드에서 재현했다.
- 되돌림 검사: 쌍을 다시 숨기면 새 검사 세 개가 모두 실패한다(`help_lists_only_the_webui_pair_under_its_heading`, `webui_pair_is_listed_and_the_rest_of_the_group_stays_hidden`, `manifest_option_claims_hold_on_both_server_binaries`).
- `tests/cli_help_consistency.rs`(27), `make verify-llama-compat`, `-D warnings` clippy, fmt, 계약 검사가 통과한다.
- 리뷰에서 HIGH 문서 오류 하나를 찾았다. 삭제 확인은 `DELETE`가 아니라 모델의 `mdl_` ID를 요구한다. 작은 오기재와 함께 두 언어 모두 수정했다.

## 미검증 및 후속

- 리버스 프록시 절은 실제 nginx가 아니라 `scripts/webui/verify_reverse_proxy.py`로 확인했다.
- 통합 매트릭스의 일부 행은 아직 not-run이며 목록은 #1834에 있다.
- 워크스페이스 전체 테스트 스위트는 CI에 맡겼다.
- 범위 밖 관찰: 비대화형 셸에서 `&`로 띄운 단일 모델 서버는 SIGINT를 무시해 SIGTERM이 필요했고, 라우터 모드는 SIGINT로 종료됐다. 별도 이슈가 필요하다.
