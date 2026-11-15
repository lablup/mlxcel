# 기술 보고서: PR #3079 - side-model join 테스트가 worker 시작 후에 실패하도록 수정

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 낮음

## 요약

main에서 `server_exit_joins_worker::embedding_worker_is_joined_when_a_later_side_model_fails_to_load`가 실패하고 있었다. 이 테스트의 `--image-model`은 존재하지 않는 경로였는데, 지금은 모델 resolution이 side-model worker를 하나도 시작하기 전에 이 경로를 거부한다. 그래서 테스트가 확인하려던 worker join 종료 경로가 실행되지 않았다. 이제 테스트는 `--image-model`에 잘못된 `config.json`이 든 디렉터리를 준다. 이 디렉터리는 resolution을 통과하고, embedding worker가 시작된 뒤 로드 단계에서 실패한다.

## 1. 문제 정의

#2827에서 추가한 이 테스트는 앞선 worker가 시작된 뒤 side-model 로드가 실패해도 프로세스가 종료되기 전에 그 worker를 join하는지 확인한다("side model workers exited before process exit"). image-model 인자의 resolution(`src/downloader/resolver.rs`)은 이제 모델이 아닌 경로에서 즉시 실패하며, 이는 의도된 동작이다. 그래서 서버는 resolver 오류로 종료했고, 테스트는 join 줄이 없다고 보고했다(#3076). root 통합 테스트를 돌리는 PR job이 없어 CI가 잡지 못했다.

## 3. 기술적 선택과 그 이유

서버 동작은 올바르므로 테스트 입력만 바꿨다. `config.json` 내용이 `{ not valid json`인 임시 디렉터리는 resolution을 통과한 뒤 loader에서 실패해, 이 테스트가 원래 겨냥한 늦은 실패를 재현한다. 확인 항목은 그대로다. join 로그 줄, 종료 코드 1, C++ abort 문구 없음, "failed to load --image-model" 사유를 본다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 1개 (`tests/server_exit_joins_worker.rs`) |
| 테스트 | `MLXCEL_REQUIRE_MODELS=1 cargo test --test server_exit_joins_worker`: 실제 checkpoint로 5개 통과 |

관련: #2827 (테스트), #3076 (이슈).
