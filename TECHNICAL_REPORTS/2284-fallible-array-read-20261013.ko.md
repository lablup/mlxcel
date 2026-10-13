# PR #2284: 프로세스를 중단하지 않고 MLX 배열 readback 오류 전파

**날짜**: 2026-10-13
**상태**: 구현 및 독립 리뷰 완료, 집중 테스트 통과
**위험도**: 중간 (FFI 오류 전파와 speculative scheduler cleanup 변경)

## 요약

PR #2284는 C++/Rust cxx 경계의 `array_evaluated_bytes`를 fallible하게 변경합니다. 이제 실패한 MLX event는 C++ 예외로 프로세스를 중단시키지 않고 engine과 scheduler 호출자에게 일반 오류로 전달됩니다. Lookahead decoding은 이미 제출된 다음 evaluation을 drain하고 speculative row position을 복구한 뒤, 영향을 받은 cohort에서 아직 완료되지 않은 row만 중단합니다. 관련 없는 활성 요청은 계속 처리하고 실패한 row를 replay하지 않습니다.

## 문제

기존 bridge는 `Vec<u8>`를 직접 반환했습니다. MLX 배열을 synchronize할 때 event 오류가 발생하면 C++ 예외가 infallible cxx 경계를 넘어 프로세스를 종료했습니다. Scheduler에도 lookahead token 수집 중 발견한 실패를 일관되게 처리하는 cleanup 경로가 필요했습니다. Speculative state에는 현재 lookahead append 하나와 이미 제출한 next prime의 append 하나가 추가로 들어 있을 수 있습니다.

## 설계

- cxx 선언은 이제 `Result<Vec<u8>>`를 반환합니다. C++ 구현의 native 반환형은 유지해 생성된 wrapper가 C++ 예외를 Rust 오류로 변환하게 합니다.
- Engine token readback은 fused-row와 prompt-lookup 호출자를 포함해 `Result<Vec<i32>, String>`을 반환합니다.
- Lookahead 실패 cleanup은 제출된 `next.tokens`가 있으면 상태를 trim하기 전에 raw fallible 경로로 평가하고, 주 실패를 한 번만 기록하며, speculative position을 한 단계 또는 두 단계 되돌립니다. 그 뒤 lookahead를 제거하고 영향을 받은 요청 중 아직 완료되지 않은 요청만 중단합니다.
- C++ fixture는 실제 오류가 signal된 MLX `Event`를 평가 완료 배열에 연결합니다. MLX가 raw pointer를 저장하므로 오류 객체는 의도적으로 프로세스 수명 동안 유지합니다. Direct FFI와 engine 테스트는 이 event를 정확히 한 번 소비합니다.
- Scheduler 검증은 수집 경계에서 thread-local one-shot readback fault를 사용합니다. 실패 event fixture를 lookahead 입력으로 그대로 사용하면 다음 decoding step을 만드는 더 이른 시점에 one-shot event 오류를 소비해 scheduler readback cleanup을 검증하지 못합니다. 이 seam은 제출된 next step을 유효하게 유지하면서 검증 대상 경계에서 오류를 주입합니다.

## 검증

소스 커밋은 `9622b8c6`입니다. 게시 후보는 base `a1c58265`를 병합한 local head `d9a8c4ea`이며 tree는 `67d54aee7acc63c90a5395fda7673c6bc72a680f`입니다. 이 base를 기준으로 리뷰한 16개 경로의 binary diff SHA-256은 그대로 `60e33dd052fdb4405f281377c5875b80700573fdf0146c8bec95d9edd38489e7`입니다.

- `cargo test --locked -p mlxcel-core -p mlxcel --lib --profile test-fast --features cuda -j4 --no-run`: 통과. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/build-no-run-restored-final.log`.
- 실제 failed-event FFI 테스트: 1개 통과. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/ffi-failed-event-test-restored-final.log`.
- Engine failed-event 전파 테스트: 1개 통과. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/engine-failed-event-test-restored-final.log`.
- 제출된 next prime이 있는 경우와 없는 경우의 scheduler cleanup: 1개 통과. Health failure가 한 번만 기록되는지, model-owned offset이 복원되는지, pending token이 방출되지 않는지, 서버가 종료되지 않는지, 관련 없는 survivor가 완료되는지도 함께 확인했습니다. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/scheduler-read-failure-test-restored-final.log`.
- `cargo fmt --all -- --check`: 통과. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/fmt-check.log`.
- 최신 repository contract gate: 현재 `main`을 병합한 뒤 통과했습니다. Version, kernel dtype-key, kernel port-dispatch, llama 호환성, ROCm overlay, Python tooling 테스트 123개, WebUI fixture 49개를 포함합니다. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/contracts-1791-a1c-refreshed.log`.
- Negative control: 기존 infallible bridge를 격리해 컴파일하고 direct FFI 테스트를 child process에서 실행했습니다. Child가 정확히 `-SIGABRT`인 `-6`으로 종료되어, 이 변경이 없으면 회귀가 프로세스를 중단시킨다는 점을 확인했습니다. 로그: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/negative-old-signature-status.log`, `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/negative-old-signature-child.log`, `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/negative-old-signature-build.log`.

16개 소유 경로를 건드리지 않은 후속 `main` 갱신들을 병합하기 전 source commit `9622b8c6`에서 만든 집중 테스트 binary는 SHA-256 `79ae35f3b958baab057da86d3abf2cc85058706a30840b2647cdb440e5dc1413`인 `mlxcel_core-be0ee8003b018a01`과 SHA-256 `e9f19800ae9efc50ffaf2e2f96986a4123b8f601809ae621a76903c19288690a`인 `mlxcel-d5e5498c5ad868df`입니다.

## 리뷰와 제한 사항

독립 구현 리뷰, 보안·성능 리뷰, static finalization 리뷰에서 이 범위에 남은 문제를 발견하지 못했습니다. 첫 번째 이슈 contract gate는 최종 scheduler test-only fault seam을 추가하기 전에 통과했고, 위의 최신 gate는 그 seam과 현재 `main` 병합을 포함한 상태에서도 통과했습니다. CI와 전체 통합 검사는 central merge 전 orchestration gate가 수행합니다. 광범위한 workspace 테스트와 Clippy도 해당 gate에서 수행하도록 의도적으로 남겼습니다.
