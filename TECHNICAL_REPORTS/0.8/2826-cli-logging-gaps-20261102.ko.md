# PR #2826: 오프라인 CLI 경고 복구와 startup-notice 가드 확장

**Date**: 2026-11-02
**Status**: Completed (PR open; closes #2725)

## 요약

PR #2826은 #2724가 남긴 진단 유실 구멍 두 개를 닫는다. 이제 오프라인 CLI 명령은 `RUST_LOG`가 없으면 `warn`으로 필터링한 stderr 구독자를 설치하므로, 환경 폴백과 폐기 플래그 경고가 별도 설정 없이 사용자에게 도달하고 stdout은 바이트 단위로 그대로 유지된다. startup-notice 가드 테스트는 고정 파일 목록과 이름만 비교하는 방식에서 벗어나, `src/` 아래 컴파일되는 모든 `.rs`를 훑고 모듈 경로와 감싸는 `impl` 타입을 기록한 뒤 호출 형태별로 해석한다. 변경은 3개 파일, +608/-56이며 그중 약 570줄이 가드와 픽스처다.

## 1. 문제 정의

`init_cli_tracing`은 `RUST_LOG`가 설정된 경우에만 구독자를 설치했다. `generate`, `inspect`, `tune` 등 오프라인 명령이 지나는 모든 `tracing::warn!`과 `tracing::error!`가 기본값으로 버려졌고, 사용자가 먼저 옵트인해야만 볼 수 있었다.

#2724가 추가한 가드는 고정 파일 목록만 따라가고 호출 대상을 이름만으로 비교했다. 스캔 목록 밖 파일의 pre-subscriber 헬퍼가 자기 `tracing::` 호출을 숨길 수 있었고, 다른 모듈의 동명 메서드는 거짓 양성을 만들 수 있었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/main.rs` | `init_cli_tracing`이 serve가 아닌 모든 명령에 구독자를 설치하고, 순수 함수 `cli_tracing_filter(Option<&OsStr>) -> EnvFilter`를 분리 |
| `src/main_tests.rs` | `RUST_LOG` 미설정, `debug`, `error` 세 경우의 필터 선택 테스트 |
| `src/server/startup_notice_tests.rs` | `src/` 전체 스캔, 모듈과 `impl` 타입을 담은 `FnDef`, `use` 파싱, 호출 형태별 해석, 픽스처 두 개 |

## 3. 기술적 선택

### 3.1 기본 필터를 `error`가 아니라 `warn`으로

`EnvFilter::from_default_env()`는 `RUST_LOG`가 없을 때 `error`로 떨어진다. 이 이슈의 목적이 유실되던 WARN 줄 자체이므로 기본값을 명시적으로 `warn`으로 둔다. 설정된 `RUST_LOG`는 그대로 전달해 기존 지시어 문법을 보존하고, `autotune: applied ...`는 `info`로 남겨 옵트인 상태를 유지한다.

### 3.2 `serve` 경로는 계속 건너뜀

`server::startup`이 자체 구독자를 설치하므로 두 번째 전역 `init()`은 패닉이다. 가드는 `matches!(command, Commands::Serve(_))` 그대로 두고, `try_init`은 앞으로 생길 이중 초기화를 프로세스 중단 대신 무동작으로 낮춘다.

### 3.3 호출 해석은 crate 전체 이름 매칭보다 의도적으로 좁게

`.f(`를 crate 전체 메서드 이름으로 맞추면 실제 트리에서 거짓 양성 470건이 나왔다. 예를 들어 `into_startup_config`의 `.resolve()` 호출이 무관한 `SpeculativeDispatch::resolve`와 매칭됐다. 소스 텍스트만으로는 수신자 타입을 알 수 없으므로, `.f(`는 호출자와 같은 모듈의 `impl` 메서드로만 해석한다. `*_tests.rs` 파일은 `#[cfg(test)]`에서만 컴파일되고 `env_fallback_*` 같은 가짜 루트를 만들므로 제외한다.

맨 이름 `f(`는 같은 파일의 자유 함수나 `use`로 들여온 자유 함수로, `module::f(`는 해당 모듈의 자유 함수로, `Type::f(`는 그 타입의 메서드로 해석한다. `is_root`의 루트 목록은 바꾸지 않았다.

## 4. 검증

| 검사 | 결과 |
|---|---|
| `cargo test --lib startup_notice` | 8 passed, 실제 트리 가드 포함 |
| `cargo test --bin mlxcel cli_tracing` | 1 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 통과 |
| 수정 파일 `rustfmt --edition 2024 --check` | 통과 |

## 5. 학습 포인트

가드의 가치는 양방향 해석 정확도에 달려 있다. 스캔만 넓히고 매칭을 좁히지 않았다면 거짓 음성 하나를 수백 개의 거짓 양성과 맞바꿨을 것이고, 실제 트리에서 실패하는 가드는 결국 꺼진다. 선택한 규칙은 요구된 픽스처 두 개와 실제 트리를 모두 통과시킨다.

## 6. 미검증

기본 `warn` 구독자는 순수 필터 헬퍼로 검증했고, 오프라인 명령을 실제로 실행해 stderr를 캡처하지는 않았다. 실 실행 검증은 포매팅 계층이 stdout에 쓰지 않는 것까지 확인해 줄 것이다.

## 7. 위험

낮음. 진단 출력만 바꾼다. 구독자가 stderr에 쓰므로 stdout은 구조적으로 그대로이고, `serve`는 기존 경로를 유지한다.
