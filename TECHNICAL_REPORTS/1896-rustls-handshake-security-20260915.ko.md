# 기술 보고서: PR #1896 — rustls 핸드셰이크 경계 수정

**날짜:** 2026-09-15
**이슈:** #1895
**범위:** 의존성 보안 패치. Rust 소스와 추론 연산은 변경하지 않는다.

## 문제와 결정

[RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285)와 [GHSA-2mjx-qc3c-rqvc](https://github.com/rustls/rustls/security/advisories/GHSA-2mjx-qc3c-rqvc)는 rustls 0.23.13–0.23.44에서 TLS 1.3 핸드셰이크 메시지가 암호화 수준 경계를 넘어 수락되는 문제를 명시한다. 핸드셰이크 기록의 인증은 유지되므로 이를 인증 우회로 주장하지 않는다. 새 권고가 공개되면서 rustls 0.23.43을 사용하는 WebUI 에픽의 cargo-deny 검사가 실패했다.

`cargo update -p rustls --precise 0.23.45`로 최초의 호환 패치 버전을 선택한다. 파싱한 잠금 파일을 비교하여 rustls 단일 항목의 버전과 체크섬만 바뀌었음을 확인했다. 매니페스트, TLS 기능, 명시적 ring 공급자와 권고 정책은 그대로 유지한다. 기존 묶음 의존성 PR #1880에는 이 rustls 패치가 없으며, 이번 변경으로 그 PR의 범위를 확장하지 않는다.

## 검증

- 최신 데이터베이스를 사용한 `cargo deny --locked check`: advisories, bans, licenses, sources 모두 통과했다. 권고 데이터베이스 `e2e640471715167f73e22eaf761f2e547adafeec`에 해당 권고가 포함되어 있다. 기존 중복 의존성 및 사용되지 않은 허용 항목의 경고는 억제하지 않았다.
- `cargo tree --locked -i rustls`: 서버 TLS와 HTTP 클라이언트가 단일 rustls 0.23.45를 공유한다. 잠금 파일 구조 비교와 `git diff --check`도 통과했다.
- `cargo test --locked --profile test-fast --lib server::tls::tls_tests -- --test-threads=1`: 설정·인증서 검사 10개가 통과했다.
- `cargo test --locked --profile test-fast --lib server::listen::listen_tests::a_tls_listener_completes_a_handshake_and_answers -- --exact --test-threads=1`: 실제 루프백 TLS 핸드셰이크·HTTP 응답 검사 1개가 통과했다. 두 실행 모두 CPU 테스트만 선택했으며 무시하거나 실패한 테스트는 없다. 빌드 시 기존 `BITLINEAR_HIP_SOURCE` 미사용 C++ 경고가 출력된다.
- 소스 커밋의 호스팅 CI [34912229649](https://github.com/lablup/mlxcel/actions/runs/34912229649): cargo-deny, WebUI bundle/contract, OpenXLA feature compile이 통과했다. 중앙 머지 전에 최종 브랜치 상태를 별도로 확인하며, 조건부 생략 작업을 런타임 증거로 계산하지 않는다.
- 독립적인 코드·보안·성능 검토에서 지적 사항이 없었다. 검토는 읽기 전용이었으며 실행 증거와 구분한다.

## 한계

기존 TLS 테스트는 애플리케이션 호환성을 확인하며, 해당 권고의 잘못된 레코드를 새롭게 재현한 테스트는 아니다. 보안 수정 자체는 확인된 업스트림 릴리스가 제공한다. 권고 예외, 광범위한 의존성 변경, GPU 테스트, 실제 모델 추론 또는 성능 벤치마크를 수행했다고 주장하지 않는다. GPU 타임아웃 원인 조사와 최종 통합 Safari 검증은 별도의 에픽 작업으로 남는다.
