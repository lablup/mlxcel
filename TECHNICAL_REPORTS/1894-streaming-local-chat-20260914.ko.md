# 기술 보고서: PR #1894 — 제한된 로컬 스트리밍 채팅

**갱신:** 2026-09-15 · **상태:** 구현 및 범위 내 실제 검증 완료, 최종 에픽·수동 게이트 잔여 · **위험도:** 중간

## 통합 동작

Chat은 기존 인증 `/v1/chat/completions?autoload=false` 전송 경로와 npm 배포 `@lablup/ui-common@0.1.0-alpha.19` 어댑터를 사용합니다. 각 턴의 opaque/inference 식별자, revision, 검증된 수치 매개변수를 고정합니다. Settings의 canonical 메모리 전용 기본값에 명시적 다음 한 턴 override를 결합합니다. 빈 입력은 Settings를 상속하고, 양쪽 모두 없으면 서버 기본값을 사용하며, 잘못된 입력은 전송을 막고 승인된 요청만 override를 소비합니다. 모델 선택은 관찰과 다음 턴만 바꾸며 진행 중 추론을 중단하지 않습니다. Settings·Activity·Models의 세션 fence와 reader abort/EOF 경쟁 수정도 유지합니다.

IME 안전 입력기는 취소·중단·오류의 부분 출력을 보존하고 접힌 reasoning과 실행하지 않는 tool 정보를 분리합니다. 서버 usage와 클라이언트 추정 지표도 구분합니다. 크기가 제한된 escaped Markdown은 raw HTML·원격 이미지를 차단하며 하이라이터는 로컬에서 지연 로드합니다. 이미지는 provider-confirmed vision 기능이 있어야 허용되고, 전체 UTF-8/base64 요청은 실제 모드별 서버 한도와 별도 브라우저 16 MiB 한도를 검사합니다. 관리 API의 2 MiB 한도를 추론 한도로 오인하지 않습니다.

대화는 명시적으로 활성화하기 전까지 메모리에만 존재합니다. 제한된 버전형 IndexedDB의 import·quota·Clear All·늦은 callback을 보호하고, 인증정보는 저장하지 않으며 이미지 저장은 별도 동의가 필요합니다. 시스템 프롬프트는 공통 Settings가 아닌 개별 대화에 속합니다.

## 검증

측정 소스: `9385bcb7ef64386180ea62b3324540eef713a4e2`; 서버 바이너리 SHA256: `b32df976d255ca69d420368dd96130bb93748c5ade02d9f4596abbf580a26058`. 호스트: Apple M1 Ultra, macOS 27.0 build 26A428. Metal·Accelerate·기본 WebUI 기능으로 두 test-fast 실행 파일을 빌드했습니다.

- 통과: Vitest 266개/34파일, Node helper 17개, type/lint, 엄격한 계약 fixture 49개, 구성한 브라우저 응답 18종, 결정적 번들 검증, 로컬 브라우저 테스트 28개. 초기 JS gzip은 145,637바이트이며 raw chunk 경고를 숨기지 않았습니다.
- `988493fc`에서 WebUI 활성·비활성 workspace all-target Clippy가 통과했습니다. `9385bcb7`은 생성 manifest 메타데이터와 테스트·하네스 외에 Rust 소스 및 제공 자산 경로·SHA256이 모두 동일합니다. 실행 파일 전체가 동일하다는 주장은 아닙니다.
- 소스 CI [run 34922431893](https://github.com/lablup/mlxcel/actions/runs/34922431893)의 실제 실행 작업은 OpenXLA feature compile을 포함해 모두 통과했습니다. MLX pin extraction·OpenXLA feature link·CUDA sm_70 compile 세 작업은 조건부 skip이며 런타임 통과가 아닙니다.

인증된 격리 router에서 동시 모델 1개, context 8192, parallelism 1, autoload 비활성 상태로 세 체크포인트를 순차 실행했습니다. Llama·Granite는 각각 “Hello.”로 올바르게 답했습니다. Qwen3-VL은 실제 확인한 로컬 이미지의 빨간 사각형·파란 원·초록 삼각형을 정확히 설명했습니다. 모두 활성 요청 증가 → UI Stop → cancelled → 활성 요청 0을 재시도 없이 관측하고 명시적 unload의 `worker_exit_observed`를 확인했으며 서버는 0으로 종료했습니다. 실제 제공 CSP와 완료·취소 후 axe·레이아웃 검사에서 위반·외부 요청이 없었습니다. 선택한 Llama·Qwen 스크린샷도 잘림 없이 확인했지만 네이티브 접근성 통과를 의미하지 않습니다.

| 체크포인트 | 캐시 config revision | `model.safetensors` SHA256 |
|---|---|---|
| Meta-Llama-3.1-8B-Instruct-4bit | `241a666dad6cb93c8ff213d39a7f34a36bf26db4` | `08eff50fa4eb1fe499ead69eaeb4c5177ae82f218da0b55e0e31d7a989debb92` |
| Granite-4.0-H-Tiny-4bit | `02c8783da0cb171942e30a7dde8acce11fe82e0d` | `fc076be2631a2a6b6ca15cbe742d034f9a2217a73ab1238c351b0358085738ea` |
| Qwen3-VL-2B-Instruct-4bit | 미확인 | `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb` |

캐시 config revision은 기록된 출처 정보이며 체크포인트 전체 revision을 독립적으로 검증했다는 뜻은 아닙니다. Qwen의 값은 없어 추정하지 않았습니다. 유지관리자가 보관한 `chat-9385-actual/result.json`은 원래 캡처 상태를 유지하며 별도 `acceptance-review.json`이 이후 의미·시각 검토 PASS를 기록합니다.

## 계측 수정 및 남은 범위

`988493fc`의 첫 Llama 검증은 스크린샷 caret 복원이 빈 `style=""`을 남겨 실패했습니다. 설치된 실제 Playwright 함수를 jsdom에서 실행해 재현했으며 초기 axe 가설은 확인되지 않았습니다. helper는 trim-empty 속성만 예외로 두고 유효하지 않은 선언을 포함한 비어 있지 않은 CSS는 계속 거부합니다. 명시적 mode-0600 증거 파일로 axe 전 응답과 후속 레이아웃 검사 전 Stop 관측을 보존합니다. 제품 코드나 CSP의 엄격함은 변경하지 않았고 실패한 첫 시도를 전체 통과로 세지 않았습니다.

독립 구현·보안·성능·접근성 검토에서 범위 내 잔여 blocker는 없었습니다. 최종 전체 workspace Metal 검증은 #1848에 속하며 이 보고서는 그 통과를 주장하지 않습니다. GPU timeout 근본 원인 조사는 별도로 유보했습니다. Safari/VoiceOver·실제 네이티브 페이지 확대 200%는 통합 수동 검증까지 명시적으로 유보되어 있으며 통과가 아닙니다. 보고서 전용 후속 커밋은 측정된 구현을 변경하지 않습니다.
