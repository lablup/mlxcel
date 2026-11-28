# PR #2819: Rust, 웹, pnpm 의존성 최신화

**작성일**: 2026-11-02 (머지 전 작성)
**상태**: 리뷰 중
**언어**: Rust, TypeScript, Python
**위험도**: 중간 (TLS 루트, 템플릿 의미, HTTP 라우팅이 새 메이저 버전과 함께 바뀜)

## 요약

v0.8.0 전에 선언된 모든 의존성을 최신으로 올리고 pnpm을 12.16.1로 올렸습니다. 예외는 hf-hub(0.5)와 TypeScript(6.0.3)입니다. `make verify`를 통과했고, 실제 체크포인트 12개의 출력, 템플릿, 토큰이 이전 빌드와 같습니다.

## 1. 문제 정의

보안 이슈가 계속 보고되고 선언된 요구 버전 다수가 메이저 버전 하나 이상 뒤처져 있어, 메인테이너가 v0.8.0 태그 전에 전체 갱신을 요청했습니다.

## 3. 기술적 선택과 그 이유

- **rustls 제공자는 ring을 유지합니다.** reqwest 0.13의 `rustls` 기능은 저장소가 피하는 C 빌드인 aws-lc-rs를 선택합니다. `rustls-no-provider`와 `src/http_tls.rs`로 ring을 유지합니다. 클라이언트를 만드는 모든 프로세스와 테스트 바이너리는 이 헬퍼를 거쳐야 하며, 그렇지 않으면 클라이언트 생성이 패닉합니다.
- **hf-hub는 0.5에 둡니다.** 1.0은 hf-xet을 통해 aws-lc-rs를 강제하고 동기 API를 없앴습니다. 그래서 hf-hub 때문에 reqwest 0.12와 native-tls가 의존성 그래프에 남습니다.
- **TypeScript는 6.0.3에 둡니다.** typescript-eslint 8.72는 6.1 미만을 요구하고, TypeScript 7은 `tests/models-fixtures.ts`가 쓰는 컴파일러 API를 제거했습니다.
- **실패한 라우트 청크는 실패 상태로 유지합니다.** 이전 재시도는 렌더마다 새 프로미스를 돌려줘 `use()`가 영원히 대기했습니다. React 19.3.1에서 이 문제가 안정적으로 드러났습니다. 브라우저는 실패한 동적 import를 문서 수명 동안 유지하므로, 새로고침 없는 재시도는 원래 효과가 없었습니다.

## 5. 학습 포인트

- pnpm 12 lockfile은 자체 `snapshots:` 키를 가진 환경 문서로 시작하므로, 텍스트 파서는 마지막 키를 읽어야 합니다.
- reqwest 0.13은 OS 신뢰 저장소로 검증합니다. 사내 CA가 있는 호스트는 이제 성공할 수 있고, 저장소가 손상된 호스트는 이제 실패할 수 있습니다. 내장 루트일 때와 반대입니다.
- minijinja 3은 Python 의미에 가까워졌습니다(`//` 내림, half-even `round`, Python식 리스트 표기). 체크포인트 12개의 출력이 같아도 모든 채팅 템플릿을 보장하지는 않습니다.

## 7. 변경 요약

| 영역 | 요약 |
|---|---|
| Rust | axum 0.8, tower 0.5, tower-http 0.7, reqwest 0.13, sha2 0.11, symphonia 0.6, minijinja 3, tokenizers 0.23, safetensors 0.8, serde-saphyr 2, toml 1 등 |
| 웹 | pnpm 12.16.1, React 19.3.1, Vite 8.3.5, axe 4.14, lucide-react 1.66, 사이트의 Next 16.4 |
| 수정 | 라우트 청크 실패, WCAG 2.5.3 이름 2건, 사이트 브랜드 아이콘 |

## 8. 후속 조치

- hf-hub 1.x가 aws-lc-rs 없이 동작하면, TypeScript는 typescript-eslint가 7을 받아들이면 올립니다.
- `webpage/site`의 `minimatch`에 개발 전용 중간 등급 권고 1건이 남아 있으며, eslint 쪽 수정을 기다립니다.
