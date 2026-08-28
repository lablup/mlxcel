# 기술 보고서: PR #1488 - b10621 speculative 플래그 매핑

**작성일**: 2026-08-28

**상태**: 완료

**언어**: Rust

**위험도**: Medium

## 요약

이슈 #1433이 소유한 b10621 speculative 매니페스트 엔트리 52건(옵션 45건, 네이티브 `/completion` 점 표기 필드 7건)을 전부 분류했다. `--spec-draft-model`은 정식 env와 함께 `supported`, `--spec-type`, `--spec-draft-n-max`, 그리고 제거된 `--draft` 철자군은 번역 테스트와 함께 `aliased`, 나머지는 #1445 GGML 패턴을 따르는 새 inert-or-reject 분류 모듈로 `not_applicable`이다. 디코드 산술 변경은 없다.

## 1. 문제 정의

b10621은 드래프트 모델 선택, 드래프트 샘플러 임계값, GGML 드래프트 측 프로세스 배치, n-gram 추측, lookup 디코딩까지 40여 개의 speculative 플래그를 노출한다. mlxcel의 추측 디코딩은 서버 전역 설정의 MTP / DFlash이고, 이 PR 전에는 대부분의 표면이 파싱되지 않거나 조용히 오파싱됐다(`-md`가 `-m` 값 `d`로 삼켜졌다).

## 2. 기술적 선택과 그 이유

### 2.1 셀렉터는 번역, 번역 불가는 거부

리뷰가 핵심 오독을 교정했다. `--spec-type`은 b10621의 추측 서브시스템 셀렉터이고, 명시적 `none`은 드래프트 모델이 있어도 추측을 끈다(셀렉터가 업스트림의 타입 추론을 멈춘다). `resolved_spec_type`은 mlxcel이 실행할 수 있는 것만 정확히 번역한다. `none`은 경고와 함께 드래프트 모델을 내려놓고, `draft-mtp` / `draft-dflash`는 `--draft-kind`로 매핑되며(명시적 kind와 충돌하면 결정적으로 실패), n-gram 모드·draft-simple·draft-eagle3·draft-dspark·다중 서브시스템 목록은 값별 진단과 함께 시작을 중단한다.

### 2.2 전체 철자 패리티, 값 분류

정식 철자 외 22개의 롱 별칭과 17개의 숏 철자가 두 바이너리 모두에서 파싱된다. 숨김 `SpecCompatArgs` 표면이 값을 분류한다: 업스트림 기본값, `--spec-draft-ngl`의 전체 오프로드 철자(역사적 음수 포함), `f16` 드래프트 캐시 타입, n-gram 셀렉터가 선택될 수 없는 동안의 모든 n-gram 튜닝 값은 inert로 수용하고, mlxcel이 재현할 수 없는 값은 모델 로드 전에 한계와 대안을 명시하며 거부한다. 페어의 운영자 활성 절반인 `--no-spec-draft-backend-sampling`은 거부되고, 환경 변수는 b10621의 bool 페어 규칙을 따른다.

### 2.3 네이티브 점 표기 필드: 업스트림의 수용 후 무시를 그대로

b10621은 `speculative.n_max`와 그 여섯 형제를 컴파일 아웃된 스키마 블록 뒤의 평면 점 표기 최상위 키로 등록하므로, 업스트림은 수용하고 무시한다. mlxcel은 serde rename으로 같은 점 표기 키를 선언하고 동일하게 inert로 처리한다. 거부하면 b10621이 응답하는 요청을 거절하게 된다. 필드 유무와 무관하게 해석된 옵션이 동일함을 테스트로 증명했다.

### 2.4 정식 env와 방어된 레거시 폴백

드래프트 토큰 상한은 정식 `LLAMA_ARG_SPEC_DRAFT_N_MAX`에 바인딩되고, 제거된 `LLAMA_ARG_DRAFT_MAX`는 순수 함수로 분리되어 우선순위 테이블 테스트를 갖춘 폴백으로 유지된다(모든 CLI 철자와 정식 변수가 이긴다. `--draft-n` 누락이 리뷰의 회귀 지적이었다). `--draft` 엔트리는 실제 차이를 기록한다: b10621은 제거된 변수가 export되어 있으면 시작을 중단하고, mlxcel은 기존 배포를 살린다.

## 3. 변경 요약

| 항목 | 값 |
|-----|---|
| 변경 파일 | 커밋당 15개, 두 커밋 |
| 라인 | 합산 +1995 / -426 |
| 매니페스트 | supported 1, aliased 3, not_applicable 48, deferred 잔여 0 |

검증: 분류 테스트 11개, 두 바이너리의 파싱 별칭 테스트, 숏 플래그 테이블 테스트, 재빌드 바이너리 대상 compat 통합 4/4, 매니페스트 체커 green, `-md` 재작성·n-gram 거부 진단·draft-kind 충돌의 라이브 확인, 그리고 실체크포인트 greedy 추측 패리티(Qwen3-4B 타깃, Qwen3-0.6B 드래프터, 64 결정 위치 토큰 동일, 수락률 0.37).

## 4. 후속 조치

- `/props`가 해석된 `speculative` 블록(드래프트 모델 basename, kind, n_max)을 보고한다.
- 분산·OpenXLA 서빙 경로는 서버 전역 speculative 설정을 그대로 물려받는다. 요청 단위 speculative 경로는 어디에도 없으며, 네이티브 점 표기 필드의 inert 처리가 그것을 문서화한다.
