# 기술 보고서: PR #2682 — 비전 파라미터 양자화 로딩

**작성일**: 2026-10-27
**상태**: 구현 완료; 검증 결과는 아래 참조
**위험도**: 중간

## 요약

비전/VLM 임베딩 테이블과 학습 파라미터를 로드할 때 MLX 양자화 sidecar를 확인하도록 수정했다. #2676의 A/B/C 그룹 전체가 실제 로딩 경로에 연결됐으며, 잘못된 입력은 MLX FFI 호출 전에 오류로 반환한다.

## 문제 정의

일부 체크포인트는 위치 테이블, class/query 토큰, 구분자, latent, 텍스트 확장 임베딩까지 양자화한다. 기존 로더는 packed U32 배열을 dense 값처럼 복사하고 압축된 차원에서 논리 폭을 추론해 잘못된 임베딩이나 복구 불가능한 C++ reshape/broadcast 오류를 일으켰다. 양자화된 비전 텐서를 모두 거부하면 유효한 외부 체크포인트도 로드하지 못한다.

## 기술적 선택과 이유

행 조회에는 `UnifiedEmbedding`을 유지하고, 전체 배열을 사용하는 경로는 `load_dense_param`으로 생성 시 한 번만 역양자화한다. 기대 논리 차원과 packed/scales 형상에서 group size, bit width, affine/MXFP4/MXFP8/NVFP4 모드를 결정한다. rank, 선행 차원, dense 실수 dtype, packed/sidecar dtype, bias 일관성, 정수 오버플로, 백엔드 지원 모드, 최종 형상을 검증한다. InternVL과 GOT-OCR을 포함해 호출부가 가진 양자화 설정을 전달하고, 키 변환 및 선택적 가중치 추출에서도 sidecar를 보존한다.

Dense 배열은 값 변경 없이 복사한다. 리샘플링할 학습 격자는 로드 시 한 번 dense로 변환한다. SAM3 block-float 프롬프트 임베딩은 UINT8 scales 저장 dtype 대신 실제 BF16 활성값 dtype을 사용한다. 실제 Moondream2 체크포인트는 `[1,729,1152]`, 기존 fixture는 2차원 테이블을 사용하므로 두 형식 모두 허용하고 선행 singleton 차원이 있는 3차원으로 정규화한다. 합성 테스트만으로는 놓쳤던 호환성 조건을 실제 모델 실행으로 확인했다.

## 검증 및 측정

Apple M5 Max 128 GiB에서 release `metal,accelerate`를 사용했다. 기준 커밋은 `ffea91111977751e409a8cb03ae1945ef372b327`, 최종 통합 코드는 `46b344b6`(main `b4cdf023`), 입력은 `tests/fixtures/test_image.png`, 프롬프트는 `Describe the image.`이다. 고정 continuation corpus와 상위 8개 logit trace를 `benchmarks/issue-2676/`에 보존했다. 이미지 프롬프트는 원래 prefill 폭으로 실행하고, 뒤따르는 teacher-forced 토큰 63개를 같은 문맥 뒤에서 한 번에 처리했다.

| 체크포인트 | 이미지 프롬프트 토큰 | 측정 위치 | Top-1 불일치 | 확신 위치 불일치, gap >= 2 | 기준 선택 토큰 logit 차이 |
|---|---:|---:|---:|---:|---:|
| HunyuanOCR 4비트 | 282 | 64 | 0/64 | 0/16 | 0 |
| Gemma4 12B 4비트 | 275 | 64 | 0/64 | 0/28 | 0 |

기록된 trace 데이터는 완전히 일치한다. Perplexity도 각각 15.7281818418과 14.6555180299로 동일하다. 최대 64토큰의 greedy 생성에서 HunyuanOCR은 64토큰, Gemma4는 9토큰 후 종료했으며 변경 전후 디코딩된 출력이 같다. 요청 목록에 대응하는 로컬 모델 6개와 GLM-4.1V, InternVL3, LFM2-VL, Kimi-VL, Moondream2, Qwen2.5, Gemma3 등 총 13개 체크포인트가 로딩·생성에 성공했다. Gemma4 assistant는 대상 모델과 함께 텍스트 MTP drafter로 검증했다. 기존 CLI는 assistant 단독 실행 및 오프라인 이미지+MTP를 지원하지 않는다.

새 테스트는 Group A의 각 로더, dense/양자화 1D·2D·3D 파라미터, 전체 배열 덧셈, slice/reshape, 토큰 연결, 지원 양자화 모드와 잘못된 레이아웃을 검증한다. 과거 packed-as-dense 동작을 잠시 재현했을 때 Group A 여섯 경우와 공유 사용 패턴을 포함한 테스트 10개가 안전하게 실패했고, 이후 정상 구현을 복원했다. 전체 워크스페이스/all-target clippy와 포맷 검사가 통과했다. 최종 통합 코드의 직렬 게이트는 통과 14,193개, 무시 455개, 별도 기존 실패 2개(서버 읽기 전용 필드 fixture coverage, Step3p7 텍스트 배치)를 기록했다. 집계는 subprocess harness를 포함한 Cargo 결과 요약의 합이다.

## 리뷰 결과 및 후속 조치

초깃값에 의존한 양자화 모드 추론, InternVL/GOT 설정 전달 누락, sidecar dtype 검사, SAM3 활성값 dtype, dense 정수 입력 허용, Phi4 차원 오버플로를 수정했다. 처음의 병렬 테스트 실행에서는 MLX의 프로세스 전역 기본 장치를 바꾸는 테스트들이 서로 간섭했다. 수치 허용치를 변경하지 않고 단일 스레드로 재실행해 통과했으며, `--no-fail-fast -- --test-threads=1`을 포함한 `make verify-test`를 사용해야 한다.

첫 직렬 게이트에서 기존 서버 스키마 coverage 실패(선언 114개, 기대값 113개)가 드러났다. 동시에 진행되던 PR #2681이 필드 수 불일치를 수정해 최종 통합 검증 전에 이 브랜치에 반영했다. 다만 해당 변경의 기존 읽기 전용 필드 fixture도 갱신되어야 하며, 이것이 남은 서버 테스트 실패다. 변경하지 않은 Step3p7 텍스트 배치 비교도 1e-5 기준에서 실패했으며, 분리된 기준 커밋 `ffea9111` 빌드에서도 동일한 실패를 재현했다(통과 0, 실패 1). 수치 허용치는 변경하지 않았다.

## 변경 요약

학습 파라미터 검증을 공통화하고, 열거된 모든 비전/VLM 소비 경로와 sidecar 키 변환, 회귀 테스트, 모델 포팅 지침 및 재현 가능한 출력 비교 자료를 추가했다. 의존성이나 체크포인트 변환 단계는 추가하지 않았다. 관련: #2676, #2459, #2659.
