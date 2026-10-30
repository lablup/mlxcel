# 기술 보고서: PR #2672 — 모델 소유 캐시의 배치 처리

**작성일**: 2026-10-27
**상태**: 측정 체크포인트 완료, Step3p5/Step3p7은 합성 검증만 수행
**위험도**: 중간

## 요약

Ministral3, ExaOne4, Mellum, Laguna, Step3p5의 요청별 캐시를 분리하고 디코드의 투영·정규화·FFN/expert·출력 헤드를 배치화했다. Step3p7은 텍스트와 임베딩 입력을 시퀀스 소유 텍스트 타워에 위임한다. 측정한 동시성 4의 총 처리량은 1.56~1.92배 증가했다.

## 문제 정의

기존 공유 `RefCell` 캐시는 새 요청마다 초기화되어 서버가 배치를 한 시퀀스로 제한했다. 올바른 동시 실행에는 길이가 다른 전체·슬라이딩 attention, 동일한 단일 요청 동작, 잘못된 입력의 원자적 거부, 페이지 메모리 계정도 필요하다.

## 기술적 선택과 그 이유

`ModelOwnedSequenceState`와 행별 attention을 사용하고 나머지 transformer 계산을 공유 배치로 실행한다. B=1 그래프와 지연 초기화를 유지한다. Ministral의 전체 캐시 window 힌트, ExaOne의 local/global 동작, Laguna의 sink와 gate, Step3p5의 부분 RoPE와 head gate를 보존한다. 패딩 prefill은 비활성 상태를 유지하며 Laguna DFlash는 독립적으로 유지한다.

리뷰에서 입력 차원·행 수 검증과 B>=2 fallback의 중복·누락 ID 원자적 거부를 추가했다. 각 래퍼는 전체 캐시 live length와 슬라이딩 캐시 visible length를 페이지 저장소에 동기화하고 Step3p7도 해당 훅을 위임한다. 이 동기화가 없으면 동시 KV 사용량을 과소 계산할 수 있다.

## 검증 및 측정

혼합 캐시 격리, 6단계 교차·네이티브 패리티, 정확한 B=1, 빈 입력·잘못된 호출, 새 요청·초기화, 페이지 저장소, Step3p7 임베딩·디코드 위임, Laguna DFlash, coverage, 포맷과 범위를 제한한 clippy가 통과했다. #2670 통합 후 최종 배치 경로 검사 12개도 통과했다.

release `metal,accelerate`, 512 GB Apple M3 Ultra에서 기준 `a29a2692`와 구현 `4cb59177`을 측정했다. 이후 모델 코드 변경 없이 rebase했다. 네 체크포인트 모두 greedy 출력과 B=1 추적 256개 위치가 기준과 일치했고, 기록된 B=1 로짓 차이는 0이었다.

| 체크포인트 | 기준 총 1/2/4 tok/s | 네이티브 총 1/2/4 tok/s | B=4 top-1 불일치 | 확신도 높은 위치 불일치 |
|---|---|---|---:|---:|
| exaone-4.0-1.2b-4bit | 226.2 / 226.0 / 226.1 | 219.0 / 332.3 / 434.0 | 2/256 | 0/100 |
| Ministral-3-3B-Instruct-2512-4bit 텍스트 뷰 | 121.4 / 119.0 / 115.5 | 120.9 / 168.5 / 203.3 | 3/256 | 0/114 |
| Laguna-XS.2-4bit | 93.8 / 94.4 / 94.1 | 89.8 / 119.2 / 170.6 | 21/256 | 0/123 |
| Mellum2-12B-A2.5B-Base | 23.0 / 51.0 / 52.3 | 33.2 / 69.3 / 81.8 | 0/256 | 0/184 |

확신도 기준은 상위 두 토큰 로짓 차이 >= 2이다. 워밍업 후 B=1을 5회 반복한 디코드 중앙값은 ExaOne 272.3→271.0 tok/s, Laguna 109.1→109.9였으며 관측 범위가 겹쳤다. 최초 ladder의 변동을 단일 요청 속도의 정확한 보장으로 해석하면 안 된다. 모든 측정 서버는 배치 제한 없이 네이티브 배치 경로를 기록했다.

제공된 Ministral VLM 체크포인트는 기본적으로 다른 텍스트 구현을 선택한다. 측정용 텍스트 뷰는 원본 파일을 링크하고 `vision_config`를 제거하며 `architectures`를 `Mistral3ForCausalLM`으로 바꾼다. 외부 모델 타입과 중첩 텍스트 설정은 유지한다. Mellum의 기준 `docs/architecture.md` 추적은 비유한 NLL로 실패했고 BF16 유지 진단에서는 통과했다. 실제 기본 설정 비교에는 양쪽 모두 통과하는 고정 Rust 소스 `git show a29a2692:src/models/mellum.rs`를 사용한다. 문서 입력의 기존 정밀도 문제는 이번 변경에서 해결하지 않았다. Step3p5/Step3p7의 실제 로컬 체크포인트는 없었다.

`--parallel 4` 서버에서 `scripts/bench_serving_concurrency.py --concurrency 1,2,4 --prompt-tokens 512 --max-tokens 128`로 재현한다. 로짓은 `MLXCEL_TRACE_BATCH_ROWS=1`과 `=4`, 인자 `MODEL CORPUS 64 4 8 512`로 추적한다. 요약 근거는 `docs/benchmark_results/data/native-batched-decode-m3ultra-2026-10-27/mixed-cache-families.json`에 있다.

## 변경 요약

다섯 모델 래퍼, Step3p7 위임, 공통 검증·harness, coverage 목록과 연속 배치 문서를 수정했다. 새 의존성이나 체크포인트 변환은 필요하지 않다. 관련: #2569, #2574, #2555, #2670.
