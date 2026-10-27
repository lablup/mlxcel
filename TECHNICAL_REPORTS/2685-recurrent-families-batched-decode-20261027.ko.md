# 기술 보고서: PR #2685 — recurrent 계열 배치 디코드

**작성일**: 2026-10-27
**상태**: 측정 및 대상 검증 완료; 최종 통합 검사 대기
**위험도**: 중간

## 요약

Mamba1, Jamba, LFM2, RecurrentGemma, RWKV7에 독립적인 시퀀스 소유 recurrent 상태를 이용하는 네이티브 디코드를 추가한다. 실제 체크포인트 네 개에서 동시성 4의 총 처리량은 기준 대비 2.22–3.27배였다. LFM2-VL의 텍스트 요청 두 개와 이미지 요청 한 개를 동시 실행한 결과는 모두 단독 실행과 정확히 일치했다.

## 문제 정의

기존 recurrent 상태는 이미 시퀀스별로 저장됐지만 서버는 이 계열의 디코드 행 수를 1로 제한했다. 올바른 배치 처리는 recurrence 상태 결합·분리, 서로 다른 attention history 유지, scalar 동작 보존, 상태 변경 전 잘못된 ID 거부를 요구한다.

## 기술적 선택과 리뷰

#2571의 공유 `recurrent_batch` helper를 재사용한다. 임베딩, projection, recurrence, feed-forward/expert 연산과 출력 head는 행을 묶어 실행하고 attention은 시퀀스별 KV offset을 유지한다. 초기화된 상태와 새 상태가 섞이면 ID를 전달하는 행별 fallback을 사용한다. 분리 상태를 복사해 한 요청이 다른 요청의 배치 버퍼를 붙잡지 않게 한다. Paged shadow 계산은 attention 저장소만 반영하고 recurrent 슬롯은 0으로 유지하며 B=1은 기존 scalar 경로를 보존한다.

구현 및 보안·성능 리뷰는 MEDIUM 문제 한 개를 수정한 뒤 통과했다. LFM2-VL의 non-ID `forward_batched`는 ID 없는 다중 행 호출을 거부하는 텍스트 tower에 위임해야 한다. 그렇지 않으면 trait fallback이 공유 recurrent 상태를 재사용할 수 있다. `a9395b7c`에서 위임을 복원했고 `898108fe`는 거부된 호출 후 identified 출력까지 비교하도록 회귀 테스트를 강화해 거부의 원자성을 검증한다. 이 변경은 측정한 ID 기반 서빙 경로를 바꾸지 않는다.

## 측정

환경은 Apple M3 Ultra, 512 GB, macOS 26.6.2와 release `metal,accelerate`다. 기준 모델 소스는 `a29a2692`이며 trace-only scalar-reference 수정을 적용했다. 후보 바이너리는 모두 `2bd2eff00fbf9dc0f8b8a1c0a5c1c310ff31ffd9`에서 측정했다. 이후 변경은 위의 non-ID wrapper 수정과 테스트 전용 회귀 검사다. 네 greedy 출력은 모두 동일하며 B=1 trace 256개 위치의 모든 데이터 행도 기준과 정확히 일치한다.

| 체크포인트 | 기준 총 처리량 C1/C2/C4 tok/s | 후보 총 처리량 C1/C2/C4 tok/s | B4 top-1 불일치 | Decided 불일치 | PPL 변화 |
|---|---|---|---:|---:|---:|
| AI21 Jamba Reasoning 3B 4-bit | 128.6 / 167.0 / 167.4 | 144.5 / 290.3 / 475.9 | 1/256 | 0/136 | +0.061% |
| Falcon Mamba 7B 4-bit Instruct | 38.8 / 37.5 / 78.8 | 38.8 / 45.8 / 195.5 | 2/256 | 0/105 | -0.027% |
| LFM2 350M 8-bit | 399.6 / 405.9 / 434.1 | 430.0 / 712.1 / 1417.4 | 4/256 | 0/94 | -0.457% |
| LFM2 8B A1B 4-bit | 88.3 / 109.9 / 202.7 | 111.9 / 137.5 / 450.3 | 0/256 | 0/117 | 0.000% |

서빙은 `--parallel 4`, 명목상 약 512토큰의 요청 prompt와 생성 상한 128토큰을 사용한다. 실제 tokenizer 토큰 수는 달라진다. Trace는 64개 위치의 chunk 네 개와 최대 history 512를 사용하며 고정된 512토큰 context가 아니다. Decided는 기준 상위 두 logit 차이가 2 이상인 위치다. Decided 불일치가 0이어도 전체 logit이나 자유 생성 텍스트가 동일하다는 의미는 아니다. 네 후보 서버 모두 clamp 없이 네이티브 배치를 기록했다. 단일 ladder 측정만으로 통계적 처리량 동등성을 주장하지 않는다.

LFM2-VL 450M 4-bit는 원래 자유 생성 workload에서 기준과 후보 모두 3/3이 일치했다. 기준은 한 행으로 clamp되고 후보는 clamp 없는 네이티브 디코드를 보고했다. 재현용 runner는 정확한 요청 설정, reasoning을 포함한 전체 메시지, 복사한 이미지와 SHA-256, 서버 명령, 시각, 소스·바이너리·라이브러리 해시와 성공·오류 상태를 보존한다. 기록된 driver 해시와 일치하는 과거 script 버전도 보관한다. 현재 runner는 포트 bind 대신 실제 listener에 연결을 시도해, 닫힌 연결의 TIME_WAIT를 사용 중인 서버로 잘못 판단하지 않는다.

## 검증, 한계와 근거

Recurrent 상태 격리, 혼합·새 상태, scalar/native 전환, 잘못된 입력 거부의 원자성, 캐시 계산과 wrapper 위임 대상 테스트 및 대상 clippy·포맷이 통과했다. VLM runner의 CPU 테스트 8개는 입력 복원, reasoning을 포함한 동일성, 오류, 덮어쓰기 방지, 정리와 포트 검사 회귀를 다룬다. RecurrentGemma와 RWKV7은 synthetic 검증만 수행했으며 실제 체크포인트 성능이나 출력에 대한 주장은 하지 않는다. 최종 통합 workspace 검사는 #2572 merge 이후에 진행할 예정이며, 이 보고서는 PR merge나 전체 suite 통과를 주장하지 않는다.

기계 판독 결과는 `benchmarks/epic-2574/recurrent-families.json`에 있다. 기준·후보 텍스트 원본과 LFM2-VL 기록은 원래 runner manifest가 있으면 `run-metadata.json`으로 보존하며, 오래된 텍스트 기준 실행은 runtime 해시 부재를 명시한다. 모든 보관 파일은 SHA-256으로 색인한다. VLM workload는 `scripts/bench_batched_decode_vlm.py`로 재현하며 바이너리를 옮길 때 실제 로드한 Metal 라이브러리도 유지해야 한다.

## 변경 요약

Recurrent 모델 계열 다섯 개, LFM2-VL 위임, 배치 coverage와 문서에 네이티브 디코드 지원을 추가한다. 새로운 kernel이나 의존성 없이 공유 recurrent helper를 재사용한다. 관련 이슈는 #2573, #2574, #2571, #2555다.
