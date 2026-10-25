# 기술 보고서: PR #2592 — Native TP 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2565, #2566, #2546, #2587

## 요약

Llama, Qwen 3, Qwen 3.5, Gemma 3, ERNIE 4.5, Hunyuan v1 Dense의 tensor-parallel runtime이 rank·layer별로 하나의 native 배치 디코드를 실행한다. 정확한 단일 행 라우팅, 기존 rank reduction 순서와 정밀도, 요청별로 분리된 캐시와 recurrent state를 보존한다. Metal 체크포인트 parity와 처리량 검증은 #2587로 명시적으로 이관했다.

## 문제 정의

기존 TP runtime은 batching을 지원한다고 표시하면서도 각 요청 행에 단일 행 forward를 반복했다. 따라서 서버의 매 step에서 rank별 weight를 요청 수만큼 다시 읽고 collective도 반복했다. Qwen 3.5에는 더 어려운 경계가 있다. Full-attention KV cache와 gated-delta convolution/recurrent state가 생성, 신규·기존 행의 혼합 배치, 해제, 재사용 과정 내내 외부 sequence cache와 정확히 연결되어야 한다.

## 변경 요약

변환한 runtime은 요청별 캐시를 rank·layer별 열로 전치한다. 각 rank에서 norm, attention, MLP, 출력 projection이 전체 요청 행을 한 번에 처리한다. Rank 출력은 원래 단일 행 forward와 같은 rank 순서 및 기존 일반/f32 reduction 정책으로 결합해 산술 계약을 보존한다. 공통 배치 router는 B=1을 원래 forward로 보내고 B=0에는 정의된 빈 결과를 반환한다.

Qwen 3.5는 gated-delta projection을 배치로 실행하되 convolution과 recurrent 갱신은 요청별 state에 분리해 적용한다. Full-attention layer는 batched split attention을 사용한다. State 조회는 외부 cache identity를 계속 키로 사용하며 서버 sequence-ID lifecycle이 행 제거 시 state를 해제한다. Single-device Qwen 3.5는 per-row fallback 분류를 유지한다. TP Gemma 4도 선행 이슈 #2546이 아직 열려 있어 제외했다.

`MLXCEL_TRACE_TP_SIZE`는 `examples/logit_trace`가 production tensor-parallel loader를 사용하게 한다. 같은 TP size에 `MLXCEL_TRACE_BATCH_ROWS=1`과 `=4`를 적용하면 향후 하드웨어 인수 검증에서 직접 비교 가능한 teacher-forced trace를 만들 수 있다. 환경변수를 설정하지 않으면 기존 single-device loader를 그대로 사용한다.

## 검증

합성 runtime 테스트 12개는 여섯 계열 각각의 TP size 1과 2를 검증한다. 모든 경우에 서로 다른 길이의 prompt history 3개, 정확한 B=1 라우팅, B=0, 서버 sequence-ID 진입점을 포함한다. 별도 Qwen 3.5 테스트는 신규·기존 행 혼합과 해제된 state 재사용을 검증한다. Warm debug CPU 빌드에서 Qwen3Next 테스트 2개, GatedDelta cache 테스트 3개, logit-trace 예제 테스트 7개, 전체 coverage ledger 테스트 6개가 통과했다.

포맷과 저장소 버전, 환경변수 문서, Llama 호환성 검사를 포함한 source gate가 통과했다. Warnings-denied 전체 workspace·all-target Clippy는 1분 2초에 통과했다. 최종 architecture, 수학적 정확성, security, performance 리뷰에는 지적 사항이 없었다.

초기 cold `test-fast` 시도는 호스트가 MLX 의존성 endpoint의 DNS를 해석하지 못해 테스트 실행 전에 중단됐다. 이를 테스트 근거로 계산하지 않았다. Metal 체크포인트 trace, greedy text 비교, serving ladder는 실행하지 않았으며 #2587에 남아 있다. Native 분류는 실측 속도 향상을 주장하지 않는다.

## 기술적 선택과 학습 포인트

Tensor parallelism은 weight의 배치를 바꾸지만 sequence 소유권 계약은 바꾸지 않는다. Rank 내부의 weight-heavy graph를 배치화하고 요청별 cache·recurrent 변경은 격리하며 collective 순서와 정밀도를 보존해야 한다. Batch width만 바꾼 single-device 모델은 TP 코드와 collective를 실행하지 않으므로 production loader를 사용하는 trace selector가 필요하다.
