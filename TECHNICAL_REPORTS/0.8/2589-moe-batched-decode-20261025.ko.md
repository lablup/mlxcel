# 기술 보고서: PR #2589 — Mixtral 방식 MoE 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2558, #2566, #2587

## 요약

7개 routed-expert 모델 계열이 임베딩, projection, 정규화, expert 및 출력 head를 여러 디코드 행에 대해 함께 실행한다. DeepSeek-OCR과 OCR-2도 텍스트 디코드를 같은 구현에 위임한다. 하드웨어 처리량 인수 검증은 #2587에 남아 있다.

## 문제 정의

기존 모델들은 trait의 행별 fallback을 사용해 동시 요청마다 전체 forward와 가중치 읽기를 반복했다. native 경로를 추가하면서 계열별 수학적 차이와 기존 단일 행 fused-expert 경로를 유지해야 한다.

## 변경 요약

#2555의 공통 split-attention 및 라우팅 헬퍼를 재사용한다. Mixtral, PhiMoE, Phixtral, Qwen2-MoE, OLMoE, DBRX, DeepSeek v1은 dense 연산과 기존 expert 모듈을 배치 hidden tensor에 실행하고, attention은 각 요청의 캐시를 갱신한다. B=1은 기존 단일 요청 forward를 호출하며 빈 배치는 정의된 shape를 유지한다.

PhiMoE는 요청별 LongRoPE 표를 선택하고, Phixtral은 f32 attention과 병렬 residual 순서를 유지한다. DBRX는 fused QKV를 분리하기 전에 clipping하며, OLMoE는 Q/K 정규화 배치를 보존한다. Qwen2-MoE와 DeepSeek는 shared expert를 유지하고 DeepSeek는 초기 dense 층에서 후속 MoE 층으로의 분기를 보존한다. 두 OCR 래퍼는 텍스트 디코드와 경로 표시를 내부 모델에 위임한다.

## 검증

CPU 집중 테스트 15개가 통과했다. 공용 MoE 테스트 6개, 실제 OCR 래퍼 테스트 2개, DBRX 계약 1개, Phixtral 계약 1개와 커버리지 5개다. 서로 다른 길이의 3개 요청, B=1 정확한 일치, B=0, 서로 다른 LongRoPE 구간, 실제 clipping, shared expert, dense→MoE 구성, 동일·분리 expert 라우팅을 검증한다. 컴파일, 포맷, 소스·라이선스 검사와 경고를 오류로 처리한 workspace 전체 타깃 Clippy가 통과했다. 구현 및 보안·성능 리뷰에서도 차단 이슈가 없었다.

사용자 지시에 따라 Metal, 실제 체크포인트의 greedy·로그잇 비교 및 서버 성능 검증은 #2587로 이관했다. native 표시는 구현된 실행 경로를 설명하며 실측 성능 향상을 주장하지 않는다.

## 기술적 선택과 학습 포인트

라우팅 로직을 복제하지 않고 기존 다중 토큰 expert 구현을 재사용한다. 테스트에는 실제 경계 조건과 의미 있는 expert 가중치가 필요하다. clipping이 발생하지 않거나 LongRoPE 표가 전환되지 않는 fixture로는 해당 분기를 검증할 수 없다. 실제 래퍼 호출 테스트는 소스 문구 일치가 아니라 호출 경계에서 위임을 확인한다.
