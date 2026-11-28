# 기술 보고서: PR #2590 — Dense 계열 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2556, #2566, #2587

## 요약

20개 dense decoder 계열이 임베딩, projection, 정규화, feed-forward 및 출력 head 연산을 디코드 행 사이에서 함께 실행한다. Phi3-VL, Phi4-SigLIP-VL, GraniteVision은 텍스트 디코드를 변환된 내부 모델에 위임한다. Metal 성능 인수 검증은 #2587에서 별도로 추적한다.

## 문제 정의

기존 trait 기본 경로는 동시 요청마다 모델의 전체 forward를 반복했다. 배치 shape만 맞추는 것으로는 충분하지 않다. residual 순서, 정규화 위치, rotary 위치 처리, attention window와 logit scaling이 기존 단일 요청 그래프와 일치해야 한다.

## 변경 요약

각 계열은 공통 split-attention 및 B=0/1/다중 행 라우팅 헬퍼를 사용하고 B=1에서는 기존 forward를 유지한다. 대상은 Apertus, Cohere, Cohere2, ERNIE 4.5, ExaOne, Gemma, Gemma2, GLM4, Granite, Hunyuan v1 dense, InternLM2, InternLM3, MiMo, OLMo2, OLMo3, Phi3, Seed-OSS, SmolLM3, StarCoder2, TeleChat3이다. 커버리지 목록과 연속 배치 문서도 실제 native·위임 경로를 반영한다.

## 기술적 선택

Phi3는 short/long LongRoPE 표에 따라 행을 묶어 배치 rotary 연산을 실행하고 원래 행 순서를 복원하며 partial rotary scaling을 보존한다. Cohere는 병렬 attention/feed-forward residual을 유지하고 Cohere2는 global NoPE attention과 window 층을 구분한다. Gemma2의 attention·출력 softcap, Granite 배율, InternLM의 요청 길이별 rotary 매개변수도 보존한다.

리뷰에서 Hunyuan의 학습된 Q/K 정규화는 RoPE 뒤에 와야 한다는 점을 발견해 수정했다. 테스트는 값이 서로 다른 norm gain으로 순서 회귀를 감지한다. OLMo3는 실제 기존 코드의 의미를 유지한다. mask 없는 단일 토큰 디코드는 전체 live cache를 사용하며, 명시적으로 더 짧은 다중 토큰 mask가 주어지면 window를 잘라 사용한다. 이는 이슈의 단순화된 설명과 다르며 기존 의미 변경은 이번 이식 범위가 아니다.

## 검증

20계열 모두 실제 작은 모델로 서로 다른 길이의 3개 요청, B=1 정확한 일치, B=0을 검사한다. 실제 vision 래퍼 객체 3개도 vision encoder를 실행하지 않고 동일한 위임 계약을 검증한다. mixed LongRoPE 표, OLMo3의 두 mask 경로와 소스 등록 검사도 포함한다. 최종 리뷰 전 집중 테스트 28개와 컴파일·포맷 검사가 통과했다. 추가 부정 경로 테스트 2개는 누락된 캐시에 모델·행·층 정보를, 잘못된 attention metadata에 모델·층·원인을 포함하는지 확인한다. 이 진단 변경 후 집중 테스트 전체를 재실행했다. 초기 구현에서 전체 커버리지 게이트 6개가 통과했고, 최종 `d7c746df` 트리에서 전체 workspace·all-target Clippy와 라이선스 헤더 검사가 통과했다.

사용자는 로컬 CI와 리뷰를 병합 기준으로 승인했다. 실제 체크포인트 출력, 시작 로그, Metal 처리량·로그잇 측정은 #2587에 남아 있다. native 표시는 실측 속도 향상을 의미하지 않는다.

## 학습 포인트

모든 norm 가중치가 1이면 잘못된 연산 순서가 드러나지 않을 수 있다. 서로 다른 학습 gain과 의미 있는 projection 가중치는 parity 테스트가 이러한 변화를 감지하도록 한다. 실제 래퍼 호출은 소스 문구 검사보다 강한 위임 증거다. 이슈 설명이 현재 코드와 달라졌다면 검증된 기존 동작을 보존하고 차이를 명시해야 한다.
