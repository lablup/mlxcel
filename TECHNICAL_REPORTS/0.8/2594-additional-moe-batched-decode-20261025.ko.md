# 기술 보고서: PR #2594 — 추가 MoE 배치 디코드

**날짜:** 2026-10-25
**관련 이슈:** #2559, #2566, #2587

## 요약

GLM-4 MoE, ERNIE 4.5 MoE, Hunyuan MoE, Bailing MoE, Dots1, MiniMax, Klear, Solar Open, Exaone MoE, Cohere 2 MoE에 네이티브 배치 디코드를 추가합니다. 새 커널 없이 기존 expert 구현이 펼쳐진 배치를 처리합니다. Exaone의 기존 trait 어댑터에서 발견한 캐시 손실도 수정합니다.

## 문제

행별 디코드는 projection과 expert 호출을 반복합니다. 배치화하면서 모델별 Q/K 정규화 순서, 부분 RoPE, 공유 expert, sliding attention, Hunyuan의 층간 K/V 공유와 요청별 캐시 분리를 유지해야 합니다.

## 변경 사항

공통 split attention이 projection을 배치 처리하고 길이가 다른 캐시 attention을 요청별로 수행합니다. Sliding 층은 기존 window 연산을 유지하며, 리뷰에서 Exaone과 Cohere의 전역 층은 공통 배치 캐시 디스패치를 사용하도록 수정했습니다. Hunyuan은 각 소유 층의 원래 batched K/V를 공유 층에 전달한 뒤 층별 RoPE·정규화를 적용하고, 각 층의 요청별 캐시를 따로 갱신합니다. 진단 오류에는 층 번호가 포함됩니다.

기존 Exaone trait 경로는 호출마다 임시 캐시를 새로 만들고 offset만 복사했습니다. 최초 회귀 테스트에서 예상 offset 4 대신 1이 관측됐습니다. 수정 경로는 표준 KV 캐시에 이력을 유지하며, window 4를 넘는 7단계 디코드에서 기존의 정상적인 persistent `AnyKVCache` 경로와 일치합니다. 따라서 과거의 B=1 출력 동일성은 의도적으로 보장하지 않으며, 단일 행 라우팅 계약은 수정된 경로를 기준으로 합니다. Sliding 층의 실제 저장 공간은 회전 버퍼의 O(window) 대신 O(context)이며 기존 KV 회계와 제한에 포함됩니다.

## 검증

10개 계열의 서로 다른 길이 3행·B1·B0 routed fixture를 포함한 native-batch 필터 52개가 통과했고 리뷰 수정 후에도 다시 통과했습니다. Coverage 8개, Hunyuan 소유 층 누락 진단, Exaone persistent-cache 기준 경로 테스트도 통과했습니다. Hunyuan은 소유·공유 층 두 그룹을 검사하고 GLM은 부분 RoPE와 Q/K 정규화를, sliding 계열은 window 경계를 검사합니다. 컴파일·포맷·소스 계약도 통과했습니다. 전체 workspace/all-target Clippy와 최종 리뷰 결과는 PR에 기록합니다.

Metal 커널, 실제 체크포인트, 서버 동시성 및 처리량은 #2587에서 검증합니다. 해당 이슈에 Exaone 기준선 예외를 명시했습니다. 사용자 지시에 따라 로컬 CI와 리뷰가 머지 기준이며 원격 CI를 기다리지 않습니다.

## 기술적 결정과 학습

배치와 단독 실행의 비교는 기존 단독 실행의 상태 손실도 드러낼 수 있습니다. 수정된 두 메서드끼리만 비교하지 않고 독립적인 정상 캐시 경로로 검증합니다. 행별 custom attention은 실제로 필요한 층에만 적용하여 전역 층의 공통 캐시 디스패치를 유지합니다.
