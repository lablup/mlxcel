# 기술 보고서: PR #2591 — MLA 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2560, #2566, #2587

## 요약

DeepSeek V2/V3/V3.2, MiniCPM3, Mistral4, GLM-4 MoE Lite, Youtu에 native 다중 행 디코드를 추가하고 GLM MoE DSA와 DeepSeek-VL2는 변환된 내부 모델에 위임한다. projection과 feed-forward/expert 연산은 배치로 실행하며 요청별로 서로 다른 길이의 캐시를 유지한다. 최종 DeepSeek V2 수정은 absorbed·expanded 캐시 배치가 서로 교차해도 호환되는 absorbed 행의 folded projection 양쪽을 배치로 실행한다. Metal 성능 인수 검증은 #2587에 남아 있다.

## 문제 정의

MLA 캐시 배치는 dense split-attention 헬퍼의 단일 head dimension으로 표현할 수 없다. 일부 경로는 확장된 key/value를 저장하고 다른 경로는 압축 latent와 rotary key를 유지한다. DSA는 요청별 sparse 선택을 추가하며 캐시 정책에 따라 absorbed·expanded 행이 같은 배치에 섞일 수 있다.

## 변경 요약

Expanded 경로는 공통 배치 캐시 attention을 사용한다. 새 latent MLA 헬퍼는 행별 캐시 추가, 선택적 gather, 위치 점수 계산과 latent attention을 수행하며 `embed_q`, `unembed_out`, 출력 projection은 배치로 실행한다. 단일 행과 빈 배치는 공통 라우팅 계약을 사용한다.

DeepSeek V2는 FP16 latent와 Int8 expanded가 섞여도 행을 기존 absorbed·expanded 캐시 배치에 따라 안정적으로 나눈다. Absorbed 행은 folded query projection을 한 번에 실행하고 행별 캐시 갱신과 Stage 2 split-KV 선택 또는 fallback을 그대로 유지한 뒤, 결합한 latent 결과에 folded output projection을 한 번 실행한다. Expanded 행도 `kv_b_proj`와 attention을 배치로 실행한다. 결과를 원래 요청 순서로 복원하므로 absorbed/expanded/absorbed 배치에서도 출력 순서와 캐시 표현이 바뀌지 않는다. DeepSeek V3.2는 앞부분이 잘린 뒤의 실제 캐시 길이로 행별 top-k 선택을 수행하고 latent·rotary 캐시를 같은 인덱스로 모은다. MiniCPM3와 Mistral4는 rotary·residual·출력 scaling을 보존한다. GLM 일반 디코드는 별도의 MTP drafting 상태를 변경하지 않는다. 두 래퍼 모두 실제 forward 호출로 위임을 검증한다.

## 검증

CPU 집중 검사 15개는 경로 분류, expanded/absorbed ragged attention, 혼합 캐시 정책, trim 후 DSA dense/top-k, 7개 작은 전체 모델의 B>=3/B1/B0 하네스, 실제 래퍼 2개와 잘못된 입력의 진단을 검증했다. Core의 CPU-safe 검사는 단일 행 경로, latent 캐시 동작, causal mask, absorbed/decompressed parity도 검증한다. Preprojected one-chunk split 검사는 backend merge kernel 없이 새 경계를 고정하고, absorbed/expanded/absorbed 교차 회귀는 안정적인 그룹화, 배치 folded projection, 원래 순서 복원을 확인한다.

전체 `split_kv` 필터에서는 CPU-safe 검사 5개가 통과했다. Custom GPU merge 검사 4개는 `paged_attention_merge_states`의 CPU port가 없어 이 CPU 호스트에서 실행할 수 없다. 이 backend 경계와 실제 체크포인트 출력, Metal 로그잇 추적, 동시 요청 성능은 #2587로 명시적으로 이관했다. 포맷, 소스, 라이선스 검사가 통과했고 warnings-denied 전체 workspace·all-target Clippy도 32.90초에 통과했다.

리뷰에서 캐시 불일치 오류가 모델명, 요청 행, 처음 누락되거나 초과한 층을 식별하도록 강화했고 회귀 테스트도 층 번호를 확인한다. 최종 재리뷰에는 남은 문제가 없었다. 안정적인 그룹·순서, 배치 folded projection, 기존 Stage 2 행별 plan·fallback, 원래 단일 행 수학을 모두 보존했다. 승인된 병합 기준은 로컬 CI와 리뷰이며 native 분류는 실측 성능 향상을 주장하지 않는다.

## 기술적 선택과 학습 포인트

서로 다른 MLA 표현을 dense attention 인터페이스에 억지로 넣지 않는다. latent 경계의 캐시·attention 연산을 공유하고 주변의 가중치 연산을 배치화한다. 서로 다른 길이와 캐시 배치를 가진 요청에서는 projection 주변에서만 호환 행을 그룹화하고 캐시 변경과 score/softmax는 행별로 유지한 뒤 원래 순서를 복원한다. 캐시 trim 이후 논리적 위치와 실제 캐시 길이는 역할이 다르다. rotary offset은 전자를, DSA fallback·선택은 후자를 사용한다. 다른 요청 때문에 기존 행의 캐시 표현이 바뀌지 않도록 해야 한다.
