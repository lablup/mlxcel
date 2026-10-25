# 기술 보고서: PR #2595 — Long-tail dense 배치 디코드

**날짜:** 2026-10-25

**관련 이슈:** #2557, #2566, #2587

## 요약

이전에 행별 fallback으로 디코드하던 13개 계열이 native 배치 디코드 graph를 실행한다. 대상은 BitNet, GPT-2, GPT-BigCode, GPT-NeoX, MiniCPM, MiniMax M3, NanoChat, Nemotron, OLMo, OpenELM, Phi, Phi-3-small, StableLM이다. MiniMax M3 VL도 text-only decode를 같은 native text-tower 경로에 위임한다. B=1의 정확한 라우팅과 각 계열의 position, normalization, residual, scaling, attention 동작을 보존하면서 weight 사용량이 큰 projection을 전체 active-row 배치에서 실행한다.

“Long-tail dense”는 이슈 wave의 분류이지 모든 architecture가 문자 그대로 dense라는 뜻은 아니다. MiniMax M3는 현재 구현대로 block-sparse attention을 사용하는 hybrid dense/MoE decoder이며, MoE selection과 expert는 바꾸지 않고 attention과 sparse-index projection에 native batch 경로를 추가했다. #2587의 Metal 인수 검증 전에는 처리량 향상을 주장하지 않는다.

## 문제 정의

이 모델들은 batching 지원을 표시했지만 active row마다 single-sequence forward를 호출하는 trait fallback을 상속했다. 따라서 서버의 decode tick은 같은 layer weight를 다시 읽고 projection graph를 B번 만들었다. 공통 구현은 동시에 서로 다른 세부 동작을 보존해야 했다. GPT-2와 GPT-BigCode의 learned absolute position, GPT-NeoX·Phi·StableLM의 partial RoPE, depth 및 muP scaling, OpenELM의 layer별 head width, Phi-3-small의 dense와 block-sparse attention 혼합이 그 예다.

MiniMax M3에는 추가 제약이 있다. Sparse index key와 query를 배치에서 한 번 projection하되 올바른 요청별 cache row와 결합해야 한다. 기존 single-row sparse 경로에는 과거의 shape 오류도 있었다. Index-key head를 추가한 뒤 `KVCache`가 증가한 key-head 수로 value storage까지 할당해, attention이 실제 K/V head 하나와 value head 두 개를 받을 수 있었다.

## 기술적 선택과 그 이유

### 공통 split-attention 계약 보존

각 계열은 `SplitAttention`을 구현하고 family·layer 진단 정보를 포함하는 layer cache metadata를 만든 뒤, 별도 행 loop 대신 `forward_split_attention`을 사용한다. `route_batched_decode`는 B=0을 정의된 빈 결과로 보내고 B=1은 각 계열의 원래 single-sequence graph로 보낸다. B>=2 native closure는 embedding, normalization, QKV와 output projection, MLP, 최종 logits를 배치로 처리한다.

GPT-2와 GPT-BigCode는 설정된 position 한계에 가까운 offset을 포함해 서로 다른 cache offset에서 행별 learned position을 가져온다. GPT-NeoX, Phi, StableLM은 설정된 선두 RoPE 차원만 회전하고 passthrough channel은 bit-exact하게 유지한다. OpenELM은 model 전체의 고정 가정 대신 각 layer에서 split shape를 구해 layer마다 다른 query/KV head width를 보존한다. Phi-3-small은 주변 projection을 배치로 유지하면서 각 행을 해당 layer가 선택한 동일한 dense 또는 block-sparse attention 구현으로 보낸다.

### 하위 호환 indexed-row hook 추가

`SplitAttention::attend_row_indexed(row, ...)`의 기본 구현은 기존 `attend_row(...)`에 위임하므로 이전 구현은 모두 같은 동작을 유지한다. MiniMax M3만 이 hook을 override한다. Normalized batch에서 index-key와 index-query projection을 한 번 계산하고 공통 helper가 전달한 row index로 slice한 뒤, 기존 요청별 cache와 sparse-mask 로직을 적용한다.

MiniMax cache에는 scoring에 필요한 추가 index-key head를 계속 보관한다. Attention에 전달하는 value tensor view만 `num_kv_heads`로 다시 slice한다. 따라서 sparse index state를 제거하지 않으면서 관측된 single-row K/V head shape 실패와 새 batch 경로를 함께 수정한다. 이 수정된 single-row 경로는 나머지 B=1 계산이 유지되는 원칙의 의도적인 예외이며, 실제 MiniMax M3 VL wrapper도 수정된 native text model에 위임한다.

## 검증

Serial native-decode filter의 테스트 53개가 통과했다. 새 계열 regression은 유의미한 deterministic weight와 서로 다른 길이의 prompt history 3개를 사용하고, cache가 채워진 B=3 decode를 sequential 실행과 비교하며 exact B=1 routing과 B=0도 검증한다. 별도 focused check는 learned position, partial-RoPE passthrough, OpenELM layer별 shape, Phi-3-small dense·sparse layer, MiniMax mixed cache length, 실제 MiniMax M3 VL wrapper, 공통 indexed hook의 custom·default 동작을 다룬다. Coverage ledger 7개와 core indexed-hook filter 2개도 모두 통과했다.

첫 Phi-3-small fixture는 짧은 history에 대해 tiny block-window 설정이 퇴화해 sequential과 batched reference 양쪽에서 non-finite 값을 만들었다. 강화한 parity 진단은 finite max-absolute difference가 0임을 보여 batch만의 divergence가 아님을 확인했다. Fixture를 finite prefill history와 실제 block-sparse attention에 진입하는 decode 위치로 교체했고, 허용오차를 완화하지 않은 민감한 parity 테스트가 통과했다.

MiniMax value-view 수정 전 full-model witness는 sparse index head가 cache에 들어간 뒤 key/value head 불일치로 실패했다. Index-key cache head는 유지하고 attention value view만 줄인 뒤에는 서로 다른 prompt 길이 3개의 cached decode가 수정된 sequential reference와 일치했다. Sparse additive-mask 경로를 실제로 실행하며, 선택적인 fused paged 경로는 tiny test budget이 serviceable하지 않음을 올바르게 보고 fallback한다.

최종 repository source gate가 통과했다. Workspace crate version, 320개 `MLXCEL_*` 이름의 문서 coverage, environment-documentation 테스트 15개, Llama compatibility manifest와 validator case가 모두 통과했다. Warnings-denied 전체 workspace·all-target Clippy는 55.33초에 통과했고 format과 diff 검사도 통과했다. 최종 architecture, correctness, security, performance 리뷰에는 지적 사항이 없었다.

BitNet의 CPU-safe 테스트는 embedding, normalization, logits를 통해 native 분류, B=3 routing, exact B=1, B=0을 검증하지만 CPU 빌드에는 BitLinear ternary kernel이나 graph fallback이 없다. 전체 ternary-layer parity 테스트는 지원되는 Metal, CUDA, ROCm backend feature에서만 실행되도록 설정했다. 해당 runtime 근거와 실제 checkpoint logit trace, greedy generation, serving throughput은 #2587에 남아 있다.

## 학습 포인트

Batching marker는 weight 사용량이 큰 graph가 실제로 배치화되고 요청별 mutable state가 row와 정확히 연결될 때만 신뢰할 수 있다. Sparse indexer 같은 auxiliary projection은 row split 전에 실행하고, cache mutation과 행별 mask는 split 뒤에 남겨야 한다. 기본 구현이 있는 trait hook을 사용하면 관련 없는 계열이 기존 동작을 다시 구현하지 않아도 공통 메커니즘을 확장할 수 있다.

Parity 진단은 산술 차이와 잘못된 fixture를 구분해야 한다. Finite maximum absolute·relative difference를 표시해 초기 Phi-3-small 실패가 허용오차 문제가 아니라 양쪽의 공통 non-finite 출력임을 확인했다. 또한 exact B=1 호환은 명백히 shape-invalid한 과거 경로를 유지한다는 뜻이 아니다. MiniMax 수정은 이 예외를 명시적으로 기록하고 runtime으로 검증했다.

## 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 28 |
| 추가 라인 | 2,387 |
| 삭제 라인 | 35 |
| 추가된 native 계열 | 13 |
| 추가된 delegating wrapper | 1 |
| 커밋 | `3c58bcee` |

구현은 공통 batching helper, 13개 model runtime, MiniMax sparse index/cache adapter, MiniMax M3 VL wrapper, focused runtime test, classification ledger, continuous-batching 문서를 갱신한다. 필요한 후속 작업은 Metal 및 실제 checkpoint 인수 검증뿐이며 #2587에서 추적한다.
